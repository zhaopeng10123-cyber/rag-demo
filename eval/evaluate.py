"""评估 RAG：检索指标（Hit Rate@k、MRR）+ 回答指标（关键点覆盖率、拒答准确率、可选忠实度）。

用法（在 rag-demo 目录下）：
  python eval/evaluate.py                 # 检索 + 回答（未配置 LLM_API_KEY 时只评检索）
  python eval/evaluate.py --retrieval-only
  python eval/evaluate.py --judge         # 额外让 LLM 给每个答案的忠实度打分
  python eval/evaluate.py --experiment    # 对比 chunk_size x k x 是否加检索指令（只评检索）
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from rag_chain import REFUSAL, build_answer_chain, format_docs  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

JUDGE_PROMPT = """请判断【答案】中的每个事实陈述是否都能在【参考资料】中找到依据。
只输出一个 0 到 1 之间的小数：1 表示完全有依据，0 表示完全是编造，不要输出其他内容。

【参考资料】
{context}

【答案】
{answer}"""


def load_qa(path: Path = EVAL_DIR / "qa.jsonl") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_hit(doc: Document, item: dict) -> bool:
    sources = item["expected_source"]
    sources = [sources] if isinstance(sources, str) else sources
    return (doc.metadata.get("source") in sources
            and any(s in doc.metadata.get("section", "") for s in item["expected_section"]))


def first_hit_rank(docs: list[Document], item: dict) -> int | None:
    return next((i for i, d in enumerate(docs, 1) if is_hit(d, item)), None)


def retrieval_metrics(ranks: list[int | None], k: int) -> dict:
    n = len(ranks)
    return {
        f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n,
        f"mrr@{k}": sum(1 / r for r in ranks if r is not None and r <= k) / n,
    }


def search(store, embed_query, question: str, k: int) -> list[Document]:
    vec = embed_query(question)
    return [d for d, _ in store.similarity_search_by_vector_with_relevance_scores(vec, k=k)]


def key_point_coverage(answer: str, key_points: list[str]) -> tuple[float, list[str]]:
    text = answer.lower()
    missed = [kp for kp in key_points if not any(alt.lower() in text for alt in kp.split("|"))]
    return (1 - len(missed) / len(key_points)) if key_points else 1.0, missed


def judge_faithfulness(llm, context: str, answer: str) -> float | None:
    reply = llm.invoke(JUDGE_PROMPT.format(context=context, answer=answer)).content
    m = re.search(r"[01](?:\.\d+)?", reply)
    return float(m.group()) if m else None


def evaluate(k: int, with_llm: bool, judge: bool) -> dict:
    qa = load_qa()
    store = config.get_vectorstore()
    embed_query = config.get_embeddings().embed_query
    llm = config.get_llm() if with_llm else None
    answer_chain = build_answer_chain(llm) if llm else None

    rows = []
    for item in qa:
        docs = search(store, embed_query, item["question"], k)
        row = {"question": item["question"], "answerable": item["answerable"],
               "retrieved": [f"{d.metadata['source'][:2]} {d.metadata['section']}" for d in docs]}
        if item["answerable"]:
            row["rank"] = first_hit_rank(docs, item)
        if answer_chain:
            context = format_docs(docs)
            answer = answer_chain.invoke({"context": context, "question": item["question"]})
            refused = REFUSAL in answer
            row.update(answer=answer, refused=refused, refusal_correct=refused != item["answerable"])
            if item["answerable"]:
                row["coverage"], row["missed"] = key_point_coverage(answer, item["key_points"])
            if judge and item["answerable"] and not refused:
                row["faithfulness"] = judge_faithfulness(llm, context, answer)
        rows.append(row)

    answerable = [r for r in rows if r["answerable"]]
    summary = {"k": k, "chunk_size": config.CHUNK_SIZE, "n": len(rows),
               **retrieval_metrics([r["rank"] for r in answerable], k)}
    if answer_chain:
        summary["key_point_coverage"] = sum(r["coverage"] for r in answerable) / len(answerable)
        summary["refusal_accuracy"] = sum(r["refusal_correct"] for r in rows) / len(rows)
        scores = [r["faithfulness"] for r in rows if r.get("faithfulness") is not None]
        if scores:
            summary["faithfulness"] = sum(scores) / len(scores)
    return {"summary": summary, "rows": rows}


def print_report(result: dict) -> None:
    s = result["summary"]
    print("\n=== 汇总 ===")
    for key, value in s.items():
        print(f"  {key:<20} {value:.3f}" if isinstance(value, float) else f"  {key:<20} {value}")

    print("\n=== 失败用例 ===")
    failed = False
    for r in result["rows"]:
        problems = []
        if r["answerable"] and r.get("rank") is None:
            problems.append("检索未命中期望章节")
        if r.get("refusal_correct") is False:
            problems.append("误拒答" if r["answerable"] else "应拒答却作答")
        if r.get("missed"):
            problems.append(f"缺少关键点 {r['missed']}")
        if r.get("faithfulness") is not None and r["faithfulness"] < 0.8:
            problems.append(f"忠实度 {r['faithfulness']}")
        if problems:
            failed = True
            print(f"- {r['question']}\n    问题：{'；'.join(problems)}")
            print(f"    检索到：{r['retrieved']}")
    if not failed:
        print("  无")


def run_experiment(chunk_sizes=(300, 500, 800), ks=(3, 5)) -> list[dict]:
    from ingest import build_index
    from langchain_huggingface import HuggingFaceEmbeddings

    qa = [q for q in load_qa() if q["answerable"]]
    with_instr = config.get_embeddings().embed_query
    no_instr = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL, encode_kwargs={"normalize_embeddings": True}
    ).embed_query

    table = []
    for cs in chunk_sizes:
        collection = f"exp_cs{cs}"
        store, chunks = build_index(cs, int(cs * 0.16), collection, verbose=False)
        for instr_name, embed_query in [("加指令", with_instr), ("不加", no_instr)]:
            ranks = [first_hit_rank(search(store, embed_query, q["question"], max(ks)), q) for q in qa]
            for k in ks:
                table.append({"chunk_size": cs, "片段数": len(chunks), "检索指令": instr_name, "k": k,
                              **retrieval_metrics(ranks, k)})
        store.delete_collection()

    print(f"\n=== 检索对比实验（可回答题 {len(qa)} 道）===")
    print(f"{'chunk_size':>10} {'片段数':>6} {'检索指令':>6} {'k':>3} {'Hit Rate':>9} {'MRR':>7}")
    for row in table:
        hit, mrr = (v for key, v in row.items() if key.startswith(("hit", "mrr")))
        print(f"{row['chunk_size']:>10} {row['片段数']:>8} {row['检索指令']:>6} {row['k']:>3} {hit:>9.3f} {mrr:>7.3f}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--experiment", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    if args.experiment:
        table = run_experiment()
        out = RESULTS_DIR / f"experiment-{stamp}.json"
        out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已保存到 {out}")
        return

    with_llm = not args.retrieval_only
    if with_llm:
        try:
            config.get_llm()
        except RuntimeError as e:
            print(f"（{e}，本次只评估检索）")
            with_llm = False

    result = evaluate(args.k, with_llm, args.judge)
    print_report(result)
    out = RESULTS_DIR / f"eval-{stamp}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存到 {out}")


if __name__ == "__main__":
    main()
