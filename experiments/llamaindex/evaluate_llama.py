"""对照实验：LangChain vs LlamaIndex 的检索效果，以及 reranker、sentence-window 两个检索技巧。

所有实验组共用：eval/evaluate.py 的评估函数（load_qa / first_hit_rank / retrieval_metrics）、
同一个 Embedding 模型和检索指令、同一份知识库，chunk_size 都设为 300。

用法（在 rag-demo 目录下）：
  python experiments/llamaindex/evaluate_llama.py             # 全部实验组，只评检索，不需要 LLM
  python experiments/llamaindex/evaluate_llama.py --answers   # 再用 LLM 对比两个框架的问答
"""
import argparse
import json
import sys
import time
from collections.abc import Callable
from functools import lru_cache
from math import comb
from pathlib import Path
from statistics import median

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parent.parent
sys.path[:0] = [str(EXP_DIR), str(ROOT)]

import chromadb  # noqa: E402
import config  # noqa: E402
import llama_rag  # noqa: E402
from eval.evaluate import EVAL_DIR, first_hit_rank, key_point_coverage, load_qa, retrieval_metrics  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank  # noqa: E402
from llama_index.core.schema import NodeWithScore, TextNode  # noqa: E402

Retrieve = Callable[[str, int], list[Document]]
KS = (3, 5)
CANDIDATES = 20  # reranker 先粗召回 20 条再重排；只重排 top-5 的话 Hit@5 从定义上就不会变
RERANKER = ROOT / "models" / "bge-reranker-base"
RESULTS_DIR = EXP_DIR / "results"
# 评估不需要落盘；同一进程里反复删建多个持久化集合时，Chroma 偶尔报 "hnsw segment reader: Nothing found on disk"
CLIENT = chromadb.EphemeralClient()


def load_items() -> dict[str, list[dict]]:
    sets = {"原评估集": [q for q in load_qa(EVAL_DIR / "qa.jsonl") if q["answerable"]]}
    extra = EXP_DIR / "qa_extra.jsonl"
    if extra.exists():
        sets["补充集"] = load_qa(extra)
        sets["合计"] = sets["原评估集"] + sets["补充集"]
    return sets


# ---------- 实验组 ----------

def langchain_arm() -> Retrieve:
    from ingest import load_markdown, split_documents
    from langchain_chroma import Chroma

    kwargs = dict(client=CLIENT, collection_name="lc_300", embedding_function=config.get_embeddings(),
                  collection_configuration=llama_rag.HNSW)
    Chroma(**kwargs).delete_collection()
    store = Chroma(**kwargs)
    store.add_documents(split_documents(load_markdown(), llama_rag.CHUNK_SIZE, llama_rag.CHUNK_OVERLAP))
    return lambda q, k: [d for d, _ in store.similarity_search_with_score(q, k=k)]


def llama_arm(nodes, collection: str, postprocessors=()) -> Retrieve:
    index = llama_rag.build_index(nodes, collection, client=CLIENT)

    def retrieve(q: str, k: int) -> list[Document]:
        found = index.as_retriever(similarity_top_k=k).retrieve(q)
        for p in postprocessors:
            found = p.postprocess_nodes(found, query_str=q)
        return [llama_rag.to_document(n) for n in found]
    retrieve.index = index  # 问答对比要用同一个索引建 query engine
    return retrieve


@lru_cache
def get_reranker() -> SentenceTransformerRerank:
    if not RERANKER.is_dir():
        raise SystemExit(f"缺少 reranker 模型 {RERANKER}，先下载 BAAI/bge-reranker-base")
    return SentenceTransformerRerank(model=str(RERANKER), top_n=CANDIDATES)


def with_rerank(base: Retrieve) -> Retrieve:
    """同一个 reranker 挂在任意实验组后面，LangChain 组也一样，这样比较的才是 reranker 本身。"""
    def retrieve(q: str, k: int) -> list[Document]:
        nodes = [NodeWithScore(node=TextNode(text=d.page_content, metadata=d.metadata)) for d in base(q, CANDIDATES)]
        return [llama_rag.to_document(n) for n in get_reranker().postprocess_nodes(nodes, query_str=q)[:k]]
    return retrieve


def build_arms(only: set[str] | None) -> dict[str, tuple[str, Retrieve]]:
    window = MetadataReplacementPostProcessor(target_metadata_key="window")
    specs = {
        "lc": ("LangChain 基线（主线 ingest.py 的切分）", langchain_arm),
        "li_aligned": ("LlamaIndex 对齐版：复用 LangChain 的片段",
                       lambda: llama_arm(llama_rag.aligned_nodes(), "li_aligned")),
        "li_native": ("LlamaIndex 惯用管线（默认把文件路径拼进向量化文本）",
                      lambda: llama_arm(llama_rag.native_nodes(), "li_native")),
        "li_native_nopath": ("LlamaIndex 惯用管线，去掉文件路径",
                             lambda: llama_arm(llama_rag.native_nodes(keep_file_path=False), "li_native_nopath")),
        "sentence_window": ("LlamaIndex sentence-window（命中小句，返回前后 3 句）",
                            lambda: llama_arm(llama_rag.sentence_window_nodes(), "li_window", [window])),
    }
    arms = {}
    for name, (desc, build) in specs.items():
        if only and name not in only and f"{name}+rerank" not in only:
            continue
        start = time.perf_counter()
        arms[name] = (desc, build())
        print(f"  建好 {name}（{time.perf_counter() - start:.0f}s）")
    for name in [n for n in list(arms) if n in ("lc", "li_native_nopath", "sentence_window")]:
        if not only or f"{name}+rerank" in only:
            arms[f"{name}+rerank"] = (f"{name} + bge-reranker-base（先召回 {CANDIDATES} 条）", with_rerank(arms[name][1]))
    if only:
        arms = {n: a for n, a in arms.items() if n in only}
    return arms


# ---------- 统计 ----------

def reciprocal(rank: int | None) -> float:
    return 1 / rank if rank else 0.0


def sign_test(wins: int, losses: int) -> float:
    """双侧符号检验：排名变好和变差的题数是否显著不同（持平的题不计入）。"""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(base: list[int | None], other: list[int | None]) -> dict:
    wins = sum(reciprocal(o) > reciprocal(b) for b, o in zip(base, other))
    losses = sum(reciprocal(o) < reciprocal(b) for b, o in zip(base, other))
    return {"变好": wins, "变差": losses, "持平": len(base) - wins - losses, "p": round(sign_test(wins, losses), 3)}


def evaluate_retrieval(arms: dict, sets: dict[str, list[dict]]) -> dict:
    items = sets.get("合计", sets["原评估集"])
    ranks = {}
    for name, (_, retrieve) in arms.items():
        start = time.perf_counter()
        ranks[name] = [first_hit_rank(retrieve(q["question"], max(KS)), q) for q in items]
        print(f"  检索完成 {name}（{time.perf_counter() - start:.1f}s）")

    report = {}
    for set_name, set_items in sets.items():
        idx = [items.index(q) for q in set_items]
        rows = {}
        for name in arms:
            r = [ranks[name][i] for i in idx]
            row = {k: v for kk in KS for k, v in retrieval_metrics(r, kk).items()}
            base = "lc" if "+rerank" not in name else name.replace("+rerank", "")
            if name != "lc" and base in ranks:
                row[f"对比 {base}"] = paired([ranks[base][i] for i in idx], r)
            rows[name] = row
        report[set_name] = {"n": len(set_items), "rows": rows}
    per_question = [{"question": q["question"], **{n: ranks[n][i] for n in arms}} for i, q in enumerate(items)]
    return {"report": report, "per_question": per_question}


def print_retrieval(result: dict, arms: dict) -> None:
    for set_name, block in result["report"].items():
        print(f"\n=== {set_name}（{block['n']} 题）===")
        print(f"| 实验组 | Hit@3 | MRR@3 | Hit@5 | MRR@5 | 逐题对比（变好/变差/持平，符号检验 p） |")
        print("|---|---:|---:|---:|---:|---|")
        for name, row in block["rows"].items():
            cmp = next(((k, v) for k, v in row.items() if k.startswith("对比")), None)
            cmp_text = f"{cmp[0]}：{cmp[1]['变好']}/{cmp[1]['变差']}/{cmp[1]['持平']}，p={cmp[1]['p']}" if cmp else "—"
            print(f"| {name} | {row['hit@3']:.3f} | {row['mrr@3']:.3f} | {row['hit@5']:.3f} | {row['mrr@5']:.3f} "
                  f"| {cmp_text} |")
    print("\n实验组说明：")
    for name, (desc, _) in arms.items():
        print(f"  {name:<24} {desc}")
    changed = [q for q in result["per_question"] if len({v for k, v in q.items() if k != "question"}) > 1]
    print(f"\n各组首个命中排名不同的题（{len(changed)} 道，None 表示前 5 名未命中）：")
    for q in changed:
        print(f"  {q['question'][:28]:<30} " + "  ".join(f"{k}={v}" for k, v in q.items() if k != "question"))


# ---------- 问答对比（需要 LLM） ----------

def evaluate_answers(lc_retrieve: Retrieve, li_index) -> dict:
    from rag_chain import REFUSAL, build_answer_chain, format_docs

    items = load_qa(EVAL_DIR / "qa.jsonl")
    lc_chain = build_answer_chain(config.get_llm())
    engine = llama_rag.build_query_engine(li_index)
    rows = []
    for i, q in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {q['question']}")
        row = {"question": q["question"], "answerable": q["answerable"]}
        for name, answer_fn in [
            ("langchain", lambda: lc_chain.invoke({"context": format_docs(lc_retrieve(q["question"], config.TOP_K)),
                                                   "question": q["question"]})),
            ("llamaindex", lambda: str(engine.query(q["question"]))),
        ]:
            start = time.perf_counter()
            answer = answer_fn().strip()
            refused = REFUSAL in answer[:40]
            entry = {"answer": answer, "refusal_correct": refused != q["answerable"],
                     "latency": round(time.perf_counter() - start, 1)}
            if q["answerable"]:
                entry["coverage"], entry["missed"] = key_point_coverage(answer, q["key_points"])
            row[name] = entry
        rows.append(row)

    summary = {}
    for name in ("langchain", "llamaindex"):
        answerable = [r[name] for r in rows if r["answerable"]]
        summary[name] = {
            "key_point_coverage": round(sum(e["coverage"] for e in answerable) / len(answerable), 3),
            "refusal_accuracy": round(sum(r[name]["refusal_correct"] for r in rows) / len(rows), 3),
            "avg_latency": round(sum(r[name]["latency"] for r in rows) / len(rows), 1),
            # 免费接口偶尔排队一两分钟，平均值会被个别请求拉高
            "median_latency": round(median(r[name]["latency"] for r in rows), 1),
        }
    print("\n=== 问答对比（qa.jsonl 全部题目）===")
    for name, s in summary.items():
        print(f"  {name:<11} 关键点覆盖率 {s['key_point_coverage']:.3f}  拒答准确率 {s['refusal_accuracy']:.3f}  "
              f"耗时中位数 {s['median_latency']}s（平均 {s['avg_latency']}s）")
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", help="只跑指定的实验组，例如 lc li_native lc+rerank")
    parser.add_argument("--answers", action="store_true", help="额外用 LLM 对比两个框架的问答")
    args = parser.parse_args()

    sets = load_items()
    print(f"评估集：" + "，".join(f"{k} {len(v)} 题" for k, v in sets.items()))
    arms = build_arms(set(args.only) if args.only else None)
    result = {"retrieval": evaluate_retrieval(arms, sets)}
    print_retrieval(result["retrieval"], arms)

    if args.answers:
        if not {"lc", "li_native_nopath"} <= set(arms):
            raise SystemExit("--answers 需要 lc 和 li_native_nopath 两个实验组")
        result["answers"] = evaluate_answers(arms["lc"][1], arms["li_native_nopath"][1].index)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"compare-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存到 {out}")


if __name__ == "__main__":
    main()
