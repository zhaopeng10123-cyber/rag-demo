"""对比评估：chain（固定检索的 RAG 链） vs agent（create_agent） vs graph（LangGraph 状态图）。

用法（在 rag-demo 目录下）：
  python eval/evaluate_agent.py --mode all
  python eval/evaluate_agent.py --mode graph --limit 5     # 只跑前 5 题，调 Prompt 时用

题目来自 qa.jsonl（知识库题和库外题）+ qa_agent.jsonl（直答、库外、检索后拒答、多轮指代）。
路由分四类：retrieve / retrieve_refuse / direct / refuse。
"""
import argparse
import json
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(EVAL_DIR.parent), str(EVAL_DIR)]

import openai  # noqa: E402

import config  # noqa: E402
import tracing  # noqa: E402
from chat import build_rewrite_chain  # noqa: E402
from evaluate import RESULTS_DIR, first_hit_rank, key_point_coverage, load_qa, retrieval_metrics  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from rag_chain import build_answer_chain, doc_label, format_docs, get_retriever  # noqa: E402
from tools import AgentResult, CallCounter, is_refusal  # noqa: E402

ROUTES = ["retrieve", "retrieve_refuse", "direct", "refuse"]
REFUSE_ROUTES = {"refuse", "retrieve_refuse"}


class ChainRunner:
    """对照组：每个问题都检索；多轮时沿用 chat.py 的改写链。"""

    def __init__(self, llm) -> None:
        self.retriever = get_retriever()
        self.answer_chain = build_answer_chain(llm)
        self.rewrite_chain = build_rewrite_chain(llm)
        self.histories: dict[str, list] = {}

    def __call__(self, question: str, thread_id: str) -> AgentResult:
        counter = CallCounter()
        trace_id = tracing.new_trace_id()
        cfg = tracing.traced({"callbacks": [counter]}, "kb_chain", trace_id=trace_id, session_id=thread_id,
                             tags=["chain", "eval"])
        history = self.histories.setdefault(thread_id, [])
        start = time.perf_counter()
        query = question
        if history:
            query = self.rewrite_chain.invoke({"history": history, "question": question}, cfg).strip()
        docs = self.retriever.invoke(query, cfg)
        answer = self.answer_chain.invoke({"context": format_docs(docs), "question": query}, cfg)
        history += [HumanMessage(question), AIMessage(answer)]
        return AgentResult(route="retrieve_refuse" if is_refusal(answer) else "retrieve", queries=[query],
                           docs=docs, answer=answer, trace=["retrieve", "generate"], llm_calls=counter.calls,
                           tokens=counter.tokens, latency=time.perf_counter() - start, trace_id=trace_id)


def build_runner(mode: str, llm):
    if mode == "chain":
        return ChainRunner(llm)
    if mode == "agent":
        import agent
        app = agent.build_agent(llm=llm)
        return lambda q, tid: agent.ask(app, q, tid, tags=["eval"])
    import graph_agent
    app = graph_agent.build_graph(llm=llm)
    return lambda q, tid: graph_agent.ask(app, q, tid, tags=["eval"])


def load_items(limit: int | None) -> list[dict]:
    items = load_qa(EVAL_DIR / "qa.jsonl") + load_qa(EVAL_DIR / "qa_agent.jsonl")
    return items[:limit] if limit else items


def evaluate_mode(mode: str, items: list[dict], llm) -> dict:
    runner = build_runner(mode, llm)
    rows = []
    for i, item in enumerate(items, 1):
        print(f"\r[{mode}] {i}/{len(items)} {item['question'][:20]:<20}", end="", flush=True)
        thread_id = uuid.uuid4().hex
        try:
            for previous in item.get("history", []):
                runner(previous, thread_id)
            r = runner(item["question"], thread_id)
        except openai.APIStatusError as e:
            if e.status_code in (401, 402, 403, 429):
                raise SystemExit(f"\nLLM 接口不可用（{e.status_code}）：{e.message}")
            rows.append({"question": item["question"], "expected_route": item["expected_route"], "error": str(e)})
            continue

        expected = item["expected_route"]
        row = {
            "question": item["question"], "history": item.get("history", []),
            "expected_route": expected, "route": r["route"],
            "route_ok": r["route"] == expected,
            "refusal_ok": (r["route"] in REFUSE_ROUTES) == (expected in REFUSE_ROUTES),
            "queries": r["queries"], "trace": r["trace"], "answer": r["answer"],
            "retrieved": [doc_label(d) for d in r["docs"]],
            "llm_calls": r["llm_calls"], "tokens": r["tokens"], "latency": round(r["latency"], 2),
        }
        if expected == "retrieve":
            row["rank"] = first_hit_rank(r["docs"], item) if r["route"] == "retrieve" else None
            row["coverage"], row["missed"] = (key_point_coverage(r["answer"], item["key_points"])
                                              if r["route"] == "retrieve" else (0.0, item["key_points"]))
        if item.get("query_must_contain"):
            row["rewrite_ok"] = any(all(t.lower() in q.lower() for t in item["query_must_contain"])
                                    for q in r["queries"])
        row["trace_id"] = r["trace_id"]
        write_scores(row)
        rows.append(row)
    print()
    return {"mode": mode, "summary": summarize(rows), "rows": rows}


def write_scores(row: dict) -> None:
    trace_id = row["trace_id"]
    tracing.score(trace_id, "route_ok", row["route_ok"], comment=f"{row['expected_route']} -> {row['route']}")
    tracing.score(trace_id, "refusal_ok", row["refusal_ok"])
    if "rank" in row:
        tracing.score(trace_id, "retrieval_hit", row["rank"] is not None)
        tracing.score(trace_id, "key_point_coverage", row["coverage"],
                      comment=f"缺少 {row['missed']}" if row["missed"] else None)
    if "rewrite_ok" in row:
        tracing.score(trace_id, "rewrite_ok", row["rewrite_ok"])


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    retrieve_rows = [r for r in ok if r["expected_route"] == "retrieve"]
    rewrite_rows = [r for r in ok if "rewrite_ok" in r]
    k = config.TOP_K
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {
        "n": len(rows), "errors": len(rows) - len(ok),
        "route_accuracy": mean([r["route_ok"] for r in ok]),
        "refusal_accuracy": mean([r["refusal_ok"] for r in ok]),
        # 期望检索的题里，路由没走检索也算检索未命中
        **retrieval_metrics([r["rank"] for r in retrieve_rows], k),
        "key_point_coverage": mean([r["coverage"] for r in retrieve_rows]),
        "rewrite_accuracy": mean([r["rewrite_ok"] for r in rewrite_rows]),
        "avg_llm_calls": mean([r["llm_calls"] for r in ok]),
        "avg_tokens": mean([r["tokens"] for r in ok]),
        "avg_latency": mean([r["latency"] for r in ok]),
        "confusion": dict(Counter(f"{r['expected_route']}->{r['route']}" for r in ok)),
    }


def print_confusion(result: dict) -> None:
    conf = result["summary"]["confusion"]
    print(f"\n[{result['mode']}] 路由混淆矩阵（行=期望，列=实际）")
    print(f"{'':>16}" + "".join(f"{c:>16}" for c in ROUTES))
    for expected in ROUTES:
        print(f"{expected:>16}" + "".join(f"{conf.get(f'{expected}->{actual}', 0):>16}" for actual in ROUTES))


def print_failures(result: dict) -> None:
    print(f"\n[{result['mode']}] 失败用例")
    failed = False
    for r in result["rows"]:
        if "error" in r:
            problems = [f"报错 {r['error'][:80]}"]
        else:
            problems = []
            if not r["route_ok"]:
                problems.append(f"路由 {r['expected_route']} -> {r['route']}")
            if r["expected_route"] == "retrieve" and r["route"] == "retrieve" and r.get("rank") is None:
                problems.append("检索未命中期望章节")
            if r.get("missed") and r["route"] == "retrieve":
                problems.append(f"缺少关键点 {r['missed']}")
            if r.get("rewrite_ok") is False:
                problems.append(f"指代改写失败 {r['queries']}")
        if problems:
            failed = True
            print(f"- {r['question']}：{'；'.join(problems)}")
            if r.get("trace"):
                print(f"    轨迹：{' -> '.join(r['trace'])}")
    if not failed:
        print("  无")


def print_comparison(results: list[dict]) -> None:
    k = config.TOP_K
    metrics = ["route_accuracy", "refusal_accuracy", f"hit@{k}", f"mrr@{k}", "key_point_coverage",
               "rewrite_accuracy", "avg_llm_calls", "avg_tokens", "avg_latency", "errors"]
    print("\n=== 三种模式对比 ===")
    print(f"{'指标':<20}" + "".join(f"{r['mode']:>12}" for r in results))
    for m in metrics:
        print(f"{m:<20}" + "".join(f"{r['summary'][m]:>12.3f}" if isinstance(r['summary'][m], float)
                                    else f"{r['summary'][m]:>12}" for r in results))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["chain", "agent", "graph", "all"], default="all")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    llm = config.get_llm()
    items = load_items(args.limit)
    modes = ["chain", "agent", "graph"] if args.mode == "all" else [args.mode]
    results = [evaluate_mode(m, items, llm) for m in modes]

    for result in results:
        print_confusion(result)
        print_failures(result)
    print_comparison(results)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"agent-{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n结果已保存到 {out}")
    tracing.flush()


if __name__ == "__main__":
    main()
