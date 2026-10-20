"""用真实 LLM 跑全部 case，和标准答案比对。

  python -m ops_agent.evaluate
  python -m ops_agent.evaluate --case case01 case05 --pause 30

指标：根因类别、影响范围是否正确；报告里命中了多少条关键证据；标准答案要求的工具是否都调用过；
以及工具轮数、LLM 调用次数、tokens、耗时。结果保存在 ops_agent/results/。
"""
import argparse
import json
import time
from pathlib import Path

import config
import openai
import tracing

from ops_agent.__main__ import progress
from ops_agent.data import Case, load_cases
from ops_agent.graph import MAX_TOOL_ROUNDS, ask, build_graph

RESULTS_DIR = Path(__file__).parent / "results"


def evidence_hits(text: str, evidence: list[str]) -> list[bool]:
    text = text.lower()
    return [any(w.lower() in text for w in group.split("|")) for group in evidence]


def score(case: Case, r: dict) -> dict:
    answer, report = case.answer, r["report"]
    text = " ".join([report.root_cause, *report.evidence]) if report else ""
    hits = evidence_hits(text, answer.evidence)
    return {
        "case_id": case.id,
        "expected": f"{answer.category}/{answer.scope}",
        "got": f"{report.category}/{report.scope}" if report else "无报告",
        "category_ok": bool(report) and report.category == answer.category,
        "scope_ok": bool(report) and report.scope == answer.scope,
        "evidence_recall": sum(hits) / len(hits),
        "missed_evidence": [g for g, h in zip(answer.evidence, hits) if not h],
        "tools_ok": set(answer.required_tools) <= set(r["tools_used"]),
        "missed_tools": sorted(set(answer.required_tools) - set(r["tools_used"])),
        "tool_rounds": len(r["rounds"]),
        "tool_calls": sum(len(x) for x in r["rounds"]),
        "llm_calls": r["llm_calls"], "tokens": r["tokens"], "latency": round(r["latency"], 1),
        "rounds": r["rounds"], "conclusion": r["conclusion"], "error": r["error"],
        "report": report.model_dump() if report else None,
        "trace_id": r.get("trace_id"),
    }


def write_scores(row: dict) -> None:
    trace_id = row["trace_id"]
    tracing.score(trace_id, "category_ok", row["category_ok"], comment=f"{row['expected']} -> {row['got']}")
    tracing.score(trace_id, "scope_ok", row["scope_ok"])
    tracing.score(trace_id, "evidence_recall", row["evidence_recall"],
                  comment=f"缺证据 {row['missed_evidence']}" if row["missed_evidence"] else None)
    tracing.score(trace_id, "tools_ok", row["tools_ok"])


def run_case(app, case: Case, retries: int, run_id: str) -> dict:
    for attempt in range(retries + 1):
        try:
            return ask(app, case, on_step=progress("    "), session_id=run_id, tags=["eval"])
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError) as e:
            if attempt == retries:
                raise
            wait = 60 * (attempt + 1)
            print(f"\n  {type(e).__name__}，{wait}s 后重试", flush=True)
            time.sleep(wait)


def summarize(rows: list[dict]) -> dict:
    mean = lambda key: sum(r[key] for r in rows) / len(rows)
    return {k: round(mean(k), 3) for k in ("category_ok", "scope_ok", "evidence_recall", "tools_ok",
                                           "tool_rounds", "tool_calls", "llm_calls", "tokens", "latency")}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m ops_agent.evaluate")
    parser.add_argument("--case", nargs="*", help="只跑这些 case，默认全部")
    parser.add_argument("--pause", type=float, default=20, help="两个 case 之间暂停的秒数，避免触发限流")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    cases = load_cases()
    ids = args.case or list(cases)
    llm = config.get_llm()
    app = build_graph(llm)
    rows = []
    # 一次评测的所有 trace 归到同一个 Langfuse 会话里
    run_id = f"ops-eval-{time.strftime('%Y%m%d-%H%M%S')}"
    for i, cid in enumerate(ids):
        if i:
            time.sleep(args.pause)
        print(f"[{i + 1}/{len(ids)}] {cid} {cases[cid].alert.title}", flush=True)
        row = score(cases[cid], run_case(app, cases[cid], args.retries, run_id))
        write_scores(row)
        rows.append(row)
        print(f"  => {row['got']}（期望 {row['expected']}），证据 {row['evidence_recall']:.0%}，"
              f"{row['tool_rounds']} 轮 {row['tool_calls']} 次工具，{row['latency']}s", flush=True)

    print(f"\n{'case':<8}{'类别':>6}{'范围':>6}{'证据':>8}{'工具':>6}{'轮数':>6}{'LLM':>6}{'tokens':>9}{'耗时':>8}")
    for r in rows:
        print(f"{r['case_id']:<8}{'✓' if r['category_ok'] else '✗':>7}{'✓' if r['scope_ok'] else '✗':>7}"
              f"{r['evidence_recall']:>9.0%}{'✓' if r['tools_ok'] else '✗':>7}{r['tool_rounds']:>8}"
              f"{r['llm_calls']:>6}{r['tokens']:>9}{r['latency']:>9.1f}")
    summary = summarize(rows)
    print(f"\n类别准确率 {summary['category_ok']:.0%}，范围准确率 {summary['scope_ok']:.0%}，"
          f"证据召回 {summary['evidence_recall']:.0%}，必需工具覆盖 {summary['tools_ok']:.0%}；"
          f"平均 {summary['tool_rounds']} 轮工具、{summary['llm_calls']} 次 LLM、"
          f"{summary['tokens']:.0f} tokens、{summary['latency']:.1f}s")
    for r in rows:
        problems = []
        if not r["category_ok"] or not r["scope_ok"]:
            problems.append(f"{r['expected']} -> {r['got']}")
        if r["missed_evidence"]:
            problems.append(f"缺证据 {r['missed_evidence']}")
        if r["missed_tools"]:
            problems.append(f"没用工具 {r['missed_tools']}")
        if r["error"]:
            problems.append(f"报告失败 {r['error'][:80]}")
        if problems:
            print(f"- {r['case_id']}：{'；'.join(problems)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    meta = {
        "model": llm.model_name,
        "temperature": llm.temperature,
        "max_tool_rounds": MAX_TOOL_ROUNDS,
        "langfuse_session": run_id if tracing.enabled() else None,
    }
    out.write_text(json.dumps({"meta": meta, "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n模型 {meta['model']}，结果已保存到 {out}")
    if tracing.enabled():
        tracing.flush()
        print(f"Langfuse 会话：{run_id}（trace 上带 category_ok / scope_ok / evidence_recall / tools_ok 分数）")


if __name__ == "__main__":
    main()
