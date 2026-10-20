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

from ops_agent.__main__ import progress
from ops_agent.data import Case, load_cases
from ops_agent.graph import ask, build_graph

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
    }


def run_case(app, case: Case, retries: int) -> dict:
    for attempt in range(retries + 1):
        try:
            return ask(app, case, on_step=progress("    "))
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
    app = build_graph(config.get_llm())
    rows = []
    for i, cid in enumerate(ids):
        if i:
            time.sleep(args.pause)
        print(f"[{i + 1}/{len(ids)}] {cid} {cases[cid].alert.title}", flush=True)
        row = score(cases[cid], run_case(app, cases[cid], args.retries))
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
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存到 {out}")


if __name__ == "__main__":
    main()
