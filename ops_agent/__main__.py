"""命令行：列出 case、跑单个 case、导出状态图。

  python -m ops_agent --list
  python -m ops_agent --case case01
  python -m ops_agent --draw
"""
import argparse
import time
from collections.abc import Callable

from ops_agent.data import CATEGORY_NAMES, SCOPE_NAMES, load_cases


def progress(prefix: str = "") -> Callable[[str], None]:
    start = time.perf_counter()
    return lambda step: print(f"{prefix}[{time.perf_counter() - start:5.0f}s] {step}", flush=True)


def print_result(result: dict, case) -> None:
    report = result["report"]
    if report is None:
        print(f"\n没有得到结构化报告：{result['error']}")
    else:
        print(f"\n根因类别：{CATEGORY_NAMES[report.category]}（{report.category}）"
              f"    影响范围：{SCOPE_NAMES[report.scope]}（{report.scope}）    把握：{report.confidence}")
        print(f"根因：{report.root_cause}")
        print("证据：\n" + "\n".join(f"  - {e}" for e in report.evidence))
        print("建议：\n" + "\n".join(f"  - {a}" for a in report.suggested_actions))
    answer = case.answer
    print(f"\n参考答案：{CATEGORY_NAMES[answer.category]} / {SCOPE_NAMES[answer.scope]}。{answer.summary}")
    print(f"（LLM 调用 {result['llm_calls']} 次，{result['tokens']} tokens，{result['latency']:.1f}s）")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m ops_agent")
    parser.add_argument("--case", help="要排查的 case id，如 case01")
    parser.add_argument("--list", action="store_true", help="列出所有 case")
    parser.add_argument("--draw", action="store_true", help="导出状态图到 docs/ops_agent.mmd")
    args = parser.parse_args()
    cases = load_cases()

    if args.draw:
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from tools import export_graph

        from ops_agent.graph import build_graph
        export_graph(build_graph(llm=FakeListChatModel(responses=[""])), "ops_agent")
        return
    if args.list or not args.case:
        for c in cases.values():
            print(f"{c.id}  {c.alert.time:%m-%d %H:%M}  {c.alert.title}")
        return
    if args.case not in cases:
        parser.error(f"没有 {args.case}，可选：{', '.join(cases)}")

    from ops_agent.graph import ask, build_graph
    case = cases[args.case]
    print(f"告警：{case.alert.title}\n", flush=True)
    print_result(ask(build_graph(), case, on_step=progress()), case)


if __name__ == "__main__":
    main()
