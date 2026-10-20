"""ops_agent 的离线测试：case 数据自洽且可解、诊断工具行为、ReAct 图的三种收尾路径。"""
import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from ops_agent.data import CATEGORY_NAMES, DEFAULT_METRICS, HISTORY_MINUTES, load_cases, series
from ops_agent.diag_tools import TOOLS, OpsContext
from ops_agent.evaluate import score
from ops_agent.graph import REPORT_PROMPT, RootCauseReport, alert_message, ask, build_graph
from tests.fake_models import ScriptedChatModel, tool_call

CASES = load_cases()
TOOL_NAMES = {t.name for t in TOOLS}


@pytest.fixture(scope="module")
def tool_app():
    graph = StateGraph(MessagesState, context_schema=OpsContext)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    return graph.compile()


@pytest.fixture
def run(tool_app):
    """经 ToolNode 调用工具，和图里一样由 context 注入 case。"""
    def call(case_id: str, name: str, **args) -> str:
        state = tool_app.invoke({"messages": [tool_call(name, **args)]}, context=OpsContext(case_id))
        return state["messages"][-1].content
    return call


def test_cases_consistent():
    assert len(CASES) == 7
    assert {c.answer.category for c in CASES.values()} == set(CATEGORY_NAMES)
    for c in CASES.values():
        a = c.alert
        assert c.host(a.host) and a.metric in DEFAULT_METRICS, c.id
        assert set(c.answer.required_tools) <= TOOL_NAMES, c.id
        assert set(c.metrics) | set(c.logs) <= {h.name for h in c.hosts}, c.id
        assert all(h in {x.name for x in c.hosts} for ch in c.changes for h in ch.hosts), c.id
        # 告警值和生成的指标对得上，且确实超过阈值
        now = series(c.id, a.host, a.metric)[-5:]
        assert min(now) > a.threshold, c.id
        assert abs(sum(now) / 5 - a.value) / a.value < 0.08, c.id


def test_series_deterministic_and_bounded():
    a = series("case01", "web-03", "cpu_usage")
    series.cache_clear()
    assert series("case01", "web-03", "cpu_usage") == a
    assert len(a) == HISTORY_MINUTES and max(a) <= 100
    # case07 的 200% 是故意构造的采集错误，不截断
    assert max(series("case07", "cache-02", "mem_usage")) > 190


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_case_solvable(run, case_id):
    """只按"告警主机 → 服务 → 日志里出现的对象"这条路线调用工具，标准答案要求的证据都能查到。"""
    c = CASES[case_id]
    a = c.alert
    service = c.host(a.host).service
    outputs = [
        run(case_id, "query_metrics", host=a.host, metric=a.metric, minutes=4320),
        run(case_id, "compare_peers", host=a.host, metric=a.metric),
        run(case_id, "search_logs", host=a.host, minutes=4320),
        run(case_id, "list_changes", target=a.host),
        run(case_id, "list_changes", target=service),
    ]
    seen = "\n".join(outputs)
    for target in {ch.target for ch in c.changes} - {h.name for h in c.hosts} - {h.service for h in c.hosts}:
        if target in seen:
            outputs.append(run(case_id, "list_changes", target=target))
    text = "\n".join(outputs).lower()
    for group in c.answer.evidence:
        assert any(word.lower() in text for word in group.split("|")), f"{case_id} 查不到证据 {group}"


def test_unknown_host_and_metric(run):
    assert "可查询的主机" in run("case01", "query_metrics", host="web-99", metric="cpu_usage")
    assert "可用指标" in run("case01", "compare_peers", host="web-03", metric="cpu")
    assert "已知的主机和服务" in run("case01", "list_changes", target="redis-x")


def test_query_metrics_shape(run):
    out = run("case06", "query_metrics", host="pg-03", metric="disk_io_util", minutes=60)
    assert "最近 5 分钟均值 9" in out and "前一天同时段均值" in out
    segments = re.findall(r"^  \d\d-\d\d \d\d:\d\d", out, re.M)
    assert len(segments) == 12 and segments[0].strip() == "10-19 01:13"
    assert "前一天同时段" not in run("case06", "query_metrics", host="pg-03", metric="disk_io_util", minutes=4320)


def test_list_changes_matching(run):
    by_host = run("case05", "list_changes", target="inv-01")
    assert "重启" in by_host and "ALTER" not in by_host
    assert "ALTER TABLE" in run("case05", "list_changes", target="mysql-inv-01")
    assert "没有" in run("case05", "list_changes", target="inv-01", hours=1)
    # 按主机查只返回涉及该主机的服务级变更；按服务查能看到灰度到其他主机的变更；新的在前
    out = run("case01", "list_changes", target="web-01")
    assert "nginx" in out and "v2.3.1" not in out
    out = run("case01", "list_changes", target="web-03")
    assert out.index("v2.3.1") < out.index("nginx")
    assert "pay-api" not in out and "pay-api" in run("case01", "list_changes", target="pay-api")


def test_search_logs_filters(run):
    assert "Out of memory" not in run("case02", "search_logs", host="batch-02")
    old = run("case02", "search_logs", host="batch-02", level="ERROR", minutes=4320)
    assert old.count("Out of memory") == 3 and "WARN" not in old
    assert "ExportCache" in run("case02", "search_logs", host="batch-02", keyword="exportcache")
    assert "没有匹配" in run("case02", "search_logs", host="batch-01")


def test_compare_peers_groups(run):
    out = run("case04", "compare_peers", host="pay-02", metric="net_latency_ms")
    service, az = out.split("同可用区 az-c 的其他服务：")
    assert "pay-02（参照）" in service and "pay-01" in service
    assert "user-04" in az and "pay-" not in az
    # 缓慢上涨的问题要靠 1 天前的值才看得出其他主机也在涨
    assert re.search(r"log-01 .*1 小时前 8\d\.\d+，1 天前 6\d\.", run("case03", "compare_peers", host="log-02",
                                                                metric="disk_usage"))


def report(**kw) -> RootCauseReport:
    fields = dict(category="downstream_dependency", scope="whole_service", root_cause="mysql-inv-01 在线 DDL",
                  evidence=["slow query on mysql-inv-01"], suggested_actions=["暂停 DDL"], confidence="high")
    return RootCauseReport(**fields | kw)


def test_graph_react_loop():
    llm = ScriptedChatModel(responses=[
        tool_call("compare_peers", host="inv-01", metric="http_p99_ms"),
        tool_call("search_logs", "call_2", host="inv-01"),
        AIMessage("三台都慢，日志里是 mysql-inv-01 的慢查询。"),
        report(),
    ])
    result = ask(build_graph(llm), CASES["case05"])
    assert result["rounds"] == [["compare_peers(host=inv-01, metric=http_p99_ms)"], ["search_logs(host=inv-01)"]]
    assert result["tools_used"] == ["compare_peers", "search_logs"]
    assert result["conclusion"].startswith("三台都慢")
    assert result["report"].category == "downstream_dependency" and result["error"] == ""
    assert llm.bound_tools[0] == [t.name for t in TOOLS]
    # 系统提示之后是告警，工具结果来自 case05 的数据（context 注入生效）
    _, first, *_, last = llm.received[-1]
    assert isinstance(first, HumanMessage) and first.content == alert_message(CASES["case05"])
    assert isinstance(last, ToolMessage) and "slow query" in last.content


def test_graph_round_limit_goes_to_report():
    calls = [tool_call("query_metrics", f"call_{i}", host="inv-01", metric="cpu_usage") for i in range(2)]
    llm = ScriptedChatModel(responses=[*calls, report()])
    result = ask(build_graph(llm, max_tool_rounds=2), CASES["case05"])
    # 到达上限后 tools 直接进 report，agent 只被调用了 2 次
    assert len(result["rounds"]) == 2 and result["conclusion"] == ""
    assert result["report"] is not None and len(llm.received) == 2


def test_graph_report_failure():
    llm = ScriptedChatModel(responses=[AIMessage("直接结论"), AIMessage("不是结构化输出")])
    result = ask(build_graph(llm), CASES["case07"])
    assert result["rounds"] == [] and result["report"] is None
    assert "AssertionError" in result["error"]


def test_score():
    result = {"report": report(scope="single_host"), "rounds": [["search_logs(host=inv-01)"]],
              "tools_used": ["search_logs"], "conclusion": "", "error": "", "llm_calls": 3, "tokens": 100,
              "latency": 1.0}
    row = score(CASES["case05"], result)
    assert row["category_ok"] and not row["scope_ok"]
    assert row["evidence_recall"] == 1.0 and row["missed_tools"] == ["compare_peers"]
    empty = score(CASES["case05"], result | {"report": None, "error": "x"})
    assert empty["got"] == "无报告" and empty["evidence_recall"] == 0


def test_report_prompt_sent():
    seen = []

    class Recorder(ScriptedChatModel):
        def with_structured_output(self, schema, **kwargs):
            inner = super().with_structured_output(schema, **kwargs)
            return inner.with_listeners(on_start=lambda run: seen.append(run.inputs))

    ask(build_graph(Recorder(responses=[AIMessage("结论"), report()])), CASES["case01"])
    messages = seen[0]["input"]
    assert messages[-1].content == REPORT_PROMPT
