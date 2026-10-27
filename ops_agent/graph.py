"""手写 ReAct 状态图：模型自己决定调哪些诊断工具，查够了再由单独的节点产出结构化根因报告。

    START -> agent --有工具调用--> tools --未到轮数上限--> agent
               |                     |--到达上限--> report
               |--不再调用工具--> report -> scope_rule -> END

agent 只负责"想下一步查什么"，report 只负责"把查到的东西整理成固定结构"。拆开后，
排查过程可以用自由文本推理，报告字段又能稳定地拿来评估；轮数上限保证一定能收敛。
scope_rule 按主机数据规则重算影响范围，报告里模型自己给的范围保留不动，两者可以对照。
"""
import time
from collections.abc import Callable
from typing import Literal

import config
import openai
import tracing
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field
from tools import CallCounter

from ops_agent.data import Case, Category, Scope, category_choices, load_cases, scope_choices
from ops_agent.diag_tools import TOOLS, OpsContext
from ops_agent.scope_rule import rule_scope

MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = f"""你是一名资深 SRE，正在排查一条云主机告警。你只能通过工具获取信息，工具都是只读的。

排查方法：
1. 先确认告警指标本身：是突变、缓慢上涨、周期性出现还是数值本身就不合理；
2. 用同类主机对比确定影响范围：单机、部分主机、整个服务还是整个可用区，异常主机和正常主机在版本等属性上有什么区别；
3. 查变更和日志找直接原因，注意变更时间和异常开始时间是否吻合，变更涉及的主机和异常主机是否一致；
4. 日志里出现的数据库等下游对象也可以用 list_changes 查它的变更。

注意：
- 时间上接近的变更不一定是原因，要和影响范围、异常形态对得上才算；
- 一轮可以同时调用多个工具；最多 {MAX_TOOL_ROUNDS} 轮工具调用，证据足够时就停止调用工具，用一段话说明你的结论和依据。"""

REPORT_PROMPT = "根据以上排查过程，输出这条告警的根因报告。证据必须来自工具返回的内容，写出具体的主机、版本、时间或日志原文。"


class RootCauseReport(BaseModel):
    """告警根因报告。"""
    category: Category = Field(description=f"根因类别，只能选一个：{category_choices()}")
    scope: Scope = Field(description=f"影响范围：{scope_choices()}。按指标异常的主机分布判断（以 compare_peers 的结果为准），"
                                     "不是按根因所在的对象：同服务所有主机都异常就是整个服务，"
                                     "多个服务的主机都异常且集中在一个可用区就是整个可用区，指标异常但业务实际正常就是无实际影响")
    root_cause: str = Field(description="一两句话说明根因")
    evidence: list[str] = Field(description="支撑结论的关键证据，每条一句")
    suggested_actions: list[str] = Field(description="建议的处置动作，按优先级排列")
    confidence: Literal["high", "medium", "low"] = Field(description="对结论的把握程度")


class OpsState(MessagesState):
    report: RootCauseReport | None
    error: str
    rule_scope: Scope | None
    rule_reason: str


def alert_message(case: Case) -> str:
    a = case.alert
    host = case.host(a.host)
    return (f"告警：{a.title}\n主机 {a.host}（服务 {host.service}，可用区 {host.az}），指标 {a.metric}，"
            f"当前值 {a.value}，阈值 {a.threshold}，告警时间 {a.time:%Y-%m-%d %H:%M}。\n请定位根因。")


def tool_rounds(state: OpsState) -> int:
    return sum(1 for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls)


def build_graph(llm=None, max_tool_rounds: int = MAX_TOOL_ROUNDS):
    llm = llm or config.get_llm()

    def agent(state: OpsState) -> dict:
        reply = llm.bind_tools(TOOLS).invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [reply]}

    def report(state: OpsState) -> dict:
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"], HumanMessage(REPORT_PROMPT)]
        try:
            result = llm.with_structured_output(RootCauseReport, method="function_calling").invoke(messages)
        except openai.APIError:
            raise
        except Exception as e:
            # 模型输出不符合 schema 时不让整张图失败，评估里记为没有报告
            return {"report": None, "error": f"{type(e).__name__}: {e}"}
        return {"report": result, "error": "" if result else "模型没有输出报告"}

    def scope_rule(state: OpsState, runtime: Runtime[OpsContext]) -> dict:
        case = load_cases()[runtime.context.case_id]
        scope, reason = rule_scope(case, state["report"].category if state["report"] else None)
        return {"rule_scope": scope, "rule_reason": reason}

    def after_agent(state: OpsState) -> str:
        return "tools" if state["messages"][-1].tool_calls else "report"

    def after_tools(state: OpsState) -> str:
        return "report" if tool_rounds(state) >= max_tool_rounds else "agent"

    graph = StateGraph(OpsState, context_schema=OpsContext)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("report", report)
    graph.add_node("scope_rule", scope_rule)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, {"tools": "tools", "report": "report"})
    graph.add_conditional_edges("tools", after_tools, {"agent": "agent", "report": "report"})
    graph.add_edge("report", "scope_rule")
    graph.add_edge("scope_rule", END)
    return graph.compile(name="ops_agent")


def format_call(call: dict) -> str:
    args = ", ".join(f"{k}={v}" for k, v in call["args"].items())
    return f"{call['name']}({args})"


def ask(app, case: Case, on_step: Callable[[str], None] | None = None, session_id: str | None = None,
        tags: list[str] | None = None) -> dict:
    """跑一个 case，返回报告、工具调用轨迹和成本。on_step 在每个节点结束时收到一行进度。"""
    counter = CallCounter()
    start = time.perf_counter()
    inputs = {"messages": [HumanMessage(alert_message(case))], "report": None, "error": "",
              "rule_scope": None, "rule_reason": ""}
    trace_id = tracing.new_trace_id()
    cfg = tracing.traced({"callbacks": [counter], "recursion_limit": 4 * MAX_TOOL_ROUNDS + 10}, "ops_agent",
                         trace_id=trace_id, session_id=session_id, tags=["ops_agent", case.id, *(tags or [])],
                         metadata={"case_id": case.id, "alert": case.alert.title})
    state = inputs
    for mode, chunk in app.stream(inputs, cfg, context=OpsContext(case.id), stream_mode=["updates", "values"]):
        if mode == "values":
            state = chunk
        elif on_step and "agent" in chunk:
            reply = chunk["agent"]["messages"][-1]
            on_step("；".join(map(format_call, reply.tool_calls)) if reply.tool_calls else "结束排查")
        elif on_step and "report" in chunk:
            on_step("生成报告")
    latency = time.perf_counter() - start

    rounds = [[format_call(c) for c in m.tool_calls]
              for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls]
    finals = [m for m in state["messages"] if isinstance(m, AIMessage) and not m.tool_calls]
    return {
        "case_id": case.id,
        "report": state["report"],
        "rule_scope": state["rule_scope"],
        "rule_reason": state["rule_reason"],
        "error": state["error"],
        "rounds": rounds,
        "tools_used": sorted({c.split("(")[0] for r in rounds for c in r}),
        "conclusion": finals[-1].text if finals else "",
        "llm_calls": counter.calls,
        "tokens": counter.tokens,
        "latency": latency,
        "trace_id": trace_id,
    }
