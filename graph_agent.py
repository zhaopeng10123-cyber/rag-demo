"""实现 B：LangGraph 显式状态图。

    router --search--> retrieve -> grade --相关--> generate
       |                  ^          |--不相关且未重试--> rewrite --> retrieve
       |--direct--> direct           |--不相关且已重试--> refuse
       |--refuse--> refuse

router 用 function calling 在三个 tool 里强制三选一；retrieve 用 ToolNode 执行检索 tool。

用法：
  python graph_agent.py "你的问题"   单轮问答，打印经过的节点
  python graph_agent.py --draw        导出状态图到 docs/graph_agent.mmd
"""
import argparse
import time
import uuid
from typing import Annotated, TypedDict

import config
import openai
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from rag_chain import build_answer_chain, format_docs, print_sources
from tools import (DIRECT_PROMPT, GRADE_PROMPT, REWRITE_PROMPT, ROUTER_PROMPT, AgentResult, CallCounter,
                   answer_directly, export_graph, refusal_text, refuse, search_knowledge_base)

MAX_RETRIES = 1
ROUTES = {search_knowledge_base.name: "retrieve", answer_directly.name: "direct", refuse.name: "refuse"}


class AgentState(TypedDict, total=False):
    # 只存每轮的用户问题和最终答案；路由用的 tool call 不进历史，
    # 否则下一轮会出现没有对应 ToolMessage 的 tool call，违反 OpenAI 消息协议
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    standalone: str   # router 改写出的独立问题，generate 用它作答
    query: str        # 实际检索用的问题，rewrite 会修改它
    route: str
    reason: str
    docs: list[Document]
    relevant: bool
    retries: int
    answer: str


class Grade(BaseModel):
    """判断检索到的资料能否回答问题。"""
    relevant: bool = Field(description="资料中是否包含能回答问题的信息")
    reason: str = Field(description="一句话说明判断依据")


def build_graph(llm=None, checkpointer=None):
    llm = llm or config.get_llm()
    router_tools = [search_knowledge_base, answer_directly, refuse]
    tool_node = ToolNode([search_knowledge_base])
    answer_chain = build_answer_chain(llm)
    tool_choice = {"value": "required"}

    def router(state: AgentState) -> dict:
        messages = [SystemMessage(ROUTER_PROMPT), *state["messages"]]
        try:
            reply = llm.bind_tools(router_tools, tool_choice=tool_choice["value"]).invoke(messages)
        except openai.BadRequestError:
            # 部分 OpenAI 兼容服务不支持 tool_choice="required"，退回 auto 并记住
            tool_choice["value"] = "auto"
            reply = llm.bind_tools(router_tools, tool_choice="auto").invoke(messages)

        if not reply.tool_calls:
            # auto 模式下模型没调用任何 tool，按直答处理，它的回复就是答案
            return {"route": "direct", "answer": reply.text, "reason": "模型未调用工具"}
        call = reply.tool_calls[0]
        route = ROUTES.get(call["name"], "refuse")
        if route == "retrieve":
            query = call["args"].get("query") or state["question"]
            return {"route": route, "query": query, "standalone": query}
        return {"route": route, "reason": call["args"].get("reason", "")}

    def retrieve(state: AgentState, config: RunnableConfig) -> dict:
        call = {"name": search_knowledge_base.name, "args": {"query": state["query"]},
                "id": f"search_{state.get('retries', 0)}", "type": "tool_call"}
        result = tool_node.invoke({"messages": [AIMessage(content="", tool_calls=[call])]}, config)
        return {"docs": result["messages"][-1].artifact or []}

    def grade(state: AgentState) -> dict:
        if not state["docs"]:
            return {"relevant": False, "reason": "没有检索到任何资料"}
        prompt = GRADE_PROMPT.format(question=state["standalone"], context=format_docs(state["docs"]))
        try:
            result = llm.with_structured_output(Grade, method="function_calling").invoke(prompt)
        except Exception:
            # 解析失败时宁可当作相关去生成，也不要误拒答
            return {"relevant": True, "reason": "相关性判断失败，默认相关"}
        return {"relevant": result.relevant, "reason": result.reason}

    def rewrite(state: AgentState) -> dict:
        query = llm.invoke(REWRITE_PROMPT.format(query=state["query"])).text.strip()
        return {"query": query or state["query"], "retries": state.get("retries", 0) + 1}

    def generate(state: AgentState) -> dict:
        answer = answer_chain.invoke({"context": format_docs(state["docs"]), "question": state["standalone"]})
        return {"answer": answer, "messages": [AIMessage(answer)]}

    def direct(state: AgentState) -> dict:
        answer = state.get("answer") or llm.invoke([SystemMessage(DIRECT_PROMPT), *state["messages"]]).text
        return {"answer": answer, "messages": [AIMessage(answer)]}

    def refuse_node(state: AgentState) -> dict:
        reason = state.get("reason", "")
        if state["route"] == "retrieve":
            reason = f"检索到的资料不足以回答：{reason}"
        answer = refusal_text(reason)
        return {"answer": answer, "messages": [AIMessage(answer)]}

    def after_router(state: AgentState) -> str:
        return state["route"]

    def after_grade(state: AgentState) -> str:
        if state["relevant"]:
            return "generate"
        return "rewrite" if state.get("retries", 0) < MAX_RETRIES else "refuse"

    graph = StateGraph(AgentState)
    graph.add_node("router", router)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.add_node("direct", direct)
    graph.add_node("refuse", refuse_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", after_router,
                                {"retrieve": "retrieve", "direct": "direct", "refuse": "refuse"})
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade,
                                {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"})
    graph.add_edge("rewrite", "retrieve")
    for node in ("generate", "direct", "refuse"):
        graph.add_edge(node, END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver(), name="kb_graph_agent")


def describe(node: str, delta: dict) -> str:
    if node == "router":
        detail = delta.get("query") or delta.get("reason", "")
        return f"router({delta['route']}: {detail})"
    if node == "retrieve":
        return f"retrieve({len(delta['docs'])} 个片段)"
    if node == "grade":
        return f"grade({'相关' if delta['relevant'] else '不相关'}: {delta['reason']})"
    if node == "rewrite":
        return f"rewrite({delta['query']})"
    return node


def turn_inputs(question: str) -> dict:
    # checkpointer 会保留上一轮的所有字段，除 messages 外每轮都要显式重置
    return {"messages": [HumanMessage(question)], "question": question, "standalone": "", "query": "",
            "route": "", "reason": "", "docs": [], "relevant": False, "retries": 0, "answer": ""}


def final_route(route: str, trace: list[str]) -> str:
    return "retrieve_refuse" if route == "retrieve" and trace and trace[-1] == "refuse" else route


def ask(app, question: str, thread_id: str) -> AgentResult:
    counter = CallCounter()
    cfg = {"configurable": {"thread_id": thread_id}, "callbacks": [counter]}

    start = time.perf_counter()
    trace, queries = [], []
    for update in app.stream(turn_inputs(question), cfg, stream_mode="updates"):
        for node, delta in update.items():
            trace.append(describe(node, delta or {}))
            if node in ("router", "rewrite") and delta.get("query"):
                queries.append(delta["query"])
    latency = time.perf_counter() - start

    state = app.get_state(cfg).values
    route = final_route(state["route"], trace)
    return AgentResult(route=route, queries=queries, docs=state.get("docs", []), answer=state["answer"],
                       trace=trace, llm_calls=counter.calls, tokens=counter.tokens, latency=latency)


def print_result(result: AgentResult) -> None:
    print(f"[路由] {result['route']}    " + " -> ".join(result["trace"]))
    print(f"助手：{result['answer']}")
    if result["docs"] and result["route"] == "retrieve":
        print_sources(result["docs"])
    print(f"（LLM 调用 {result['llm_calls']} 次，{result['tokens']} tokens，{result['latency']:.1f}s）")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*")
    parser.add_argument("--draw", action="store_true")
    args = parser.parse_args()

    if args.draw:
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        export_graph(build_graph(llm=FakeListChatModel(responses=[""])), "graph_agent")
        return
    question = " ".join(args.question) or "LCEL 是什么？"
    print_result(ask(build_graph(), question, thread_id=uuid.uuid4().hex))


if __name__ == "__main__":
    main()
