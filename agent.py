"""实现 A：create_agent 版 Function Calling Agent。

检索是一个 tool，模型在 ReAct 循环里自己决定调不调用：调用了就是“检索”，
没调用就是“直答”或“拒答”。路由是隐式的，只能在事后从消息里推断。

用法：
  python agent.py "你的问题"     单轮问答，打印路由轨迹
  python agent.py --draw          导出内置图到 docs/agent.mmd
"""
import argparse
import time
import uuid

import config
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from rag_chain import print_sources
from tools import (AGENT_SYSTEM_PROMPT, AgentResult, CallCounter, export_graph, is_refusal,
                   search_knowledge_base)

MAX_SEARCHES = 2


def build_agent(llm=None, checkpointer=None):
    return create_agent(
        model=llm or config.get_llm(),
        tools=[search_knowledge_base],
        system_prompt=AGENT_SYSTEM_PROMPT,
        # 每个问题最多检索 2 次；超出时 tool 返回错误信息，模型只能基于已有资料作答
        middleware=[ToolCallLimitMiddleware(tool_name=search_knowledge_base.name, run_limit=MAX_SEARCHES)],
        checkpointer=checkpointer or InMemorySaver(),
        name="kb_agent",
    )


def infer_route(queries: list[str], answer: str) -> str:
    refused = is_refusal(answer)
    if queries:
        return "retrieve_refuse" if refused else "retrieve"
    return "refuse" if refused else "direct"


def ask(agent, question: str, thread_id: str) -> AgentResult:
    counter = CallCounter()
    cfg = {"configurable": {"thread_id": thread_id}, "callbacks": [counter]}
    before = len(agent.get_state(cfg).values.get("messages", []))

    start = time.perf_counter()
    state = agent.invoke({"messages": [HumanMessage(question)]}, cfg)
    latency = time.perf_counter() - start

    trace, queries, docs = [], [], []
    for msg in state["messages"][before + 1:]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                queries.append(tc["args"].get("query", ""))
                trace.append(f"model -> {tc['name']}(query={queries[-1]!r})")
        elif isinstance(msg, ToolMessage):
            # 多次检索时引用编号以最后一次检索为准，和最终答案里的 [n] 对应
            if msg.artifact:
                docs = msg.artifact
            trace.append(f"tools -> {len(msg.artifact or [])} 个片段")
        elif isinstance(msg, AIMessage):
            trace.append("model -> 最终答案")

    answer = state["messages"][-1].text
    return AgentResult(route=infer_route(queries, answer), queries=queries, docs=docs, answer=answer, trace=trace,
                       llm_calls=counter.calls, tokens=counter.tokens, latency=latency)


def print_result(result: AgentResult) -> None:
    print(f"[路由] {result['route']}    " + "  |  ".join(result["trace"]))
    print(f"助手：{result['answer']}")
    if result["docs"]:
        print_sources(result["docs"])
    print(f"（LLM 调用 {result['llm_calls']} 次，{result['tokens']} tokens，{result['latency']:.1f}s）")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*")
    parser.add_argument("--draw", action="store_true")
    args = parser.parse_args()

    if args.draw:
        # 只导出结构，不需要真实 LLM
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        export_graph(build_agent(llm=FakeListChatModel(responses=[""])), "agent")
        return
    question = " ".join(args.question) or "LCEL 是什么？"
    print_result(ask(build_agent(), question, thread_id=uuid.uuid4().hex))


if __name__ == "__main__":
    main()
