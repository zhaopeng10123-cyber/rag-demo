"""三种实现的流式问答，统一产出 (事件名, 数据)，由 api.py 编码成 SSE。

事件顺序：step*（处理进度）-> token*（答案增量）-> sources?（引用，仅检索作答时）-> done（路由、完整答案和成本）

答案的逐字输出来自 LangGraph 的 stream_mode="messages"，只转发作答节点（graph 的 generate/direct、
agent 的最终回复）里的 LLM 增量；路由、相关性判断、改写这些中间调用的输出不发给客户端。
作答节点没调 LLM（拒答节点、router 未调工具时的直答）或模型不支持流式时，整段答案作为一个 token 补发。
"""
import time
from collections.abc import AsyncIterator
from typing import Any

import agent as agent_impl
import config
import graph_agent
import tools
import tracing
from chat import MAX_TURNS, build_rewrite_chain
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from rag_chain import build_answer_chain, format_docs
from tools import CallCounter, is_refusal

Event = tuple[str, dict[str, Any]]
MODES = ("graph", "agent", "chain")
GRAPH_ANSWER_NODES = {"generate", "direct"}


def sources_payload(docs) -> list[dict]:
    # 编号和 format_docs 一致，对应答案里的 [n]
    return [{"id": i, "source": d.metadata.get("source", ""), "section": d.metadata.get("section", "")}
            for i, d in enumerate(docs, 1)]


def done_payload(route: str, answer: str, queries: list[str], counter: CallCounter, start: float,
                 trace_id: str | None) -> dict:
    return {"route": route, "answer": answer, "queries": queries, "llm_calls": counter.calls,
            "tokens": counter.tokens, "latency": round(time.perf_counter() - start, 2), "trace_id": trace_id}


def stream_config(name: str, session_id: str, user_id: str | None, counter: CallCounter,
                  thread: bool = True) -> tuple[dict, str | None]:
    trace_id = tracing.new_trace_id()
    cfg = {"configurable": {"thread_id": session_id}} if thread else {}
    cfg = tracing.traced({**cfg, "callbacks": [counter]}, name, trace_id=trace_id, session_id=session_id,
                         user_id=user_id, tags=["api"])
    return cfg, trace_id


def answer_token(msg: Any, meta: dict, nodes: set[str]) -> str:
    """messages 流里属于作答节点的文本增量；工具调用的增量和节点返回的完整消息都不算。"""
    if isinstance(msg, AIMessageChunk) and not msg.tool_call_chunks and meta.get("langgraph_node") in nodes:
        return msg.text
    return ""


class GraphBackend:
    def __init__(self, llm) -> None:
        self.app = graph_agent.build_graph(llm=llm)

    async def stream(self, question: str, session_id: str, user_id: str | None = None) -> AsyncIterator[Event]:
        counter = CallCounter()
        cfg, trace_id = stream_config("api.graph", session_id, user_id, counter)
        start = time.perf_counter()
        trace, queries, streamed = [], [], False

        async for mode, chunk in self.app.astream(graph_agent.turn_inputs(question), cfg,
                                                  stream_mode=["updates", "messages"]):
            if mode == "messages":
                if text := answer_token(*chunk, GRAPH_ANSWER_NODES):
                    streamed = True
                    yield "token", {"text": text}
                continue
            for node, delta in chunk.items():
                delta = delta or {}
                trace.append(graph_agent.describe(node, delta))
                yield "step", {"node": node, "detail": trace[-1]}
                if node in ("router", "rewrite") and delta.get("query"):
                    queries.append(delta["query"])
                if node in ("generate", "direct", "refuse") and not streamed:
                    yield "token", {"text": delta["answer"]}

        state = (await self.app.aget_state(cfg)).values
        route = graph_agent.final_route(state["route"], trace)
        if route == "retrieve":
            yield "sources", {"sources": sources_payload(state["docs"])}
        yield "done", done_payload(route, state["answer"], queries, counter, start, trace_id)

    def clear(self, session_id: str) -> None:
        self.app.checkpointer.delete_thread(session_id)


class AgentBackend:
    def __init__(self, llm) -> None:
        self.agent = agent_impl.build_agent(llm=llm)

    async def stream(self, question: str, session_id: str, user_id: str | None = None) -> AsyncIterator[Event]:
        counter = CallCounter()
        cfg, trace_id = stream_config("api.agent", session_id, user_id, counter)
        start = time.perf_counter()
        queries, docs, answer, streamed = [], [], "", False

        async for mode, chunk in self.agent.astream({"messages": [HumanMessage(question)]}, cfg,
                                                    stream_mode=["updates", "messages"]):
            if mode == "messages":
                if text := answer_token(*chunk, {"model"}):
                    streamed = True
                    yield "token", {"text": text}
                continue
            # 中间件节点的 delta 可能是 None
            for delta in chunk.values():
                for msg in (delta or {}).get("messages", []):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            queries.append(tc["args"].get("query", ""))
                            yield "step", {"node": "model", "detail": f"{tc['name']}({queries[-1]})"}
                    elif isinstance(msg, ToolMessage):
                        # 多次检索时引用编号以最后一次为准
                        if msg.artifact:
                            docs = msg.artifact
                        yield "step", {"node": "tools", "detail": f"{len(msg.artifact or [])} 个片段"}
                    elif isinstance(msg, AIMessage):
                        answer = msg.text
                        if not streamed:
                            yield "token", {"text": answer}

        route = agent_impl.infer_route(queries, answer)
        if route == "retrieve" and docs:
            yield "sources", {"sources": sources_payload(docs)}
        yield "done", done_payload(route, answer, queries, counter, start, trace_id)

    def clear(self, session_id: str) -> None:
        self.agent.checkpointer.delete_thread(session_id)


class ChainBackend:
    """RAG 链本身无状态，多轮历史存在进程内字典里，做法和 chat.py 一致。"""

    def __init__(self, llm) -> None:
        self.rewrite_chain = build_rewrite_chain(llm)
        self.answer_chain = build_answer_chain(llm)
        self.histories: dict[str, list[BaseMessage]] = {}

    async def stream(self, question: str, session_id: str, user_id: str | None = None) -> AsyncIterator[Event]:
        counter = CallCounter()
        # 改写、检索、作答是三次独立调用，靠同一个 trace id 归到一条 trace 下
        cfg, trace_id = stream_config("api.chain", session_id, user_id, counter, thread=False)
        start = time.perf_counter()
        history = self.histories.get(session_id, [])

        standalone = question
        if history:
            standalone = (await self.rewrite_chain.ainvoke({"history": history, "question": question}, cfg)).strip()
            standalone = standalone or question
            yield "step", {"node": "rewrite", "detail": f"rewrite({standalone})"}
        docs = await tools._get_retriever().ainvoke(standalone, cfg)
        yield "step", {"node": "retrieve", "detail": f"retrieve({len(docs)} 个片段)"}

        answer = ""
        async for text in self.answer_chain.astream({"context": format_docs(docs), "question": standalone}, cfg):
            answer += text
            yield "token", {"text": text}
        self.histories[session_id] = (history + [HumanMessage(question), AIMessage(answer)])[-MAX_TURNS * 2:]

        route = "retrieve_refuse" if is_refusal(answer) else "retrieve"
        if route == "retrieve":
            yield "sources", {"sources": sources_payload(docs)}
        yield "done", done_payload(route, answer, [standalone], counter, start, trace_id)

    def clear(self, session_id: str) -> None:
        self.histories.pop(session_id, None)


BACKENDS = {"graph": GraphBackend, "agent": AgentBackend, "chain": ChainBackend}


class Backends:
    """按需构建各模式的后端并缓存，三种模式共用一个 LLM 实例。"""

    def __init__(self, llm=None) -> None:
        self._llm = llm
        self._built: dict[str, Any] = {}

    def get(self, mode: str):
        if mode not in self._built:
            self._llm = self._llm or config.get_llm()
            self._built[mode] = BACKENDS[mode](self._llm)
        return self._built[mode]

    def clear(self, session_id: str) -> None:
        for backend in self._built.values():
            backend.clear(session_id)
