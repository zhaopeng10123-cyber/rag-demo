"""Langfuse 追踪的离线测试：span 导出到内存，检查 trace 的结构和属性，不连 Langfuse 服务。"""
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import api
import graph_agent
import tracing
from api_client import iter_sse
from graph_agent import Grade
from tests.fake_models import ScriptedChatModel, tool_call

EXPORTER = InMemorySpanExporter()
ANSWER = "K、V 缓存起来复用。\n来源：[1]"


@pytest.fixture
def spans(monkeypatch):
    """打开追踪；同一个 public key 的 Langfuse 资源是进程级单例，所以 exporter 也只建一次。"""
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-offline-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-offline-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://127.0.0.1:9")
    tracing.client.cache_clear()
    client = Langfuse(span_exporter=EXPORTER)
    EXPORTER.clear()
    scores = []
    monkeypatch.setattr(client, "create_score", lambda **kw: scores.append(kw))
    monkeypatch.setattr(tracing, "client", lambda: client)

    def finished():
        client.flush()
        return EXPORTER.get_finished_spans()

    finished.scores = scores
    return finished


def trace_hex(span) -> str:
    return format(span.context.trace_id, "032x")


def attr(span, key: str):
    return span.attributes.get(key)


def test_disabled_is_noop(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    cfg = {"callbacks": ["counter"]}
    assert not tracing.enabled()
    assert tracing.traced(cfg, "x", session_id="s") == cfg
    assert tracing.new_trace_id() is None and tracing.trace_url("abc") is None
    tracing.score("abc", "ok", True)  # 不报错也不连网


def test_graph_agent_full_trace(spans, fake_retriever):
    llm = ScriptedChatModel(responses=[tool_call("search_knowledge_base", query="KV Cache 是什么"),
                                       Grade(relevant=True, reason="有定义"), AIMessage(ANSWER)])
    app = graph_agent.build_graph(llm=llm)
    result = graph_agent.ask(app, "KV Cache 是什么", thread_id="thread-1", tags=["eval"])

    finished = spans()
    assert result["trace_id"] and {trace_hex(s) for s in finished} == {result["trace_id"]}
    names = [s.name for s in finished]
    assert "kb_graph_agent" in names
    # 状态图的每个节点、检索 tool、LLM 调用都在同一条 trace 里
    assert {"router", "retrieve", "grade", "generate", "search_knowledge_base"} <= set(names)
    generations = [s for s in finished if attr(s, "langfuse.observation.type") == "generation"]
    assert len(generations) == 2  # router 和 generate；grade 走 with_structured_output，剧本模型不产生 LLM 调用
    root = next(s for s in finished if s.name == "kb_graph_agent")
    assert attr(root, "session.id") == "thread-1"
    assert set(attr(root, "langfuse.trace.tags")) >= {"graph", "eval"}


def test_chain_mode_api_groups_calls_and_hashes_api_key(spans, fake_retriever):
    llm = ScriptedChatModel(responses=[AIMessage(ANSWER)])
    client = TestClient(api.create_app(llm=llm, api_keys=["secret-key-1"]))
    with client.stream("POST", "/chat/stream", json={"question": "KV Cache 是什么", "mode": "chain",
                                                     "session_id": "s1"},
                       headers={"X-API-Key": "secret-key-1"}) as r:
        events = list(iter_sse(r))
    done = events[-1][1]

    finished = spans()
    # 检索和作答是两次独立调用，靠预先生成的 trace id 归到同一条 trace
    assert done["trace_id"] and {trace_hex(s) for s in finished} == {done["trace_id"]}
    roots = [s for s in finished if s.name == "api.chain"]
    assert len(roots) == 2
    # user / session 是 trace 级属性，根 span 带上即可；流式调用的子 span 上没有
    assert {(attr(s, "user.id"), attr(s, "session.id")) for s in roots} == {(tracing.anonymize("key:secret-key-1"),
                                                                             "s1")}
    assert all("secret-key-1" not in str(dict(s.attributes)) for s in finished)


def test_scores(spans):
    tracing.score("t1", "route_ok", True, comment="retrieve -> retrieve")
    tracing.score("t1", "coverage", 0.5)
    tracing.score("t1", "label", "partial_hosts")
    assert [(s["name"], s["value"], s["data_type"]) for s in spans.scores] == [
        ("route_ok", 1.0, "BOOLEAN"), ("coverage", 0.5, "NUMERIC"), ("label", "partial_hosts", "CATEGORICAL")]
    tracing.score(None, "x", 1)
    assert len(spans.scores) == 3


def test_anonymize():
    assert tracing.anonymize("ip:127.0.0.1") == "ip:127.0.0.1"
    hashed = tracing.anonymize("key:abc")
    assert hashed.startswith("key:") and "abc" not in hashed and hashed == tracing.anonymize("key:abc")
