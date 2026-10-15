"""离线测试 api.py：三种模式的 SSE 事件流、逐字输出、限流和会话管理。"""
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, LLMResult

import api
import config
from api_client import iter_sse
from graph_agent import Grade
from rag_chain import REFUSAL
from ratelimit import TokenBucket, parse_rate
from tools import CallCounter
from tests.fake_models import ScriptedChatModel, tool_call

KV_ANSWER = "K、V 缓存起来复用。\n来源：[1]"


def make_client(responses=(), **kwargs) -> tuple[ScriptedChatModel, TestClient]:
    llm = ScriptedChatModel(responses=list(responses))
    return llm, TestClient(api.create_app(llm=llm, **kwargs))


def post(client: TestClient, question: str, **body):
    with client.stream("POST", "/chat/stream", json={"question": question, **body}) as r:
        if r.status_code != 200:
            r.read()
            return r, []
        return r, list(iter_sse(r))


def tokens(events) -> list[str]:
    return [d["text"] for e, d in events if e == "token"]


def steps(events) -> list[str]:
    return [d["node"] for e, d in events if e == "step"]


def last(events, name: str) -> dict:
    assert events[-1][0] == name, f"最后一个事件是 {events[-1]}"
    return events[-1][1]


def test_graph_streams_answer_token_by_token():
    llm, client = make_client([tool_call("search_knowledge_base", query="KV Cache 是什么"),
                               Grade(relevant=True, reason="有定义"), AIMessage(KV_ANSWER)])
    r, events = post(client, "KV Cache 是什么")

    assert r.headers["content-type"].startswith("text/event-stream")
    assert events[0][0] == "session" and events[0][1]["mode"] == "graph"
    assert steps(events) == ["router", "retrieve", "grade", "generate"]
    # 逐字输出，且路由节点的工具调用增量没有混进答案
    assert len(tokens(events)) == len(KV_ANSWER)
    assert "".join(tokens(events)) == KV_ANSWER
    [sources] = [d["sources"] for e, d in events if e == "sources"]
    assert [(s["id"], s["source"]) for s in sources] == [(1, "02.md"), (2, "02.md")]
    done = last(events, "done")
    assert done["route"] == "retrieve" and done["answer"] == KV_ANSWER
    assert done["queries"] == ["KV Cache 是什么"] and done["llm_calls"] == 2
    assert 0 < done["first_token_latency"] <= done["latency"] + 0.1
    assert llm.responses == []


def test_call_counter_uses_max_of_cumulative_stream_usage():
    counter, run_id = CallCounter(), uuid4()
    usage = lambda total: {"input_tokens": 5, "output_tokens": total - 5, "total_tokens": total}
    for total in (10, 11, 12):
        counter.on_llm_new_token("", chunk=ChatGenerationChunk(message=AIMessageChunk(
            content="", usage_metadata=usage(total))), run_id=run_id)
    merged = AIMessage("x", usage_metadata=usage(33))  # LangChain 把三个分块的累计用量相加后的结果
    counter.on_llm_end(LLMResult(generations=[[ChatGeneration(message=merged)]]), run_id=run_id)
    assert counter.tokens == 12

    counter.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage("y", usage_metadata=usage(7)))]]),
                       run_id=uuid4())
    assert counter.tokens == 19


def test_graph_refuse_sends_whole_answer_without_sources():
    _, client = make_client([tool_call("refuse", reason="Java 不在知识库范围")])
    _, events = post(client, "Java 快速排序")
    assert steps(events) == ["router", "refuse"]
    assert len(tokens(events)) == 1 and tokens(events)[0].startswith(REFUSAL)
    assert "sources" not in [e for e, _ in events]
    assert last(events, "done")["route"] == "refuse"


def test_agent_mode_streams_final_answer():
    _, client = make_client([tool_call("search_knowledge_base", query="KV Cache 是什么"), AIMessage(KV_ANSWER)])
    _, events = post(client, "KV Cache 是什么", mode="agent")
    assert steps(events) == ["model", "tools"]
    assert "".join(tokens(events)) == KV_ANSWER and len(tokens(events)) > 1
    assert any(e == "sources" for e, _ in events)
    assert last(events, "done")["route"] == "retrieve"


def test_agent_mode_direct_answer():
    _, client = make_client([AIMessage("你好！我是知识库助手。")])
    _, events = post(client, "你好", mode="agent")
    assert steps(events) == [] and "".join(tokens(events)) == "你好！我是知识库助手。"
    assert last(events, "done")["route"] == "direct"


def test_chain_mode_rewrites_follow_up_with_session_history(fake_retriever):
    _, client = make_client([AIMessage("Retriever 是检索抽象。"),
                             AIMessage("Retriever 和 Vector Store 有什么区别"), AIMessage(f"{REFUSAL}。")])
    _, first = post(client, "Retriever 是什么", mode="chain")
    session_id = first[0][1]["session_id"]
    assert steps(first) == ["retrieve"] and last(first, "done")["route"] == "retrieve"

    _, second = post(client, "它和 Vector Store 有什么区别", mode="chain", session_id=session_id)
    assert second[0][1]["session_id"] == session_id
    assert steps(second) == ["rewrite", "retrieve"]
    assert fake_retriever[-1] == "Retriever 和 Vector Store 有什么区别"
    done = last(second, "done")
    assert done["route"] == "retrieve_refuse" and "sources" not in [e for e, _ in second]


def test_delete_session_clears_history():
    _, client = make_client([tool_call("answer_directly", reason="寒暄"), AIMessage("你好")])
    post(client, "你好", session_id="s1")
    graph = client.app.state.backends.get("graph").app
    cfg = {"configurable": {"thread_id": "s1"}}
    assert len(graph.get_state(cfg).values["messages"]) == 2
    assert client.delete("/sessions/s1").status_code == 204
    assert graph.get_state(cfg).values == {}


def test_rate_limit_returns_429_with_retry_after():
    llm, client = make_client([tool_call("refuse", reason="x")] * 2, rate_limit="2/minute")
    remaining = [post(client, "问题")[0].headers["X-RateLimit-Remaining"] for _ in range(2)]
    assert remaining == ["1", "0"]

    r, _ = post(client, "问题")
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1 and "频繁" in r.json()["detail"]
    assert llm.responses == []


def test_concurrency_limit_returns_503():
    _, client = make_client(max_concurrent=1)
    client.app.state.slots.active = 1
    r, _ = post(client, "问题")
    assert r.status_code == 503 and r.headers["Retry-After"] == "5"


def test_busy_session_returns_409():
    _, client = make_client()
    client.app.state.busy.add("graph:s1")
    assert post(client, "问题", session_id="s1")[0].status_code == 409


def test_slot_and_session_released_after_stream_even_on_error():
    _, client = make_client([])  # 剧本为空，router 调 LLM 时就会抛异常
    _, events = post(client, "问题", session_id="s1")
    assert last(events, "error")["status"] == 500
    assert client.app.state.slots.active == 0 and client.app.state.busy == set()


def test_api_key_required_when_configured():
    _, client = make_client([tool_call("refuse", reason="x")], api_keys=["k1"])
    assert post(client, "问题")[0].status_code == 401
    with client.stream("POST", "/chat/stream", json={"question": "问题"}, headers={"X-API-Key": "k1"}) as r:
        assert r.status_code == 200 and list(iter_sse(r))[-1][0] == "done"


def test_missing_llm_config_returns_503(monkeypatch):
    def no_key():
        raise RuntimeError("请先配置 LLM_API_KEY")

    monkeypatch.setattr(config, "get_llm", no_key)
    client = TestClient(api.create_app())
    r, _ = post(client, "问题")
    assert r.status_code == 503 and "LLM_API_KEY" in r.json()["detail"]


@pytest.mark.parametrize("body", [{"question": ""}, {"question": "x" * 1001},
                                  {"question": "问题", "session_id": "../etc"}, {"question": "问题", "mode": "x"}])
def test_invalid_request_returns_422(body):
    _, client = make_client()
    assert client.post("/chat/stream", json=body).status_code == 422


def test_token_bucket_burst_refill_and_isolation():
    now = [0.0]
    bucket = TokenBucket(3, 60, clock=lambda: now[0])
    assert [bucket.acquire("a").allowed for _ in range(4)] == [True, True, True, False]
    assert bucket.acquire("a").retry_after == pytest.approx(20)
    assert bucket.acquire("b").allowed
    now[0] = 20
    assert bucket.acquire("a").allowed and not bucket.acquire("a").allowed


def test_token_bucket_prunes_idle_keys():
    now = [0.0]
    bucket = TokenBucket(1, 1, clock=lambda: now[0], max_keys=2)
    bucket.acquire("a"), bucket.acquire("b")
    now[0] = 5
    bucket.acquire("c")
    assert list(bucket._buckets) == ["c"]


def test_parse_rate():
    assert parse_rate("10/minute") == (10, 60.0)
    assert parse_rate(" 2 / second ") == (2, 1.0)
    for bad in ["10", "0/minute", "10/week", "abc"]:
        with pytest.raises(ValueError):
            parse_rate(bad)
