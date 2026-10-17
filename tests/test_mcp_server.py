"""MCP server 的工具测试：直接调用 FastMCP 的 list_tools / call_tool，走参数校验和结构化输出，不起子进程。"""
import config
import mcp_server
import pytest
from langchain_core.messages import AIMessage
from mcp.server.fastmcp.exceptions import ToolError
from rag_chain import REFUSAL

from tests.fake_models import DOCS, ScriptedChatModel

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def fake_search(monkeypatch):
    """检索词里带“空”时返回空列表，其余返回 DOCS 的前 k 条，不加载 Embedding 模型。"""
    calls = []

    def search(query: str, k: int):
        calls.append((query, k))
        return [] if "空" in query else DOCS[:k]

    monkeypatch.setattr(mcp_server, "search_docs", search)
    monkeypatch.setattr(mcp_server, "_chain", None)
    return calls


def use_llm(monkeypatch, *replies: str) -> None:
    llm = ScriptedChatModel(responses=[AIMessage(r) for r in replies])
    monkeypatch.setattr(config, "get_llm", lambda temperature=0: llm)


async def call(name: str, args: dict) -> dict:
    _content, structured = await mcp_server.mcp.call_tool(name, args)
    return structured


async def test_tools_schema():
    tools = {t.name: t for t in await mcp_server.mcp.list_tools()}
    assert set(tools) == {"search_knowledge", "answer"}
    for t in tools.values():
        assert t.annotations.readOnlyHint
        assert "LangGraph" in t.description  # 说明里带上了知识库主题
    top_k = tools["search_knowledge"].inputSchema["properties"]["top_k"]
    assert (top_k["minimum"], top_k["maximum"], top_k["default"]) == (1, 10, config.TOP_K)
    assert set(tools["answer"].outputSchema["properties"]) == {"answer", "refused", "sources"}


async def test_search_returns_numbered_chunks(fake_search):
    result = await call("search_knowledge", {"query": "什么是 KV Cache", "top_k": 2})
    assert fake_search == [("什么是 KV Cache", 2)]
    assert [c["id"] for c in result["chunks"]] == [1, 2]
    assert result["chunks"][0]["source"] == "02.md"
    assert result["chunks"][0]["section"] == DOCS[0].metadata["section"]
    assert result["chunks"][0]["content"] == DOCS[0].page_content


async def test_search_empty_and_invalid_args():
    assert (await call("search_knowledge", {"query": "空的结果"}))["chunks"] == []
    with pytest.raises(ToolError):
        await call("search_knowledge", {"query": "KV Cache", "top_k": 0})


async def test_answer_with_sources(monkeypatch, fake_search):
    use_llm(monkeypatch, "KV Cache 把 K、V 算一次就缓存起来。\n来源：[1]")
    result = await call("answer", {"question": "什么是 KV Cache？"})
    assert fake_search == [("什么是 KV Cache？", config.TOP_K)]
    assert result["refused"] is False
    assert result["answer"].endswith("来源：[1]")
    assert [s["id"] for s in result["sources"]] == [1, 2]


async def test_answer_refusal(monkeypatch):
    use_llm(monkeypatch, REFUSAL)
    result = await call("answer", {"question": "今天北京天气怎么样？"})
    assert result["refused"] is True


async def test_answer_without_llm_config(monkeypatch):
    def missing_key(temperature=0):
        raise RuntimeError("请先在 rag-demo/.env 中配置 LLM_API_KEY")

    monkeypatch.setattr(config, "get_llm", missing_key)
    with pytest.raises(ToolError, match="LLM_API_KEY"):
        await call("answer", {"question": "什么是 LCEL？"})


async def test_topics_resource():
    [content] = await mcp_server.mcp.read_resource("kb://topics")
    assert "SGLang" in content.content
    assert "01-LangChain与LlamaIndex从零入门.md" in content.content
