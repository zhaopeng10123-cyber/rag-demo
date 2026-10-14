"""离线测试：用剧本模型和假检索器覆盖状态图的全部分支、多轮记忆，以及 create_agent 版的路由推断。

运行：python -m pytest tests -q
"""
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

import agent
import graph_agent
import tools
from graph_agent import Grade
from rag_chain import REFUSAL, format_docs
from tests.fake_models import ScriptedChatModel, tool_call

DOCS = [
    Document(page_content="【2.3 什么是 KV Cache】K、V 算一次就缓存。",
             metadata={"source": "02.md", "section": "第 2 章 > 2.3 什么是 KV Cache"}),
    Document(page_content="【2.4 显存】KV Cache 占显存。",
             metadata={"source": "02.md", "section": "第 2 章 > 2.4 KV Cache 带来的新问题：显存"}),
]


@pytest.fixture(autouse=True)
def fake_retriever(monkeypatch):
    """检索词里带“空”时返回空列表，其余返回固定的 DOCS，不加载 Embedding 模型。"""
    queries = []

    def search(query: str):
        queries.append(query)
        return [] if "空" in query else DOCS

    monkeypatch.setattr(tools, "_retriever", RunnableLambda(search))
    return queries


def router_inputs(llm: ScriptedChatModel) -> list[list]:
    return [msgs for msgs in llm.received
            if isinstance(msgs[0], SystemMessage) and msgs[0].content == tools.ROUTER_PROMPT]


def nodes(result) -> list[str]:
    return [step.split("(")[0] for step in result["trace"]]


def run_graph(responses, questions, thread_id="t"):
    llm = ScriptedChatModel(responses=list(responses))
    app = graph_agent.build_graph(llm=llm)
    results = [graph_agent.ask(app, q, thread_id) for q in questions]
    assert llm.responses == [], f"剧本没有用完，图少调用了 LLM：{llm.responses}"
    return llm, app, results


def test_retrieve_branch_generates_with_artifact_docs(fake_retriever):
    llm, _, [r] = run_graph(
        [tool_call("search_knowledge_base", query="KV Cache 是什么"),
         Grade(relevant=True, reason="2.3 有定义"),
         AIMessage("K、V 缓存起来复用。\n来源：[1]")],
        ["KV Cache 是什么"],
    )
    assert r["route"] == "retrieve"
    assert nodes(r) == ["router", "retrieve", "grade", "generate"]
    assert r["docs"] == DOCS
    assert fake_retriever == ["KV Cache 是什么"]
    assert llm.bound_tools[0] == ["search_knowledge_base", "answer_directly", "refuse"]
    generate_prompt = llm.received[-1][0].content
    assert format_docs(DOCS) in generate_prompt


def test_direct_branch():
    _, _, [r] = run_graph(
        [tool_call("answer_directly", reason="寒暄"), AIMessage("你好！我是知识库助手。")],
        ["你好"],
    )
    assert r["route"] == "direct"
    assert nodes(r) == ["router", "direct"]
    assert r["answer"] == "你好！我是知识库助手。"
    assert r["docs"] == []


def test_refuse_branch_needs_no_extra_llm_call():
    llm, _, [r] = run_graph([tool_call("refuse", reason="Spring Boot 不在知识库范围")], ["Spring Boot 定时任务"])
    assert r["route"] == "refuse"
    assert nodes(r) == ["router", "refuse"]
    assert r["answer"].startswith(REFUSAL) and "Spring Boot" in r["answer"]
    assert len(llm.received) == 1


def test_irrelevant_docs_rewrite_once_then_refuse(fake_retriever):
    _, app, [r] = run_graph(
        [tool_call("search_knowledge_base", query="vLLM 是哪家公司开发的"),
         Grade(relevant=False, reason="没提到公司"),
         AIMessage("vLLM 开发团队"),
         Grade(relevant=False, reason="仍没有提到开发者")],
        ["vLLM 是哪家公司开发的"],
    )
    assert r["route"] == "retrieve_refuse"
    assert nodes(r) == ["router", "retrieve", "grade", "rewrite", "retrieve", "grade", "refuse"]
    assert r["queries"] == ["vLLM 是哪家公司开发的", "vLLM 开发团队"]
    assert fake_retriever == r["queries"]
    assert r["answer"].startswith(REFUSAL) and "检索到的资料不足以回答" in r["answer"]
    assert app.get_state({"configurable": {"thread_id": "t"}}).values["retries"] == graph_agent.MAX_RETRIES


def test_rewrite_then_relevant_generates():
    _, _, [r] = run_graph(
        [tool_call("search_knowledge_base", query="KV 缓存"),
         Grade(relevant=False, reason="不够具体"),
         AIMessage("KV Cache 是什么"),
         Grade(relevant=True, reason="有定义"),
         AIMessage("答案。来源：[1]")],
        ["KV 缓存"],
    )
    assert r["route"] == "retrieve"
    assert nodes(r) == ["router", "retrieve", "grade", "rewrite", "retrieve", "grade", "generate"]


def test_empty_retrieval_skips_grade_llm():
    llm, _, [r] = run_graph(
        [tool_call("search_knowledge_base", query="空查询"), AIMessage("还是空的")],
        ["空查询"],
    )
    assert r["route"] == "retrieve_refuse"
    assert "没有检索到任何资料" in r["answer"]
    assert len(llm.received) == 2


def test_no_tool_call_falls_back_to_direct_without_extra_call():
    llm, _, [r] = run_graph([AIMessage("你好呀")], ["你好"])
    assert r["route"] == "direct"
    assert r["answer"] == "你好呀"
    assert len(llm.received) == 1


def test_multi_turn_history_and_per_turn_reset():
    llm, app, [first, second] = run_graph(
        [tool_call("search_knowledge_base", query="LangChain 的 Retriever 是什么"),
         Grade(relevant=True, reason="有"), AIMessage("Retriever 是检索抽象。"),
         tool_call("refuse", reason="天气是实时信息")],
        ["LangChain 的 Retriever 是什么", "今天天气怎么样"],
    )
    second_router = router_inputs(llm)[1]
    assert [type(m) for m in second_router] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert second_router[2].content == "Retriever 是检索抽象。"
    # 上一轮的 docs 不能泄漏到这一轮
    assert second["docs"] == [] and second["route"] == "refuse"
    assert len(app.get_state({"configurable": {"thread_id": "t"}}).values["messages"]) == 4


def test_separate_threads_do_not_share_history():
    llm = ScriptedChatModel(responses=[tool_call("answer_directly", reason="寒暄"), AIMessage("你好"),
                                       tool_call("answer_directly", reason="寒暄"), AIMessage("你好")])
    app = graph_agent.build_graph(llm=llm)
    graph_agent.ask(app, "你好", "a")
    graph_agent.ask(app, "你好", "b")
    assert len(router_inputs(llm)[1]) == 2


@pytest.mark.parametrize("responses, expected", [
    ([tool_call("search_knowledge_base", query="KV Cache 是什么"), AIMessage("答案。来源：[1]")], "retrieve"),
    ([tool_call("search_knowledge_base", query="vLLM 公司"), AIMessage(f"{REFUSAL}，资料没提到公司。")],
     "retrieve_refuse"),
    ([AIMessage("你好！我是知识库助手。")], "direct"),
    ([AIMessage(f"**{REFUSAL}**，这是 Java 问题。")], "refuse"),
])
def test_create_agent_route_inference(responses, expected):
    llm = ScriptedChatModel(responses=list(responses))
    result = agent.ask(agent.build_agent(llm=llm), "问题", "t")
    assert result["route"] == expected
    assert llm.responses == []
    if expected.startswith("retrieve"):
        assert result["docs"] == DOCS and result["queries"]
