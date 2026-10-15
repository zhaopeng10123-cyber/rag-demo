"""Agent 共享层：检索 tool、路由决策 tool、统一结果结构、Prompt 和 LLM 调用计数。

agent.py（create_agent）和 graph_agent.py（StateGraph）都从这里取 tool 和 Prompt，保证两者可比。
"""
from pathlib import Path
from typing import Any, Literal, TypedDict
from uuid import UUID

import config
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.tools import tool
from rag_chain import REFUSAL, format_docs, get_retriever

KB_TOPICS = ("LangChain、LlamaIndex、RAG（检索增强生成）、Agent、Function Calling、LangGraph、"
             "Embedding、向量数据库、SGLang、vLLM、KV Cache、连续批处理、Triton、TensorRT、大模型推理与部署")

DOCS_DIR = config.ROOT / "docs"

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


@tool(response_format="content_and_artifact", parse_docstring=True)
def search_knowledge_base(query: str) -> tuple[str, list[Document]]:
    """检索个人技术笔记知识库。知识库覆盖：LangChain、LlamaIndex、RAG、Agent、Function Calling、
    LangGraph、Embedding、向量数据库、SGLang、vLLM、KV Cache、连续批处理、Triton、TensorRT、
    大模型推理与部署。
    只要问题和这些主题有关就调用；寒暄、与这些主题无关的问题不要调用。

    Args:
        query: 脱离对话上下文也能看懂的完整检索问题，把“它”“这个”等指代替换成具体名称。
    """
    docs = _get_retriever().invoke(query)
    if not docs:
        return "没有检索到任何资料。", []
    return format_docs(docs), docs


@tool(parse_docstring=True)
def answer_directly(reason: str) -> str:
    """问题是寒暄、致谢、告别，或者在问助手自己是谁、能做什么，不需要查资料时调用。

    Args:
        reason: 一句话说明为什么可以直接回答。
    """
    return reason


@tool(parse_docstring=True)
def refuse(reason: str) -> str:
    """问题超出知识库范围时调用：和知识库主题无关的专业问题（如 Java、前端、数据库运维）、
    实时信息（天气、新闻、今天几号）、个人隐私，或其他知识库不可能包含的内容。

    Args:
        reason: 一句话说明为什么拒答。
    """
    return reason


Route = Literal["retrieve", "direct", "refuse", "retrieve_refuse"]


class AgentResult(TypedDict):
    route: Route
    queries: list[str]
    docs: list[Document]
    answer: str
    trace: list[str]
    llm_calls: int
    tokens: int
    latency: float


def refusal_text(reason: str = "") -> str:
    return f"{REFUSAL}。{reason.strip()}" if reason.strip() else f"{REFUSAL}。"


def is_refusal(answer: str) -> bool:
    # 模型偶尔会在拒答话术前加 Markdown 加粗或一句客套话，只看开头一小段
    return REFUSAL in answer[:40]


def export_graph(app, name: str) -> Path:
    """把编译好的图导出为 docs/<name>.mmd；mermaid.ink 可访问时再额外导出 PNG。"""
    DOCS_DIR.mkdir(exist_ok=True)
    graph = app.get_graph()
    mmd = DOCS_DIR / f"{name}.mmd"
    mmd.write_text(graph.draw_mermaid(), encoding="utf-8")
    try:
        (DOCS_DIR / f"{name}.png").write_bytes(graph.draw_mermaid_png())
        print(f"已导出 {mmd} 和 {name}.png")
    except Exception as e:
        print(f"已导出 {mmd}（PNG 生成失败：{type(e).__name__}，可把 .mmd 粘贴到 mermaid.live 查看）")
    return mmd


AGENT_SYSTEM_PROMPT = f"""你是一个个人技术笔记知识库助手。知识库覆盖：{KB_TOPICS}。

按问题类型选择行为：
1. 和知识库主题有关的问题：必须先调用 search_knowledge_base 检索，query 写成脱离上下文的完整问题。
   只根据检索到的资料回答，不要用你自己的知识补充；答案末尾另起一行写“来源：”并列出用到的资料编号，例如“来源：[1][3]”。
   如果资料不足以回答，回答以“{REFUSAL}”开头，再用一句话说明缺少什么。
2. 寒暄、致谢，或者问你是谁、能做什么：不要调用工具，简短回答，并说明你是基于技术笔记的知识库助手。
3. 其他问题（和知识库主题无关的专业问题、天气新闻等实时信息）：不要调用工具，
   回答以“{REFUSAL}”开头，再用一句话说明原因。不要用你自己的知识作答。"""

ROUTER_PROMPT = f"""你是知识库问答系统的路由器，只负责决定如何处理用户的最新问题，不要回答问题本身。
知识库覆盖：{KB_TOPICS}。

必须且只能调用一个工具：
- search_knowledge_base：问题和知识库主题有关。结合对话历史，把问题改写成完整、独立的检索问题作为 query；
- answer_directly：寒暄、致谢、告别，或者问助手是谁、能做什么；
- refuse：和知识库主题无关的专业问题、实时信息（天气、新闻、日期）等知识库不可能包含的内容。"""

DIRECT_PROMPT = f"""你是一个个人技术笔记知识库助手，知识库覆盖：{KB_TOPICS}。
用户在和你寒暄或询问你的身份、能力。用一两句话友好地回答，可以顺便提示用户能问哪些主题。
不要回答任何专业问题。"""

GRADE_PROMPT = """判断【参考资料】是否包含能回答【问题】的信息。只要有部分资料能直接支撑回答就算相关；
资料只是提到了相似的词、但没有回答问题所需的内容，算不相关。

【问题】
{question}

【参考资料】
{context}"""

REWRITE_PROMPT = """用下面的检索问题在技术笔记知识库里没有找到相关资料。请把它改写成一个更适合检索的问题：
补全缩写、换成更常见的说法、或者聚焦到最核心的概念。只输出改写后的问题，不要任何解释。

原问题：{query}"""


class CallCounter(BaseCallbackHandler):
    """统计一次问答里的 LLM 调用次数和 token 消耗，挂在 config["callbacks"] 上即可。"""

    def __init__(self) -> None:
        self.calls = 0
        self.tokens = 0
        self._streamed: dict[UUID, int] = {}

    def on_chat_model_start(self, serialized: dict, messages: list, **kwargs: Any) -> None:
        self.calls += 1

    def on_llm_new_token(self, token: str, *, chunk=None, run_id: UUID, **kwargs: Any) -> None:
        usage = getattr(getattr(chunk, "message", None), "usage_metadata", None) or {}
        if usage:
            self._streamed[run_id] = max(self._streamed.get(run_id, 0), usage.get("total_tokens", 0))

    def on_llm_end(self, response, *, run_id: UUID, **kwargs: Any) -> None:
        if run_id in self._streamed:
            # 部分 OpenAI 兼容服务（如硅基流动）每个流式分块都带累计用量，LangChain 合并分块时会把它们相加，
            # 所以流式调用取分块里的最大值
            self.tokens += self._streamed.pop(run_id)
            return
        for gens in response.generations:
            for gen in gens:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                self.tokens += usage.get("total_tokens", 0)
