"""MCP server：把知识库检索和 RAG 问答暴露成 MCP 工具，任何 MCP 客户端（Inspector、Claude Desktop、Cursor 等）都能调用。

工具：
  search_knowledge(query, top_k)  只检索，返回片段和来源，由调用方自己组织答案
  answer(question)                走完整的 RAG 链（检索 + LLM 生成 + 拒答），返回答案和引用
资源：
  kb://topics                     知识库覆盖的主题和文档列表

用法（在 rag-demo 目录下）：
  mcp dev mcp_server.py           打开 MCP Inspector 调试
  python mcp_server.py            stdio 方式运行，供客户端以子进程启动
  python mcp_server.py --http     Streamable HTTP 方式运行：http://127.0.0.1:8001/mcp

stdio 模式下 stdout 是协议通道，这个文件里不能 print，日志一律走 stderr。
"""
import argparse
from typing import Annotated

import anyio
import config
# Windows 上 stdio server 有一个线程一直阻塞在 stdin 上读消息，这时再首次导入 numpy、torch 这类带 C 扩展的库，
# 实测会一直卡住。所以要在开始读 stdin 之前、也就是模块导入时就把检索和问答用到的库导入（约 10 秒），
# 模型本身仍然在第一次调用时才加载（不到 1 秒）
import langchain_chroma  # noqa: F401
import langchain_huggingface  # noqa: F401
import langchain_openai  # noqa: F401
import sentence_transformers  # noqa: F401
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field
from rag_chain import REFUSAL, build_rag_chain
from tools import KB_TOPICS, is_refusal

HTTP_PORT = 8001  # api.py 的 uvicorn 默认占用 8000

_store = None
_chain: Runnable | None = None


def search_docs(query: str, k: int) -> list[Document]:
    global _store
    if _store is None:
        _store = config.get_vectorstore()
    return _store.similarity_search(query, k=k)


def _rag_chain() -> Runnable:
    global _chain
    if _chain is None:
        retriever = RunnableLambda(lambda q: search_docs(q, config.TOP_K))
        _chain = build_rag_chain(retriever=retriever, llm=config.get_llm())
    return _chain


mcp = FastMCP(
    "rag-demo",
    instructions=f"个人技术笔记知识库（中文），覆盖：{KB_TOPICS}。"
                 "需要引用资料自己作答时用 search_knowledge；只想要一个现成的带引用答案时用 answer。",
    port=HTTP_PORT,
)


class Chunk(BaseModel):
    id: int = Field(description="片段编号，从 1 开始，按相关度从高到低排列")
    source: str = Field(description="来源文件名")
    section: str = Field(description="章节路径，例如“第 2 章 推理的两个阶段与 KV Cache > 2.3 什么是 KV Cache”；文档开头为空")
    content: str = Field(description="片段正文，开头的【】里是章节路径")


class SearchResult(BaseModel):
    query: str
    chunks: list[Chunk] = Field(description="检索到的片段；为空表示知识库里没有找到任何资料")


class Source(BaseModel):
    id: int = Field(description="对应答案里的 [n]")
    source: str
    section: str


class Answer(BaseModel):
    answer: str = Field(description="中文答案，末尾的“来源：[n]”对应 sources 里的编号")
    refused: bool = Field(description=f"为 true 表示知识库资料不足，答案以“{REFUSAL}”开头，不要再据此作答")
    sources: list[Source] = Field(description="生成答案时提供给模型的全部片段，不一定每条都被引用")


def to_chunks(docs: list[Document]) -> list[Chunk]:
    return [Chunk(id=i, source=d.metadata.get("source", ""), section=d.metadata.get("section", ""),
                  content=d.page_content) for i, d in enumerate(docs, 1)]


# 工具说明是给调用方的模型看的：什么时候用、什么时候不用、返回什么、拿到结果后该怎么做。
# 要插入 KB_TOPICS，f-string 不能当 docstring，所以通过 description 传入
SEARCH_DESCRIPTION = f"""在个人技术笔记知识库里做向量检索，返回最相关的原文片段和来源，不调用 LLM。

知识库覆盖：{KB_TOPICS}。问题和这些主题有关时调用；寒暄、和这些主题无关的问题不要调用。

返回的是原文片段，需要你自己阅读后组织答案，并用片段的 id 标注引用。
片段按相关度排序，但不保证一定能回答问题：如果所有片段都答不上，应告诉用户知识库里没有相关信息，不要用自己的知识补充。"""

ANSWER_DESCRIPTION = f"""用知识库回答问题：先检索 {config.TOP_K} 个片段，再由服务端的 LLM 只根据这些片段生成带引用的答案。

知识库覆盖：{KB_TOPICS}。
资料不足以回答时会拒答（refused 为 true），这时不要再用自己的知识补答。

这个工具内部会调用一次 LLM，耗时十几秒。如果你自己能阅读资料并作答，优先用 search_knowledge，避免重复调用模型；
只有需要一个现成的、带引用的答案时才用这个工具。"""


@mcp.tool(description=SEARCH_DESCRIPTION,
          annotations=ToolAnnotations(title="检索知识库", readOnlyHint=True, openWorldHint=False))
async def search_knowledge(
    query: Annotated[str, Field(min_length=1, max_length=500,
                                description="完整、独立的中文检索问题。把对话里的“它”“这个”替换成具体名称，"
                                            "例如“RadixAttention 是怎么复用 KV Cache 的”，不要只写“它怎么复用”")],
    top_k: Annotated[int, Field(ge=1, le=10, description="返回的片段数")] = config.TOP_K,
) -> SearchResult:
    docs = await anyio.to_thread.run_sync(search_docs, query, top_k)
    return SearchResult(query=query, chunks=to_chunks(docs))


@mcp.tool(description=ANSWER_DESCRIPTION, annotations=ToolAnnotations(title="知识库问答", readOnlyHint=True))
async def answer(
    question: Annotated[str, Field(min_length=1, max_length=1000,
                                   description="用户的完整中文问题，脱离对话上下文也能看懂")],
) -> Answer:
    result = await anyio.to_thread.run_sync(_rag_chain().invoke, question)
    sources = [Source(id=c.id, source=c.source, section=c.section) for c in to_chunks(result["docs"])]
    text = result["answer"].strip()
    return Answer(answer=text, refused=is_refusal(text), sources=sources)


@mcp.resource("kb://topics", name="知识库主题", mime_type="text/markdown")
def kb_topics() -> str:
    """知识库覆盖的主题和包含的文档。"""
    docs = "\n".join(f"- {p.name}" for p in sorted(config.KB_DIR.glob("*.md")))
    return f"# 知识库主题\n\n{KB_TOPICS}\n\n# 文档\n\n{docs}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help=f"用 Streamable HTTP 运行在 {HTTP_PORT} 端口")
    args = parser.parse_args()
    mcp.run("streamable-http" if args.http else "stdio")


if __name__ == "__main__":
    main()
