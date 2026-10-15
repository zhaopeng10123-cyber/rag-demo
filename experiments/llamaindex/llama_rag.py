"""LlamaIndex 版 RAG，用于和主线 LangChain 实现做对照实验。

和主线共用：同一份知识库（config.KB_DIR）、同一个 Embedding 模型和检索指令、同一个 top-k、同一个 LLM 配置。
向量库单独放在 experiments/llamaindex/chroma_db/，不碰主线的 chroma_db/。

两种建库方式：
  aligned  直接复用 ingest.py 切好的片段，只换成 LlamaIndex 的 Embedding / 向量库 / 检索器。
           检索结果应该和 LangChain 完全一致，用来证明对照实验的"尺子"是公平的
  native   LlamaIndex 惯用管线：SimpleDirectoryReader -> MarkdownNodeParser -> SentenceSplitter

用法（在 rag-demo 目录下）：
  python experiments/llamaindex/llama_rag.py --build           建库（native，300 字）
  python experiments/llamaindex/llama_rag.py "什么是 LCEL？"   带引用的问答
"""
import argparse
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent.parent))

import chromadb  # noqa: E402
import config  # noqa: E402
from chromadb.api import ClientAPI  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from llama_index.core import PromptTemplate, SimpleDirectoryReader, StorageContext, VectorStoreIndex  # noqa: E402
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata  # noqa: E402
from llama_index.core.llms.callbacks import llm_completion_callback  # noqa: E402
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter, SentenceWindowNodeParser  # noqa: E402
from llama_index.core.query_engine import CitationQueryEngine  # noqa: E402
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode, TransformComponent  # noqa: E402
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: E402
from llama_index.vector_stores.chroma import ChromaVectorStore  # noqa: E402
from pydantic import Field, PrivateAttr  # noqa: E402
from rag_chain import REFUSAL  # noqa: E402

CHROMA_DIR = EXP_DIR / "chroma_db"
CHUNK_SIZE, CHUNK_OVERLAP = 300, 48  # 和 eval/evaluate.py --experiment 里 300 那一组一致
COLLECTION = "li_native"
# Chroma 默认的 HNSW 参数下，sentence-window 那组（2132 个节点，211 个重复向量）top-5 召回率只有 0.93~0.94，
# 每次重建漏掉的近邻还不一样，同一组实验两次能跑出不同排名。图建得更密后召回率 0.992 且可复现
HNSW = {"hnsw": {"space": "cosine", "ef_search": 500, "ef_construction": 400, "max_neighbors": 64}}


@lru_cache
def get_embed_model() -> HuggingFaceEmbedding:
    # 只有 model_name 恰好等于官方模型名时 LlamaIndex 才会自动加 bge 检索指令，
    # 用本地路径加载时会悄悄不加，所以必须显式传入
    return HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL, query_instruction=config.QUERY_INSTRUCTION,
                                normalize=True, device="cpu")


def get_vector_store(collection: str, reset: bool = False, client: ClientAPI | None = None) -> ChromaVectorStore:
    client = client or chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset and collection in [c.name for c in client.list_collections()]:
        client.delete_collection(collection)
    coll = client.get_or_create_collection(collection, configuration=HNSW)
    return ChromaVectorStore(chroma_collection=coll)


class AddSection(TransformComponent):
    """给 MarkdownNodeParser 切出的节点加上 section 元数据（h2 > h3 …，和 ingest.py 的 section_of 同一口径）。

    只给评估判断命中用，不参与向量化。必须在 SentenceSplitter 之前做：
    MarkdownNodeParser 的 header_path 不含当前节点自己的标题，二次切分后后面的片段就找不回标题了。
    """

    def __call__(self, nodes: list[BaseNode], **kwargs: Any) -> list[BaseNode]:
        stack: list[str] = []
        doc_id = None
        for node in nodes:
            if node.ref_doc_id != doc_id:
                stack, doc_id = [], node.ref_doc_id
            m = re.match(r"(#{1,6})\s+(.+)", node.get_content().lstrip())
            if m:
                level = len(m[1])
                stack = stack[:level - 1] + [""] * (level - 1 - len(stack)) + [m[2].strip()]
            node.metadata["section"] = " > ".join(t for t in stack[1:] if t)
            node.metadata["source"] = node.metadata.get("file_name", "")
            node.excluded_embed_metadata_keys = [*node.excluded_embed_metadata_keys, "section", "source"]
            node.excluded_llm_metadata_keys = [*node.excluded_llm_metadata_keys, "section", "source"]
        return nodes


def load_documents():
    return SimpleDirectoryReader(input_dir=str(config.KB_DIR), required_exts=[".md"], encoding="utf-8").load_data()


def native_nodes(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP,
                 keep_file_path: bool = True) -> list[BaseNode]:
    """LlamaIndex 惯用管线。

    keep_file_path=True 是默认行为：文件绝对路径会被拼进向量化文本。SentenceSplitter 切分时会先从 chunk_size
    里扣掉元数据长度（取向量化和 LLM 两份元数据里较长的），所以路径越长，片段正文越短。
    """
    docs = load_documents()
    if not keep_file_path:
        for doc in docs:
            doc.excluded_embed_metadata_keys.append("file_path")
            doc.excluded_llm_metadata_keys.append("file_path")
    nodes = MarkdownNodeParser().get_nodes_from_documents(docs)
    nodes = AddSection()(nodes)
    # tokenizer=list 让 chunk_size 按字符计，和 LangChain 的 RecursiveCharacterTextSplitter 同一口径；
    # 默认按 tiktoken 的 token 计，中文下长度对不上
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=list)
    return splitter(nodes)


def split_zh_sentences(text: str) -> list[str]:
    """按中文句末标点和换行分句。SentenceWindowNodeParser 默认的分句器是为英文设计的，中文段落会整段成为一句。"""
    return [s for s in re.split(r"(?<=[。！？!?\n])", text) if s.strip()]


def sentence_window_nodes(window_size: int = 3) -> list[BaseNode]:
    """每个句子一个节点，检索命中小句；前后各 window_size 句存在 metadata["window"] 里，作答前再替换回去。"""
    docs = load_documents()
    for doc in docs:
        doc.excluded_embed_metadata_keys.append("file_path")
        doc.excluded_llm_metadata_keys.append("file_path")
    nodes = AddSection()(MarkdownNodeParser().get_nodes_from_documents(docs))
    parser = SentenceWindowNodeParser(sentence_splitter=split_zh_sentences, window_size=window_size)
    return parser(nodes)


def aligned_nodes(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[BaseNode]:
    """复用 ingest.py 的切分结果，向量化的文本和 LangChain 完全相同。"""
    from ingest import load_markdown, split_documents

    keys = ["source", "section"]
    return [TextNode(text=c.page_content, metadata={k: c.metadata[k] for k in keys},
                     excluded_embed_metadata_keys=keys, excluded_llm_metadata_keys=keys)
            for c in split_documents(load_markdown(), chunk_size, chunk_overlap)]


def build_index(nodes: list[BaseNode], collection: str, client: ClientAPI | None = None) -> VectorStoreIndex:
    storage = StorageContext.from_defaults(vector_store=get_vector_store(collection, reset=True, client=client))
    return VectorStoreIndex(nodes, storage_context=storage, embed_model=get_embed_model())


def load_index(collection: str = COLLECTION) -> VectorStoreIndex:
    return VectorStoreIndex.from_vector_store(get_vector_store(collection), embed_model=get_embed_model())


def to_document(n: NodeWithScore | BaseNode) -> Document:
    """转成 LangChain Document，这样能直接用 eval/evaluate.py 的 first_hit_rank 判断命中。"""
    node = n.node if isinstance(n, NodeWithScore) else n
    return Document(page_content=node.get_content(), metadata={"source": node.metadata.get("source", ""),
                                                               "section": node.metadata.get("section", "")})


class OpenAICompatibleLLM(CustomLLM):
    """最小的 OpenAI 兼容 LLM。

    官方的 llama-index-llms-openai-like 依赖 openai<3，装上会把主环境里 langchain-openai 用的 openai 3.x 降级。
    """

    model: str
    api_base: str
    api_key: str = Field(repr=False)
    temperature: float = 0
    context_window: int = 32768
    _client: Any = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from openai import OpenAI

        self._client = OpenAI(base_url=self.api_base, api_key=self.api_key)

    @classmethod
    def from_env(cls) -> "OpenAICompatibleLLM":
        lc = config.get_llm()  # 复用主线的 .env 读取和校验
        return cls(model=lc.model_name, api_base=str(lc.openai_api_base), api_key=lc.openai_api_key.get_secret_value())

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=self.context_window, num_output=2048, model_name=self.model)

    def _create(self, prompt: str, stream: bool):
        return self._client.chat.completions.create(model=self.model, temperature=self.temperature, stream=stream,
                                                    messages=[{"role": "user", "content": prompt}])

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=self._create(prompt, stream=False).choices[0].message.content or "")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        def gen():
            text = ""
            for chunk in self._create(prompt, stream=True):
                delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                text += delta
                yield CompletionResponse(text=text, delta=delta)
        return gen()


# 规则和 rag_chain.SYSTEM_PROMPT 一致；CitationQueryEngine 会把资料编号成 "Source n:"
RULES = f"""你是一个知识库问答助手，只能根据下面编号的参考资料回答问题。

规则：
1. 只使用参考资料中的信息，不要用你自己的知识补充；
2. 如果参考资料不足以回答，直接回答“{REFUSAL}”，不要编造；
3. 回答用中文，条理清晰；
4. 在答案末尾另起一行，写“来源：”并列出你实际用到的资料编号（Source n 写成 [n]），例如“来源：[1][3]”。"""

QA_TEMPLATE = PromptTemplate(RULES + "\n\n【参考资料】\n{context_str}\n\n问题：{query_str}\n回答：")
REFINE_TEMPLATE = PromptTemplate(
    RULES + "\n\n已有的回答：\n{existing_answer}\n\n下面是补充的参考资料，请据此完善已有的回答；"
            "如果补充资料没有帮助，原样输出已有的回答。\n\n【参考资料】\n{context_msg}\n\n问题：{query_str}\n回答：")


def build_query_engine(index: VectorStoreIndex, llm=None, top_k: int = config.TOP_K) -> CitationQueryEngine:
    return CitationQueryEngine.from_args(
        index, llm=llm or OpenAICompatibleLLM.from_env(), similarity_top_k=top_k,
        # 足够大，检索到的片段不会被再切一次，一个片段对应一个编号
        citation_chunk_size=2048, citation_qa_template=QA_TEMPLATE, citation_refine_template=REFINE_TEMPLATE,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    if args.build:
        nodes = native_nodes()
        build_index(nodes, COLLECTION)
        lengths = [len(n.get_content()) for n in nodes]
        print(f"片段 {len(nodes)} 个，平均 {sum(lengths) / len(lengths):.0f} 字 -> {CHROMA_DIR} / {COLLECTION}")
        return

    question = " ".join(args.question) or "LCEL 是什么？"
    response = build_query_engine(load_index()).query(question)
    print(f"问：{question}\n答：{str(response).strip()}\n检索到的片段：")
    for i, n in enumerate(response.source_nodes, 1):
        print(f"  [{i}] {n.node.metadata.get('source')} > {n.node.metadata.get('section')}  (相似度 {n.score:.3f})")


if __name__ == "__main__":
    main()
