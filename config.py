import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
# 必须在导入 huggingface 相关库之前加载，.env 里的 HF_ENDPOINT（镜像）才会生效
load_dotenv(ROOT / ".env")

KB_DIR = Path(os.getenv("KB_DIR", ROOT / "data" / "knowledge-base"))
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "knowledge_base"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
# download_model.py 下载到本地的模型优先，避免每次联网检查 HuggingFace
_local_model = ROOT / "models" / EMBEDDING_MODEL.split("/")[-1]
if _local_model.is_dir():
    EMBEDDING_MODEL = str(_local_model)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("TOP_K", "5"))
# bge 中文模型官方推荐：短查询加检索指令，文档端不加
QUERY_INSTRUCTION = os.getenv("QUERY_INSTRUCTION", "为这个句子生成表示以用于检索相关文章：")


def get_llm(temperature: float = 0):
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("LLM_API_KEY")
    if not api_key or api_key == "sk-xxx":
        raise RuntimeError("请先在 rag-demo/.env 中配置 LLM_API_KEY（参考 .env.example）")
    return ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        api_key=api_key,
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        temperature=temperature,
    )


_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        # query_encode_kwargs 会整体替换 encode_kwargs，所以归一化要写两遍
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            query_encode_kwargs={"normalize_embeddings": True, "prompt": QUERY_INSTRUCTION},
        )
    return _embeddings


def get_vectorstore(collection: str = COLLECTION):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
