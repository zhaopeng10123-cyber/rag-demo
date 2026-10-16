"""Milvus / PgVector 实操的共用部分：从主线 Chroma 读出现成的片段和向量，查询向量化，暴力检索参照。

直接读 Chroma 里存好的向量，不重新向量化，三个库里的向量逐位相同，检索结果才有可比性。
需要先在 rag-demo 目录下运行过 python ingest.py。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chromadb  # noqa: E402
import config  # noqa: E402

DIM = 512  # bge-small-zh-v1.5
TOP_K = config.TOP_K
# 02 号（SGLang 的 RadixAttention）和 03 号（vLLM 的 PagedAttention）文档都讲到，适合演示按 source 过滤
DEMO_QUESTION = "KV Cache 是怎么管理和复用的？"
QUESTIONS = [
    "什么是 LCEL？",
    "RadixAttention 是怎么复用 KV Cache 的？",
    "vLLM 的连续批处理和静态批处理有什么区别？",
    "什么情况下需要用 LangGraph？",
    "TensorRT-LLM 和 vLLM、SGLang 相比该怎么选？",
]


def load_chunks() -> list[dict]:
    """返回 [{id, text, source, section, vector}]，id 就是 ingest.py 里的 chunk_id。"""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    coll = client.get_collection(config.COLLECTION)
    res = coll.get(include=["embeddings", "documents", "metadatas"])
    if not res["ids"]:
        raise SystemExit("主线 Chroma 是空的，先在 rag-demo 目录下运行 python ingest.py")
    rows = [
        {"id": int(m["chunk_id"]), "text": doc, "source": m["source"], "section": m["section"],
         "vector": np.asarray(vec, dtype=np.float32)}
        for doc, m, vec in zip(res["documents"], res["metadatas"], res["embeddings"])
    ]
    return sorted(rows, key=lambda r: r["id"])


def embed_query(question: str) -> list[float]:
    # 和主线一样：查询端加 bge 检索指令、归一化
    return config.get_embeddings().embed_query(question)


def exact_top_k(rows: list[dict], qvec: list[float], k: int = TOP_K) -> list[int]:
    """暴力计算余弦相似度（向量已归一化，点积即余弦），作为"正确答案"。"""
    matrix = np.stack([r["vector"] for r in rows])
    scores = matrix @ np.asarray(qvec, dtype=np.float32)
    return [rows[i]["id"] for i in np.argsort(-scores)[:k]]


def chroma_top_k(qvec: list[float], k: int = TOP_K) -> list[int]:
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    res = client.get_collection(config.COLLECTION).query(query_embeddings=[qvec], n_results=k, include=["metadatas"])
    return [int(m["chunk_id"]) for m in res["metadatas"][0]]


def preview(text: str, n: int = 60) -> str:
    # ingest.py 把【章节】拼在了正文开头，打印时已经单独显示了 section，这里去掉
    if text.startswith("【"):
        text = text.split("\n", 1)[-1]
    return text.replace("\n", " ")[:n]
