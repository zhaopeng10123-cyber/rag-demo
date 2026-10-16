"""Milvus 实操：建 collection -> 插入主线现成的片段和 bge 向量 -> 按主键取、按标量条件查 -> 向量检索。

先启动 Milvus standalone（在 experiments/vector-db 目录下）：
  docker compose -f milvus-standalone-docker-compose.yaml up -d
再在 rag-demo 目录下：
  python experiments/vector-db/milvus_demo.py
  python experiments/vector-db/milvus_demo.py "SGLang 和 vLLM 有什么区别？"
"""
import argparse
import os

from common import DEMO_QUESTION, DIM, TOP_K, embed_query, load_chunks, preview
from pymilvus import DataType, MilvusClient

URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION = "rag_demo_chunks"
# VARCHAR 的 max_length 按字节计，中文一个字占 3 字节，500 字的片段约 1.5 KB
TEXT_MAX_BYTES = 8192
OUTPUT_FIELDS = ["source", "section", "text"]


def connect() -> MilvusClient:
    try:
        return MilvusClient(uri=URI, timeout=10)
    except Exception as e:
        raise SystemExit(f"连不上 Milvus（{URI}）：{type(e).__name__}。先在 experiments/vector-db 下运行 "
                         "docker compose -f milvus-standalone-docker-compose.yaml up -d") from e


def build(client: MilvusClient, rows: list[dict]) -> None:
    longest = max(len(r["text"].encode("utf-8")) for r in rows)
    assert longest <= TEXT_MAX_BYTES, f"最长片段 {longest} 字节，超过了 text 字段的 max_length"

    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("text", DataType.VARCHAR, max_length=TEXT_MAX_BYTES)
    schema.add_field("source", DataType.VARCHAR, max_length=256)
    schema.add_field("section", DataType.VARCHAR, max_length=1024)

    # 两百多条数据用 FLAT（暴力精确检索）就够了；向量已归一化，COSINE 和 IP 结果相同
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
    # 传了 index_params，create_collection 会顺带建索引并 load；
    # 不传的话要自己 create_index + load_collection，否则 search 会报 collection not loaded。
    # 默认一致性级别是 Bounded，刚插入的数据可能暂时搜不到，演示用 Strong
    client.create_collection(COLLECTION, schema=schema, index_params=index_params, consistency_level="Strong")

    data = [{"id": r["id"], "vector": r["vector"].tolist(), "text": r["text"],
             "source": r["source"], "section": r["section"]} for r in rows]
    res = client.insert(COLLECTION, data)
    print(f"[存] 插入 {res['insert_count']} 条 -> collection '{COLLECTION}'（{DIM} 维，FLAT + COSINE）")


def search(client: MilvusClient, qvec: list[float], k: int = TOP_K, filter: str = "") -> list[dict]:
    return client.search(COLLECTION, data=[qvec], limit=k, filter=filter,
                         output_fields=OUTPUT_FIELDS, search_params={"metric_type": "COSINE"})[0]


def search_ids(client: MilvusClient, qvec: list[float], k: int = TOP_K) -> list[int]:
    return [hit["id"] for hit in search(client, qvec, k)]


def print_hits(hits: list[dict]) -> None:
    # COSINE 下 distance 其实是相似度，越大越相似；Chroma 返回的是距离，越小越相似
    for hit in hits:
        e = hit["entity"]
        print(f"  {hit['distance']:.3f}  #{hit['id']:<4} [{e['source']}] {e['section']}")
        print(f"         {preview(e['text'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*")
    args = parser.parse_args()

    client = connect()
    rows = load_chunks()
    build(client, rows)

    print("\n[取] 按主键 get：")
    for r in client.get(COLLECTION, ids=[0, 1], output_fields=["source", "section"]):
        print(f"  #{r['id']} [{r['source']}] {r['section'] or '（文档开头）'}")
    count = client.query(COLLECTION, filter="", output_fields=["count(*)"])[0]["count(*)"]
    print(f"[取] count(*) = {count}")
    print('[取] 标量条件 query：source like "03-%"，前 3 条')
    for r in client.query(COLLECTION, filter='source like "03-%"', output_fields=["section"], limit=3):
        print(f"  #{r['id']} {r['section'] or '（文档开头）'}")

    question = " ".join(args.question) or DEMO_QUESTION
    qvec = embed_query(question)
    print(f"\n[查] search：{question}")
    print_hits(search(client, qvec))

    print(f'\n[查] 带过滤的 search（只在 03 号文档里找）：{question}')
    print_hits(search(client, qvec, filter='source like "03-%"'))


if __name__ == "__main__":
    main()
