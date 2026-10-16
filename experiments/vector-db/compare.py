"""正确性检查：同一批向量、同一个查询向量，比较 Milvus、PgVector、Chroma 的 top-k 和暴力检索是否一致。

Milvus 用 FLAT、PgVector 不建索引，两者都是精确检索，结果应该和暴力检索完全相同；
Chroma 用的是 HNSW 近似检索，个别位置可能不同。

先运行过 milvus_demo.py 和 pgvector_demo.py（建好库），再在 rag-demo 目录下：
  python experiments/vector-db/compare.py
哪个库没启动就跳过哪个。
"""
import milvus_demo
import pgvector_demo
from common import QUESTIONS, TOP_K, chroma_top_k, embed_query, exact_top_k, load_chunks


def try_backend(name: str, connect):
    try:
        return connect()
    except SystemExit as e:
        print(f"（跳过 {name}：{e}）")
        return None


def main() -> None:
    rows = load_chunks()
    milvus = try_backend("Milvus", milvus_demo.connect)
    if milvus is not None and not milvus.has_collection(milvus_demo.COLLECTION):
        print("（跳过 Milvus：还没建库，先运行 milvus_demo.py）")
        milvus = None
    pg = try_backend("PgVector", pgvector_demo.connect)
    if pg is not None and pg.execute("SELECT to_regclass(%s)", (pgvector_demo.TABLE,)).fetchone()[0] is None:
        print("（跳过 PgVector：还没建表，先运行 pgvector_demo.py）")
        pg = None

    backends = {"Chroma": chroma_top_k}
    if milvus is not None:
        backends["Milvus"] = lambda q, k: milvus_demo.search_ids(milvus, q, k)
    if pg is not None:
        backends["PgVector"] = lambda q, k: pgvector_demo.search_ids(pg, q, k)

    same = {name: 0 for name in backends}
    for question in QUESTIONS:
        qvec = embed_query(question)
        expected = exact_top_k(rows, qvec, TOP_K)
        print(f"\n{question}")
        print(f"  {'暴力检索':<10}{expected}")
        for name, top_k in backends.items():
            got = top_k(qvec, TOP_K)
            same[name] += got == expected
            print(f"  {name:<10}{got}  {'一致' if got == expected else '不一致'}")

    print(f"\n=== top-{TOP_K} 和暴力检索完全一致的题数（共 {len(QUESTIONS)} 题）===")
    for name, n in same.items():
        print(f"  {name:<10}{n}")


if __name__ == "__main__":
    main()
