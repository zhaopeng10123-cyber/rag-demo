"""PgVector 实操：建表 -> 插入主线现成的片段和 bge 向量 -> 按主键取、按 SQL 条件查 -> 向量检索。

先启动 Postgres + pgvector（在 experiments/vector-db 目录下）：
  docker compose -f pgvector-docker-compose.yaml up -d
再在 rag-demo 目录下：
  python experiments/vector-db/pgvector_demo.py
  python experiments/vector-db/pgvector_demo.py "SGLang 和 vLLM 有什么区别？"
"""
import argparse
import os

import numpy as np
import psycopg
from common import DEMO_QUESTION, DIM, TOP_K, embed_query, load_chunks, preview
from pgvector.psycopg import register_vector

DSN = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5432/rag")
TABLE = "rag_demo_chunks"
# <=> 余弦距离，<-> L2 距离，<#> 负内积。没建向量索引时就是精确的全表扫描
SEARCH_SQL = f"""
    SELECT id, source, section, content, embedding <=> %(q)s AS distance
    FROM {TABLE}
    WHERE source LIKE %(source)s
    ORDER BY embedding <=> %(q)s
    LIMIT %(k)s
"""


def connect() -> psycopg.Connection:
    try:
        conn = psycopg.connect(DSN, autocommit=True, connect_timeout=10)
    except psycopg.OperationalError as e:
        raise SystemExit(f"连不上 Postgres（{DSN}）。先在 experiments/vector-db 下运行 "
                         "docker compose -f pgvector-docker-compose.yaml up -d") from e
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # 要在扩展创建之后注册，它会去库里查 vector 类型的 OID，之后 numpy 数组就能直接当参数传
    register_vector(conn)
    return conn


def build(conn: psycopg.Connection, rows: list[dict]) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
    conn.execute(f"""
        CREATE TABLE {TABLE} (
            id        integer PRIMARY KEY,
            source    text NOT NULL,
            section   text NOT NULL,
            content   text NOT NULL,
            embedding vector({DIM}) NOT NULL
        )""")
    with conn.cursor() as cur:
        cur.executemany(f"INSERT INTO {TABLE} (id, source, section, content, embedding) VALUES (%s, %s, %s, %s, %s)",
                        [(r["id"], r["source"], r["section"], r["text"], r["vector"]) for r in rows])
    # 数据量大时再建 HNSW 索引：CREATE INDEX ON rag_demo_chunks USING hnsw (embedding vector_cosine_ops)
    print(f"[存] 插入 {len(rows)} 行 -> 表 {TABLE}（vector({DIM})，无索引，精确检索）")


def search(conn: psycopg.Connection, qvec: list[float], k: int = TOP_K, source: str = "%") -> list[tuple]:
    params = {"q": np.asarray(qvec, dtype=np.float32), "source": source, "k": k}
    return conn.execute(SEARCH_SQL, params).fetchall()


def search_ids(conn: psycopg.Connection, qvec: list[float], k: int = TOP_K) -> list[int]:
    return [row[0] for row in search(conn, qvec, k)]


def print_hits(hits: list[tuple]) -> None:
    for id_, source, section, content, distance in hits:
        print(f"  {1 - distance:.3f}  #{id_:<4} [{source}] {section}")
        print(f"         {preview(content)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*")
    args = parser.parse_args()

    conn = connect()
    rows = load_chunks()
    build(conn, rows)

    print("\n[取] 按主键 SELECT：")
    for id_, source, section in conn.execute(f"SELECT id, source, section FROM {TABLE} WHERE id = ANY(%s)", ([0, 1],)):
        print(f"  #{id_} [{source}] {section or '（文档开头）'}")
    print(f"[取] count(*) = {conn.execute(f'SELECT count(*) FROM {TABLE}').fetchone()[0]}")
    print("[取] 普通 SQL 条件：source LIKE '03-%'，前 3 条")
    for id_, section in conn.execute(f"SELECT id, section FROM {TABLE} WHERE source LIKE %s ORDER BY id LIMIT 3",
                                     ("03-%",)):
        print(f"  #{id_} {section or '（文档开头）'}")

    question = " ".join(args.question) or DEMO_QUESTION
    qvec = embed_query(question)
    # 打印的是 1 - 余弦距离，和 Milvus COSINE 的分数同一口径
    print(f"\n[查] 向量检索：{question}")
    print_hits(search(conn, qvec))

    print(f"\n[查] 加 WHERE 条件的向量检索（只在 03 号文档里找）：{question}")
    print_hits(search(conn, qvec, source="03-%"))


if __name__ == "__main__":
    main()
