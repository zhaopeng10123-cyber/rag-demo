# Milvus / PgVector 实操

把主线 Chroma 里现成的 234 个片段和 bge 向量（512 维）分别灌进 Milvus 和 PgVector，跑通"存、取、查"：

- **存**：建 collection / 表，插入片段、向量和元数据
- **取**：按主键取、按标量条件查、count
- **查**：向量相似检索，以及带元数据过滤的向量检索

不接评估集、不调索引参数。最后用 `compare.py` 检查三个库的 top-5 和暴力检索是否一致，确认数据存对了。

这个目录下全部是新增文件，主线代码一行没改。

## 怎么跑

需要 Docker。Windows 上装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)，后端选 WSL2。国内拉镜像慢的话，在 Docker Desktop 的 Docker Engine 设置里配镜像加速；Milvus 的 etcd 镜像在 quay.io 上，也可能需要代理。

```bash
# 在 rag-demo 目录下。需要主线已经运行过 python ingest.py
pip install -r experiments/vector-db/requirements.txt

cd experiments/vector-db
docker compose -f milvus-standalone-docker-compose.yaml up -d   # etcd + minio + milvus 三个容器
docker compose -f pgvector-docker-compose.yaml up -d            # 一个 Postgres 容器
cd ../..

python experiments/vector-db/milvus_demo.py       # 存、取、查
python experiments/vector-db/pgvector_demo.py
python experiments/vector-db/compare.py           # 和暴力检索、Chroma 对比 top-5
```

两个 demo 脚本都可以跟一个问题，例如 `python experiments/vector-db/milvus_demo.py "SGLang 和 vLLM 有什么区别？"`。每次运行都会删掉旧的 collection / 表再重建。

Milvus 第一次启动要等健康检查通过（`docker ps` 里 `milvus-standalone` 显示 `healthy`），大约一两分钟。

连接地址可以用环境变量覆盖：`MILVUS_URI`（默认 `http://localhost:19530`）、`PG_DSN`（默认 `postgresql://postgres:postgres@localhost:5432/rag`）。

用完清理：

```bash
cd experiments/vector-db
docker compose -f milvus-standalone-docker-compose.yaml down    # 数据在 volumes/ 下（已加入 .gitignore），删掉这个目录才会清空
docker compose -f pgvector-docker-compose.yaml down -v          # -v 同时删除命名卷里的数据
```

## 数据怎么对应

向量直接从主线 Chroma 读出来，不重新向量化，三个库里的向量逐位相同。查询向量用主线的 `config.get_embeddings().embed_query()`，同样加 bge 检索指令、归一化。

| 字段 | Chroma（主线） | Milvus | PgVector |
|---|---|---|---|
| 主键 | `knowledge_base-{chunk_id}` | `id INT64`，取 `chunk_id` | `id integer PRIMARY KEY` |
| 向量 | 自动管理 | `vector FLOAT_VECTOR(dim=512)` | `embedding vector(512)` |
| 正文 | document | `text VARCHAR(max_length=8192)` | `content text` |
| 元数据 | metadata 字典 | `source`、`section` 两个 VARCHAR 字段 | `source`、`section` 两个 text 列 |
| 索引 / 度量 | HNSW + cosine | FLAT + COSINE | 不建索引，`<=>` 余弦距离 |
| 过滤写法 | `where={"source": ...}` | `filter='source like "03-%"'` | `WHERE source LIKE '03-%'` |

## 踩坑点

1. **Milvus 建完 collection 不能直接搜**。必须有索引并且 load 到内存。`create_collection` 传了 `index_params` 会顺带完成这两步；只传 schema 的话，要自己调用 `create_index` 和 `load_collection`，否则 search 会报 collection not loaded。
2. **Milvus 的 VARCHAR `max_length` 按字节计**，范围 1～65535。中文一个字占 3 字节，500 字的片段约 1.5 KB；按字数设成 500 会插入失败。更长的正文可以用 `TEXT` 类型。
3. **Milvus 默认一致性级别是 Bounded**，刚插入的数据可能短时间内搜不到。演示里建 collection 时设成了 `Strong`；也可以只在某一次 search / query 里传 `consistency_level`。
4. **分数方向不一样**。Milvus 的 COSINE 返回的 `distance` 实际上是相似度，越大越相似；pgvector 的 `<=>` 和 Chroma 返回的是距离，越小越相似。脚本里 PgVector 打印的是 `1 - 距离`，和 Milvus 同一口径。
5. **pgvector 的 `register_vector` 要在 `CREATE EXTENSION vector` 之后调用**。它会去库里查 `vector` 类型的 OID，注册后 numpy 数组可以直接当 SQL 参数传。
6. **Postgres 数据目录不要挂载 Windows 目录**，会遇到权限问题，所以 `pgvector-docker-compose.yaml` 用的是命名卷。Milvus 的官方 compose 文件挂的是 `./volumes/`，在 Docker Desktop 上可以正常使用。
7. **同一目录下的两个 compose 文件会被当成同一个项目**。Docker Compose 默认用目录名做项目名，先起 pgvector 再起 Milvus 时，Milvus 会把 pgvector 容器当成孤儿容器，试图删除它所在的网络，然后启动失败。所以 `pgvector-docker-compose.yaml` 里用 `name: rag-pgvector` 单独指定了项目名。启动 Milvus 时提示 `version` 属性已过时，是官方文件自带的，可以忽略。

## 索引，各一句话

- **FLAT**：暴力精确检索，几千条以内直接用它。这里 Milvus 用的就是 FLAT。
- **HNSW**：多层近邻图，召回高、延迟低、不需要训练，代价是内存大、建图慢。Chroma 默认用它；pgvector 是 `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)`。
- **IVF_FLAT**：先用 k-means 聚成 `nlist` 个桶，查询时只扫 `nprobe` 个桶。省内存，但要先有数据才能建；`nprobe` 太小会漏召回。
- **IVF_PQ**：在 IVF 的基础上用乘积量化压缩向量，用精度换内存。

## Milvus 还是 PgVector

- **Milvus**：专用向量数据库，存储和计算分离，可以水平扩展到亿级向量，索引类型多。代价是组件多：standalone 也要 etcd（元数据）和 MinIO（对象存储）两个依赖。
- **PgVector**：Postgres 的一个扩展，一个容器就能跑。元数据就是普通的 SQL 列，过滤、join、事务都是现成的。百万级以内，或者项目里本来就有 Postgres 时最省事。

## 正确性检查

`compare.py` 对 5 个问题算出暴力检索的 top-5 `chunk_id`，再和三个库的结果逐个比较：

- Milvus（FLAT）和 PgVector（无索引）都是精确检索，结果应该和暴力检索完全一致。
- Chroma 用 HNSW 近似检索，理论上可能有个别不同；数据量大、图稀疏时才容易漏，见 [LlamaIndex 实验](../llamaindex/README.md)的第 9 条。

实测结果（Milvus v3.0.3、pgvector 0.8.7，Docker Desktop on Windows）：

| | Chroma | Milvus | PgVector |
|---|---:|---:|---:|
| top-5 和暴力检索完全一致的题数（共 5 题） | 5 | 5 | 5 |

不只是排名，Milvus 的 COSINE 分数和 PgVector 的 `1 - 距离` 也逐位相同，例如默认演示问题的第一名两边都是 0.740。在 234 条向量的规模下，Chroma 的 HNSW 也没有漏召回。

哪个库没启动，`compare.py` 就跳过哪个。

## 文件

```
milvus-standalone-docker-compose.yaml  Milvus 官方 v3.0.3 standalone 编排文件，未改动
pgvector-docker-compose.yaml           Postgres 17 + pgvector 0.8.7
common.py                              从主线 Chroma 读片段和向量、查询向量化、暴力检索、Chroma 检索
milvus_demo.py                         Milvus 的存、取、查
pgvector_demo.py                       PgVector 的存、取、查
compare.py                             三个库和暴力检索的 top-5 对比
requirements.txt                       额外依赖
```
