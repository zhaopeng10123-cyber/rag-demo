# LlamaIndex 对照实验

用 LlamaIndex 平行实现一遍主线的 RAG，在同一把"尺子"下对比两个框架的检索效果，再顺带测两个检索技巧：bge-reranker 重排序和 sentence-window。

这个目录下全部是新增文件，主线代码一行没改，`python -m pytest tests -q` 的 33 个测试保持全绿。

## 怎么跑

在 `rag-demo` 目录下：

```bash
pip install -r experiments/llamaindex/requirements.txt
python download_model.py BAAI/bge-reranker-base          # reranker 实验要用，约 1.1 GB

python experiments/llamaindex/evaluate_llama.py           # 全部实验组，只评检索，不需要 LLM（CPU 约 4 分钟）
python experiments/llamaindex/evaluate_llama.py --only lc li_native_nopath --answers   # 再用 LLM 对比问答
python experiments/llamaindex/llama_rag.py --build        # 单独建一个 LlamaIndex 库
python experiments/llamaindex/llama_rag.py "RadixAttention 是怎么复用 KV Cache 的？"
```

结果会存到 `experiments/llamaindex/results/`（已加入 `.gitignore`）。

## 对齐了哪些变量

| 变量 | 两边的取值 |
|---|---|
| 知识库 | `data/knowledge-base/` 的 3 篇笔记 |
| Embedding | 本地 `bge-small-zh-v1.5`，归一化，查询端加同一句 bge 检索指令 |
| 切分长度 | chunk_size=300、overlap=48，都按**字符**计 |
| top-k | 5（reranker 组先召回 20 条再重排） |
| 向量库 | Chroma，cosine 距离，同一套 HNSW 参数 |
| LLM 和 Prompt | 同一个 `.env` 配置；拒答句和"来源：[n]"引用规则一字不差 |
| 评估 | 直接复用 `eval/evaluate.py` 的 `load_qa`、`first_hit_rank`、`retrieval_metrics`、`key_point_coverage` |

**对齐版（`li_aligned`）是这把尺子的校验**：它直接用 `ingest.py` 切好的片段，只把 Embedding、向量库、检索器换成 LlamaIndex 的。48 道题的命中排名和 LangChain 基线逐题完全相同，说明在片段相同的前提下，两个框架的检索没有任何差别，后面各组的差异都来自切分方式和检索技巧。

## 对齐时发现的隐藏变量

照着官方示例写，下面这些默认行为会让对照实验悄悄失效。这是整个实验最有价值的部分：

1. **bge 检索指令会被悄悄跳过**。`HuggingFaceEmbedding` 只有在 `model_name` 恰好等于官方模型名时才自动加 bge 的查询指令，用本地路径加载时不会加，也不报错。必须显式传 `query_instruction`。主线 README 的实验表明，这句指令对 Hit@3 有影响。
2. **文件绝对路径被拼进了向量化文本**。`SimpleDirectoryReader` 默认把 `file_path`（例如 `D:\CV\rag-demo\data\knowledge-base\03-vLLM….md`）作为元数据拼到每个片段前面一起向量化，只排除了 `file_name`。结果每个片段都带上了文件名里的关键词，"RecursiveCharacterTextSplitter"这道题的前 5 名全部来自 03 号文件。
3. **chunk_size 并不是正文长度**。`SentenceSplitter` 会先从 chunk_size 里扣掉元数据的长度（取向量化和 LLM 两份元数据里较长的那份）。同样设成 300，带路径时片段正文平均只有 114 字，去掉路径后是 168 字，而 LangChain 是 254 字。只在向量化元数据里排除路径还不够，LLM 那份也得排除，长度才会变化。
4. **chunk_size 默认按 token 计**。`SentenceSplitter` 默认用 tiktoken 计长度，中文下和 LangChain 的字符数对不上，要传 `tokenizer=list` 才按字符计；默认 overlap 是 200，不是 LangChain 常用的几十。
5. **`MarkdownNodeParser` 的标题路径不含节点自己的标题**，而且用 "/" 分隔，但标题本身就可能含 "/"（例如"6.4 它和 vLLM / SGLang 的关系"）。为了和 LangChain 用同一口径判断命中，自己写了 `AddSection` 打 section 标签（只供评估，不参与向量化）。
6. **`SentenceWindowNodeParser` 默认的分句器是为英文设计的**，中文段落会整段变成一句。换成按"。！？"和换行分句。
7. **`CitationQueryEngine` 的模板是英文的**，而且有 qa 和 refine 两个模板，拒答和引用规则两个都要写，否则检索结果一多、触发 refine 时规则就丢了。
8. **`llama-index-llms-openai-like` 要求 `openai<3`**，装上会把主线 langchain-openai 依赖的 openai 3.x 降级。所以自己写了一个 36 行的 `OpenAICompatibleLLM(CustomLLM)`。
9. **Chroma 默认的 HNSW 参数召回不全**（这一条和框架无关）。sentence-window 组有 2132 个节点、211 个重复向量（"）"、代码块标记这类短句），默认参数下 top-5 召回率只有 0.93～0.94（以暴力检索为准），而且每次重建索引漏掉的近邻不一样，同一组实验两次能跑出不同排名。把图建得更密（`ef_search=500, ef_construction=400, max_neighbors=64`）后召回率为 0.992，两次运行结果完全一致。所有实验组都用这套参数。

## 概念映射

| 环节 | LangChain（主线） | LlamaIndex |
|---|---|---|
| 文档 | `Document` | `Document`，切分后是 `TextNode` |
| 加载 | `Path.read_text`（`ingest.load_markdown`） | `SimpleDirectoryReader` |
| 按标题切 | `MarkdownHeaderTextSplitter` | `MarkdownNodeParser` |
| 按长度切 | `RecursiveCharacterTextSplitter` | `SentenceSplitter` |
| 元数据是否参与向量化 | 不参与，需要的话自己拼进正文（主线拼了【章节】） | 默认参与，用 `excluded_embed_metadata_keys` / `excluded_llm_metadata_keys` 排除 |
| Embedding | `HuggingFaceEmbeddings` | `HuggingFaceEmbedding` |
| 向量库 | `langchain_chroma.Chroma` | `ChromaVectorStore` + `StorageContext` + `VectorStoreIndex` |
| 检索器 | `vectorstore.as_retriever()` | `index.as_retriever()` |
| 检索后处理 | 自己写 | `node_postprocessors`（`SentenceTransformerRerank`、`MetadataReplacementPostProcessor`） |
| Prompt | `ChatPromptTemplate`（system + human） | `PromptTemplate`，qa 和 refine 两个 |
| 生成链 | LCEL：`prompt \| llm \| StrOutputParser()` | `QueryEngine`（检索器 + 响应合成器） |
| 带编号引用 | `format_docs` 自己编号 | `CitationQueryEngine` 编号成 "Source n" |
| LLM | `ChatOpenAI` | `OpenAILike`，这里改用自写的 `CustomLLM` |

## 代码量

只数非空、非注释的行：

| | LangChain | LlamaIndex |
|---|---:|---:|
| 建库 + 问答（含命令行入口） | `ingest.py` 67 + `rag_chain.py` 64 = 131 | `llama_rag.py` 约 100 |
| LLM 适配 | 0 | 36（`OpenAICompatibleLLM`） |
| 评估用的 section 标签 | 0（切分时顺带得到） | 20（`AddSection`） |
| 实验变体（对齐版、sentence-window） | — | 19 |

两边的核心代码量差不多。LlamaIndex"5 行代码 RAG"的简洁来自默认值，而这次实验里拖后腿的恰恰就是默认值：一旦要对齐切分长度、排除路径、加检索指令、换中文模板，就得把每个默认值都显式写出来。

## 检索结果

48 道题 = 主线评估集的 15 道知识库题 + 新增的 [`qa_extra.jsonl`](qa_extra.jsonl) 33 道。补充集重点覆盖原评估集完全没涉及的 03 号文档（vLLM 连续批处理、Triton、TensorRT），题目用自己的话改写，不照抄标题。新增它是因为原来 15 题的基线已经是 Hit@5 = 1.000，到了天花板，reranker 这类改进根本显示不出来。

"逐题对比"把每道题的首个命中排名和对照组比较，统计变好、变差、持平的题数，并做双侧符号检验。题目少的时候，这比看平均数更能说明问题。

**合计 48 题：**

| 实验组 | Hit@3 | MRR@3 | Hit@5 | MRR@5 | 逐题对比（变好/变差/持平） | p |
|---|---:|---:|---:|---:|---|---:|
| `lc` LangChain 基线 | 0.917 | 0.781 | 0.958 | 0.792 | — | |
| `li_aligned` 对齐版 | 0.917 | 0.781 | 0.958 | 0.792 | 对比 lc：0 / 0 / 48 | 1.0 |
| `li_native` 惯用管线 | 0.812 | 0.701 | 0.896 | 0.720 | 对比 lc：5 / 13 / 30 | 0.096 |
| `li_native_nopath` 惯用管线去掉路径 | 0.854 | 0.760 | 0.938 | 0.780 | 对比 lc：9 / 9 / 30 | 1.0 |
| `sentence_window` | 0.812 | 0.688 | 0.917 | 0.709 | 对比 lc：9 / 13 / 26 | 0.523 |
| `lc+rerank` | **0.958** | **0.903** | **0.979** | **0.908** | 对比 lc：11 / 3 / 34 | 0.057 |
| `li_native_nopath+rerank` | 0.917 | 0.872 | 0.938 | 0.876 | 对比 nopath：11 / 3 / 34 | 0.057 |
| `sentence_window+rerank` | 0.875 | 0.865 | 0.917 | 0.875 | 对比 sentence_window：15 / 2 / 31 | 0.002 |

**只看原来的 15 题**，惯用管线的劣势更明显：`li_native` 的 Hit@5 从 1.000 掉到 0.800，7 题变差、0 题变好（p=0.016）；去掉路径后是 0.933（2 题变好、5 题变差）。

几点结论：

- **框架本身不影响检索**：片段相同时两个框架逐题一致。差异全部来自"怎么切、往向量里塞了什么"。
- **LlamaIndex 默认管线更差，主要是文件路径的锅**：带路径时 13 题变差、5 题变好；只去掉路径，就变成 9 题变好、9 题变差，和 LangChain 打平。剩下的差别来自片段长度（168 字对 254 字），片段短了，命中的常常是"面试高频问题"这类摘要片段，而不是期望的正文章节。
- **reranker 是唯一稳定的正收益**：挂在三个组后面全部提升，MRR@5 提升 0.10～0.17。在 LangChain 基线上 11 题变好、3 题变差（p=0.057，48 题下接近但没到 0.05）。代价是 CPU 上每题多约 1.7 秒（20 对 query-片段过一遍 cross-encoder）。
- **sentence-window 单独用不划算**：以句子为单位检索，平均每句 36 字，语义太碎，比基线还差（9 好 13 差）。但它和 reranker 是互补的：小句召回得杂，reranker 拿着前后 3 句的窗口重排，15 题变好、2 题变差（p=0.002），是所有组合里提升最显著的。即便如此，它最终也没超过 `lc+rerank`。
- 有一道题所有组都没命中：「TensorRT-LLM 和 vLLM、SGLang 相比该怎么选？」，答案在一张对比表里，表格行被切开后单独看语义很弱。

## 问答对比

`--answers` 在主线 `eval/qa.jsonl` 的 17 题上（15 道知识库题、2 道库外题）各跑一遍真实问答，LLM 为硅基流动的 `Qwen/Qwen3-8B`（思考模式），temperature=0：

- LangChain：基线检索 top-5，加主线的 `build_answer_chain`；
- LlamaIndex：惯用管线去掉路径（`li_native_nopath`）的 top-5，加 `CitationQueryEngine`。

| 指标 | LangChain | LlamaIndex |
|---|---:|---:|
| 关键点覆盖率（15 道知识库题） | **0.950** | 0.900 |
| 拒答准确率（17 题） | **1.000** | 0.941 |
| 耗时中位数（秒） | 18.6 | 15.6 |
| 平均答案长度（字） | 270 | 201 |

两边只差一道题：「RecursiveCharacterTextSplitter 里的"递归"是什么意思？」LlamaIndex 检索时第 4 名才命中，片段平均只有 168 字，模型判断资料不够，回答了"知识库中没有相关信息"。这正是检索结论里"片段太短"在问答上的体现。另外 LlamaIndex 有一道题比 LangChain 多覆盖了一个关键点，其余题目覆盖率相同。

耗时没有可比性：免费接口偶尔排队一两分钟，LangChain 那边恰好碰上 3 次（130～166 秒），平均耗时被拉到 44 秒，中位数两边差不多。17 道题的问答对比只能说明"两边都能按规则拒答和引用"，分不出高下。

## 文件

```
llama_rag.py        LlamaIndex 版 RAG：三种建库方式（对齐版、惯用管线、sentence-window）、自写 LLM、带引用的查询引擎
evaluate_llama.py   对照实验：8 个实验组的检索指标、逐题对比和符号检验，--answers 再对比真实问答
qa_extra.jsonl      补充的 33 道检索题，格式和 eval/qa.jsonl 相同
requirements.txt    额外依赖，以及为什么不用 llama-index-llms-openai-like
```
