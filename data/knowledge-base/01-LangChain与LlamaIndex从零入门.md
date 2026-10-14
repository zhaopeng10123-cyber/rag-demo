# LangChain 与 LlamaIndex 从零入门：为什么用框架构建 RAG 知识库与智能 Agent

> 适合人群：会一点 Python，但还没系统接触过大模型应用开发的同学。
> 阅读方式：按顺序读，每一章都依赖前一章的概念，不要跳。
> 说明：文中代码基于 LangChain 1.x、LlamaIndex 0.12+ 风格的 API。这两个框架迭代很快，如果某个导入路径报错，请以官方文档为准，但概念和流程是稳定的。

---

## 目录

- [第 0 章 先建立全局认知：我们到底要解决什么问题](#第-0-章-先建立全局认知我们到底要解决什么问题)
- [第 1 章 大模型（LLM）基础：从一次 API 调用说起](#第-1-章-大模型llm基础从一次-api-调用说起)
- [第 2 章 直接调用大模型有哪些问题](#第-2-章-直接调用大模型有哪些问题)
- [第 3 章 RAG：让大模型"先查资料再回答"](#第-3-章-rag让大模型先查资料再回答)
- [第 4 章 Agent：让大模型"自己决定做什么"](#第-4-章-agent让大模型自己决定做什么)
- [第 5 章 不用框架，纯手写一个 RAG 和一个 Agent](#第-5-章-不用框架纯手写一个-rag-和一个-agent)
- [第 6 章 LangChain 是什么](#第-6-章-langchain-是什么)
- [第 7 章 LlamaIndex 是什么](#第-7-章-llamaindex-是什么)
- [第 8 章 LangChain vs LlamaIndex：区别与选择](#第-8-章-langchain-vs-llamaindex区别与选择)
- [第 9 章 核心问题：为什么要用框架构建 RAG 和 Agent](#第-9-章-核心问题为什么要用框架构建-rag-和-agent)
- [第 10 章 进阶：生产级 RAG 的常见优化手段](#第-10-章-进阶生产级-rag-的常见优化手段)
- [第 11 章 动手路线：从环境搭建到跑通第一个项目](#第-11-章-动手路线从环境搭建到跑通第一个项目)
- [第 12 章 术语表](#第-12-章-术语表)
- [第 13 章 面试高频问题](#第-13-章-面试高频问题)

---

## 第 0 章 先建立全局认知：我们到底要解决什么问题

一句话概括：

> **大模型很聪明，但它不知道你的私有数据，也不能主动去做事。RAG 解决"不知道"，Agent 解决"不能做"，LangChain / LlamaIndex 则是把这两件事工程化的工具箱。**

它们之间的关系可以这样理解：

```
                    ┌────────────────────────────┐
                    │        你的业务应用         │
                    │  (客服机器人/企业知识库/助手) │
                    └──────────────┬─────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
   ┌──────▼──────┐          ┌──────▼──────┐          ┌──────▼──────┐
   │     RAG     │          │    Agent    │          │   记忆/对话  │
   │ 检索增强生成 │          │  智能体/工具 │          │   多轮上下文 │
   └──────┬──────┘          └──────┬──────┘          └──────┬──────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                    ┌──────────────▼─────────────┐
                    │ 开发框架：LangChain / LlamaIndex │
                    └──────────────┬─────────────┘
                                   │
                    ┌──────────────▼─────────────┐
                    │ 大模型 API：GPT/Claude/DeepSeek/通义… │
                    └────────────────────────────┘
```

> 最底层的"大模型 API"既可以是厂商提供的云端服务，也可以是用 SGLang、vLLM 等推理框架在自己服务器上部署的开源模型，详见 [02-SGLang与大模型推理部署](./02-SGLang与大模型推理部署.md)。

接下来我们从最底层的"大模型调用"开始，一层一层往上搭。

---

## 第 1 章 大模型（LLM）基础：从一次 API 调用说起

### 1.1 什么是 LLM

LLM（Large Language Model，大语言模型）本质上是一个**根据前文预测下一个词**的模型。你给它一段文字，它一个词一个词地往后续写，直到它认为该结束了。

ChatGPT、Claude、DeepSeek、通义千问、文心一言都是 LLM。

### 1.2 什么是 Token

模型不是按"字"或"单词"处理文本的，而是按 **Token**（词元）。

- 英文里，1 个 Token 大约是 0.75 个单词；
- 中文里，1 个汉字通常是 1～2 个 Token。

为什么要关心 Token？因为：

1. **计费按 Token 算**：输入 Token + 输出 Token。
2. **上下文窗口按 Token 算**：每个模型一次能"看到"的 Token 数有上限（比如 128K）。

### 1.3 一次 API 调用长什么样

现在主流大模型都提供"聊天补全（Chat Completions）"接口。以 OpenAI 兼容格式为例（DeepSeek、通义、Kimi 等国内模型基本都兼容这个格式）：

```python
# pip install openai
from openai import OpenAI

client = OpenAI(api_key="你的API_KEY")  # 国内模型额外传 base_url 即可

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "你是一个乐于助人的助手。"},
        {"role": "user", "content": "什么是RAG？"},
    ],
)
print(response.choices[0].message.content)
```

这里有三个关键概念：

| 角色 role | 含义 |
|---|---|
| `system` | 系统提示词，设定模型的身份和规则 |
| `user` | 用户说的话 |
| `assistant` | 模型之前的回答（多轮对话时需要传回去） |

### 1.4 关键事实：大模型 API 是"无状态"的

这一点非常重要，很多初学者会误解：

> **模型本身不会记住你上一次说了什么。**

你在 ChatGPT 网页上感觉它"记得"前文，是因为网页每次都把**整个历史对话**重新发给模型。所以"记忆"其实是应用层自己维护的一个消息列表。

```python
messages = [{"role": "system", "content": "你是助手"}]

messages.append({"role": "user", "content": "我叫小明"})
reply = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
messages.append({"role": "assistant", "content": reply.choices[0].message.content})

messages.append({"role": "user", "content": "我叫什么？"})  # 能答对，因为历史在 messages 里
reply = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
```

记住这一点，后面理解"记忆（Memory）"组件就很容易了。

### 1.5 什么是 Prompt（提示词）

Prompt 就是你发给模型的全部文本。模型的回答质量很大程度上取决于 Prompt 怎么写。后面你会看到，RAG 的本质就是**自动化地往 Prompt 里塞入相关资料**。

---

## 第 2 章 直接调用大模型有哪些问题

只会调 API 的话，做真实业务很快就会碰到下面这些问题：

| 问题 | 具体表现 | 举例 |
|---|---|---|
| **知识截止** | 模型只知道训练数据截止日期之前的事 | 问它"今天的新闻"答不出来 |
| **不懂私有数据** | 训练时没见过你公司的文档 | 问"我们公司年假几天"它只能瞎猜 |
| **幻觉（Hallucination）** | 不知道时会一本正经地编造 | 编出不存在的论文、法条 |
| **上下文窗口有限** | 不能把几万份文档一次全塞进去 | 1000 份 PDF 远超 128K Token |
| **成本和延迟** | 塞得越多，越贵越慢 | 每次问答都带 10 万 Token 不现实 |
| **不能行动** | 只能输出文字，不能查数据库、发邮件、调接口 | "帮我查一下订单状态"做不到 |
| **无记忆** | API 无状态，需要自己维护历史 | 见 1.4 节 |

解决方案分两大类：

- **"不知道"类问题**（知识截止、私有数据、幻觉、窗口有限）→ 用 **RAG**；
- **"不能做"类问题**（不能行动、多步骤任务）→ 用 **Agent**。

> 补充：还有一个方案是**微调（Fine-tuning）**，即用你的数据再训练模型。但微调成本高、更新慢、不容易溯源，更适合"改变模型说话风格/格式"，而不是"注入经常变化的知识"。注入知识首选 RAG。

---

## 第 3 章 RAG：让大模型"先查资料再回答"

### 3.1 RAG 的定义

RAG = **R**etrieval-**A**ugmented **G**eneration，检索增强生成。

用一个比喻解释：

- **不用 RAG**：闭卷考试，模型只能凭记忆答题，不会就编。
- **用 RAG**：开卷考试，先去资料里翻到相关的几页，再根据这几页答题。

### 3.2 RAG 的完整流程

RAG 分为两个阶段：**离线建库（Indexing）** 和 **在线问答（Querying）**。

```
【阶段一：离线建库，只需做一次（文档更新时再做）】

 原始文档          加载             切分              向量化             存储
(PDF/Word/   ──►  Load   ──►  Split/Chunk  ──►   Embedding   ──►  Vector Store
 网页/数据库)      读成文本       切成小段           每段变成向量        存入向量数据库


【阶段二：在线问答，每次提问都执行】

 用户问题  ──►  问题向量化  ──►  在向量库中检索  ──►  拼接 Prompt  ──►  LLM 生成答案
                Embedding         Top-K 相似片段       问题+资料
```

下面把每一步都拆开讲清楚。

### 3.3 第 1 步：加载文档（Load）

把各种格式的文件读成纯文本，同时保留**元数据（metadata）**，比如文件名、页码、作者、更新时间。

- PDF → 需要 PDF 解析库（pypdf、PyMuPDF 等），扫描件还需要 OCR；
- Word → python-docx；
- 网页 → 爬取后去掉 HTML 标签；
- 数据库 → SQL 查询结果转文本。

元数据很重要：后面回答时可以告诉用户"答案出自《员工手册》第 12 页"，这就是**可溯源**。

### 3.4 第 2 步：切分文档（Split / Chunk）

**为什么要切分？**

1. 一份文档可能有几万字，整篇塞给模型太长；
2. 检索的粒度越细，找到的内容越精准；
3. Embedding 模型本身也有输入长度限制。

**怎么切？** 常见参数：

- `chunk_size`：每段多长，例如 500 字；
- `chunk_overlap`：相邻两段重叠多少，例如 50 字。重叠是为了防止一句话被从中间切断，导致语义丢失。

```
原文：[ AAAAAAAAAA BBBBBBBBBB CCCCCCCCCC ]

chunk_size=10, overlap=2 切分后：
  段1: AAAAAAAAAA
  段2:         AA BBBBBBBB      ← 开头带上一段结尾的 2 个字符
  段3:                 BB CCCCCCCC
```

更好的做法是**按语义边界切**：优先按段落切，段落太长再按句子切，句子还太长再按字符切。这就是后面 LangChain 里 `RecursiveCharacterTextSplitter` 的思路。

### 3.5 第 3 步：向量化（Embedding）

这是 RAG 里最核心、也是最需要理解的一步。

**什么是 Embedding？**

Embedding 是把一段文字转换成一串数字（向量）的过程，例如：

```
"猫喜欢吃鱼"   →  [0.12, -0.53, 0.88, ..., 0.07]   (比如 1536 个数字)
"小猫爱吃鱼"   →  [0.11, -0.50, 0.85, ..., 0.09]   ← 和上面很接近
"今天股市大跌" →  [-0.72, 0.31, -0.10, ..., 0.64]  ← 和上面差很远
```

关键性质：**语义相近的文字，向量也相近**。这样"找相关资料"就变成了一个数学问题：**找距离最近的向量**。

**为什么不用关键词搜索？**

关键词搜索要求字面匹配。用户问"怎么请假"，文档里写的是"休假申请流程"，关键词搜索可能找不到；但 Embedding 能识别出两者语义相近。

**怎么衡量"相近"？** 最常用的是**余弦相似度**：

\[
\text{cos}(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \, \|\vec{b}\|}
\]

结果在 -1 到 1 之间，越接近 1 越相似。

**常用的 Embedding 模型：**

- 商用：OpenAI `text-embedding-3-small/large`、通义 `text-embedding-v3` 等；
- 开源（中文效果好，可本地部署）：`BAAI/bge-m3`、`BAAI/bge-large-zh-v1.5` 等。

> 注意：建库和查询必须使用**同一个** Embedding 模型，否则向量不在同一个"坐标系"里，比较没有意义。

### 3.6 第 4 步：存入向量数据库（Vector Store）

每个片段变成向量后，需要存起来并支持快速的"最近邻搜索"。

- 数据量小时：直接放内存里，用 numpy 暴力计算相似度就行；
- 数据量大时（百万级以上）：需要专门的向量数据库，它们用 HNSW、IVF 等近似最近邻（ANN）算法加速。

| 向量库 | 特点 |
|---|---|
| FAISS | Meta 开源的库，本地使用，速度快 |
| Chroma | 轻量，上手最简单，适合原型 |
| Milvus | 国产开源，分布式，适合大规模生产 |
| Qdrant / Weaviate | 功能完善的开源向量数据库 |
| pgvector | PostgreSQL 插件，已有 PG 的团队首选 |
| Elasticsearch | 同时支持关键词检索和向量检索 |

每条记录通常存三样东西：**向量 + 原文片段 + 元数据**。

### 3.7 第 5 步：检索（Retrieve）

用户提问时：

1. 用同一个 Embedding 模型把问题转成向量；
2. 在向量库里找出与问题向量最相似的 K 个片段（Top-K，K 通常取 3～10）；
3. 返回这些片段的原文和元数据。

### 3.8 第 6 步：拼接 Prompt 并生成（Augment + Generate）

把检索到的资料和用户问题一起塞进 Prompt：

```
你是公司的HR助手。请仅根据下面的【参考资料】回答问题。
如果资料中没有答案，请直接说"我不知道"，不要编造。

【参考资料】
片段1：员工入职满一年后，每年享有5天带薪年假……（来源：员工手册 p.12）
片段2：年假可以分次休，每次不少于半天……（来源：员工手册 p.13）

【问题】
我入职两年了，年假有几天？
```

模型基于资料作答，从而：

- 能回答私有数据的问题；
- 大幅减少幻觉（因为被要求"只根据资料"）；
- 可以给出出处；
- 文档更新后只需重新建库，不用重新训练模型。

### 3.9 小结：RAG 的 6 个步骤

```
加载 → 切分 → 向量化 → 存储 → 检索 → 生成
└──────── 离线建库 ────────┘   └─ 在线问答 ─┘
```

---

## 第 4 章 Agent：让大模型"自己决定做什么"

### 4.1 Agent 的定义

Agent（智能体）= **LLM（大脑） + 工具（手脚） + 循环（自主决策） + 记忆**。

RAG 的流程是**写死的**：永远是"检索 → 生成"。
Agent 的流程是**模型自己决定的**：它根据问题判断需要调用什么工具、调用几次、什么时候可以给出最终答案。

举例，用户问："帮我查一下北京明天的天气，如果下雨就给张三发邮件提醒他带伞。"

Agent 的执行过程：

```
思考：我需要先查天气            → 调用工具 get_weather("北京")
观察：明天有雨
思考：下雨了，需要发邮件         → 调用工具 send_email("张三", "明天下雨记得带伞")
观察：发送成功
思考：任务完成                   → 输出最终答案："北京明天有雨，已提醒张三带伞。"
```

这就是经典的 **ReAct 模式（Reasoning + Acting，推理 + 行动）**。

### 4.2 模型是怎么"调用工具"的？—— Function Calling

这是理解 Agent 的关键。**模型自己并不能执行代码**，真实过程是：

1. 你在请求里告诉模型："你有这些工具可用"，并用 JSON Schema 描述每个工具的名字、用途、参数；
2. 模型判断需要用工具时，不直接回答，而是返回一段结构化数据：`{"name": "get_weather", "arguments": {"city": "北京"}}`；
3. **你的程序**解析这段数据，真正去执行 `get_weather("北京")`；
4. 把执行结果作为一条 `tool` 消息发回给模型；
5. 模型看到结果后，决定是继续调用其他工具，还是给出最终回答。

```
┌──────────┐  ① 问题 + 工具列表   ┌──────────┐
│          │ ───────────────────► │          │
│  你的程序 │  ② 我要调用工具X(参数) │   LLM    │
│          │ ◄─────────────────── │          │
│ ③执行工具X│  ④ 工具X的返回结果    │          │
│          │ ───────────────────► │          │
│          │  ⑤ 最终答案(或继续②)  │          │
│          │ ◄─────────────────── │          │
└──────────┘                      └──────────┘
        ↑_____________ 循环，直到模型不再调用工具 ____________│
```

### 4.3 Agent 的四个组成部分

| 组成 | 作用 | 举例 |
|---|---|---|
| **大脑（LLM）** | 理解意图、规划步骤、决定调用哪个工具 | GPT-4o、Claude、DeepSeek |
| **工具（Tools）** | 与外部世界交互的能力 | 搜索、查数据库、调 API、执行代码、**RAG 检索** |
| **记忆（Memory）** | 短期：本次对话历史；长期：跨会话的用户偏好等 | 消息列表、数据库存储 |
| **规划/循环（Planning/Loop）** | 反复"思考 → 行动 → 观察"直到完成 | ReAct、Plan-and-Execute |

### 4.4 RAG 和 Agent 的关系

它们不是二选一，而是可以组合：

- **RAG 可以作为 Agent 的一个工具**：Agent 判断"这个问题需要查公司文档"时，就调用 RAG 检索工具；
- 这种模式叫 **Agentic RAG**：由 Agent 决定要不要检索、检索几次、换什么关键词重新检索。

---

## 第 5 章 不用框架，纯手写一个 RAG 和一个 Agent

为了真正理解框架帮我们做了什么，先不用任何框架，只用 `openai` 和 `numpy` 手写一遍。

### 5.1 手写 RAG

```python
# pip install openai numpy
import numpy as np
from openai import OpenAI

client = OpenAI()  # 读取环境变量 OPENAI_API_KEY

# ---------- 第1步：加载 ----------
with open("company_handbook.txt", encoding="utf-8") as f:
    text = f.read()

# ---------- 第2步：切分 ----------
def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = split_text(text)

# ---------- 第3步：向量化 ----------
def embed(texts: list[str]) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in resp.data])

# ---------- 第4步：存储（这里直接存在内存的 numpy 数组里）----------
chunk_vectors = embed(chunks)

# ---------- 第5步：检索 ----------
def retrieve(query: str, k: int = 3) -> list[str]:
    q = embed([query])[0]
    sims = chunk_vectors @ q / (np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(q))
    top_k = np.argsort(-sims)[:k]
    return [chunks[i] for i in top_k]

# ---------- 第6步：生成 ----------
def answer(query: str) -> str:
    context = "\n\n".join(retrieve(query))
    prompt = (
        "请仅根据以下参考资料回答问题，资料中没有的内容请回答“我不知道”。\n\n"
        f"【参考资料】\n{context}\n\n【问题】\n{query}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

print(answer("入职两年年假有几天？"))
```

50 行代码就能跑通。**但请注意，这只是玩具级别**，离真实业务还差很远：

- 只支持 txt，PDF、Word、网页、表格怎么办？
- 按固定字符数切分，会把句子、表格切烂；
- 没有元数据，无法溯源；
- 向量存在内存，程序一重启就没了，百万级数据根本放不下；
- 每次启动都要重新调用 Embedding（花钱）；
- 想换成 DeepSeek / 通义 / 本地模型，要改很多地方；
- 没有流式输出、没有异步、没有错误重试；
- 检索效果不好时，没有重排序、混合检索、查询改写等优化手段；
- 出了问题不知道是检索错了还是生成错了，无法调试和评估。

### 5.2 手写 Agent

```python
import json
from openai import OpenAI

client = OpenAI()

# ---------- 定义真正的工具函数 ----------
def get_weather(city: str) -> str:
    return f"{city}明天有小雨，18~24℃"  # 实际项目里这里调用天气 API

TOOL_FUNCTIONS = {"get_weather": get_weather}

# ---------- 用 JSON Schema 向模型描述工具 ----------
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市明天的天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名"}},
            "required": ["city"],
        },
    },
}]

messages = [{"role": "user", "content": "北京明天需要带伞吗？"}]

# ---------- Agent 循环 ----------
for _ in range(10):  # 设上限，防止死循环
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
    msg = resp.choices[0].message

    if not msg.tool_calls:          # 模型不再调用工具 → 给出最终答案
        print(msg.content)
        break

    messages.append(msg)            # 把"我要调用工具"这条消息加入历史
    for call in msg.tool_calls:     # 逐个执行工具，把结果发回给模型
        args = json.loads(call.function.arguments)
        result = TOOL_FUNCTIONS[call.function.name](**args)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
```

同样能跑，但真实业务里还要处理：

- 每加一个工具都要手写一份 JSON Schema，容易和函数签名不一致；
- 工具执行报错、模型给出非法参数怎么办？
- 对话历史越来越长超出窗口怎么办？跨会话的记忆怎么持久化？
- 需要"执行敏感操作前让人工确认"（Human-in-the-loop）怎么办？
- 多个 Agent 协作（一个负责检索、一个负责写报告）怎么编排？
- 执行到一半程序崩了，能否从断点恢复？
- 每一步调用了什么工具、花了多少 Token，怎么追踪？

**这些"脏活累活"，正是 LangChain 和 LlamaIndex 帮你做的。**

---

## 第 6 章 LangChain 是什么

### 6.1 一句话定义

> **LangChain 是一个用于开发大模型应用的开源框架，它把模型调用、提示词、文档处理、检索、工具、Agent 等环节抽象成标准化的可组合组件，让你像搭积木一样"链（Chain）"起来。**

- 2022 年 10 月由 Harrison Chase 发布，是最早、生态最大的 LLM 应用框架之一；
- 支持 Python 和 JavaScript/TypeScript；
- 名字含义：Lang（Language Model）+ Chain（把多个步骤串成链）。

### 6.2 LangChain 生态全景

现在的 LangChain 已经不是一个单独的包，而是一个产品家族：

| 组成 | 作用 |
|---|---|
| `langchain-core` | 核心抽象：消息、提示词、Runnable 接口、LCEL 等 |
| `langchain` | 上层功能，如 `create_agent` 等开箱即用的 Agent |
| `langchain-openai`、`langchain-deepseek` 等 | 各家模型/服务的官方集成包 |
| `langchain-community` | 社区维护的大量第三方集成（文档加载器、向量库等） |
| `langchain-text-splitters` | 文本切分器 |
| **LangGraph** | 底层编排框架，用"图（Graph）"描述有状态、可循环的 Agent 工作流 |
| **LangSmith** | 可观测性平台：追踪每一步调用、调试、评估、监控 |

> 从 LangChain 1.0 开始，Agent 统一建立在 LangGraph 之上。简单 Agent 用 `langchain.agents.create_agent` 即可；需要精细控制流程时直接用 LangGraph。

### 6.3 核心组件逐个讲解

我们按照第 3 章 RAG 的流程，把 LangChain 的组件一一对应上。

#### 组件 1：模型（Chat Models）—— 统一的模型接口

```python
# pip install langchain langchain-openai
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print(llm.invoke("用一句话解释RAG").content)
```

换成 DeepSeek（OpenAI 兼容接口）只需要改参数，其他代码完全不用动：

```python
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com",
    api_key="你的DeepSeek Key",
)
```

或者使用 LangChain 的统一初始化方法：

```python
from langchain.chat_models import init_chat_model
llm = init_chat_model("openai:gpt-4o-mini")
```

**价值：模型可替换。** 所有模型都实现了同一套接口（`invoke` / `stream` / `batch` / `ainvoke`），切换供应商不影响业务代码。

#### 组件 2：提示词模板（Prompt Templates）

把 Prompt 中会变化的部分做成变量：

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是公司HR助手，只根据参考资料回答，不知道就说不知道。\n\n参考资料：\n{context}"),
    ("human", "{question}"),
])

messages = prompt.invoke({"context": "年假5天", "question": "年假几天？"})
```

#### 组件 3：输出解析器（Output Parsers）/ 结构化输出

把模型输出的文本转成程序能用的格式：

```python
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

parser = StrOutputParser()  # 把 AIMessage 转成纯字符串

class Resume(BaseModel):
    name: str
    years_of_experience: int
    skills: list[str]

structured_llm = llm.with_structured_output(Resume)  # 直接让模型返回 Pydantic 对象
result = structured_llm.invoke("张三，5年Python经验，熟悉LangChain和FastAPI")
print(result.skills)
```

#### 组件 4：文档加载器（Document Loaders）—— 对应 RAG 第 1 步

```python
# pip install langchain-community pypdf
from langchain_community.document_loaders import PyPDFLoader, TextLoader

docs = PyPDFLoader("员工手册.pdf").load()
# 每个元素是一个 Document 对象：
#   doc.page_content  → 文本内容
#   doc.metadata      → {"source": "员工手册.pdf", "page": 0}
```

LangChain 社区提供了上百种加载器：PDF、Word、Excel、Markdown、网页、Notion、飞书、GitHub、数据库……

#### 组件 5：文本切分器（Text Splitters）—— 对应 RAG 第 2 步

```python
# pip install langchain-text-splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],  # 中文友好的分隔符
)
splits = splitter.split_documents(docs)  # 元数据会自动继承到每个片段
```

"Recursive（递归）"的含义：先尝试用 `\n\n`（段落）切，如果某段还超长，再用 `\n` 切，再不行用 `。` 切……尽量在语义边界处断开。

#### 组件 6：Embedding 模型 —— 对应 RAG 第 3 步

```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 或者使用本地开源模型（pip install langchain-huggingface sentence-transformers）
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
```

#### 组件 7：向量存储（Vector Stores）—— 对应 RAG 第 4 步

```python
from langchain_core.vectorstores import InMemoryVectorStore
vectorstore = InMemoryVectorStore.from_documents(splits, embeddings)

# 换成 Chroma 并持久化到磁盘（pip install langchain-chroma）
from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")
```

FAISS、Milvus、Qdrant、pgvector、Elasticsearch 等都有对应集成，接口完全一致。

#### 组件 8：检索器（Retrievers）—— 对应 RAG 第 5 步

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
relevant_docs = retriever.invoke("年假有几天？")
```

Retriever 是比 Vector Store 更通用的抽象：任何"输入问题、返回文档列表"的东西都是 Retriever。所以关键词检索（BM25）、混合检索、多路召回等都可以用统一的方式接入。

#### 组件 9：LCEL（LangChain 表达式语言）—— 把组件串成链

LangChain 里几乎所有组件都实现了 **Runnable** 接口，可以用 `|`（管道符）串起来，前一个的输出就是后一个的输入，类似 Linux 管道。

```python
chain = prompt | llm | StrOutputParser()
chain.invoke({"context": "年假5天", "question": "年假几天？"})
```

用 LCEL 串起来的链，**自动获得**以下能力，不需要额外写代码：

- `chain.stream(...)`：流式输出（打字机效果）；
- `chain.batch([...])`：批量并发处理；
- `await chain.ainvoke(...)`：异步调用；
- 自动接入 LangSmith 追踪每一步的输入输出。

### 6.4 用 LangChain 完整实现 RAG

把上面的组件组合起来，对比第 5 章的手写版：

```python
# pip install langchain langchain-openai langchain-community langchain-text-splitters pypdf
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 加载
docs = PyPDFLoader("员工手册.pdf").load()

# 2. 切分
splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

# 3 + 4. 向量化并存储
vectorstore = InMemoryVectorStore.from_documents(splits, OpenAIEmbeddings(model="text-embedding-3-small"))

# 5. 检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6. 生成
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是HR助手，只根据参考资料回答，不知道就说不知道，并注明页码。\n\n参考资料：\n{context}"),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(f"[第{d.metadata.get('page', 0) + 1}页] {d.page_content}" for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)

# 普通调用
print(rag_chain.invoke("入职两年年假有几天？"))

# 流式调用（一行都不用改链本身）
for token in rag_chain.stream("病假怎么请？"):
    print(token, end="", flush=True)
```

逐行解释这条链的数据流：

1. 输入字符串 `"入职两年年假有几天？"`；
2. 字典里的两个分支**并行**执行：
   - `context`：问题 → `retriever` 检索出 3 个 Document → `format_docs` 拼成字符串；
   - `question`：`RunnablePassthrough()` 原样传递问题；
3. 得到 `{"context": "...", "question": "..."}`，填入 `prompt` 模板；
4. 填好的消息发给 `ChatOpenAI`，得到 `AIMessage`；
5. `StrOutputParser` 把 `AIMessage` 转成字符串。

### 6.5 用 LangChain 实现 Agent

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

@tool
def search_handbook(query: str) -> str:
    """在公司员工手册中检索与问题相关的规章制度。涉及请假、薪资、报销等公司制度时使用。"""
    return format_docs(retriever.invoke(query))

@tool
def get_weather(city: str) -> str:
    """查询指定城市明天的天气。"""
    return f"{city}明天有小雨，18~24℃"

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[search_handbook, get_weather],
    system_prompt="你是公司的智能助手，可以查询员工手册和天气。",
    checkpointer=InMemorySaver(),  # 短期记忆：同一个 thread_id 下自动保存对话历史
)

config = {"configurable": {"thread_id": "user-001"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "我想明天请一天年假，需要提前几天申请？顺便看下北京明天天气"}]},
    config=config,
)
print(result["messages"][-1].content)
```

对比第 5.2 节手写版，框架帮你做了：

- `@tool` 装饰器**自动根据函数签名和 docstring 生成 JSON Schema**（所以 docstring 一定要写清楚，模型靠它决定何时调用）；
- 自动执行"思考 → 调用工具 → 观察 → 再思考"的循环；
- 工具报错处理、最大步数限制；
- `checkpointer` 自动持久化对话状态（换成 Postgres/Redis 版本即可跨进程保存）；
- 通过中间件（middleware）支持人工审批、对话历史自动摘要等能力。

### 6.6 什么时候需要 LangGraph

当流程比较复杂，例如：

- 有分支和循环："检索结果不相关 → 改写问题 → 重新检索"；
- 多个 Agent 协作：研究员 Agent + 写作 Agent + 审核 Agent；
- 需要在某一步暂停，等待人工审批后继续；
- 需要断点续跑。

LangGraph 用 **节点（Node）+ 边（Edge）+ 状态（State）** 来描述流程：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    question: str
    docs: list
    answer: str

def retrieve_node(state: State):
    return {"docs": retriever.invoke(state["question"])}

def generate_node(state: State):
    context = format_docs(state["docs"])
    msg = llm.invoke(f"根据资料回答：\n{context}\n\n问题：{state['question']}")
    return {"answer": msg.content}

graph = StateGraph(State)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
app = graph.compile()

print(app.invoke({"question": "年假几天？"})["answer"])
```

这个例子只是线性流程，但你可以用 `add_conditional_edges` 加入"判断检索结果是否相关，不相关就回到改写节点"的条件分支，实现更智能的 Agentic RAG。

---

## 第 7 章 LlamaIndex 是什么

### 7.1 一句话定义

> **LlamaIndex 是一个专注于"把你的数据连接到大模型"的数据框架，它在文档解析、索引构建、检索和查询方面做得特别深入，是构建 RAG 知识库最省事的框架之一。**

- 2022 年 11 月由 Jerry Liu 发布，最初叫 **GPT Index**，后改名 LlamaIndex；
- 定位：如果说 LangChain 是"通用的 LLM 应用积木"，LlamaIndex 就是"以数据和检索为中心的 RAG 专家"；
- 现在也提供了完善的 Agent 和 Workflow 能力。

### 7.2 LlamaIndex 的核心概念

LlamaIndex 有一套自己的术语，和 RAG 流程的对应关系如下：

| RAG 步骤 | LlamaIndex 概念 | 说明 |
|---|---|---|
| 加载 | **Reader** / `SimpleDirectoryReader` | 读取文件，生成 `Document` |
| 原始文档 | **Document** | 一份完整的文档 + 元数据 |
| 切分 | **Node Parser** / `SentenceSplitter` | 把 Document 切成 `Node` |
| 文档片段 | **Node** | 切分后的片段，记录与原文档、前后片段的关系 |
| 向量化 + 存储 | **Index** / `VectorStoreIndex` | 对 Node 建立索引（向量索引最常用） |
| 存储层 | **StorageContext** | 管理文档存储、索引存储、向量存储 |
| 检索 | **Retriever** | 从索引中找出相关 Node |
| 后处理 | **Node Postprocessor** | 对检索结果重排序、过滤 |
| 生成 | **Response Synthesizer** | 把检索结果和问题合成最终答案 |
| 一问一答 | **Query Engine** | = Retriever + Postprocessor + Synthesizer |
| 多轮对话 | **Chat Engine** | 带对话记忆的 Query Engine |
| 全局配置 | **Settings** | 全局设置默认 LLM、Embedding 模型、切分参数 |

### 7.3 最经典的"5 行代码 RAG"

LlamaIndex 最出名的就是它的高层 API 极其简洁：

```python
# pip install llama-index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()   # 1. 加载 data 目录下所有文件
index = VectorStoreIndex.from_documents(documents)       # 2~4. 切分 + 向量化 + 存储
query_engine = index.as_query_engine()                   # 5~6. 检索 + 生成
print(query_engine.query("入职两年年假有几天？"))
```

这 4 行代码在背后完成了第 3 章 RAG 的全部 6 个步骤：

1. `SimpleDirectoryReader` 自动识别目录中的 PDF、Word、Markdown、txt 等格式并加载；
2. `from_documents` 内部用默认的 `SentenceSplitter` 切分成 Node；
3. 调用默认的 Embedding 模型给每个 Node 生成向量；
4. 存入默认的内存向量库；
5. `as_query_engine` 创建检索器（默认 Top-2）和响应合成器；
6. `query` 时自动检索、拼 Prompt、调 LLM 生成答案。

### 7.4 一步步拆开：自定义每个环节

高层 API 适合快速验证，真实项目通常需要自定义每一步：

```python
# pip install llama-index llama-index-embeddings-huggingface
from llama_index.core import (
    Settings, SimpleDirectoryReader, VectorStoreIndex,
    StorageContext, load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

# ---------- 全局配置：LLM、Embedding、切分参数 ----------
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

PERSIST_DIR = "./storage"

if not os.path.exists(PERSIST_DIR):
    # ---------- 第一次运行：建库并持久化到磁盘 ----------
    documents = SimpleDirectoryReader("data", recursive=True).load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # ---------- 之后运行：直接从磁盘加载，不用重新 Embedding ----------
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# ---------- 自定义查询引擎 ----------
query_engine = index.as_query_engine(
    similarity_top_k=5,                                                  # 检索 5 个片段
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],  # 过滤低相关度片段
)

response = query_engine.query("入职两年年假有几天？")
print(response)

# ---------- 查看答案的出处（溯源）----------
for node in response.source_nodes:
    print(node.metadata.get("file_name"), node.score, node.text[:50])
```

### 7.5 LlamaIndex 的多种索引类型

这是 LlamaIndex 区别于其他框架的一大特色，不只有向量索引：

| 索引 | 原理 | 适用场景 |
|---|---|---|
| `VectorStoreIndex` | 向量相似度检索 | 最通用，绝大多数问答场景 |
| `SummaryIndex` | 按顺序遍历所有节点 | "总结这份文档"类需要看全文的任务 |
| `KeywordTableIndex` | 关键词倒排表 | 关键词精确匹配 |
| `PropertyGraphIndex` | 知识图谱 | 实体关系复杂的问题（如"A公司的子公司的CEO是谁"）|
| `DocumentSummaryIndex` | 先给每篇文档生成摘要，按摘要检索 | 文档数量多、需要先定位到文档 |

### 7.6 多轮对话：Chat Engine

```python
chat_engine = index.as_chat_engine(chat_mode="condense_plus_context")
print(chat_engine.chat("年假有几天？"))
print(chat_engine.chat("那病假呢？"))  # 能理解"那"指代的是"假期天数"
```

`condense_plus_context` 模式会先把"那病假呢？"结合历史**改写**成独立问题"病假有几天？"，再去检索。这一步非常关键，否则直接拿"那病假呢"去检索，效果会很差。

### 7.7 用 LlamaIndex 实现 Agent

LlamaIndex 可以把 Query Engine 包装成工具交给 Agent：

```python
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool

def get_weather(city: str) -> str:
    """查询指定城市明天的天气。"""
    return f"{city}明天有小雨，18~24℃"

handbook_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="employee_handbook",
    description="查询公司员工手册，涉及请假、薪资、报销等公司制度时使用。",
)

agent = FunctionAgent(
    tools=[handbook_tool, get_weather],  # 普通函数会被自动包装成工具
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="你是公司的智能助手。",
)

async def main():
    response = await agent.run("请年假要提前几天申请？北京明天天气如何？")
    print(str(response))

asyncio.run(main())
```

LlamaIndex 还提供：

- **AgentWorkflow**：多个 Agent 协作和相互交接（handoff）；
- **Workflows**：基于事件驱动的流程编排（类似 LangGraph 的定位）；
- **LlamaParse**：高质量的复杂文档解析服务（表格、图表、多栏排版的 PDF），这是很多 RAG 项目效果的关键瓶颈。

---

## 第 8 章 LangChain vs LlamaIndex：区别与选择

### 8.1 核心区别

| 维度 | LangChain | LlamaIndex |
|---|---|---|
| **定位** | 通用的 LLM 应用开发框架 | 以数据为中心的 RAG / 检索框架 |
| **核心优势** | 组件丰富、编排灵活、Agent 生态强 | 数据接入、索引、检索、查询优化 |
| **Agent 能力** | 很强（LangGraph 是业界主流 Agent 编排方案之一）| 不错（FunctionAgent、AgentWorkflow、Workflows）|
| **RAG 上手难度** | 中等，需要自己组装组件 | 很低，几行代码跑通 |
| **RAG 深度** | 足够，高级技巧需要自己组合 | 内置大量高级检索策略和索引类型 |
| **流程编排** | LCEL（简单链）+ LangGraph（复杂图）| Workflows（事件驱动）|
| **可观测性** | LangSmith（官方，功能完善）| 内置 instrumentation，可对接 Arize Phoenix、Langfuse 等 |
| **抽象程度** | 较灵活，可以控制每个细节 | 高层 API 非常简洁，底层也可定制 |

### 8.2 怎么选

```
你的需求主要是什么？
│
├─ 主要是"基于大量文档问答"（企业知识库、文档助手）
│     └─► 优先 LlamaIndex：数据处理和检索更省事
│
├─ 主要是"让 AI 执行多步骤任务、调用各种工具、多 Agent 协作"
│     └─► 优先 LangChain + LangGraph：编排能力更强
│
├─ 两者都很重要
│     └─► 组合使用：LlamaIndex 负责数据和检索，
│         包装成工具交给 LangGraph 编排的 Agent
│
└─ 需求非常简单、追求极致可控、团队不想引入依赖
      └─► 直接调用模型 SDK 手写（见第 5 章）
```

> 两个框架并不互斥。LlamaIndex 的 Query Engine 可以作为 LangChain 的工具使用，反过来 LangChain 的组件也能在 LlamaIndex 中使用。

---

## 第 9 章 核心问题：为什么要用框架构建 RAG 和 Agent

这是本文最核心的一章。我们已经在第 5 章看到，不用框架也能写出 RAG 和 Agent，那为什么业界普遍使用框架？下面从十个方面分析。

### 9.1 原因一：统一抽象，模型和组件可随时替换

**问题**：每家大模型、每个向量库的 API 都不一样。OpenAI、Claude、DeepSeek、通义的请求格式、工具调用格式、流式返回格式都有差异；Chroma、Milvus、pgvector 的接口也各不相同。

**框架的做法**：定义统一接口。所有 Chat Model 都有 `invoke/stream`，所有 Vector Store 都有 `add_documents/similarity_search`，所有 Retriever 都有 `invoke`。

**带来的价值**：

- 开发阶段用便宜的模型，上线换成效果好的模型，改一行配置；
- 因合规要求从海外模型切到国产模型，业务代码不用动；
- 原型用 Chroma，上线换 Milvus，只改初始化代码；
- 避免被单一供应商锁定。

### 9.2 原因二：庞大的集成生态，不用重复造轮子

**问题**：真实企业数据分散在 PDF、Word、Excel、网页、Confluence、飞书、Notion、数据库、邮件里。每种格式都写一个解析器，工作量巨大且容易出错。

**框架的做法**：LangChain 和 LlamaIndex（含 LlamaHub）都提供了数百个现成的文档加载器、向量库、模型、工具集成。

**带来的价值**：把精力放在业务逻辑上，而不是写第 N 个 PDF 解析器。

### 9.3 原因三：内置 RAG 最佳实践，效果起点更高

**问题**：手写 RAG 很容易跑通，但**效果很难做好**。第 5 章的手写版按固定字符数切分、只做向量检索，在真实数据上召回率往往不理想。

**框架的做法**：内置了业界验证过的方案：

- 按语义边界的递归切分、按 Markdown 标题切分、语义切分；
- 混合检索（向量 + BM25 关键词）；
- 重排序（Rerank）；
- 查询改写、多查询扩展、HyDE；
- 父子文档检索（用小块检索、返回大块上下文）；
- 多轮对话中的问题压缩改写（见 7.6 节）。

**带来的价值**：这些技巧每个自己实现都要几十到几百行代码和大量调试，框架里往往只需要换个参数或组件。

### 9.4 原因四：Agent 循环和工具调用的复杂性被封装

**问题**：第 5.2 节手写的 Agent 循环只是最简单的情况。生产环境要处理：

- 工具 JSON Schema 和函数签名保持一致；
- 模型输出非法参数、工具执行抛异常；
- 并行调用多个工具；
- 最大迭代次数，防止死循环烧钱；
- 不同模型的工具调用格式差异。

**框架的做法**：`@tool` 自动生成 Schema；`create_agent` / `FunctionAgent` 封装了完整的 ReAct 循环和错误处理。

**带来的价值**：几行代码得到一个健壮的 Agent，而不是从头处理各种边界情况。

### 9.5 原因五：状态管理与记忆

**问题**：API 无状态（1.4 节），对话历史需要自己维护。历史太长会超出窗口；多用户需要会话隔离；服务重启后要能恢复；长期记忆（如用户偏好）需要单独存储。

**框架的做法**：LangGraph 的 checkpointer（内存 / SQLite / Postgres / Redis）按 `thread_id` 自动保存和恢复状态；LlamaIndex 的 Memory 模块和 Chat Engine 自动管理对话历史；支持历史自动摘要和裁剪。

**带来的价值**：多用户、多会话、可恢复的对话系统开箱即用。

### 9.6 原因六：复杂流程编排

**问题**：真实业务很少是"一问一答"。例如一个智能客服可能是：判断意图 → 闲聊直接回答 / 业务问题去检索 / 订单问题查数据库 → 检索结果不相关则改写问题重试 → 涉及退款需要人工审批 → 生成回答。用 if-else 和 while 硬写，代码很快变得难以维护。

**框架的做法**：LangGraph 用图描述流程，LlamaIndex Workflows 用事件驱动描述流程，都支持分支、循环、并行、暂停等待人工审批（Human-in-the-loop）、断点续跑。

**带来的价值**：复杂流程可视化、可维护、可测试。

### 9.7 原因七：可观测性、调试与评估

**问题**：RAG 回答错了，到底是哪一步出了问题？是文档没解析对、切分不合理、检索没召回、还是模型没理解？Agent 为什么调用了错误的工具？每次请求花了多少 Token 和时间？手写代码基本是一个黑盒。

**框架的做法**：

- LangSmith 记录每一步的输入、输出、耗时、Token 消耗，可以逐步回放；
- LlamaIndex 内置 instrumentation，可对接 Langfuse、Arize Phoenix 等平台；
- 都提供评估工具：检索命中率、答案忠实度（是否基于资料）、答案相关性等指标。

**带来的价值**：从"凭感觉调 Prompt"变成"用数据驱动优化"，这是 Demo 走向生产的关键。

### 9.8 原因八：工程能力开箱即用

流式输出、异步、批处理、并发、失败重试、回退到备用模型（fallback）、缓存……这些都是生产必需的，框架的统一接口让每个组件自动具备这些能力（参见 6.3 节 LCEL 部分）。

### 9.9 原因九：开发效率与团队协作

- **原型快**：LlamaIndex 几行代码跑通 RAG，可以快速验证需求是否可行；
- **共同语言**：团队成员都理解 Retriever、Chain、Tool、Agent 这些概念，沟通成本低；
- **社区资源**：遇到问题有大量教程、示例和讨论；
- **跟进前沿**：新的 RAG 技巧、新模型发布后，框架通常很快就会集成。

### 9.10 原因十：从 Demo 平滑演进到生产

框架让你的系统可以渐进式升级，而不是推倒重来：

```
阶段1 原型验证：  LlamaIndex 5 行代码 + 内存向量库
      ↓  (替换组件，主体代码不变)
阶段2 效果优化：  调整切分 + 混合检索 + Rerank + 评估
      ↓
阶段3 能力扩展：  RAG 包装成工具 + Agent + 多工具
      ↓
阶段4 生产部署：  Milvus/pgvector + 持久化记忆 + 可观测性 + 人工审批
```

### 9.11 客观看待：框架的代价

使用框架也有缺点，面试和技术选型时要能说清楚：

| 代价 | 说明 | 应对 |
|---|---|---|
| **抽象过重** | 简单需求被包了好几层，不容易看懂底层发生了什么 | 先理解原理（第 3～5 章），再用框架 |
| **版本变化快** | API 经常调整，网上旧教程的代码可能跑不通 | 锁定版本号，以官方文档为准 |
| **调试困难** | 出错时调用栈很深 | 配合 LangSmith 等追踪工具 |
| **性能开销** | 多一层抽象，有一定额外开销 | 对绝大多数场景可忽略，瓶颈通常在模型调用 |
| **黑盒默认值** | 默认 Prompt、默认切分参数不一定适合你的数据 | 关键环节显式配置，不要全靠默认值 |

**结论**：

> 框架的价值不在于"能做到手写做不到的事"，而在于**把 RAG 和 Agent 中大量重复、易错、与业务无关的工程问题标准化**，让开发者专注于业务逻辑和效果优化。对于需求极其简单或对控制力要求极高的场景，直接用模型 SDK 手写也是合理的选择。

---

## 第 10 章 进阶：生产级 RAG 的常见优化手段

当基础 RAG 效果不够好时，按流程逐环节排查和优化：

### 10.1 数据处理阶段

- **文档解析质量**：PDF 中的表格、多栏排版、图片是最大痛点。可使用 LlamaParse、MinerU、Unstructured 等专业解析工具；
- **数据清洗**：去掉页眉页脚、目录、重复内容；
- **合理切分**：按文档结构（标题、章节）切分优于按固定长度切分；
- **丰富元数据**：部门、日期、文档类型等，检索时可以按元数据过滤。

### 10.2 检索阶段

- **混合检索（Hybrid Search）**：向量检索擅长语义，BM25 擅长专有名词、编号、型号，两者结合；
- **重排序（Rerank）**：先粗召回 20～50 个片段，再用 Rerank 模型（如 `bge-reranker`）精排出最相关的 3～5 个；
- **查询改写**：把口语化、有指代的问题改写成适合检索的形式；
- **多查询（Multi-Query）**：把一个问题改写成多个角度分别检索，再合并结果；
- **HyDE**：先让模型生成一个"假设答案"，用假设答案去检索；
- **父子文档（Small-to-Big）**：用小片段做检索（精准），命中后返回所属的大段落（上下文完整）。

### 10.3 生成阶段

- 在 Prompt 中明确要求"只根据资料回答，不知道就说不知道"；
- 要求标注引用来源；
- 控制上下文长度，把最相关的内容放在开头或结尾（模型对中间部分的注意力相对较弱）。

### 10.4 评估阶段

建立一个"问题 + 标准答案 + 标准出处"的测试集，量化评估：

| 指标 | 衡量什么 |
|---|---|
| 召回率 / 命中率（Hit Rate）| 正确的片段有没有被检索出来 |
| MRR | 正确片段排在第几位 |
| 忠实度（Faithfulness）| 答案是否都能在检索资料中找到依据（衡量幻觉）|
| 答案相关性（Answer Relevancy）| 答案是否回答了用户的问题 |

常用评估工具：Ragas、LangSmith、LlamaIndex 内置的 evaluation 模块。

---

## 第 11 章 动手路线：从环境搭建到跑通第一个项目

### 11.1 环境准备

```bash
# 1. 安装 Python 3.10+，创建虚拟环境
python -m venv .venv
# Windows PowerShell 激活：
.venv\Scripts\Activate.ps1
# macOS / Linux 激活：
# source .venv/bin/activate

# 2. 安装依赖（按需选择）
pip install openai numpy                                  # 第5章手写版
pip install langchain langchain-openai langchain-community langchain-text-splitters pypdf  # LangChain
pip install llama-index                                   # LlamaIndex
```

### 11.2 配置 API Key

```bash
# Windows PowerShell（当前会话有效）
$env:OPENAI_API_KEY = "sk-xxx"

# macOS / Linux
export OPENAI_API_KEY="sk-xxx"
```

更推荐在项目根目录建一个 `.env` 文件，用 `python-dotenv` 加载，并**把 `.env` 加入 `.gitignore`，绝不要把 Key 提交到代码仓库**。

使用国内模型时，大部分都兼容 OpenAI 接口，只需要额外设置 `base_url` 和对应的 `api_key`、`model` 名称。

### 11.3 推荐的学习顺序

1. **跑通第 5 章手写版**：理解每一步在做什么，这是一切的基础；
2. **用 LlamaIndex 5 行代码版**处理你自己的文档，感受框架的便利；
3. **用 LangChain LCEL 重写 RAG**，理解组件化和管道的思想；
4. **给 RAG 加上多轮对话**，理解问题改写和记忆；
5. **把 RAG 包装成工具，构建一个能查资料 + 调用其他工具的 Agent**；
6. **用 LangGraph 实现带"检索结果评估 → 不相关则改写重试"的 Agentic RAG**；
7. **接入 LangSmith 或 Langfuse，建立评估集，量化优化效果**；
8. **把向量库换成 Milvus / pgvector，用 FastAPI 封装成接口，做成完整项目**。

### 11.4 适合写进简历的练手项目

- **企业制度问答助手**：上传员工手册、报销制度等 PDF，支持多轮问答和引用出处；
- **个人论文/技术文档助手**：对一批论文做 RAG，支持跨文档对比和总结；
- **智能客服 Agent**：结合知识库检索 + 订单查询工具 + 转人工流程（LangGraph）；
- **代码仓库问答**：对一个开源项目代码建立索引，回答"某功能在哪里实现"。

写进简历时，突出**你做了哪些优化、指标提升了多少**（例如"引入混合检索和 Rerank 后，Top-5 命中率从 72% 提升到 89%"），而不仅仅是"用 LangChain 做了一个 RAG"。

---

## 第 12 章 术语表

| 术语 | 解释 |
|---|---|
| LLM | 大语言模型，根据前文预测下一个词的模型 |
| Token | 模型处理文本的最小单位，计费和上下文长度都按它计算 |
| Context Window | 上下文窗口，模型一次能处理的最大 Token 数 |
| Prompt | 发送给模型的全部输入文本 |
| System Prompt | 系统提示词，用于设定模型身份和规则 |
| Hallucination | 幻觉，模型一本正经地编造不存在的内容 |
| Fine-tuning | 微调，用特定数据继续训练模型 |
| RAG | 检索增强生成，先检索相关资料再让模型基于资料回答 |
| Embedding | 把文本转换成向量，使语义相近的文本向量也相近 |
| Vector Store | 向量数据库，存储向量并支持相似度搜索 |
| Chunk | 文档切分后的片段 |
| Top-K | 检索时返回最相似的 K 个结果 |
| Cosine Similarity | 余弦相似度，衡量两个向量方向的相似程度 |
| BM25 | 一种经典的关键词检索算法 |
| Hybrid Search | 混合检索，向量检索 + 关键词检索 |
| Rerank | 重排序，用更精确的模型对初步检索结果重新排序 |
| Agent | 智能体，能自主规划并调用工具完成任务的 LLM 应用 |
| Tool | 工具，Agent 可调用的函数（搜索、查库、调 API 等）|
| Function Calling | 模型以结构化格式输出"要调用哪个函数、传什么参数"的能力 |
| ReAct | 推理（Reasoning）+ 行动（Acting）交替进行的 Agent 范式 |
| Memory | 记忆，保存对话历史或长期信息 |
| Agentic RAG | 由 Agent 决定何时检索、如何检索的 RAG |
| Human-in-the-loop | 人在回路，关键步骤需要人工确认 |
| MCP | Model Context Protocol，一种让模型以统一方式接入外部工具和数据的开放协议 |
| LCEL | LangChain 表达式语言，用 `\|` 把组件串成链 |
| Runnable | LangChain 中所有可执行组件的统一接口 |
| LangGraph | LangChain 生态中基于图的有状态 Agent 编排框架 |
| LangSmith | LangChain 生态中的追踪、调试、评估平台 |
| Node（LlamaIndex）| 文档切分后的片段，带有与原文档和相邻片段的关系 |
| Query Engine | LlamaIndex 中负责"检索 + 生成"一问一答的组件 |
| Chat Engine | LlamaIndex 中带对话记忆的问答组件 |

---

## 第 13 章 面试高频问题

**Q1：什么是 RAG？为什么需要 RAG？**
答：检索增强生成。先从外部知识库检索与问题相关的资料，再把资料和问题一起交给大模型生成答案。用于解决大模型知识截止、不了解私有数据、容易幻觉、上下文窗口有限的问题，并且可以溯源、知识更新成本低。

**Q2：RAG 和微调怎么选？**
答：需要注入**经常变化的知识**、需要**溯源**时用 RAG；需要改变模型的**输出风格、格式、特定任务能力**时用微调。两者也可以结合。

**Q3：RAG 的完整流程是什么？**
答：离线阶段：加载 → 切分 → 向量化 → 存入向量库；在线阶段：问题向量化 → 检索 Top-K → 拼接 Prompt → 模型生成。

**Q4：chunk_size 怎么定？太大或太小有什么问题？**
答：太小会丢失上下文，片段语义不完整；太大会混入无关内容，降低检索精度，也浪费 Token。通常从 300～800 字开始，结合文档结构和评估结果调整，并设置 10%～20% 的重叠。也可以用父子文档策略兼顾两者。

**Q5：RAG 效果不好，你会怎么排查？**
答：分环节定位：先看文档解析是否正确 → 切分是否合理 → 正确片段是否被检索到（看召回率）→ 被检索到但排序靠后（加 Rerank）→ 资料正确但回答错误（优化 Prompt 或换模型）。建立评估集，用数据驱动优化。

**Q6：Agent 和普通的 Chain 有什么区别？**
答：Chain 的执行步骤是开发者预先写死的；Agent 由大模型在运行时动态决定调用哪些工具、调用几次、何时结束。

**Q7：Function Calling 的原理是什么？**
答：开发者用 JSON Schema 描述工具，模型判断需要时输出结构化的函数名和参数，由应用程序实际执行函数，并把结果返回给模型，模型再基于结果继续推理或给出最终答案。模型本身不执行代码。

**Q8：LangChain 和 LlamaIndex 有什么区别？**
答：LangChain 是通用的 LLM 应用开发框架，组件丰富、编排灵活，配合 LangGraph 在 Agent 和复杂流程上很强；LlamaIndex 以数据为中心，在文档接入、索引构建、检索优化方面更深入，构建 RAG 更省事。两者可以组合使用。

**Q9：为什么要用 LangChain / LlamaIndex，而不是直接调 API？**
答：统一抽象使模型和组件可替换；集成生态丰富；内置 RAG 最佳实践；封装 Agent 循环和工具调用；提供状态和记忆管理；支持复杂流程编排；提供可观测性和评估；流式、异步等工程能力开箱即用。代价是抽象较重、版本变化快，简单场景直接调 SDK 也可以。

**Q10：如何减少 RAG 中的幻觉？**
答：提升检索质量（混合检索、Rerank）；Prompt 中要求只根据资料回答、不知道就说不知道；要求引用出处；设置相似度阈值，检索不到相关内容时直接拒答；用忠实度指标评估。

---

> 下一步建议：先把第 5 章的手写代码在本地跑通，再进入第 11 章的学习路线。理解原理之后再用框架，你会清楚地知道框架每一行代码背后在做什么。
