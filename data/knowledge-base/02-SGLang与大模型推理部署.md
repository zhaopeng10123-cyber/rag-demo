# SGLang 从零入门：大模型推理部署，以及它和 LangChain 的关系

> 前置阅读：[01-LangChain与LlamaIndex从零入门](./01-LangChain与LlamaIndex从零入门.md)（至少读完第 1 章）。
> 阅读方式：按顺序读。第 1～3 章讲"模型是怎么跑起来的"，这是理解 SGLang 的前提，不要跳过。
> 说明：SGLang 迭代很快，安装命令和启动参数请以[官方文档](https://docs.sglang.ai/)为准，本文重点讲原理和用法思路。

---

## 目录

- [第 0 章 先回答问题：SGLang 是什么？和 LangChain 有关系吗？](#第-0-章-先回答问题sglang-是什么和-langchain-有关系吗)
- [第 1 章 从零开始：一个大模型是怎么"跑起来"的](#第-1-章-从零开始一个大模型是怎么跑起来的)
- [第 2 章 推理的两个阶段与 KV Cache](#第-2-章-推理的两个阶段与-kv-cache)
- [第 3 章 为什么需要专门的推理框架](#第-3-章-为什么需要专门的推理框架)
- [第 4 章 SGLang 是什么](#第-4-章-sglang-是什么)
- [第 5 章 SGLang 的核心技术](#第-5-章-sglang-的核心技术)
- [第 6 章 动手：用 SGLang 部署一个模型](#第-6-章-动手用-sglang-部署一个模型)
- [第 7 章 SGLang 和 LangChain / LlamaIndex 怎么配合](#第-7-章-sglang-和-langchain--llamaindex-怎么配合)
- [第 8 章 SGLang vs vLLM vs 其他推理框架](#第-8-章-sglang-vs-vllm-vs-其他推理框架)
- [第 9 章 为什么 RAG / Agent 场景特别适合 SGLang](#第-9-章-为什么-rag--agent-场景特别适合-sglang)
- [第 10 章 术语表](#第-10-章-术语表)
- [第 11 章 面试高频问题](#第-11-章-面试高频问题)

---

## 第 0 章 先回答问题：SGLang 是什么？和 LangChain 有关系吗？

### 0.1 一句话定义

> **SGLang 是一个高性能的大模型推理服务框架：它负责把开源大模型（Qwen、DeepSeek、Llama 等）加载到 GPU 上，以尽可能高的速度和并发处理请求，并对外提供 OpenAI 兼容的 API。**

### 0.2 和 LangChain 的关系

**没有直接关系，名字相似纯属巧合。** 它们处在技术栈的不同层次：

```
┌─────────────────────────────────────────────────────────┐
│ 应用层   你的业务：企业知识库、智能客服、Agent 助手          │
├─────────────────────────────────────────────────────────┤
│ 编排层   LangChain / LangGraph / LlamaIndex               │  ← 决定"做什么"
│          负责：RAG 流程、Agent 循环、工具调用、记忆          │    （调用模型 API）
├─────────────────────────────────────────────────────────┤
│              ↓  HTTP 请求（OpenAI 兼容 API）  ↓            │
├─────────────────────────────────────────────────────────┤
│ 推理层   SGLang / vLLM / TensorRT-LLM / Ollama             │  ← 决定"怎么跑得快"
│          负责：加载模型、GPU 计算、批处理、缓存、并发         │    （提供模型 API）
├─────────────────────────────────────────────────────────┤
│ 硬件层   NVIDIA / AMD GPU、昇腾 NPU 等                     │
└─────────────────────────────────────────────────────────┘
```

打个比方：

- **SGLang 是发电厂**：负责高效地"发电"（生成 Token）；
- **LangChain 是家里的电器**：负责用电去完成具体的事（问答、执行任务）。

电器不关心电是哪个电厂发的，只要插座（OpenAI 兼容 API）标准一致就能用。

| 对比项 | SGLang | LangChain |
|---|---|---|
| 所在层次 | 推理层（底层） | 编排层（上层） |
| 解决的问题 | 模型怎么在 GPU 上跑得快、扛得住高并发 | 怎么用模型搭建 RAG、Agent 等应用 |
| 输入 | 模型权重文件 + GPU | 模型 API + 业务数据 + 工具 |
| 输出 | 一个模型 API 服务 | 一个业务应用 |
| 是否需要 GPU | 需要（它就是用来驱动 GPU 的） | 不需要（只发 HTTP 请求） |
| 主要使用者 | 算法工程师、推理/平台工程师 | 应用开发工程师 |
| 同类产品 | vLLM、TensorRT-LLM、LMDeploy、Ollama | LlamaIndex、Dify、AutoGen |

### 0.3 什么时候你需要关心 SGLang

- 调用 OpenAI、DeepSeek 官方 API 时：**不需要**，推理部署是服务商的事；
- 需要**私有化部署**（数据不能出公司内网、合规要求）：**需要**；
- 调用量很大，想用开源模型**降低成本**：**需要**；
- 需要用**自己微调过的模型**：**需要**。

### 0.4 一个容易混淆的点

SGLang 的全称最初是 **Structured Generation Language（结构化生成语言）**。早期它除了推理引擎，还提供一套 Python 前端语言，用来编写"多次调用模型"的程序（见 5.5 节），这部分和 LangChain 的"链"确实有一些相似之处。

但两者的出发点完全不同：

- SGLang 的前端语言：目的是让推理引擎看清程序结构，从而**跑得更快**；
- LangChain：目的是**集成各种组件、编排业务流程**。

如今 SGLang 主要以**推理服务引擎**的身份被广泛使用，前端语言是次要功能。

---

## 第 1 章 从零开始：一个大模型是怎么"跑起来"的

### 1.1 模型到底是什么

一个大模型在磁盘上就是一堆文件：

```
Qwen2.5-7B-Instruct/
├── config.json                  # 模型结构配置：多少层、隐藏维度多大……
├── tokenizer.json               # 分词器：文字 ↔ Token ID 的映射表
├── model-00001-of-00004.safetensors   # 模型权重（参数）
├── model-00002-of-00004.safetensors
├── ...
```

其中**权重文件**存放的是模型的参数，即几十亿个浮点数。"7B"就是 70 亿（7 Billion）个参数。

### 1.2 运行一个模型需要多少显存

模型推理时，参数必须加载到 GPU 显存里。估算方法：

\[
\text{权重显存} \approx \text{参数量} \times \text{每个参数占用的字节数}
\]

| 精度 | 每个参数字节数 | 7B 模型权重 | 72B 模型权重 |
|---|---|---|---|
| FP32 | 4 字节 | ≈ 28 GB | ≈ 288 GB |
| FP16 / BF16 | 2 字节 | ≈ 14 GB | ≈ 144 GB |
| FP8 / INT8 | 1 字节 | ≈ 7 GB | ≈ 72 GB |
| INT4 | 0.5 字节 | ≈ 3.5 GB | ≈ 36 GB |

除了权重，还要留出显存给 **KV Cache**（第 2 章讲）和计算过程中的临时数据。所以 7B 模型 BF16 精度部署，24 GB 显存的显卡比较从容；72B 模型通常需要多张 80 GB 显卡。

降低精度的做法叫**量化（Quantization）**，可以大幅减少显存占用，代价是轻微的效果损失。

### 1.3 最简单的运行方式：HuggingFace Transformers

```python
# pip install transformers torch accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

messages = [{"role": "user", "content": "什么是RAG？"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

每一步在做什么：

1. `AutoTokenizer`：加载分词器；
2. `AutoModelForCausalLM`：把权重加载到 GPU；
3. `apply_chat_template`：把 `messages` 列表转换成模型训练时使用的对话格式（每个模型的格式不同，比如加上 `<|im_start|>user` 这样的特殊标记）；
4. `tokenizer(...)`：把文字转成 Token ID；
5. `model.generate`：逐个生成新 Token；
6. `tokenizer.decode`：把生成的 Token ID 转回文字。

这样能跑，**但只适合自己做实验**，原因在第 3 章说明。

---

## 第 2 章 推理的两个阶段与 KV Cache

要理解推理框架为什么能加速，必须先理解这一章。

### 2.1 模型是一个 Token 一个 Token 生成的

回顾 01 文档 1.1 节：模型根据前文预测下一个词。生成一句话的过程是这样的：

```
输入: "中国的首都是"
第1步: 模型看 "中国的首都是"       → 预测 "北"
第2步: 模型看 "中国的首都是北"     → 预测 "京"
第3步: 模型看 "中国的首都是北京"   → 预测 "。"
第4步: 模型看 "中国的首都是北京。" → 预测 <结束>
```

### 2.2 两个阶段：Prefill 和 Decode

| 阶段 | 做什么 | 特点 |
|---|---|---|
| **Prefill（预填充）** | 一次性处理整个输入 Prompt，生成第一个 Token | 输入的所有 Token 可以**并行**计算，主要受 GPU 算力限制（计算密集） |
| **Decode（解码）** | 每次生成一个新 Token，循环直到结束 | 只能**串行**一个一个生成，主要受显存读写速度限制（访存密集） |

两个衡量用户体验的指标与此对应：

- **TTFT（Time To First Token，首 Token 延迟）**：用户发出请求后多久看到第一个字，主要由 Prefill 决定；
- **TPOT（Time Per Output Token，每个输出 Token 的耗时）**：后续每个字出来的速度，主要由 Decode 决定。

### 2.3 什么是 KV Cache

在 Transformer 模型中，每个 Token 在每一层都会计算出两个向量：**Key（K）和 Value（V）**，后续 Token 需要"回头看"前面所有 Token 的 K 和 V（这就是注意力机制）。

**如果不缓存**：第 100 步生成时，要把前 99 个 Token 的 K、V 全部重新算一遍，第 101 步又重新算 100 个……大量重复计算。

**KV Cache 的做法**：每个 Token 的 K、V 算一次就存在显存里，后面直接读取。

```
不用 KV Cache：每一步都重算全部前文     → 计算量随长度平方增长
使用 KV Cache：每一步只算新 Token 的 K、V → 计算量随长度线性增长（但占用显存）
```

### 2.4 KV Cache 带来的新问题：显存

KV Cache 的大小和**上下文长度 × 并发请求数**成正比。一个长对话的 KV Cache 可能占用好几 GB 显存。所以：

> **推理框架的核心难题之一，就是如何高效地管理 KV Cache 这块显存：存得多、浪费少、能复用。**

记住这句话，第 5 章 SGLang 的核心创新 RadixAttention 就是在 KV Cache 复用上做文章。

---

## 第 3 章 为什么需要专门的推理框架

把第 1.3 节的 Transformers 代码直接包成一个 Web 服务给公司用，会遇到这些问题：

| 问题 | 原因 |
|---|---|
| **并发能力极差** | `generate` 一次处理一个请求，10 个人同时问就要排队 |
| **GPU 利用率低** | Decode 阶段每次只算 1 个 Token，GPU 大部分算力闲置 |
| **显存浪费严重** | 每个请求要按最大长度预留 KV Cache，实际用不完 |
| **重复计算** | 相同的系统提示词、相同的对话历史，每次请求都重新 Prefill |
| **没有标准 API** | 需要自己写 HTTP 服务、流式输出、OpenAI 格式兼容 |
| **大模型放不下** | 72B 模型单卡放不下，需要自己实现多卡并行 |
| **结构化输出不可靠** | 要求输出 JSON，模型偶尔格式出错 |

专门的推理框架用以下通用技术解决这些问题：

### 3.1 连续批处理（Continuous Batching）

**传统静态批处理**：凑齐一批请求一起算，必须等这一批里最长的那个生成完，才能开始下一批。短请求被迫陪跑。

**连续批处理**：以"每一步生成"为单位调度。某个请求生成完立刻退出，新请求立刻插入，GPU 始终满负荷。

```
静态批处理:  [A A A A A A] [D D D]
             [B B . . . .] [E E E]     ← B 早就完成了，但要等 A
             [C C C . . .] [F F F]

连续批处理:  [A A A A A A]
             [B B D D D E E]           ← B 完成后 D 立即补上
             [C C C F F F]
```

> 连续批处理的完整原理（与动态批处理的区别、调度器、Chunked Prefill、抢占）详见 [03-vLLM连续批处理与Triton-TensorRT](./03-vLLM连续批处理与Triton-TensorRT.md)。

### 3.2 分页注意力（PagedAttention）

借鉴操作系统虚拟内存分页的思想：把 KV Cache 切成固定大小的"页"，按需分配，不必为每个请求预留一整块连续显存。显存浪费从 60%～80% 降到很低，同样的显卡能同时服务更多请求。（这项技术由 vLLM 首先提出，现已被各推理框架普遍采用。）

### 3.3 前缀缓存（Prefix Caching）

如果多个请求的开头部分相同（例如相同的系统提示词），这部分的 KV Cache 只算一次，后续请求直接复用，跳过重复的 Prefill。**SGLang 在这方面做得最早、最系统**，见第 5 章。

### 3.4 其他常见技术

- **张量并行（Tensor Parallelism, TP）**：把一个大模型的每一层切开，分散到多张卡上同时计算；
- **量化（Quantization）**：FP8、INT4（AWQ、GPTQ）等，减少显存、提升速度；
- **投机解码（Speculative Decoding）**：用一个小模型快速"猜"后面几个 Token，大模型一次性验证，猜对了就一次前进多步；
- **约束解码（Constrained Decoding）**：强制模型输出符合 JSON Schema 或正则表达式；
- **OpenAI 兼容 API 服务**：开箱即用的 HTTP 服务，支持流式输出。

---

## 第 4 章 SGLang 是什么

### 4.1 背景

- 2023 年底由 LMSYS 组织（UC Berkeley、Stanford 等高校研究者发起，也是 Vicuna 模型和 Chatbot Arena 大模型排行榜的作者）的研究团队发布，论文为 *SGLang: Efficient Execution of Structured Language Model Programs*；
- 开源（Apache 2.0 协议），GitHub 仓库：[sgl-project/sglang](https://github.com/sgl-project/sglang)；
- 已被众多公司和云厂商用于大规模生产部署，DeepSeek 等模型发布时也会把 SGLang 列为推荐的部署方案之一。

### 4.2 SGLang 的两个组成部分

| 部分 | 作用 | 现状 |
|---|---|---|
| **后端运行时（SRT, SGLang Runtime）** | 高性能推理引擎 + API 服务 | **主要用途**，绝大多数人用的是这一部分 |
| **前端语言（Frontend Language）** | 用 Python 编写多步骤 LLM 程序的领域特定语言 | 辅助功能，适合特定场景 |

### 4.3 SGLang 能做什么

- 部署主流开源模型：Qwen、DeepSeek（含 V3/R1 等 MoE 大模型）、Llama、GLM、Mistral、Gemma 等；
- 部署多模态模型（图文理解）；
- 部署部分 Embedding 模型和 Reward 模型；
- 提供 OpenAI 兼容 API：`/v1/chat/completions`、`/v1/completions`、`/v1/embeddings`；
- 支持工具调用（Function Calling）、推理模型的思考内容解析、结构化输出；
- 支持多卡、多机分布式部署。

---

## 第 5 章 SGLang 的核心技术

### 5.1 RadixAttention：自动复用 KV Cache（最核心的创新）

**问题回顾**：第 3.3 节说过，很多请求的开头是相同的。看几个典型场景：

```
场景1 多轮对话：
  第1轮请求: [系统提示词][用户问题1]
  第2轮请求: [系统提示词][用户问题1][回答1][用户问题2]       ← 前面部分完全相同
  第3轮请求: [系统提示词][用户问题1][回答1][用户问题2][回答2][用户问题3]

场景2 同一应用的不同用户：
  用户A: [很长的系统提示词 + 工具说明][A的问题]
  用户B: [很长的系统提示词 + 工具说明][B的问题]               ← 前缀相同

场景3 RAG 对同一文档多次提问：
  请求1: [系统提示词][文档X的片段][问题1]
  请求2: [系统提示词][文档X的片段][问题2]                     ← 前缀相同
```

**SGLang 的做法**：

1. 请求处理完后，**不立即丢弃** KV Cache，而是保留在显存中；
2. 用一棵**基数树（Radix Tree）**来组织所有缓存。基数树是一种前缀树，相同前缀的序列共享同一条路径；
3. 新请求到来时，在树中查找**最长的匹配前缀**，这部分 KV Cache 直接复用，只需要计算剩余的新内容；
4. 显存不够时，按 **LRU（最近最少使用）** 策略淘汰最久没用的缓存节点。

```
                        [根]
                          │
                  [系统提示词 + 工具说明]          ← 所有请求共享，只算一次
                   ╱              ╲
           [用户A的问题1]      [用户B的问题]
                 │
           [回答1][问题2]                         ← A 的第2轮复用第1轮
                 │
           [回答2][问题3]                         ← A 的第3轮复用第2轮
```

**效果**：命中缓存的部分不再需要 Prefill，首 Token 延迟（TTFT）大幅下降，GPU 也省下算力去服务更多请求。整个过程**全自动**，使用者不需要做任何改动。

> 现在 vLLM 等框架也支持自动前缀缓存，但 RadixAttention 是 SGLang 最早系统化提出的核心设计，并且与它的调度器深度结合（例如优先调度能命中缓存的请求）。

### 5.2 高效的结构化输出（约束解码）

**问题**：Agent 调用工具、信息抽取等场景要求模型输出严格的 JSON。仅靠 Prompt 要求"请输出 JSON"，模型偶尔还是会出错（少个括号、多句废话），程序解析就会崩溃。

**约束解码的原理**：模型每生成一个 Token 前，会给词表中所有 Token 打分。约束解码会根据 JSON Schema（或正则表达式）判断"此时哪些 Token 是合法的"，把不合法的 Token 概率直接置零。这样模型**只可能**输出符合格式的内容。

```
Schema 要求: {"name": string, "age": integer}

已生成: {"name": "张三", "age":
下一个合法 Token: 只能是数字，模型想输出 "二十" 也不可能
```

**SGLang 的优化**：

- 把约束编译成**有限状态机**，并用"压缩有限状态机"技术：当后续内容是唯一确定的（比如 `{"name": "` 这种固定的格式字符），一次性跳过多个 Token，不必逐个生成；
- 集成 XGrammar 等高性能语法引擎，让约束解码几乎没有额外开销。

### 5.3 高性能调度器与其他优化

- **零开销 CPU 调度**：GPU 计算当前批次时，CPU 同时准备下一批次，避免 GPU 空等；
- **Prefill-Decode 分离（PD Disaggregation）**：第 2.2 节说过，Prefill 是计算密集、Decode 是访存密集，特性完全不同。大规模部署时把两者放在不同的 GPU 集群上，互不干扰；
- **MoE 模型优化**：针对 DeepSeek-V3/R1 这类混合专家模型，支持专家并行（Expert Parallelism）、数据并行注意力等，适合大规模部署；
- **投机解码**：支持 EAGLE 等投机解码算法；
- **多种量化格式**：FP8、INT4、AWQ、GPTQ 等。

### 5.4 前端语言：编写结构化的 LLM 程序

这就是 0.4 节提到的、和 LangChain 有一点相似的部分。它用 Python 函数描述一个多步骤的生成过程：

```python
import sglang as sgl

@sgl.function
def multi_turn_qa(s, question1, question2):
    s += sgl.system("你是一个乐于助人的助手。")
    s += sgl.user(question1)
    s += sgl.assistant(sgl.gen("answer1", max_tokens=256))
    s += sgl.user(question2)
    s += sgl.assistant(sgl.gen("answer2", max_tokens=256))

sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

state = multi_turn_qa.run(question1="什么是RAG？", question2="它和微调有什么区别？")
print(state["answer1"])
print(state["answer2"])
```

还支持：

- `sgl.select("choice", choices=["是", "否"])`：让模型从固定选项中选择；
- `s.fork(n)`：从当前状态分叉出多个分支**并行**生成，并且共享前缀的 KV Cache；
- `sgl.gen(..., regex=...)`：用正则约束输出格式。

**它的价值**：因为整个程序结构对推理引擎可见，引擎知道哪些部分共享前缀、哪些可以并行，能做出更好的优化。

**它的局限**：只能对接 SGLang 后端，没有 LangChain 那样丰富的文档加载、向量库、工具等生态集成。所以实际项目中通常是 **LangChain 负责编排，SGLang 负责推理**。

---

## 第 6 章 动手：用 SGLang 部署一个模型

### 6.1 环境要求

- **操作系统：Linux**。Windows 用户请使用 **WSL2**（Windows 子系统 Linux）或 Docker；
- **GPU**：NVIDIA GPU（也支持 AMD 及部分其他加速卡），并安装好驱动；
- **Python**：3.10 及以上。

> 如果你只有 Windows 笔记本、没有独立显卡，想在本地体验开源模型，更适合用 **Ollama**（见第 8 章），SGLang 定位是服务器上的生产级部署。

### 6.2 安装

```bash
# 方式一：pip 安装（具体命令以官方文档为准）
pip install --upgrade pip
pip install "sglang[all]"

# 方式二：Docker（推荐，免去 CUDA 环境配置的麻烦）
docker run --gpus all --shm-size 32g -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 30000
```

国内网络下载 HuggingFace 模型较慢时，可以设置镜像环境变量 `HF_ENDPOINT=https://hf-mirror.com`，或者先从 ModelScope 下载模型到本地，再把 `--model-path` 指向本地目录。

### 6.3 启动服务

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 30000
```

常用参数：

| 参数 | 作用 |
|---|---|
| `--model-path` | HuggingFace 模型名或本地模型目录 |
| `--tp` | 张量并行的 GPU 数量，例如 `--tp 4` 表示用 4 张卡跑一个模型 |
| `--mem-fraction-static` | 预留给模型权重和 KV Cache 的显存比例，显存不足报错时可调小 |
| `--context-length` | 最大上下文长度 |
| `--quantization` | 量化方式，例如 `fp8` |
| `--tool-call-parser` | 开启工具调用解析，需要与模型匹配（如 Qwen 系列、DeepSeek 系列有各自的解析器） |
| `--reasoning-parser` | 解析推理模型（如 DeepSeek-R1）的思考过程，与最终回答分开返回 |
| `--is-embedding` | 以 Embedding 模型的方式启动 |

看到类似 `The server is fired up and ready to roll!` 的日志，说明启动成功。

### 6.4 测试服务

**用 curl 测试：**

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "用一句话介绍RAG"}]
  }'
```

**用 OpenAI SDK 测试**（注意：和 01 文档第 1.3 节的代码几乎一样，只是换了 `base_url`）：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")  # 本地服务不校验 Key

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "用一句话介绍RAG"}],
    temperature=0,
)
print(resp.choices[0].message.content)
```

这就是 OpenAI 兼容 API 的意义：**你自己部署的模型，用法和调用 OpenAI 官方 API 完全一样**。

### 6.5 结构化输出

```python
import json

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "years_of_experience": {"type": "integer"},
        "skills": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "years_of_experience", "skills"],
}

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "提取信息：张三，5年Python经验，熟悉LangChain和FastAPI"}],
    response_format={"type": "json_schema", "json_schema": {"name": "resume", "schema": schema}},
)
data = json.loads(resp.choices[0].message.content)  # 由约束解码保证一定是合法 JSON
print(data["skills"])
```

---

## 第 7 章 SGLang 和 LangChain / LlamaIndex 怎么配合

这是两份文档的交汇点。核心思路只有一句话：

> **SGLang 启动一个 OpenAI 兼容的服务，LangChain / LlamaIndex 把它当作"OpenAI"来调用，只需要改 `base_url`。**

### 7.1 整体架构

```
┌──────────────────────────────────────────────────────┐
│                  你的应用（FastAPI 等）                 │
│  ┌────────────────────────────────────────────────┐  │
│  │ LangChain / LangGraph                          │  │
│  │  文档加载 → 切分 → 检索 → Prompt → Agent 循环    │  │
│  └───────────┬──────────────────────┬─────────────┘  │
└──────────────┼──────────────────────┼────────────────┘
               │ /v1/chat/completions │ /v1/embeddings
               ▼                      ▼
     ┌───────────────────┐   ┌───────────────────┐   ┌──────────────┐
     │ SGLang 服务 1      │   │ SGLang 服务 2      │   │ 向量数据库     │
     │ 对话模型 Qwen2.5-7B │   │ Embedding 模型     │   │ Milvus 等     │
     │ GPU 0              │   │ GPU 1              │   │              │
     └───────────────────┘   └───────────────────┘   └──────────────┘
                     全部部署在公司内网，数据不出域
```

### 7.2 LangChain 调用 SGLang

把 01 文档第 6.4 节的 RAG 代码中的模型替换掉即可，**其余代码一行都不用改**：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="http://localhost:30000/v1",
    api_key="EMPTY",
    temperature=0,
)
print(llm.invoke("用一句话介绍RAG").content)
```

如果 SGLang 启动时开启了 `--tool-call-parser`，这个 `llm` 也可以直接用于 01 文档第 6.5 节的 `create_agent`，构建完全私有化部署的 Agent。

### 7.3 LlamaIndex 调用 SGLang

```python
# pip install llama-index-llms-openai-like
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike

Settings.llm = OpenAILike(
    model="Qwen/Qwen2.5-7B-Instruct",
    api_base="http://localhost:30000/v1",
    api_key="EMPTY",
    is_chat_model=True,
    is_function_calling_model=True,  # 服务端开启了工具调用解析时设为 True
)
```

> 这里用 `OpenAILike` 而不是 `OpenAI` 类，是因为 LlamaIndex 的 `OpenAI` 类会校验模型名是否是 OpenAI 官方模型。

### 7.4 常见坑

| 现象 | 原因 | 解决 |
|---|---|---|
| Agent 不调用工具，或者把工具调用当成普通文字输出 | 启动 SGLang 时没有开启工具调用解析，或解析器与模型不匹配 | 加上 `--tool-call-parser`，并选择与模型匹配的解析器 |
| 报错上下文超长 | RAG 塞入的资料 + 对话历史超过了 `--context-length` | 减少 Top-K、缩短片段、裁剪历史，或调大上下文长度（需显存支持） |
| 启动时显存不足（OOM） | 模型太大或 KV Cache 预留过多 | 调小 `--mem-fraction-static`、使用量化、增加 `--tp` |
| 输出夹杂思考过程 | 推理模型（如 R1）的思考内容未被分离 | 加上 `--reasoning-parser` |

---

## 第 8 章 SGLang vs vLLM vs 其他推理框架

| 框架 | 定位 | 特点 | 适合谁 |
|---|---|---|---|
| **SGLang** | 生产级高性能推理服务 | RadixAttention 前缀缓存、结构化输出快、MoE 大模型和大规模分布式部署能力强 | 高并发、多轮对话、Agent、DeepSeek 类大模型部署 |
| **vLLM** | 生产级高性能推理服务 | PagedAttention 提出者，社区最大，模型和硬件支持最广 | 通用首选之一，生态最成熟 |
| **TensorRT-LLM** | NVIDIA 官方推理库 | 针对 NVIDIA GPU 深度优化，性能上限高，但配置复杂 | 追求极致性能、只用 NVIDIA 卡 |
| **LMDeploy** | 上海 AI Lab 推出的推理框架 | 量化和国产模型支持好 | 国内团队、量化部署 |
| **TGI** | HuggingFace 推出的推理服务 | 与 HuggingFace 生态集成好 | HuggingFace 生态用户 |
| **Ollama / llama.cpp** | 本地个人使用 | 安装简单，能在笔记本 CPU / Mac 上运行量化模型，**支持 Windows** | 个人学习、本地体验 |

**怎么选：**

```
你要做什么？
│
├─ 在自己的笔记本上体验开源模型、学习 RAG
│     └─► Ollama（一条命令启动，同样提供 OpenAI 兼容 API）
│
├─ 公司服务器上部署，给团队或业务使用
│     └─► SGLang 或 vLLM，两者都是主流选择，
│         用自己的模型和真实请求做压测再决定
│         · 多轮对话 / Agent / 大量共享前缀 / 结构化输出多 → SGLang 优势明显
│         · 需要最广泛的模型和硬件兼容性 → vLLM
│
└─ 只用 NVIDIA 卡、有专人调优、追求极致性能
      └─► TensorRT-LLM
```

> 无论选哪个，它们都提供 OpenAI 兼容 API，所以**上层的 LangChain / LlamaIndex 代码完全不用改**，推理框架可以随时替换。这也呼应了 01 文档第 9.1 节"统一抽象，组件可替换"的思想。

> vLLM、TensorRT-LLM，以及常和它们一起出现的 Triton，详见 [03-vLLM连续批处理与Triton-TensorRT](./03-vLLM连续批处理与Triton-TensorRT.md)。

---

## 第 9 章 为什么 RAG / Agent 场景特别适合 SGLang

把两份文档的知识结合起来看：

### 9.1 Agent 的请求天然有大量共享前缀

回顾 01 文档第 5.2 节的 Agent 循环：每调用一次工具，就要把**完整的消息历史**重新发给模型。

```
第1次请求: [系统提示词][工具说明][用户问题]
第2次请求: [系统提示词][工具说明][用户问题][工具调用1][工具结果1]
第3次请求: [系统提示词][工具说明][用户问题][工具调用1][工具结果1][工具调用2][工具结果2]
```

每一次请求都是上一次的"延长版"。没有前缀缓存时，每次都要从头 Prefill；有了 RadixAttention，每次只需计算新增的部分。**Agent 步骤越多、工具说明越长，收益越大。**

### 9.2 RAG 的系统提示词和热门文档会被反复使用

- 同一个应用的所有用户共享同一段系统提示词；
- 热门问题会反复检索到同一批文档片段；
- 多轮对话中，同一份资料会随着历史被反复发送。

这些都能被前缀缓存命中。

### 9.3 Agent 依赖可靠的结构化输出

工具调用的参数、路由决策（"这个问题该交给哪个 Agent"）、信息抽取结果都需要合法 JSON。SGLang 的约束解码保证格式正确且开销很小，减少 Agent 因解析失败而出错重试的情况。

### 9.4 私有化部署的现实需求

企业知识库里的文档往往涉及内部制度、客户数据、财务信息，很多公司不允许发送到外部 API。用 SGLang 在内网部署开源模型 + LangChain 构建应用，是常见的企业级私有化 RAG / Agent 方案。

---

## 第 10 章 术语表

| 术语 | 解释 |
|---|---|
| 推理（Inference） | 用训练好的模型生成结果的过程（区别于训练） |
| 推理框架 / 推理引擎 | 专门优化模型推理速度和并发的软件，如 SGLang、vLLM |
| 权重（Weights） | 模型的参数，即模型文件中存储的大量浮点数 |
| 7B / 72B | 模型参数量，B = Billion（十亿） |
| 显存（VRAM） | GPU 上的内存，模型权重和 KV Cache 都存在这里 |
| 量化（Quantization） | 降低参数精度（如 16 位降到 4 位）以减少显存、提升速度 |
| Prefill | 推理第一阶段，并行处理整个输入 Prompt |
| Decode | 推理第二阶段，逐个生成输出 Token |
| TTFT | 首 Token 延迟，从发出请求到收到第一个 Token 的时间 |
| TPOT | 每个输出 Token 的平均生成时间 |
| 吞吐量（Throughput） | 单位时间内处理的 Token 数或请求数 |
| KV Cache | 缓存每个 Token 在注意力计算中的 Key 和 Value，避免重复计算 |
| Continuous Batching | 连续批处理，以生成步为单位动态调度请求 |
| PagedAttention | 分页管理 KV Cache 显存，减少浪费 |
| Prefix Caching | 前缀缓存，复用相同前缀的 KV Cache |
| RadixAttention | SGLang 用基数树管理和自动复用 KV Cache 的技术 |
| Radix Tree | 基数树，一种压缩的前缀树 |
| Constrained Decoding | 约束解码，强制模型输出符合指定格式 |
| Speculative Decoding | 投机解码，用小模型猜测、大模型验证来加速 |
| Tensor Parallelism | 张量并行，把模型每层切分到多张 GPU 上 |
| MoE | 混合专家模型，每次只激活部分参数，如 DeepSeek-V3 |
| Expert Parallelism | 专家并行，把 MoE 模型的不同专家分布到不同 GPU |
| PD Disaggregation | Prefill 和 Decode 分离部署 |
| OpenAI 兼容 API | 与 OpenAI 接口格式一致的 API，让上层代码可以无缝切换模型服务 |
| WSL2 | Windows 上运行 Linux 环境的官方方案 |

---

## 第 11 章 面试高频问题

**Q1：SGLang 是什么？和 LangChain 有什么关系？**
答：SGLang 是高性能大模型推理服务框架，负责把模型部署在 GPU 上并提供 OpenAI 兼容 API；LangChain 是应用编排框架，负责调用模型构建 RAG、Agent。两者处于不同层次，没有直接关系，但可以配合使用：SGLang 提供模型服务，LangChain 通过改 `base_url` 调用它。

**Q2：为什么不能直接用 Transformers 的 `generate` 部署生产服务？**
答：一次只处理一个请求、并发差；GPU 利用率低；KV Cache 显存预留浪费；相同前缀重复计算；没有标准 API 和流式服务；大模型多卡并行需要自己实现。推理框架通过连续批处理、PagedAttention、前缀缓存、张量并行等技术解决这些问题。

**Q3：什么是 KV Cache？为什么它是推理优化的关键？**
答：缓存每个 Token 在每层注意力计算中的 Key 和 Value，避免生成每个新 Token 时重复计算前文，把计算量从平方级降到线性级。但它占用大量显存，且与上下文长度和并发数成正比，所以 KV Cache 的管理（分配、复用、淘汰）直接决定了推理服务的并发能力和延迟。

**Q4：RadixAttention 的原理是什么？**
答：请求结束后保留 KV Cache，用基数树按 Token 序列组织所有缓存；新请求到来时匹配最长公共前缀，直接复用这部分 KV Cache，只计算剩余部分；显存不足时按 LRU 淘汰。对多轮对话、共享系统提示词、Agent 多步调用等场景效果显著。

**Q5：Prefill 和 Decode 有什么区别？**
答：Prefill 并行处理整个输入，计算密集，决定首 Token 延迟；Decode 逐个生成 Token，访存密集，决定生成速度。因为特性不同，大规模部署时可以将两者分离到不同机器上（PD 分离）。

**Q6：如何保证模型输出合法 JSON？**
答：Prompt 约束不可靠，生产中使用约束解码：根据 JSON Schema 构建状态机，每一步把不合法 Token 的概率置零，保证输出一定符合格式。SGLang 通过压缩状态机和高性能语法引擎让这一过程开销很小。

**Q7：SGLang 和 vLLM 怎么选？**
答：两者都是主流的生产级推理框架，都支持连续批处理、分页 KV Cache、前缀缓存、OpenAI 兼容 API。SGLang 在前缀复用、结构化输出、MoE 大模型大规模部署上有优势，适合多轮对话和 Agent 场景；vLLM 社区和模型、硬件支持更广。最终应基于自己的模型和真实请求做压测决定。

**Q8：设计一个企业私有化 RAG 系统，你会怎么搭？**
答：推理层用 SGLang（或 vLLM）在内网部署对话模型和 Embedding 模型，开启工具调用解析；数据层用专业解析工具处理文档，存入 Milvus 或 pgvector；编排层用 LlamaIndex 做数据接入和检索，用 LangChain / LangGraph 编排 Agent 流程；服务层用 FastAPI 对外提供接口；再加上 LangSmith 或 Langfuse 做追踪和评估。全部组件部署在内网，数据不出域。

---

> 下一步建议：如果有 Linux + NVIDIA GPU 环境，用 SGLang 部署一个 Qwen2.5-7B，再把 01 文档第 6.4 节的 RAG 代码改成调用本地服务，你就完成了一个完整的私有化 RAG。如果只有 Windows 笔记本，先用 Ollama 走通同样的流程，思路完全一致。
