# vLLM 连续批处理、Triton 与 TensorRT 从零讲解

> 前置阅读：[02-SGLang与大模型推理部署](./02-SGLang与大模型推理部署.md) 的第 1～3 章（显存、Prefill/Decode、KV Cache）。本文直接使用这些概念。
> 阅读方式：按顺序读。第 1～4 章讲 vLLM 和连续批处理，第 5～8 章讲 Triton 和 TensorRT，第 9 章把它们放进同一张图里。
> 说明：这几个项目更新很快，命令和参数请以官方文档为准，本文重点讲原理。

---

## 目录

- [第 0 章 先给结论](#第-0-章-先给结论)
- [第 1 章 vLLM 是什么](#第-1-章-vllm-是什么)
- [第 2 章 从零理解批处理：为什么"一起算"更快](#第-2-章-从零理解批处理为什么一起算更快)
- [第 3 章 三种批处理方式：静态、动态、连续](#第-3-章-三种批处理方式静态动态连续)
- [第 4 章 vLLM 里的连续批处理是怎么实现的](#第-4-章-vllm-里的连续批处理是怎么实现的)
- [第 5 章 先澄清：Triton 是两个不同的东西](#第-5-章-先澄清triton-是两个不同的东西)
- [第 6 章 NVIDIA Triton Inference Server：模型服务器](#第-6-章-nvidia-triton-inference-server模型服务器)
- [第 7 章 OpenAI Triton：写 GPU 算子的编程语言](#第-7-章-openai-triton写-gpu-算子的编程语言)
- [第 8 章 TensorRT 与 TensorRT-LLM：NVIDIA 的推理优化器](#第-8-章-tensorrt-与-tensorrt-llmnvidia-的推理优化器)
- [第 9 章 全景图：这些东西分别在哪一层](#第-9-章-全景图这些东西分别在哪一层)
- [第 10 章 术语表](#第-10-章-术语表)
- [第 11 章 面试高频问题](#第-11-章-面试高频问题)

---

## 第 0 章 先给结论

| 名词 | 一句话解释 | 类比 |
|---|---|---|
| **vLLM** | 开源的大模型推理服务引擎，和 SGLang 是同类产品 | 一座高效的发电厂 |
| **Continuous Batching（连续批处理）** | 以"每生成一个 Token"为单位调度请求，有请求完成就立刻让新请求补位 | 公交车每到一站都可以上下客，而不是满员发车、全部到终点才换下一批 |
| **NVIDIA Triton Inference Server** | NVIDIA 的通用模型服务器，能同时托管多种框架、多种类型的模型 | 一个可以接入各种发电机的配电站 |
| **OpenAI Triton** | 一种用 Python 写 GPU 计算函数（算子）的编程语言和编译器 | 给工程师用的"发电机零件加工机床" |
| **TensorRT** | NVIDIA 的推理优化编译器，把训练好的模型编译成针对特定 GPU 的高速版本 | 把发电机针对某种燃料重新调校 |
| **TensorRT-LLM** | 基于 TensorRT 思想、专为大模型推理打造的 NVIDIA 官方库 | NVIDIA 原厂出品的大模型专用发电厂 |

它们的关系一句话概括：

> **vLLM、SGLang、TensorRT-LLM 是同一层的"大模型推理引擎"，都实现了连续批处理；NVIDIA Triton Inference Server 是更外层的"服务器"，可以把这些引擎当作后端来托管；OpenAI Triton 和 TensorRT 则在更底层，负责让每一次 GPU 计算本身更快。**

---

## 第 1 章 vLLM 是什么

### 1.1 背景

- 2023 年由 UC Berkeley 的研究团队开源，论文 *Efficient Memory Management for Large Language Model Serving with PagedAttention* 发表于 SOSP 2023；
- 最核心的贡献是 **PagedAttention**（02 文档第 3.2 节提过），配合连续批处理，让大模型推理吞吐量相比当时的 HuggingFace Transformers 提升了一个数量级；
- 现在是 PyTorch 基金会下的项目，是社区规模最大的开源推理引擎之一，支持的模型和硬件（NVIDIA、AMD、Intel、TPU、昇腾等）非常广。

### 1.2 vLLM 和 SGLang 的关系

它们是**同类竞品**，都在推理层，功能高度重合：

- 都支持连续批处理、分页 KV Cache、前缀缓存、量化、张量并行、投机解码；
- 都提供 OpenAI 兼容 API，上层的 LangChain 代码可以无缝切换；
- 两个项目互相借鉴，性能你追我赶。差异和选型见 02 文档第 8 章。

### 1.3 两种使用方式

**方式一：启动 API 服务（最常用）**

```bash
pip install vllm          # 需要 Linux + GPU，Windows 用 WSL2 或 Docker

vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

之后的调用方式和 02 文档第 6.4 节完全一样，只是端口不同：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
```

**方式二：离线批量推理（Python 内直接调用）**

适合一次性处理大量数据，例如给 10 万条评论打标签：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
params = SamplingParams(temperature=0, max_tokens=64)

prompts = [f"判断这条评论的情感（正面/负面）：{c}" for c in comments]  # comments 是一个大列表
outputs = llm.generate(prompts, params)   # vLLM 内部自动用连续批处理并发执行

for out in outputs:
    print(out.outputs[0].text)
```

注意这里你一次性把所有 prompt 交给了 vLLM，**怎么分批、怎么调度，全部由 vLLM 内部的连续批处理决定**，这就是下面几章要讲的内容。

---

## 第 2 章 从零理解批处理：为什么"一起算"更快

### 2.1 回顾：Decode 阶段的瓶颈不在算力

02 文档第 2.2 节说过：Decode 阶段每次只生成 1 个 Token，是**访存密集**的。具体来说：

- 生成 1 个 Token，GPU 需要把**全部模型权重**从显存读一遍（7B 模型 BF16 约 14 GB）；
- 但读进来之后，只对 1 个 Token 做计算，计算量很小；
- 结果就是：GPU 大部分时间在"等数据从显存搬过来"，计算单元闲着。

用一个比喻：

> 厨师（GPU 计算单元）炒菜很快，但每炒一盘菜都要去仓库（显存）把所有调料搬一遍。如果一次只炒一盘，大部分时间都花在搬调料上。

### 2.2 批处理的本质：读一次权重，服务多个请求

如果同时有 32 个请求在 Decode，GPU 把权重读进来一次，就能同时为 32 个请求各算出 1 个 Token：

```
不批处理:  读权重 → 算请求A的1个Token → 读权重 → 算请求B的1个Token → ...（读 32 次）
批处理:    读权重 → 同时算 A、B、C……共 32 个请求的各 1 个 Token        （读 1 次）
```

在批大小还不太大的时候，**处理 32 个请求的耗时和处理 1 个请求差不多**，吞吐量几乎提升 32 倍。直到批大小大到计算单元也忙不过来（变成计算密集），提升才会放缓。

> 厨师一次把调料搬过来，同时炒 32 盘菜。

### 2.3 那么问题来了：怎么"凑批"？

大模型推理和普通的图像分类模型不一样：

- **每个请求的输入长度不同**：有的问题 10 个字，有的 RAG 请求带了 5000 字资料；
- **每个请求的输出长度不同，而且事先不知道**：有的回答 5 个字，有的回答 2000 字，只有生成到结束符才知道。

"怎么把长短不一、结束时间未知的请求凑在一起算"，就是批处理策略要解决的问题。

---

## 第 3 章 三种批处理方式：静态、动态、连续

### 3.1 静态批处理（Static Batching）

**做法**：凑齐固定数量的请求（比如 4 个）组成一批，一起开始，**整批全部生成完**才返回结果，再处理下一批。

```
时间步:     1  2  3  4  5  6  7  8  9  10
请求A:      ■  ■  ■  ■  ■  ■  ■  ■  ■  ■    (需要生成 10 个 Token)
请求B:      ■  ■  ■  □  □  □  □  □  □  □    (3 个就结束了，后面 7 步白白占着位置)
请求C:      ■  ■  ■  ■  ■  □  □  □  □  □
请求D:      ■  ■  □  □  □  □  □  □  □  □
                                            ↑ 第 10 步结束后，E、F、G、H 才能开始
■ = 有效计算    □ = 浪费（陪跑）
```

**问题**：

1. **陪跑浪费**：短请求结束后，它的位置一直空着直到最长的请求结束，上图中 40 个格子里有 20 个是浪费；
2. **排队延迟**：新请求要等整批结束，B 只需要 3 步，却要等 10 步才能拿到结果；
3. **填充浪费**：输入长度不同，要把短的补齐（padding）到和最长的一样长。

### 3.2 动态批处理（Dynamic Batching）

**做法**：不要求凑齐固定数量，而是设一个**等待时间窗口**（比如 5 毫秒），窗口内到达的请求组成一批。

- 优点：低负载时不用干等凑满一批；
- 缺点：**仍然是以"请求"为单位**组批，批一旦开始就不能加入新请求，陪跑问题依然存在。

动态批处理非常适合**图像分类、Embedding、Rerank 这类"一次前向计算就出结果"的模型**，因为它们每个请求的计算时间几乎一样，不存在"有的早结束有的晚结束"的问题。NVIDIA Triton Inference Server 的 Dynamic Batching 功能就是这种（见第 6 章）。

### 3.3 连续批处理（Continuous Batching）

**核心思想**：调度的单位从"一个请求"变成"**一步生成**（iteration，迭代）"。

- 每生成一步（每个请求各生成 1 个 Token）之后，调度器都重新检查一次；
- 有请求生成了结束符 → **立即移出**批次，结果立刻返回给用户；
- 有空位 + 等待队列里有新请求 → **立即加入**批次。

```
时间步:     1  2  3  4  5  6  7  8  9  10
槽位1:      A  A  A  A  A  A  A  A  A  A
槽位2:      B  B  B  E  E  E  E  H  H  H     ← B 结束后，E 在第 4 步立刻补上
槽位3:      C  C  C  C  C  F  F  F  F  F     ← C 结束后，F 补上
槽位4:      D  D  G  G  G  G  G  G  I  I     ← D 结束后，G 补上；G 结束后，I 补上

几乎没有空闲格子；B 在第 3 步结束后马上就能拿到结果
```

**效果**：

- GPU 始终满负荷，**吞吐量**大幅提升；
- 短请求不用陪跑，**延迟**显著降低；
- 新请求不需要等一整批结束，排队时间短。

### 3.4 名字有很多，说的是同一件事

| 叫法 | 出处 |
|---|---|
| Iteration-level Scheduling（迭代级调度） | 2022 年 Orca 论文（OSDI 2022），最早系统提出这个思想 |
| Continuous Batching（连续批处理） | 社区和 vLLM 等开源项目最常用的叫法 |
| In-flight Batching（飞行中批处理） | NVIDIA TensorRT-LLM 的叫法 |

> 注意区分：**Dynamic Batching（动态批处理）是请求级的，不是连续批处理**。面试中经常被拿来对比。

### 3.5 一个关键问题：Prefill 和 Decode 怎么一起算？

新请求加入时需要先做 Prefill（一次性处理整个输入），而正在运行的请求在做 Decode（每次 1 个 Token）。两者计算特性差异很大（02 文档第 2.2 节），怎么放在同一步里？

早期做法是：有新请求时，这一步先只做 Prefill，正在 Decode 的请求暂停一步。问题是，如果新请求带了 8000 Token 的 RAG 资料，这次 Prefill 很慢，所有正在生成的用户都会感到"卡了一下"。

改进做法是 **Chunked Prefill（分块预填充）**，下一章详细讲。

---

## 第 4 章 vLLM 里的连续批处理是怎么实现的

### 4.1 为什么连续批处理需要 PagedAttention

连续批处理意味着请求随时进出，每个请求的 KV Cache 长度也在不断增长、且最终长度未知。

如果像传统做法那样给每个请求**预留一整块连续显存**（按最大长度，比如 8192 Token）：

- 大部分请求根本用不了那么长，**显存大量浪费**；
- 浪费的显存导致能同时运行的请求数变少，**批大小上不去**，连续批处理的优势就发挥不出来。

PagedAttention 的做法（类比操作系统的内存分页）：

```
KV Cache 显存被切成固定大小的块（Block），比如每块存 16 个 Token 的 K、V

请求A 的逻辑序列: [Token 1-16][Token 17-32][Token 33-40]
                       ↓            ↓            ↓
物理显存块:          块#7         块#2         块#15      ← 不需要连续
                  （块表 Block Table 记录这个映射关系）

请求A 每多生成 16 个 Token，才再申请一个新块
请求A 结束后，它占用的块立刻归还，分配给新请求
```

**连续批处理 + PagedAttention = 请求可以随时进出，显存也可以随时分配和回收**。两者是相互配合的。

### 4.2 调度器每一步在做什么

vLLM 内部有一个**调度器（Scheduler）**，维护两个队列：

- **等待队列（Waiting）**：刚到达、还没开始处理的请求；
- **运行队列（Running）**：正在生成中的请求。

每一步（每次 GPU 前向计算之前），调度器按以下逻辑决定这一步算哪些内容：

```
第1步：先安排运行队列中的请求，每个请求本步生成 1 个 Token（Decode）
       └─ 需要新的 KV 块但显存不够？→ 触发"抢占"（见 4.4 节）

第2步：如果本步还有余量（Token 预算和请求数都没超限），
       从等待队列中取新请求加入，做 Prefill
       └─ 新请求的输入太长？→ 只处理其中一块（Chunked Prefill，见 4.3 节）

第3步：把本步所有要计算的 Token 拼成一个批次，交给 GPU 执行一次前向计算

第4步：检查结果：
       ├─ 生成了结束符或达到最大长度 → 移出运行队列，释放 KV 块，返回结果
       └─ 否则 → 留在运行队列，下一步继续
```

### 4.3 Chunked Prefill（分块预填充）

**问题**：3.5 节说过，一个超长输入的 Prefill 会让所有人卡顿。

**做法**：给每一步设一个 **Token 预算**（比如每步最多处理 2048 个 Token）。

- 先把正在 Decode 的请求放进来（每个占 1 个 Token）；
- 剩余预算用来处理新请求的 Prefill；如果新请求的输入有 8000 Token，就切成几块，**分几步完成**。

```
Token 预算 = 2048，运行中有 48 个请求在 Decode，新来一个 8000 Token 的 RAG 请求

步骤1: [48 个 Decode Token] + [新请求的第 1    ~ 2000 个 Token 的 Prefill]
步骤2: [48 个 Decode Token] + [新请求的第 2001 ~ 4000 个 Token 的 Prefill]
步骤3: [48 个 Decode Token] + [新请求的第 4001 ~ 6000 个 Token 的 Prefill]
步骤4: [48 个 Decode Token] + [新请求的第 6001 ~ 8000 个 Token 的 Prefill]
步骤5: [49 个 Decode Token]   ← 新请求也开始 Decode 了
```

**效果**：正在生成的用户不会被长请求卡住；同时 Prefill（计算密集）和 Decode（访存密集）混在一起，GPU 的算力和带宽都被更充分地利用。

> 在 RAG 场景中这一点尤其重要，因为 RAG 请求普遍带着很长的检索资料。

### 4.4 抢占（Preemption）

**问题**：所有请求都在不断生成，KV 块越用越多，如果显存满了怎么办？

**做法**：调度器暂停一部分请求（通常是最晚到达的），释放它们的 KV 块，让其他请求继续。被抢占的请求回到等待队列，之后重新调度时，**重新计算**它的 KV Cache（早期版本也支持把 KV Cache 换出到 CPU 内存）。

抢占说明显存已经不够用了，频繁发生时会拖慢整体性能，需要调整参数（见 4.5 节）。

### 4.5 和连续批处理相关的 vLLM 参数

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --max-model-len 32768
```

| 参数 | 含义 | 调大的效果 | 调小的效果 |
|---|---|---|---|
| `--gpu-memory-utilization` | vLLM 可使用的显存比例（权重 + KV Cache） | KV 块更多，能同时跑更多请求 | 给同一张卡上的其他程序留空间 |
| `--max-num-seqs` | 一步中最多同时运行的请求数（批大小上限） | 吞吐量更高 | 单个请求延迟更低、显存压力更小 |
| `--max-num-batched-tokens` | 每一步的 Token 预算（Chunked Prefill 的块大小上限） | Prefill 更快，首 Token 延迟（TTFT）更低 | Decode 更平稳，生成速度（TPOT）更稳定 |
| `--max-model-len` | 单个请求的最大上下文长度 | 支持更长的 RAG 资料 | 节省显存 |

**调参的本质是吞吐量和延迟的权衡**：批越大，总吞吐量越高，但每个用户感受到的速度可能变慢。

---

## 第 5 章 先澄清：Triton 是两个不同的东西

这是初学者最容易混淆的地方。业界有两个都叫 Triton 的项目，**它们没有任何关系**：

| | NVIDIA Triton Inference Server | OpenAI Triton |
|---|---|---|
| **开发者** | NVIDIA | OpenAI |
| **是什么** | 模型**服务器**（部署和对外提供服务） | GPU **编程语言 + 编译器**（编写计算函数） |
| **所在层次** | 服务层（最上面） | 算子层（最底下） |
| **使用者** | 部署/平台工程师 | 写高性能算子的底层工程师 |
| **类比** | 餐厅的前台和调度系统 | 厨房里定制刀具的机床 |
| **和 vLLM 的关系** | 可以把 vLLM 当作后端托管 | vLLM 内部有部分算子是用它写的 |

面试或交流中听到"Triton"，先根据上下文判断：

- 和 **TensorRT、部署、模型服务、Batching** 一起出现 → 通常指 **NVIDIA Triton Inference Server**；
- 和 **Kernel、算子、CUDA、FlashAttention、torch.compile** 一起出现 → 通常指 **OpenAI Triton**。

> 另外，2025 年 NVIDIA 推出了分布式推理框架 **Dynamo**，Triton Inference Server 也被纳入 Dynamo 体系，所以在新的 NVIDIA 资料里可能看到 "Dynamo Triton" 的叫法，指的就是 Triton Inference Server。

---

## 第 6 章 NVIDIA Triton Inference Server：模型服务器

### 6.1 它解决什么问题

一个真实的 AI 系统里往往不只一个模型。以企业 RAG 系统为例：

```
用户问题 → [Embedding 模型] → 向量检索 → [Rerank 模型] → [大语言模型] → 回答
                 ↑                            ↑                ↑
             BERT 类模型                 BERT 类模型          7B LLM
             ONNX 格式                   TensorRT 引擎       vLLM / TensorRT-LLM
```

再加上 OCR 模型、图片分类模型等，可能有来自不同框架（PyTorch、ONNX、TensorFlow）的十几个模型。如果每个模型都单独写一个服务，接口、监控、扩容方式各不相同，运维会非常痛苦。

**Triton Inference Server 的定位：一个统一的服务器，托管各种框架的各种模型，用统一的方式对外提供服务。**

### 6.2 核心功能

| 功能 | 说明 |
|---|---|
| **多框架后端（Backend）** | 支持 TensorRT、ONNX Runtime、PyTorch、TensorFlow、Python 自定义逻辑，以及 vLLM、TensorRT-LLM 后端 |
| **动态批处理（Dynamic Batching）** | 在一个时间窗口内把请求合并成批（第 3.2 节），适合 Embedding、Rerank、分类等模型 |
| **并发执行** | 同一个模型可以开多个实例，多个模型可以在同一张 GPU 上并行运行 |
| **模型编排（Ensemble）** | 把多个模型串成流水线，比如"预处理 → 模型 → 后处理"在服务端一次完成 |
| **统一接口** | HTTP 和 gRPC 协议 |
| **监控指标** | 内置 Prometheus 指标：请求数、延迟、GPU 利用率等 |
| **模型管理** | 模型热加载、版本管理 |

### 6.3 模型仓库的结构

Triton 通过一个固定结构的目录来管理模型：

```
model_repository/
├── bge_embedding/              # 模型名
│   ├── config.pbtxt            # 配置：后端类型、输入输出格式、批处理策略
│   └── 1/                      # 版本号
│       └── model.plan          # TensorRT 引擎文件
└── bge_reranker/
    ├── config.pbtxt
    └── 1/
        └── model.onnx          # ONNX 模型文件
```

`config.pbtxt` 示例（开启动态批处理）：

```
name: "bge_embedding"
backend: "tensorrt"
max_batch_size: 64
dynamic_batching {
  max_queue_delay_microseconds: 2000   # 最多等 2 毫秒凑批
}
instance_group [{ count: 2, kind: KIND_GPU }]   # 开 2 个实例并发处理
```

启动（通常用官方 Docker 镜像）：

```bash
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/model_repository:/models \
    nvcr.io/nvidia/tritonserver:<版本号>-py3 \
    tritonserver --model-repository=/models
```

8000、8001、8002 端口分别对应 HTTP、gRPC 和监控指标。

### 6.4 它和 vLLM / SGLang 的关系

- **vLLM / SGLang 自带了 API 服务**，只部署一个大模型时，直接用 `vllm serve` 更简单；
- 当需要**在同一个平台统一管理大模型和其他各类模型**时，可以用 Triton 作为统一入口，大模型部分用 vLLM 后端或 TensorRT-LLM 后端；
- 注意：大模型的**连续批处理由后端引擎（vLLM / TensorRT-LLM）实现**，Triton 自己的动态批处理是给普通模型用的。

---

## 第 7 章 OpenAI Triton：写 GPU 算子的编程语言

### 7.1 先理解什么是"算子（Kernel）"

模型的每一次计算（矩阵乘法、注意力计算、激活函数、归一化……）在 GPU 上都对应一个或几个**算子（Kernel）**，也就是在 GPU 上运行的一个函数。

推理快不快，很大程度取决于这些算子写得好不好。比如著名的 **FlashAttention** 就是一个高度优化的注意力算子，通过减少显存读写大幅加速注意力计算。

### 7.2 写算子的传统方式：CUDA

CUDA 是 NVIDIA 提供的 GPU 编程方式，用 C++ 编写。问题是门槛很高：

- 需要手动管理线程、线程块、共享内存、内存合并访问等大量硬件细节；
- 写一个高性能的矩阵乘法算子可能需要资深工程师花几周时间；
- 只能在 NVIDIA GPU 上运行。

### 7.3 OpenAI Triton 的做法

OpenAI Triton 让你**用 Python 语法写算子**，以"数据块（Block）"为单位思考，线程调度、内存访问优化等细节交给编译器自动处理。

一个最简单的例子：向量加法。

```python
# pip install triton（Linux + GPU；安装 PyTorch 时通常已附带）
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                          # 我是第几个程序实例
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) # 我负责处理的数据下标
    mask = offsets < n                                    # 防止越界
    x = tl.load(x_ptr + offsets, mask=mask)               # 一次读入一整块数据
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)         # 一次写回一整块

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)  # 需要启动多少个程序实例
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
    return out

x = torch.rand(100_000, device="cuda")
y = torch.rand(100_000, device="cuda")
print(torch.allclose(add(x, y), x + y))  # True
```

逐行理解：

1. 把 10 万个数切成每块 1024 个，共需要约 98 个"程序实例"并行运行；
2. 每个实例通过 `program_id` 知道自己是第几个，算出自己负责的下标范围；
3. `mask` 处理最后一块不满 1024 个的情况；
4. 用 `tl.load` / `tl.store` 一次读写一整块，具体如何映射到 GPU 线程由编译器决定。

### 7.4 它在大模型推理中的作用

- vLLM、SGLang 内部有不少算子是用 OpenAI Triton 编写的（例如部分注意力、MoE、量化相关算子）；
- PyTorch 的 `torch.compile` 会自动把模型代码编译生成 Triton 算子；
- 除了 NVIDIA GPU，Triton 也支持 AMD GPU 等硬件，一定程度上降低了对 CUDA 的依赖。

**对应用开发者来说**：一般不需要自己写 Triton 算子，知道"它是用来写高性能 GPU 算子的、vLLM 等推理引擎内部在用"就足够了。

---

## 第 8 章 TensorRT 与 TensorRT-LLM：NVIDIA 的推理优化器

### 8.1 TensorRT 是什么

TensorRT 是 NVIDIA 的**深度学习推理优化 SDK**。它的工作方式像一个**编译器**：

```
训练好的模型                    TensorRT 编译（Build）               推理引擎文件
(PyTorch → 导出 ONNX)  ──────►  针对你的 GPU 型号做各种优化  ──────►  .engine / .plan
                                （可能需要几分钟到几十分钟）           加载后高速推理
```

类比：Python 代码直接解释执行比较慢，用编译器编译成针对某种 CPU 的机器码就快很多。TensorRT 对模型做的就是类似的事情。

### 8.2 TensorRT 做了哪些优化

| 优化 | 做法 | 效果 |
|---|---|---|
| **层融合（Layer Fusion）** | 把"卷积 → 加偏置 → 激活函数"这样连续的多个操作合并成一个算子 | 减少显存读写次数和算子启动开销 |
| **精度校准（Precision Calibration）** | 把 FP32 降为 FP16、INT8、FP8，INT8 时用一批样本数据校准量化参数 | 速度更快、显存更少，精度损失可控 |
| **算子自动调优（Kernel Auto-tuning）** | 对每一层尝试多种实现方式，在你的 GPU 上实测，选最快的 | 针对具体硬件达到最优 |
| **显存优化** | 分析张量的生命周期，复用显存 | 减少显存占用 |

**代价**：

- 编译出来的引擎文件**绑定 GPU 型号和 TensorRT 版本**，换了显卡型号通常要重新编译；
- 只能在 NVIDIA GPU 上运行；
- 遇到不支持的算子时需要自己写插件，调试难度较高。

### 8.3 TensorRT 适合什么模型

TensorRT 非常成熟，长期用于计算机视觉、语音、推荐系统等领域。在 RAG 系统里，它很适合加速 **Embedding 模型和 Rerank 模型**（BERT 类结构，输入长度相对固定），配合 Triton Inference Server 的动态批处理，能显著提升检索环节的吞吐量。

但原始的 TensorRT 是为"输入形状比较固定、一次前向就出结果"的模型设计的，对大模型那种"逐 Token 生成、KV Cache 不断增长、请求随时进出"的场景支持不够好，于是有了 TensorRT-LLM。

### 8.4 TensorRT-LLM 是什么

TensorRT-LLM 是 NVIDIA 专为大模型推理打造的开源库，可以理解为 **NVIDIA 官方版的 vLLM / SGLang**，同时融合了 TensorRT 的编译优化能力：

- **In-flight Batching**：即连续批处理（第 3.4 节）；
- **分页 KV Cache**：与 PagedAttention 思想相同；
- **丰富的量化支持**：FP8、FP4、INT4（AWQ）、INT8 等，并针对最新的 NVIDIA GPU 深度优化；
- **多卡、多机并行**：张量并行、流水线并行、专家并行；
- **投机解码、前缀缓存、Chunked Prefill** 等；
- 提供 `trtllm-serve` 命令启动 OpenAI 兼容服务，也可以作为 Triton Inference Server 的后端。

> 早期使用 TensorRT-LLM 需要先把模型转换并编译成引擎，流程繁琐；新版本提供了基于 PyTorch 的使用方式和更简洁的高层 API，上手难度已明显降低。

### 8.5 TensorRT-LLM vs vLLM / SGLang

| 维度 | TensorRT-LLM | vLLM / SGLang |
|---|---|---|
| 开发者 | NVIDIA 官方 | 开源社区（高校发起） |
| 硬件 | 只支持 NVIDIA GPU | NVIDIA、AMD 及多种加速卡 |
| 性能 | 在 NVIDIA 新硬件上通常能达到很高的性能上限 | 非常接近，部分场景更优，社区优化迭代很快 |
| 上手难度 | 较高（虽然新版本已改善） | 较低，一条命令启动 |
| 新模型支持速度 | 相对慢一些 | 通常很快，热门模型发布当天或几天内支持 |
| 适合 | 全 NVIDIA 硬件、有专人调优、追求极致性能 | 绝大多数团队的首选 |

---

## 第 9 章 全景图：这些东西分别在哪一层

把 01、02、03 三篇文档的内容放进同一张图：

```
┌─────────────────────────────────────────────────────────────────────┐
│ 应用层      企业知识库 / 智能客服 / Agent 助手                          │
├─────────────────────────────────────────────────────────────────────┤
│ 编排层      LangChain / LangGraph / LlamaIndex          （01 文档）    │
│             RAG 流程、Agent 循环、工具调用、记忆                        │
├──────────────────────────── HTTP / gRPC ─────────────────────────────┤
│ 服务层      vllm serve / SGLang server / trtllm-serve                 │
│             NVIDIA Triton Inference Server（统一托管多种模型）           │
│             NVIDIA Dynamo（大规模分布式推理调度）                       │
├─────────────────────────────────────────────────────────────────────┤
│ 推理引擎层  vLLM / SGLang / TensorRT-LLM                (02、03 文档)  │
│             连续批处理、PagedAttention、前缀缓存、调度、并行             │
├─────────────────────────────────────────────────────────────────────┤
│ 编译/优化层 TensorRT（模型编译优化）、torch.compile                      │
├─────────────────────────────────────────────────────────────────────┤
│ 算子层      CUDA、cuBLAS、FlashAttention、OpenAI Triton 编写的算子       │
├─────────────────────────────────────────────────────────────────────┤
│ 硬件层      NVIDIA / AMD GPU、昇腾 NPU 等                               │
└─────────────────────────────────────────────────────────────────────┘
```

**从上往下看一个 RAG 请求的旅程：**

1. 用户提问，LangChain 发起检索：调用 Embedding 服务（可能是 Triton Inference Server 托管的 TensorRT 引擎，动态批处理）；
2. 检索到片段后调用 Rerank 服务（同上）；
3. LangChain 拼接 Prompt，通过 OpenAI 兼容 API 发给 vLLM；
4. vLLM 调度器把这个请求放进等待队列，下一步就通过连续批处理加入运行批次；
5. 长长的 RAG 资料通过 Chunked Prefill 分几步完成 Prefill，KV Cache 分配在 PagedAttention 管理的显存块中；
6. 每一步的 GPU 计算调用底层算子（CUDA、FlashAttention、Triton 算子）；
7. 生成结束符后，请求退出批次、释放显存块，回答流式返回给 LangChain，再返回给用户。

**对应用开发者的建议：**

- 必须掌握：编排层（01 文档）；
- 应该理解：服务层和推理引擎层的原理（连续批处理、KV Cache、PagedAttention），能部署和调参；
- 了解即可：编译优化层和算子层，知道它们是做什么的、在哪一层。

---

## 第 10 章 术语表

| 术语 | 解释 |
|---|---|
| vLLM | 开源大模型推理服务引擎，以 PagedAttention 闻名 |
| Static Batching | 静态批处理，凑齐一批后一起开始、全部结束才处理下一批 |
| Dynamic Batching | 动态批处理，按时间窗口凑批，仍以请求为单位，适合非生成类模型 |
| Continuous Batching | 连续批处理，每生成一步就重新调度，请求随时进出 |
| In-flight Batching | TensorRT-LLM 对连续批处理的叫法 |
| Iteration-level Scheduling | 迭代级调度，Orca 论文中对连续批处理思想的叫法 |
| Orca | 2022 年提出迭代级调度的推理系统论文 |
| PagedAttention | 把 KV Cache 切成固定大小的块按需分配，类似操作系统内存分页 |
| Block Table | 块表，记录请求的逻辑 Token 位置到物理显存块的映射 |
| Scheduler | 调度器，每一步决定哪些请求参与计算 |
| Chunked Prefill | 分块预填充，把长输入的 Prefill 拆到多步中，与 Decode 混合执行 |
| Preemption | 抢占，显存不足时暂停部分请求、释放其 KV Cache |
| Token Budget | 每一步最多处理的 Token 数 |
| Padding | 填充，把短序列补齐到相同长度 |
| NVIDIA Triton Inference Server | NVIDIA 的通用模型服务器，托管多框架模型 |
| Backend（Triton） | Triton 中负责实际执行某类模型的插件，如 TensorRT、vLLM 后端 |
| Ensemble（Triton） | Triton 中把多个模型串成流水线的功能 |
| OpenAI Triton | 用 Python 编写 GPU 算子的语言和编译器 |
| Kernel（算子） | 在 GPU 上运行的一个计算函数 |
| CUDA | NVIDIA 的 GPU 编程平台 |
| FlashAttention | 减少显存读写的高性能注意力算子 |
| torch.compile | PyTorch 的编译功能，可自动生成 Triton 算子 |
| TensorRT | NVIDIA 推理优化编译器，做层融合、量化、算子调优 |
| Engine / Plan 文件 | TensorRT 编译生成的、绑定特定 GPU 的推理引擎文件 |
| Layer Fusion | 层融合，把多个连续操作合并为一个算子 |
| TensorRT-LLM | NVIDIA 官方大模型推理库 |
| ONNX | 一种通用的模型交换格式，常作为 PyTorch 到 TensorRT 的中间格式 |
| NVIDIA Dynamo | NVIDIA 推出的大规模分布式推理框架，可调度 vLLM、SGLang、TensorRT-LLM 等引擎 |

---

## 第 11 章 面试高频问题

**Q1：什么是连续批处理？和静态批处理有什么区别？**
答：静态批处理以请求为单位组批，整批全部生成完才处理下一批，短请求要陪跑，新请求要排队。连续批处理以每一步生成为单位调度，请求生成结束就立刻移出并返回结果，空位立刻由等待中的新请求补上，GPU 利用率和吞吐量大幅提升，延迟也更低。

**Q2：连续批处理和动态批处理有什么区别？**
答：动态批处理按时间窗口凑批，但仍以请求为单位，一个批次开始后不能加入新请求，适合 Embedding、分类这类单次前向就出结果的模型。连续批处理是迭代级的，每一步都可以有请求进出，专门解决大模型输出长度不一、结束时间未知的问题。

**Q3：为什么批处理能提升大模型推理吞吐量？**
答：Decode 阶段是访存密集的，每生成一个 Token 都要把全部权重从显存读一遍，计算单元大多闲置。批处理让读一次权重就能同时为多个请求各生成一个 Token，在达到计算瓶颈前，吞吐量几乎随批大小线性提升。

**Q4：PagedAttention 和连续批处理是什么关系？**
答：连续批处理让请求随时进出，KV Cache 长度动态增长且最终长度未知。如果按最大长度预留连续显存，浪费严重，批大小上不去。PagedAttention 把 KV Cache 切成固定大小的块按需分配、用完即还，显存利用率高，从而支撑更大的批，让连续批处理的优势充分发挥。

**Q5：什么是 Chunked Prefill？解决什么问题？**
答：把长输入的 Prefill 切成多块，和正在进行的 Decode 请求混在同一步中执行，每一步受 Token 预算限制。避免一个超长请求的 Prefill 让所有正在生成的请求卡顿，同时让计算密集的 Prefill 和访存密集的 Decode 互补，提高 GPU 利用率。RAG 请求输入普遍较长，这一点很关键。

**Q6：显存不够时 vLLM 怎么处理？**
答：触发抢占，暂停部分请求（通常是后到达的）并释放其 KV Cache，之后重新调度时重新计算。频繁抢占说明显存紧张，可以调整 `--gpu-memory-utilization`、`--max-num-seqs`、`--max-model-len`，或使用量化、增加 GPU。

**Q7：Triton 是什么？**
答：需要区分两个项目。NVIDIA Triton Inference Server 是模型服务器，统一托管 TensorRT、ONNX、PyTorch、vLLM 等多种后端的模型，提供动态批处理、并发执行、模型编排和监控。OpenAI Triton 是用 Python 编写 GPU 算子的语言和编译器，vLLM、SGLang 和 `torch.compile` 内部都在使用。

**Q8：TensorRT 做了哪些优化？有什么代价？**
答：层融合、精度校准量化（FP16/INT8/FP8）、针对具体 GPU 的算子自动调优、显存复用。代价是编译出的引擎绑定 GPU 型号和版本，只支持 NVIDIA 硬件，遇到不支持的算子需要写插件，调试门槛较高。

**Q9：TensorRT-LLM 和 vLLM 怎么选？**
答：全 NVIDIA 硬件、有专人调优、追求极致性能时可选 TensorRT-LLM；需要多硬件支持、快速支持新模型、上手简单时选 vLLM 或 SGLang。两者都提供 OpenAI 兼容 API，上层应用代码不受影响，应该用真实业务请求压测后决定。

**Q10：一个企业 RAG 系统里，这些组件你会怎么组合？**
答：Embedding 和 Rerank 模型用 TensorRT 编译加速，部署在 Triton Inference Server 上并开启动态批处理；大语言模型用 vLLM 或 SGLang 部署，利用连续批处理和前缀缓存应对高并发和长上下文；上层用 LangChain / LlamaIndex 编排检索和 Agent 流程。如果团队希望统一运维，也可以把大模型通过 vLLM 或 TensorRT-LLM 后端一起纳入 Triton 管理。

---

> 下一步建议：如果有 Linux + GPU 环境，用 `vllm serve` 部署一个模型，再写个脚本同时发 1、8、32、64 个并发请求，观察总吞吐量和单请求延迟的变化，你会直观感受到连续批处理的效果。
