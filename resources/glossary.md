# 专业名词术语表（Glossary）

> 按字母顺序排列，覆盖本知识库全部关键术语。点击「出处」可定位到详细讲解。

---

## A

| 术语 | 全称 | 解释 |
|------|------|------|
| **A2A** | Agent-to-Agent | Google 提出的 Agent 间通信协议，解决 Agent 与 Agent 之间的标准化协作 |
| **Adapter** | — | PEFT 方法之一，在 Transformer 层间插入小型适配器网络进行微调 |
| **Agent** | — | 基于 LLM 的自主智能体，具备感知、规划、行动能力，能使用工具完成复杂任务 |
| **Agent Card** | — | A2A 协议中 Agent 的"名片"，描述 Agent 的能力和接口 |
| **ALiBi** | Attention with Linear Biases | 一种位置编码方式，在注意力分数上直接加线性位置偏置，支持长度外推 |
| **ANN** | Approximate Nearest Neighbor | 近似最近邻搜索，牺牲少量精度换取大幅加速的向量检索方法 |
| **AWQ** | Activation-aware Weight Quantization | 激活感知的权重量化方法，保护对激活影响大的重要权重 |

## B

| 术语 | 全称 | 解释 |
|------|------|------|
| **BF16** | BFloat16 | 16 位浮点格式，与 FP32 有相同的指数范围，常用于训练 |
| **BPE** | Byte Pair Encoding | 一种子词分词算法，GPT 系列模型使用，将文本拆分为子词 Token |

## C

| 术语 | 全称 | 解释 |
|------|------|------|
| **Causal Attention** | — | 因果自注意力，只能看到当前及之前的 Token，用于自回归生成（Decoder-Only） |
| **Chain-of-Thought (CoT)** | — | 思维链提示，让 LLM 逐步推理再给出答案，显著提升复杂推理能力 |
| **Checkpoint** | — | 检查点，Agent 执行过程中持久化的状态快照，用于断点恢复 |
| **Chinchilla Scaling Law** | — | DeepMind 提出的最优训练配比法则：训练 Token 数 ≈ 20 × 模型参数量 |
| **Chunk / Chunking** | — | 将长文本切分为较短片段的过程，RAG 中的核心步骤 |
| **Computer Use** | — | Agent 通过截图+操作鼠标键盘来使用计算机的能力（Anthropic 首发） |
| **Continuous Batching** | — | 连续批处理，请求完成即释放资源并加入新请求，吞吐量提升 2-10 倍 |
| **CRAG** | Corrective RAG | 修正型 RAG，对检索结果进行质量评估和修正后再生成 |
| **CrewAI** | — | 基于角色扮演的多 Agent 框架，支持定义角色、任务、工具 |

## D

| 术语 | 全称 | 解释 |
|------|------|------|
| **DAN** | Do Anything Now | 一种典型的 LLM 越狱攻击模式，试图让模型扮演无限制角色 |
| **Decode** | — | LLM 推理的第二阶段，逐个生成后续 Token，内存密集型 |
| **Decoder-Only** | — | 只有解码器的 Transformer 架构，是当前主流 LLM 的标准架构（GPT/LLaMA/Qwen） |
| **DeepSpeed** | — | 微软的分布式训练框架，支持 ZeRO 优化 |
| **Dify** | — | 开源 LLM 应用平台，提供低代码 Agent 构建能力 |
| **DPO** | Direct Preference Optimization | 直接偏好优化，RLHF 的简化替代方案，直接用偏好数据训练 |
| **Draft Model** | — | 推测解码中的小模型，快速生成候选 Token 供大模型验证 |

## E

| 术语 | 全称 | 解释 |
|------|------|------|
| **Embedding** | — | 将文本/图像等转化为固定维度的稠密向量表示，用于语义检索 |
| **Encoder-Decoder** | — | 原始 Transformer 架构，编码器处理输入，解码器生成输出（如 T5） |
| **Evol-Instruct** | — | 渐进式增加指令复杂度的数据增强方法 |

## F

| 术语 | 全称 | 解释 |
|------|------|------|
| **Feature Flag** | — | 功能开关，用于灰度发布和 A/B 测试 |
| **Few-shot** | — | 少样本提示，在 Prompt 中提供几个示例来引导 LLM 输出 |
| **Fine-tuning** | — | 微调，在预训练模型基础上用特定数据进一步训练以适配目标任务 |
| **Flash Attention** | — | 优化注意力计算的 IO 操作，通过分块计算减少 GPU 显存读写，速度提升 2-4 倍 |
| **FP16** | Float16 | 半精度浮点数，每个参数 2 字节 |
| **Function Calling** | — | LLM 根据用户请求生成结构化的工具调用指令（函数名 + 参数），是 Agent 使用工具的基础 |

## G

| 术语 | 全称 | 解释 |
|------|------|------|
| **GGUF** | — | llama.cpp 使用的模型格式，支持 CPU/GPU 混合推理，适合本地部署 |
| **GoT** | Graph of Thoughts | 图思维，将推理过程建模为图结构，允许合并和优化思维路径 |
| **GPTQ** | — | 基于逆 Hessian 的模型量化方法，适合 GPU 推理 |
| **GQA** | Grouped-Query Attention | 分组查询注意力，每组注意力头共享一组 KV，节省约 75% KV Cache |
| **Guardrails** | — | 护栏/防护栏，对 LLM 输入输出进行过滤和验证的安全机制 |
| **GRPO** | Group Relative Policy Optimization | DeepSeek 用于训练推理模型的强化学习算法 |

## H

| 术语 | 全称 | 解释 |
|------|------|------|
| **Hallucination** | 幻觉 | LLM 生成看似合理但事实错误或不忠于上下文的内容 |
| **HNSW** | Hierarchical Navigable Small World | 多层小世界图索引，向量检索中查询最快、精度最高的算法之一 |
| **Human-in-the-Loop** | — | 人机协作，在 Agent 关键节点引入人工判断和审核 |
| **Hybrid Search** | 混合检索 | 同时使用向量检索和关键词检索，结合语义和精确匹配的优势 |

## I

| 术语 | 全称 | 解释 |
|------|------|------|
| **INT4 / INT8** | — | 4 位/8 位整数量化，将模型权重从高精度压缩为低精度以节省显存 |
| **IVF** | Inverted File Index | 倒排文件索引，将向量空间聚类，查询时只搜索最近的簇 |

## J

| 术语 | 全称 | 解释 |
|------|------|------|
| **Jailbreak** | 越狱攻击 | 通过特殊提示绕过 LLM 的安全对齐限制，使其生成有害内容 |
| **JSON-RPC** | — | 基于 JSON 的远程过程调用协议，MCP 协议的通信基础 |
| **JSON Schema** | — | 描述 JSON 数据结构的规范，用于定义工具参数格式 |

## K

| 术语 | 全称 | 解释 |
|------|------|------|
| **KV Cache** | Key-Value Cache | 缓存自回归生成中已计算的 Key 和 Value 矩阵，避免重复计算 |

## L

| 术语 | 全称 | 解释 |
|------|------|------|
| **LangChain** | — | 最流行的 LLM 应用开发框架，提供链式调用和工具集成 |
| **LangFuse** | — | 开源的 LLM 应用可观测性平台，支持 Tracing 和评估 |
| **LangGraph** | — | LangChain 团队的 Agent 编排框架，基于有向图的状态机 |
| **LangSmith** | — | LangChain 官方的追踪和评估 SaaS 平台 |
| **LLM** | Large Language Model | 大语言模型，基于 Transformer 的大规模预训练语言模型 |
| **LLM-as-Judge** | — | 用 LLM 评估另一个 LLM 输出质量的方法 |
| **LLM Gateway** | — | LLM 网关，统一管理模型路由、限流、降级、缓存的中间层 |
| **LlamaGuard** | — | Meta 的安全分类模型，用于检测有害输入输出 |
| **LlamaIndex** | — | 以数据和 RAG 为中心的 LLM 应用框架 |
| **LoRA** | Low-Rank Adaptation | 低秩自适应微调，将权重变化矩阵分解为两个小矩阵，参数量减少 100-1000 倍 |
| **Lost in the Middle** | — | LLM 对长上下文中间位置的信息注意力下降的现象 |

## M

| 术语 | 全称 | 解释 |
|------|------|------|
| **MCP** | Model Context Protocol | Anthropic 提出的标准化 LLM 应用与工具/数据源通信的开放协议 |
| **MCP Host** | — | 运行 MCP Client 的宿主应用（如 Cursor、Claude Desktop） |
| **MCP Server** | — | 通过 MCP 协议暴露工具和数据的服务端 |
| **Mean Pooling** | — | 对所有 Token 输出取均值作为句子 Embedding 的方法 |
| **MetaGPT** | — | 模拟软件公司的多 Agent 框架，Agent 扮演产品经理、架构师等角色 |
| **MHA** | Multi-Head Attention | 多头注意力，每个头有独立的 QKV 权重矩阵 |
| **MoE** | Mixture of Experts | 混合专家模型，每个 Token 只激活部分 Expert，实现大参数低推理成本 |
| **MQA** | Multi-Query Attention | 多查询注意力，所有注意力头共享一组 KV，KV Cache 节省约 97% |
| **MTEB** | Massive Text Embedding Benchmark | 最权威的 Embedding 模型评估基准 |

## N

| 术语 | 全称 | 解释 |
|------|------|------|
| **NF4** | 4-bit NormalFloat | QLoRA 使用的量化格式，专为正态分布的模型权重设计 |
| **NLI** | Natural Language Inference | 自然语言推理，判断两段文本间的蕴含/矛盾/中立关系，可用于幻觉检测 |
| **nprobe** | — | IVF 索引查询时搜索的聚类数量参数，越大越精确但越慢 |

## O

| 术语 | 全称 | 解释 |
|------|------|------|
| **Ollama** | — | 本地运行 LLM 的工具，最简单的本地部署方案 |
| **OpenTelemetry (OTel)** | — | 开源的可观测性框架标准，支持 Traces/Metrics/Logs |

## P

| 术语 | 全称 | 解释 |
|------|------|------|
| **P-Tuning v2** | — | PEFT 方法，在 Transformer 每层学习前缀向量 |
| **PagedAttention** | — | vLLM 核心技术，借鉴 OS 虚拟内存，将 KV Cache 分页管理，显存利用率从 ~60% 提升到 ~90%+ |
| **PEFT** | Parameter Efficient Fine-Tuning | 参数高效微调，只更新少量参数（0.1%-1%），冻结大部分原始权重 |
| **PII** | Personally Identifiable Information | 个人身份信息（姓名、手机号、身份证号等），需脱敏处理 |
| **Pipeline Parallelism** | — | 流水线并行，将模型不同层放在不同 GPU 上 |
| **Plan-and-Execute** | — | 先整体规划再逐步执行的 Agent 范式，规划器和执行器分离 |
| **PQ** | Product Quantization | 乘积量化，将高维向量分段压缩为短码，大幅减少内存占用 |
| **Pre-Norm** | — | 在注意力/FFN 之前做归一化，训练更稳定，现代 LLM 标配 |
| **Prefill** | — | LLM 推理第一阶段，处理完整输入 Prompt 生成第一个 Token，计算密集型 |
| **Prefix Tuning** | — | PEFT 方法，在输入前加可学习的虚拟 Token 前缀 |
| **Prompt Chaining** | — | 将复杂任务拆分为固定多步骤，每步一个 LLM 调用的工作流模式 |
| **Prompt Engineering** | 提示工程 | 设计和优化 LLM 输入提示以获得最佳输出的技术 |
| **Prompt Injection** | 提示注入 | 在输入中插入恶意指令使 LLM 偏离原始任务的攻击方式 |
| **Prompt Leaking** | 提示泄露 | 攻击者诱使 LLM 输出其 System Prompt 内容 |
| **PTQ** | Post-Training Quantization | 训练后量化，训练完成后直接对权重进行量化 |

## Q

| 术语 | 全称 | 解释 |
|------|------|------|
| **QAT** | Quantization-Aware Training | 量化感知训练，在训练过程中模拟量化操作 |
| **QLoRA** | Quantized LoRA | 在 4-bit 量化基座模型上进行 LoRA 微调，7B 模型仅需 ~6GB 显存 |

## R

| 术语 | 全称 | 解释 |
|------|------|------|
| **RAG** | Retrieval-Augmented Generation | 检索增强生成，先从知识库检索相关文档再让 LLM 生成回答 |
| **Rank (r)** | — | LoRA 中低秩矩阵的秩，控制新增参数量和拟合能力（典型值 8-64） |
| **RBAC** | Role-Based Access Control | 基于角色的访问控制 |
| **ReAct** | Reasoning + Acting | 推理与行动交替进行的 Agent 范式，Think → Act → Observe 循环 |
| **Recall@K** | — | Top-K 检索结果中包含正确答案的比例，衡量检索质量的核心指标 |
| **Red Teaming** | 红队测试 | 模拟攻击者对系统进行对抗性测试，发现安全漏洞 |
| **Reflexion** | — | 自我反思机制，Agent 从错误中学习并改进后续行为 |
| **Rerank** | 重排序 | 对初步检索结果用精排模型重新排序，提升检索精度 |
| **ReWOO** | Reasoning Without Observation | 先完成全部规划再执行工具调用，减少 LLM 调用次数 |
| **RLHF** | Reinforcement Learning from Human Feedback | 基于人类反馈的强化学习，让 LLM 输出对齐人类偏好 |
| **RMSNorm** | Root Mean Square Normalization | 均方根归一化，去掉了均值中心化，比 LayerNorm 更快，现代 LLM 标配 |
| **RoPE** | Rotary Position Embedding | 旋转位置编码，通过旋转操作编码相对位置信息，现代 LLM 主流方案 |
| **Routing** | 路由 | 根据输入特征将请求分发到不同处理流程/模型的 Agent 工作流模式 |

## S

| 术语 | 全称 | 解释 |
|------|------|------|
| **Sandwich Defense** | 三明治防御 | 在用户输入前后都加上系统指令，防御 Prompt Injection |
| **Scaling Law** | 缩放定律 | 模型性能与参数量、数据量、计算量之间的幂律关系 |
| **Self-Consistency** | 自一致性 | 多次采样 LLM 输出，通过投票选择最一致的答案 |
| **Self-Instruct** | — | 用 LLM 自动生成指令微调数据的方法 |
| **Self-RAG** | — | 自主决策是否需要检索的 RAG 变体 |
| **Semantic Cache** | 语义缓存 | 基于语义相似度而非完全匹配的缓存，相似查询复用结果 |
| **SFT** | Supervised Fine-Tuning | 有监督微调，用人工标注的指令-回答对训练模型 |
| **SGLang** | — | 高性能 LLM 推理框架，使用 RadixAttention 优化前缀共享 |
| **Span** | — | 可观测性中的基本单元，记录一个操作的类型、输入输出、延迟等 |
| **Speculative Decoding** | 推测解码 | 用小模型快速生成候选 Token，大模型一次验证，无损加速 1.5-3x |
| **SSE** | Server-Sent Events | 服务端向客户端单向推送事件的 HTTP 协议，用于 Streaming |
| **STAR** | Situation-Task-Action-Result | 面试回答框架：情景-任务-行动-结果 |
| **Static Batching** | 静态批处理 | 传统批处理，一批请求必须同时开始和结束，效率低 |
| **stdio** | Standard Input/Output | 标准输入输出，MCP 本地通信方式 |
| **SwiGLU** | — | 现代 LLM 使用的激活函数，比 ReLU/GELU 效果更好 |
| **System Prompt** | 系统提示 | 定义 LLM 角色、规则和行为约束的隐藏指令 |

## T

| 术语 | 全称 | 解释 |
|------|------|------|
| **Temperature** | — | 控制 LLM 输出随机性的参数，0 = 确定性输出，越高越随机 |
| **Tensor Parallelism** | 张量并行 | 将模型权重切分到多个 GPU 并行计算，降低单次推理延迟 |
| **TensorRT-LLM** | — | NVIDIA 的 LLM 推理优化框架，在 NVIDIA GPU 上性能最佳 |
| **TGI** | Text Generation Inference | HuggingFace 的 LLM 推理服务框架 |
| **Token** | — | LLM 处理文本的基本单位（通常 1 个中文字 ≈ 1-2 Token，1 个英文单词 ≈ 1 Token） |
| **Tokenizer** | 分词器 | 将文本转换为 Token 序列的组件 |
| **Tool** | 工具 | Agent 可调用的外部功能（搜索、计算、API 等），扩展 LLM 能力边界 |
| **Top-K** | — | 从概率最高的 K 个 Token 中采样，或检索最相似的 K 个结果 |
| **Top-P (Nucleus)** | — | 从累积概率达到 P 的 Token 集合中采样 |
| **ToT** | Tree of Thoughts | 树形思维，将推理扩展为树结构，探索多条路径后选择最优 |
| **TPS** | Tokens Per Second | 每秒生成的 Token 数，衡量 LLM 推理速度 |
| **Trace** | — | 可观测性中一次完整交互的记录，包含多个 Span |
| **Transformer** | — | 基于自注意力机制的深度学习架构，LLM 的核心基础 |
| **TTFT** | Time To First Token | 首 Token 延迟，从发送请求到返回第一个 Token 的时间 |

## U

| 术语 | 全称 | 解释 |
|------|------|------|
| **Unsloth** | — | 速度最快、显存最省的 LLM 微调工具 |

## V

| 术语 | 全称 | 解释 |
|------|------|------|
| **Vector Database** | 向量数据库 | 专门存储和检索高维向量的数据库（Milvus、Weaviate、Pinecone 等） |
| **VLM** | Vision-Language Model | 视觉语言模型，同时处理图像和文本的多模态模型 |
| **vLLM** | — | 高性能 LLM 推理框架，核心技术 PagedAttention，吞吐量业界领先 |

## W

| 术语 | 全称 | 解释 |
|------|------|------|
| **WebSocket** | — | 全双工通信协议，支持服务端和客户端双向实时通信 |
| **Workflow** | 工作流 | Agent 的执行流程编排，可基于 DAG（有向无环图）或状态机 |

## Y

| 术语 | 全称 | 解释 |
|------|------|------|
| **YaRN** | Yet another RoPE extensioN | 结合 NTK 和注意力缩放的 RoPE 长度扩展方法 |

## Z

| 术语 | 全称 | 解释 |
|------|------|------|
| **Zero-shot** | 零样本 | 不提供任何示例，直接让 LLM 完成任务 |
| **ZeRO** | Zero Redundancy Optimizer | DeepSpeed 的显存优化策略，将优化器状态/梯度/参数分布到多个 GPU |
