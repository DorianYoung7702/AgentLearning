# LLM 推理优化

> 大厂非常看重工程落地能力，推理优化是让 Agent 在生产环境中可用的关键技术。
> **最后更新：2026 年 4 月**

## 1. 为什么需要推理优化？

| 挑战 | 说明 |
|------|------|
| **延迟高** | 大模型推理慢，Agent 多步调用放大延迟 |
| **显存大** | 7B 模型 FP16 需要 ~14GB 显存 |
| **吞吐低** | 自回归生成串行，并发能力差 |
| **成本高** | GPU 资源昂贵 |

## 2. LLM 推理过程分析

### 2.1 两个阶段

```
阶段1: Prefill（预填充）
  处理整个输入 Prompt → 生成第一个 Token
  特点：计算密集（Compute Bound），可并行处理所有输入 Token

阶段2: Decode（解码）
  逐个生成后续 Token → 直到结束
  特点：内存密集（Memory Bound），每步只生成一个 Token
```

**关键指标**：
| 指标 | 说明 |
|------|------|
| **TTFT（Time To First Token）** | 首个 Token 延迟（Prefill 时间） |
| **TPS（Tokens Per Second）** | 每秒生成 Token 数（Decode 速度） |
| **Throughput** | 吞吐量（每秒处理的请求数） |

### 2.2 显存占用分析

```
模型推理显存构成：

模型参数：  参数量 × 精度字节数
  7B FP16:  7B × 2 bytes = 14GB
  7B INT4:  7B × 0.5 bytes = 3.5GB

KV Cache：  2 × layers × hidden_dim × seq_len × batch × 精度字节数
  长序列时可能超过模型参数本身的显存

激活值：    中间计算的临时显存

总显存 = 模型参数 + KV Cache + 激活值
```

## 3. 模型量化（Quantization）

### 3.1 什么是量化？

将模型权重从高精度（FP16/BF16）转为低精度（INT8/INT4），减少显存和加速推理。

```
FP16:  每个参数 2 字节  → 7B = 14GB
INT8:  每个参数 1 字节  → 7B = 7GB   (节省 50%)
INT4:  每个参数 0.5 字节 → 7B = 3.5GB (节省 75%)
```

### 3.2 量化方法分类

| 方法 | 说明 | 代表 |
|------|------|------|
| **训练后量化（PTQ）** | 训练后直接量化，无需重训 | GPTQ, AWQ, GGUF |
| **量化感知训练（QAT）** | 训练时模拟量化 | — |
| **动态量化** | 推理时动态计算量化参数 | LLM.int8() |

### 3.3 常用量化格式

| 格式 | 特点 | 适用场景 |
|------|------|---------|
| **GPTQ** | 基于逆 Hessian 的精确量化，GPU 推理 | 服务端部署 |
| **AWQ** | 激活感知量化，保护重要权重 | 服务端部署 |
| **GGUF** | llama.cpp 格式，CPU/GPU 混合 | 本地部署、边缘设备 |
| **BitsAndBytes** | HuggingFace 集成，4/8-bit | 开发和微调 |

### 3.4 量化对效果的影响

```
一般规律：
FP16 → INT8:  几乎无损 (< 1% 性能下降)
FP16 → INT4:  轻微损失 (1-3% 性能下降)
FP16 → INT3:  明显损失 (> 5% 性能下降)

注意：小模型对量化更敏感
  70B INT4 ≈ 70B FP16 (大模型量化几乎无损)
  7B INT4 < 7B FP16   (小模型有可感知的损失)
```

## 4. KV Cache 优化

### 4.1 KV Cache 原理

```
自回归生成时，每步都需要之前所有 Token 的 Key 和 Value。
如果不缓存，每生成一个 Token 都要重新计算所有 Token 的 K、V。

有 KV Cache:
  Step 1: 计算所有 Token 的 K1,V1 → 缓存
  Step 2: 只计算新 Token 的 K2,V2 → 追加到缓存 → 用全部 K,V 计算注意力
  Step 3: 只计算新 Token 的 K3,V3 → 追加到缓存 → ...

空间换时间：缓存大但计算快
```

### 4.2 KV Cache 显存问题

```
KV Cache 大小 = 2 × num_layers × hidden_dim × seq_len × batch_size × dtype_size

示例（LLaMA-7B, FP16, seq_len=4096, batch=32）：
  = 2 × 32 × 4096 × 4096 × 32 × 2 bytes
  = ~32 GB  （可能比模型参数还大！）
```

### 4.3 GQA 和 MQA

| 方法 | 全称 | 原理 | KV Cache 节省 |
|------|------|------|-------------|
| **MHA** | Multi-Head Attention | 每个头有独立的 K、V | 0%（基线） |
| **MQA** | Multi-Query Attention | 所有头共享一组 K、V | ~90% |
| **GQA** | Grouped-Query Attention | 每组头共享一组 K、V | ~75% |

```
MHA (标准):    32 个 head → 32 组 KV
GQA (分组):    32 个 head, 8 组 → 8 组 KV (节省 75%)
MQA (单组):    32 个 head → 1 组 KV (节省 97%)
```

- LLaMA-2 70B 使用 GQA
- LLaMA-3 全系列使用 GQA

### 4.4 PagedAttention（vLLM 核心技术）

```
问题：KV Cache 大小动态变化，传统内存分配会造成碎片和浪费

PagedAttention 思想（类比操作系统虚拟内存）：
  - 将 KV Cache 分成固定大小的"页"（Block）
  - 按需分配页，而非预分配整个序列长度
  - 不同请求可共享相同的页（如共享 System Prompt 的 KV Cache）

效果：
  - 显存利用率从 ~60% 提升到 ~90%+
  - 支持更大 batch size → 更高吞吐量
```

## 5. 推理框架（2026 最新对比）

### 5.1 主流框架横向对比

| 框架 | 核心技术 | H100 吞吐量 | 冷启动 | 最适合场景 |
|------|---------|------------|--------|----------|
| **vLLM** | PagedAttention | 2,400 tok/s | ~60s | 快速上线、生态最广 |
| **SGLang** | RadixAttention | 2,460 tok/s | ~60s | 前缀共享多、Agent 场景 |
| **TensorRT-LLM** | TensorRT 编译 | 2,780 tok/s | ~28min | NVIDIA GPU 峰值性能 |
| **LMDeploy** | TurboMind 引擎 | 2,460 tok/s | 中等 | 国内部署、华为 NPU |
| **llama.cpp** | GGUF/CPU/GPU 混合 | 中等 | 极快 | 本地部署、CPU 推理 |
| **Ollama** | 基于 llama.cpp | 中等 | 极快 | 开发调试、本地最简 |
| **TGI** | HuggingFace | 中等 | 快 | HF 生态集成 |

> **2026 结论**：SGLang 和 LMDeploy 在多数场景下比完全优化前的 vLLM 快 29%；TensorRT-LLM 峰值最高但编译冷启动 28 分钟适合固定部署；vLLM 胜在生态和开发友好度。

### 5.2 vLLM

```
核心优势：
  - PagedAttention：KV Cache 分页，显存利用率 ~90%+
  - Continuous Batching：请求完成即释放，吞吐量大幅提升
  - 生态最广：支持几乎所有主流模型
  - 开发友好：60 秒冷启动，API 兼容 OpenAI

适用：高并发在线服务、快速生产部署
```

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", tensor_parallel_size=2)
outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=256))
```

### 5.3 SGLang

```
核心优势：
  - RadixAttention：自动复用共享前缀的 KV Cache
  - 对 Agent 场景特别友好（System Prompt 前缀共享）
  - 吞吐量与 LMDeploy 持平，比 vLLM 快 ~29%

适用：Agent 应用（System Prompt 长）、批量推理
```

### 5.4 TensorRT-LLM (NVIDIA)

```
核心优势：
  - TensorRT 编译优化，算子融合
  - 峰值吞吐量最高（H100 上 2,780 tok/s）
  - FP8 量化支持

缺点：~28 分钟编译冷启动，灵活性差
适用：固定模型的生产部署，追求峰值性能
```

## 6. 批处理优化

### 6.1 Static Batching vs Continuous Batching

```
Static Batching（传统）：
  一个 batch 中所有请求必须一起开始、一起结束
  短请求等待长请求 → 资源浪费

  请求1: [====]
  请求2: [========]       ← 请求1已完成但还要等
  请求3: [============]   ← 所有请求在此时才能释放

Continuous Batching（vLLM 等）：
  请求完成后立即释放资源，新请求可随时加入

  请求1: [====]→ 释放，请求4 加入
  请求2: [========]→ 释放
  请求3: [============]
  请求4:      [======]
```

### 6.2 效果

Continuous Batching 相比 Static Batching 吞吐量提升 **2-10 倍**。

## 7. 推测解码（Speculative Decoding）

### 7.1 原理

```
传统解码：大模型每次生成 1 个 Token（慢）

推测解码：
1. 小模型（Draft Model）快速生成 N 个候选 Token
2. 大模型（Target Model）一次性验证这 N 个 Token
3. 接受匹配的 Token，拒绝不匹配的
4. 从拒绝位置重新开始

效果：一次大模型调用生成多个 Token
加速比：1.5x - 3x（取决于小模型与大模型的一致率）
```

### 7.2 关键条件
- 小模型生成足够快
- 小模型与大模型的输出一致率高
- **保证与大模型独立生成完全相同的分布**（无损加速）

## 8. 其他优化技术

### 8.1 Flash Attention

- 优化注意力计算的 IO 操作
- 通过分块计算减少 GPU HBM（高带宽内存）的读写次数
- 速度提升 2-4 倍，显存减少
- 已成为标配（Flash Attention 2/3）

### 8.2 算子融合（Operator Fusion）

将多个小算子合并为一个大算子，减少内存读写和 kernel 启动开销。

### 8.3 张量并行（Tensor Parallelism）

将模型权重切分到多个 GPU 上并行计算，单次请求延迟降低。

### 8.4 流水线并行（Pipeline Parallelism）

将模型不同层放在不同 GPU 上，适合超大模型。

## 9. 面试高频问题

### Q1: LLM 推理有哪些主要优化方向？
**要点**：① 量化（减少精度降低显存）② KV Cache 优化（GQA/MQA/PagedAttention）③ 批处理优化（Continuous Batching）④ 推测解码（小模型辅助加速）⑤ 注意力优化（Flash Attention）⑥ 并行化（Tensor/Pipeline Parallel）

### Q2: 解释 KV Cache 的原理和显存问题。
**要点**：缓存已计算的 K、V 避免重复计算。显存与 layers×hidden×seq_len×batch 成正比，长序列时可能超过模型参数。优化：GQA 减少 KV 头数，PagedAttention 按需分页分配。

### Q3: 什么是 PagedAttention？为什么 vLLM 性能好？
**要点**：借鉴 OS 虚拟内存思想，将 KV Cache 分页管理，按需分配，消除碎片。配合 Continuous Batching，请求完成即释放资源。显存利用率从 ~60% 提升到 ~90%+。

### Q4: INT8 和 INT4 量化各自适合什么场景？
**要点**：INT8 几乎无损，适合对质量要求高的场景；INT4 有轻微损失但显存节省 75%，适合资源受限场景。大模型（70B+）对量化更鲁棒，小模型更敏感。

### Q5: 解释推测解码的原理。
**要点**：用小模型快速生成 N 个候选 Token，大模型一次验证。匹配的接受，不匹配的拒绝重生成。无损加速 1.5-3x。关键条件：小模型够快且与大模型一致率高。

### Q6: 对比 vLLM、TensorRT-LLM、SGLang。（2026 最新数据）
**要点**：
- **vLLM**：PagedAttention，生态最广，60s 冷启动，H100 吞吐 ~2,400 tok/s，首选快速上线
- **SGLang**：RadixAttention（自动前缀共享），H100 吞吐 ~2,460 tok/s，比 vLLM 快 ~29%，Agent 场景友好
- **TensorRT-LLM**：峰值最高（~2,780 tok/s），但 ~28min 编译冷启动，适合固定模型生产部署
- **LMDeploy**：吞吐与 SGLang 持平，国内部署首选，支持华为 NPU
- 选择原则：追速度用 SGLang/LMDeploy；追极致性能用 TensorRT-LLM；追生态用 vLLM
