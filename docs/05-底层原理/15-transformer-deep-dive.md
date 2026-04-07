# Transformer 深入与模型架构

> 大厂面试对 Transformer 的考察远不止自注意力机制，还包括各种变体和优化技术。

## 1. Transformer 完整架构

### 1.1 Decoder-Only Transformer Block

```
输入 Token Embeddings + 位置编码
        │
        ▼
┌───────────────────┐
│   RMSNorm         │  ← Pre-Norm (LLaMA 风格)
│   ↓               │
│   Causal Self-     │
│   Attention        │  ← 因果自注意力 + RoPE
│   (GQA/MQA)       │
│   ↓               │
│   + Residual       │  ← 残差连接
│   ↓               │
│   RMSNorm         │
│   ↓               │
│   FFN / MLP       │  ← SwiGLU 激活
│   ↓               │
│   + Residual       │  ← 残差连接
└───────────────────┘
        │
        ▼ (重复 N 层)
        │
   RMSNorm → Linear → Softmax → 输出概率
```

### 1.2 与原始 Transformer 的区别

| 组件 | 原始 Transformer | 现代 LLM (LLaMA/Qwen) |
|------|-----------------|----------------------|
| 归一化 | LayerNorm, Post-Norm | RMSNorm, Pre-Norm |
| 位置编码 | 正弦/余弦 | RoPE（旋转位置编码） |
| 激活函数 | ReLU | SwiGLU |
| 注意力 | MHA | GQA（分组查询注意力） |
| 架构 | Encoder-Decoder | Decoder-Only |

## 2. 位置编码详解

### 2.1 RoPE（Rotary Position Embedding）

**核心思想**：通过旋转矩阵将位置信息编码到 Query 和 Key 中。

```
对 Query/Key 向量的每对相邻维度应用旋转：

[q_2i, q_2i+1] → [q_2i·cos(mθ) - q_2i+1·sin(mθ),
                   q_2i·sin(mθ) + q_2i+1·cos(mθ)]

其中 m 是位置索引，θ_i = 10000^(-2i/d)
```

**为什么 RoPE 好？**
- 注意力分数自然包含**相对位置**信息
- 理论上可以扩展到任意长度
- 实现高效（只需旋转操作）

### 2.2 位置编码长度扩展

| 方法 | 原理 | 效果 |
|------|------|------|
| **线性插值（PI）** | 将位置索引缩放到训练范围内 | 简单有效 |
| **NTK-Aware** | 修改 RoPE 的基频 | 保持高频信息 |
| **YaRN** | 结合 NTK 和注意力缩放 | 目前最好 |
| **LongRoPE** | 非均匀位置插值 | 超长上下文 |

### 2.3 ALiBi

```
不加位置编码，而是在注意力分数上加位置偏置：
  Attention = softmax(Q·K^T / √d - m·|i-j|)

其中 m 是每个头的斜率，|i-j| 是位置距离
```
- 优点：天然支持长度外推
- 缺点：效果略逊于 RoPE

## 3. 归一化技术

### 3.1 LayerNorm vs RMSNorm

```python
# LayerNorm
y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

# RMSNorm (去掉了均值中心化和偏置)
y = x / sqrt(mean(x^2) + eps) * gamma
```

**RMSNorm 优势**：
- 计算更简单（去掉均值计算和偏置项）
- 速度快 ~10-15%
- 实际效果与 LayerNorm 相当

### 3.2 Pre-Norm vs Post-Norm

```
Post-Norm (原始):  x → Attention → Add(x) → LayerNorm
Pre-Norm (现代):   x → LayerNorm → Attention → Add(x)
```
- Pre-Norm 训练更稳定，是现代 LLM 的标配

## 4. 激活函数

### 4.1 SwiGLU

```
SwiGLU(x) = Swish(x·W_gate) ⊙ (x·W_up)

其中:
  Swish(x) = x · sigmoid(β·x)
  ⊙ 是逐元素乘法
```

- 比 ReLU/GELU 效果更好
- FFN 有 3 个权重矩阵（gate, up, down）而非 2 个
- LLaMA、Qwen、Mistral 等均使用

## 5. Mixture of Experts（MoE）

### 5.1 原理

```
标准 Transformer: 每个 Token 都经过同一个 FFN

MoE Transformer: 每个 Token 只激活部分 Expert（FFN）

┌─────────────────────────────────────┐
│            MoE Layer                 │
│                                      │
│  Input → [Router] → 选择 Top-K Expert │
│              │                        │
│     ┌───┬───┬───┬───┬───┬───┬───┬───┐│
│     │E1 │E2 │E3 │E4 │E5 │E6 │E7 │E8 ││ ← 8 个 Expert
│     └───┘───┘───┘───┘───┘───┘───┘───┘│
│              │ 只激活 Top-2           │
│              ▼                        │
│     加权求和 → Output                 │
└─────────────────────────────────────┘
```

### 5.2 关键参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| num_experts | Expert 总数 | 8, 16, 64, 256 |
| top_k | 每个 Token 激活的 Expert 数 | 1, 2 |
| 总参数量 | 所有 Expert 的参数总和 | 很大 |
| 激活参数量 | 每次推理实际使用的参数 | 总参数的 1/N |

### 5.3 代表模型

| 模型 | 总参数 | 激活参数 | Experts |
|------|--------|---------|---------|
| Mixtral 8×7B | 47B | 12.9B | 8 experts, top-2 |
| DeepSeek-V3 | 671B | 37B | 256 experts, top-8 |
| Qwen2.5-MoE | — | 14B | — |

### 5.4 MoE 的优缺点

**优点**：
- 参数量大（容量大）但推理成本低（只激活部分）
- 相同推理成本下性能更好

**缺点**：
- 显存需要加载所有 Expert（总参数大）
- 训练不稳定（负载均衡问题）
- Expert 间通信开销（分布式场景）

### 5.5 负载均衡

问题：Router 可能总是选择少数几个 Expert，其他 Expert 闲置。

解决：在训练 loss 中加入**负载均衡正则项**，鼓励均匀使用所有 Expert。

## 6. Scaling Laws

### 6.1 Kaplan Scaling Law (OpenAI, 2020)

```
模型性能（Loss）与三个因素的幂律关系：

L(N) ∝ N^(-0.076)    N = 模型参数量
L(D) ∝ D^(-0.095)    D = 训练数据量
L(C) ∝ C^(-0.050)    C = 计算量

含义：增加参数、数据、计算都能改善性能，但收益递减
```

### 6.2 Chinchilla Scaling Law (DeepMind, 2022)

```
最优训练配比：
  训练 Token 数 ≈ 20 × 模型参数量

例：7B 模型应使用 ~140B Token 训练

影响：
  之前模型偏大数据偏少（GPT-3: 175B 参数, 300B Token）
  Chinchilla 后趋向更均衡（LLaMA: 7B 参数, 1T Token）
```

### 6.3 对 Agent 的启示

- 更大的模型通常推理能力更强 → 更好的 Agent
- 但推理成本也更高 → 需要模型分级路由
- 开源小模型 + 微调 可能 > 通用大模型 在特定任务上

## 7. 多模态架构

### 7.1 Vision-Language Model（VLM）

```
图像输入 → [Vision Encoder (ViT)] → 图像 Token
                                        │
文本输入 → [Tokenizer] → 文本 Token ──┐ │
                                       ▼ ▼
                              [LLM Decoder] → 输出
```

**代表模型**：GPT-4o、Claude 3.5、Qwen-VL

### 7.2 对 Agent 的影响

- **Computer Use**：Agent 可以"看到"屏幕截图并操作
- **文档理解**：直接理解 PDF/图片中的内容
- **多模态工具**：图像生成、语音识别等

## 8. 面试高频问题

### Q1: 现代 LLM（如 LLaMA）与原始 Transformer 有哪些架构改进？
**要点**：① Pre-Norm (RMSNorm) 替代 Post-Norm (LayerNorm) ② RoPE 替代正弦位置编码 ③ SwiGLU 替代 ReLU ④ GQA 替代 MHA ⑤ Decoder-Only 架构。

### Q2: 解释 RoPE 的原理和优势。
**要点**：对 Q/K 向量应用旋转操作编码位置，使得注意力分数包含相对位置信息。优势：① 相对位置编码更自然 ② 理论可扩展到任意长度 ③ 计算高效。

### Q3: 什么是 MoE？DeepSeek-V3 为什么用 MoE？
**要点**：MoE 将 FFN 分成多个 Expert，每个 Token 只激活 Top-K 个。DeepSeek-V3 总参数 671B 但激活只有 37B，实现大容量低成本推理。关键挑战：负载均衡和训练稳定性。

### Q4: 解释 Scaling Law 及其实际意义。
**要点**：模型性能与参数量、数据量、计算量呈幂律关系。Chinchilla Law 指出最优配比约 20 Token/参数。实际意义：指导模型训练的资源分配，过大模型+少数据不如适当模型+充足数据。

### Q5: GQA 和 MQA 分别是什么？为什么需要？
**要点**：MQA 所有注意力头共享一组 KV（节省 ~97% KV Cache），GQA 按组共享（节省 ~75%）。GQA 是 MHA 和 MQA 的折中，效果接近 MHA 但显存显著减少。LLaMA-3 全系列使用 GQA。

### Q6: 如何将 LLM 扩展到超长上下文？
**要点**：① 位置编码外推（YaRN/NTK 插值）② Flash Attention 减少内存 ③ 稀疏注意力（只关注部分 Token）④ 上下文压缩 ⑤ 分层检索。挑战："Lost in the Middle"、KV Cache 显存爆炸。
