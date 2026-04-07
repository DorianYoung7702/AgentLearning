# 🤖 AI Agent 面试学习库（大厂版）

> 系统性掌握 AI Agent 全部核心知识，目标：**全部掌握后能通过大厂技术面试**。
> 涵盖 20 篇深度知识文档 + 5 套面试题集（含代码题 + 架构题）+ 资源推荐。

## 📋 目录结构

```
├── README.md                              # 总览与学习路线
│
├── docs/                                  # 知识文档（6 阶段 × 20 篇）
│   │
│   ├── 01-基础概念/                        # 第一阶段（1-2 周）
│   │   ├── 01-agent-fundamentals.md       #   Agent 基础概念与架构
│   │   ├── 02-llm-basics.md              #   大语言模型基础
│   │   └── 03-prompt-engineering.md       #   提示工程
│   │
│   ├── 02-核心能力/                        # 第二阶段（2-3 周）
│   │   ├── 04-tool-use.md                #   工具使用与 Function Calling
│   │   ├── 05-memory-systems.md          #   记忆系统
│   │   └── 06-planning-reasoning.md      #   规划与推理
│   │
│   ├── 03-进阶专题/                        # 第三阶段（2-3 周）
│   │   ├── 07-rag.md                     #   检索增强生成 (RAG)
│   │   ├── 08-multi-agent.md             #   多智能体系统
│   │   └── 09-agent-frameworks.md        #   主流 Agent 框架
│   │
│   ├── 04-工程深入/                        # 第四阶段（2-3 周）⭐ 大厂重点
│   │   ├── 10-evaluation.md              #   Agent 评估与基准测试
│   │   ├── 11-deployment.md              #   生产部署与工程实践
│   │   ├── 12-fine-tuning.md             #   模型微调（LoRA/QLoRA/PEFT）
│   │   └── 13-mcp-protocols.md           #   MCP 协议与 Agent 互操作
│   │
│   ├── 05-底层原理/                        # 第五阶段（1-2 周）⭐ 大厂必考
│   │   ├── 14-inference-optimization.md   #   LLM 推理优化
│   │   ├── 15-transformer-deep-dive.md    #   Transformer 深入与模型架构
│   │   └── 16-vector-db-internals.md      #   向量数据库原理
│   │
│   └── 06-安全前沿与工程化/                  # 第六阶段（1-2 周）
│       ├── 17-agent-safety.md            #   Agent 安全专题
│       ├── 18-agentic-patterns.md        #   Agentic 设计模式
│       ├── 19-cutting-edge.md            #   前沿趋势与热点
│       └── 20-agent-engineering.md        #   Agent 工程化最佳实践 ⭐⭐
│
├── interview/                             # 面试题集（5 套）
│   ├── questions-basic.md                 #   基础面试题（25题）
│   ├── questions-advanced.md              #   进阶面试题（25题）
│   ├── questions-system-design.md         #   系统设计题（5题 + 答题框架）
│   ├── questions-coding.md                #   代码实战题（10题 + 完整代码）⭐
│   └── questions-architecture.md          #   架构方向面试题（15题）⭐⭐
│
└── resources/
    ├── references.md                      # 推荐论文、博客、开源项目
    └── glossary.md                        # 专业名词术语表（150+ 词条）
```

> ⭐ 标记为大厂面试加分/必考内容

## ⚡ 快速学习路线（按时间窗口）

> 根据你距离面试的时间选择对应路线，每条路线均以「能过面试」为目标。

### 🔴 7 天冲刺（已有 LLM 基础，临近面试）

```
Day 1  Agent 核心原语    01-agent-fundamentals + 04-tool-use
Day 2  RAG + 记忆        07-rag + 05-memory-systems
Day 3  框架 + 协议       09-agent-frameworks + 13-mcp-protocols（⭐ 必考）
Day 4  工程实践          20-agent-engineering
Day 5  架构题专项        interview/questions-architecture.md（全刷）
Day 6  编程题专项        interview/questions-coding.md（手写 ReAct + RAG）
Day 7  全题库冲刺        questions-basic + questions-advanced（各选 10 题）
```

### � 2 周速成（有 ML/Python 基础）

```
第 1 周（原理层）
  Day 1-2  01基础概念 全部（Agent + LLM + Prompt）
  Day 3-4  02核心能力 全部（Tool + Memory + Planning）
  Day 5    03进阶专题：07-rag + 08-multi-agent
  Day 6    03进阶专题：09-agent-frameworks（重点框架选型）
  Day 7    复习 + 刷 questions-basic（25 题）

第 2 周（工程层 + 冲刺）
  Day 8    04工程深入：13-mcp-protocols + 20-agent-engineering
  Day 9    05底层原理：14-inference-optimization
  Day 10   06安全前沿：19-cutting-edge（了解 2026 最新格局）
  Day 11   interview/questions-architecture 全刷
  Day 12   interview/questions-coding 手写全部代码题
  Day 13   questions-advanced + questions-system-design
  Day 14   模拟面试 + 查漏补缺
```

### 🟢 4 周系统学习（从零开始）

按下方「完整学习路线图」的 6 个阶段顺序推进，每阶段结束后刷对应面试题。

---

### 🎯 各岗位侧重点

| 岗位方向 | 必看模块 | 重点面试题 |
|---------|---------|----------|
| **Agent 开发工程师** | 01~09 + 13 + 20 | questions-coding + questions-basic |
| **Agent 架构师** | 08 + 09 + 13 + 20 + 11 | questions-architecture + questions-system-design |
| **LLM 推理工程师** | 14 + 15 + 12 + 16 | questions-advanced（推理优化部分） |
| **AI 安全工程师** | 17 + 20 + 13 | questions-advanced（安全部分） |

---

## �🗺️ 完整学习路线图

### 第一阶段：基础概念（1-2 周）
1. **Agent 基础概念与架构** — 什么是 Agent，核心组成，ReAct 范式
2. **大语言模型基础** — Transformer、训练流程、推理参数
3. **提示工程** — System Prompt、Few-shot、CoT、结构化输出

### 第二阶段：核心能力（2-3 周）
4. **工具使用与 Function Calling** — 让 Agent 具备行动能力
5. **记忆系统** — 短期/长期记忆、向量检索、上下文管理
6. **规划与推理** — ReAct、Plan-Execute、ToT、Reflexion

### 第三阶段：进阶专题（2-3 周）
7. **RAG 检索增强生成** — 完整 RAG 管道、Chunk、Rerank、混合检索
8. **多智能体系统** — 架构模式、通信协议、角色设计
9. **主流 Agent 框架** — LangChain / LangGraph / CrewAI / AutoGen

### 第四阶段：工程深入（2-3 周）⭐ 大厂重点
10. **评估与测试** — Agent 质量保障、LLM-as-Judge、基准测试
11. **生产部署** — 可靠性、成本优化、监控告警、发布策略
12. **模型微调** — LoRA/QLoRA/PEFT 原理与实践
13. **MCP 协议** — MCP/A2A 标准化协议（2025 热点）

### 第五阶段：底层原理（1-2 周）⭐ 大厂必考
14. **推理优化** — 量化、KV Cache、vLLM、推测解码
15. **Transformer 深入** — RoPE、MoE、GQA、Flash Attention、Scaling Law
16. **向量数据库原理** — HNSW、IVF、PQ 索引算法

### 第六阶段：安全与前沿（1 周）
17. **Agent 安全** — Prompt Injection、Jailbreak、Guardrails
18. **Agentic 设计模式** — Reflection、Routing、Orchestrator 等模式
19. **前沿趋势** — 推理模型、Computer Use、Agent 生态演进
20. **Agent 工程化实践** — 可观测性、测试工程、状态管理、Streaming、编排引擎

### 第七阶段：面试冲刺（1-2 周）
- 刷基础面试题（25 题）
- 刷进阶面试题（25 题）
- 练系统设计题（5 题）
- **手写代码题**（ReAct Agent、RAG Pipeline 等 10 题）
- **架构面试题**（15 题：可观测性、灰度发布、LLM Gateway 等）
- 准备项目经验讲述（STAR 法则）

## 🎯 大厂面试考点全景

| 考点方向 | 关键词 | 权重 |
|---------|--------|------|
| **Agent 架构** | 感知-规划-行动、ReAct、Agentic Patterns | ⭐⭐⭐⭐⭐ |
| **LLM 原理** | Transformer、RoPE、MoE、GQA、Scaling Law | ⭐⭐⭐⭐⭐ |
| **Prompt 工程** | System Prompt、CoT、结构化输出 | ⭐⭐⭐⭐ |
| **工具调用** | Function Calling、MCP 协议 | ⭐⭐⭐⭐⭐ |
| **RAG** | 向量检索、Chunk、Rerank、混合检索 | ⭐⭐⭐⭐⭐ |
| **记忆系统** | 向量数据库、HNSW、上下文管理 | ⭐⭐⭐⭐ |
| **微调** | LoRA/QLoRA 原理、数据准备、评估 | ⭐⭐⭐⭐⭐ |
| **推理优化** | 量化、KV Cache、vLLM、PagedAttention | ⭐⭐⭐⭐ |
| **多 Agent** | 架构模式、通信、协调 | ⭐⭐⭐ |
| **安全** | Prompt Injection、Guardrails | ⭐⭐⭐⭐ |
| **工程落地** | 可靠性、成本优化、监控、评估体系 | ⭐⭐⭐⭐⭐ |
| **代码能力** | 手写 ReAct Agent、RAG Pipeline | ⭐⭐⭐⭐⭐ |
| **前沿趋势** | 推理模型、Computer Use、MCP/A2A | ⭐⭐⭐ |
| **工程实践** | 可观测性、测试体系、Streaming、状态管理 | ⭐⭐⭐⭐⭐ |

## 📖 使用建议

1. **按顺序学习**：知识点之间存在依赖关系，建议按路线图顺序推进
2. **做笔记**：每个知识点学完后，用自己的话总结核心要点
3. **动手实践**：coding 题必须自己手写一遍，不能只看
4. **刷面试题**：每完成一个阶段后，做对应的面试题检验掌握程度
5. **查漏补缺**：面试题做错的部分，回到对应文档深入学习
6. **模拟面试**：找朋友互相提问，练习口头表达
7. **关注前沿**：面试前一周浏览最新论文和产品动态

## ⏰ 时间规划

| 基础 | 情况 | 建议周期 |
|------|------|---------|
| 有 LLM/NLP 基础 | 主要补 Agent 专项 | 4-6 周 |
| 有 ML 基础，LLM 不熟 | 需要从 LLM 基础开始 | 6-8 周 |
| 转行/零基础 | 需要完整学习 | 8-12 周 |
