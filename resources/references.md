# 推荐资源

> 论文、博客、开源项目、课程等学习资源汇总。

---

## 一、必读论文

### Agent 核心论文
| 论文 | 年份 | 要点 |
|------|------|------|
| **ReAct: Synergizing Reasoning and Acting in Language Models** | 2022 | ReAct 范式的开创性论文 |
| **Toolformer: Language Models Can Teach Themselves to Use Tools** | 2023 | LLM 自主学习使用工具 |
| **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends** | 2023 | LLM 作为控制器调度多模型 |
| **Generative Agents: Interactive Simulacra of Human Behavior** | 2023 | 斯坦福 AI 小镇，Agent 社会模拟 |
| **The Rise and Potential of Large Language Model Based Agents: A Survey** | 2023 | Agent 综述，体系完整 |
| **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation** | 2023 | 微软多 Agent 对话框架论文 |
| **MetaGPT: Meta Programming for Multi-Agent Collaborative Framework** | 2023 | 软件公司模拟的多 Agent |

### RAG 相关论文
| 论文 | 年份 | 要点 |
|------|------|------|
| **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** | 2020 | RAG 原始论文 |
| **Self-RAG: Learning to Retrieve, Generate, and Critique** | 2023 | 自主决策是否检索 |
| **Corrective Retrieval Augmented Generation (CRAG)** | 2024 | 检索结果质量评估和修正 |
| **Lost in the Middle: How Language Models Use Long Contexts** | 2023 | 长上下文注意力分布研究 |

### 推理与规划论文
| 论文 | 年份 | 要点 |
|------|------|------|
| **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** | 2022 | CoT 经典论文 |
| **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** | 2023 | 树形推理 |
| **Reflexion: Language Agents with Verbal Reinforcement Learning** | 2023 | 自我反思改进 |
| **Plan-and-Solve Prompting** | 2023 | 计划与求解 |

---

## 二、推荐博客和文章

### 入门必读
- **Lilian Weng - LLM Powered Autonomous Agents**
  - https://lilianweng.github.io/posts/2023-06-23-agent/
  - Agent 全面综述，必读

- **Anthropic - Building effective agents**
  - https://www.anthropic.com/research/building-effective-agents
  - Anthropic 官方 Agent 构建指南

- **OpenAI - A Practical Guide to Building Agents**
  - https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
  - OpenAI 官方 Agent 实践指南

### RAG 专题
- **LangChain RAG 教程系列**
  - https://python.langchain.com/docs/tutorials/rag/

- **Pinecone - RAG Guide**
  - https://www.pinecone.io/learn/retrieval-augmented-generation/

### 框架文档
- **LangChain 文档**: https://python.langchain.com/docs/
- **LangGraph 文档**: https://langchain-ai.github.io/langgraph/
- **LlamaIndex 文档**: https://docs.llamaindex.ai/
- **CrewAI 文档**: https://docs.crewai.com/
- **AutoGen 文档**: https://microsoft.github.io/autogen/

---

## 三、开源项目

### Agent 框架
| 项目 | Star | 说明 |
|------|------|------|
| **LangChain** | 95k+ | 最流行的 LLM 应用框架 |
| **LlamaIndex** | 37k+ | 数据/RAG 为中心的框架 |
| **AutoGen** | 35k+ | 微软多 Agent 框架 |
| **CrewAI** | 25k+ | 角色扮演多 Agent |
| **MetaGPT** | 45k+ | 模拟软件公司 |
| **Dify** | 55k+ | 开源 LLM 应用平台 |
| **smolagents** | 15k+ | HuggingFace 轻量 Agent |

### 值得学习的 Agent 项目
| 项目 | 说明 |
|------|------|
| **OpenDevin/OpenHands** | 开源编程 Agent |
| **GPT-Researcher** | 自动研究 Agent |
| **BabyAGI** | 经典任务规划 Agent |
| **AutoGPT** | 早期自主 Agent 先驱 |
| **ChatDev** | 多 Agent 软件开发 |
| **MemGPT/Letta** | 长期记忆管理 |

### RAG 相关
| 项目 | 说明 |
|------|------|
| **RAGFlow** | 开源 RAG 引擎 |
| **Quivr** | 个人知识库 RAG |
| **GraphRAG** | 微软图谱增强 RAG |

---

## 四、视频课程

### 中文课程
- **吴恩达 x LangChain 系列课程**（DeepLearning.AI）
  - Building Systems with the ChatGPT API
  - LangChain for LLM Application Development
  - Functions, Tools and Agents with LangChain

- **面向开发者的 LLM 入门教程**（吴恩达 x OpenAI）

### 英文课程
- **DeepLearning.AI - AI Agents in LangGraph**
- **DeepLearning.AI - Building Agentic RAG with LlamaIndex**
- **DeepLearning.AI - Multi AI Agent Systems with crewAI**

---

## 五、面试准备资源

### 知识体系
1. 本知识库 `docs/` 目录下的 11 篇文档
2. 面试题集 `interview/` 目录下的 3 个题集

### 准备策略
1. **基础扎实**：先通读 docs 01-06（Agent 基础 + LLM + Prompt + 工具 + 记忆 + 规划）
2. **进阶深入**：学习 docs 07-11（RAG + 多Agent + 框架 + 评估 + 部署）
3. **刷题巩固**：先做基础题，再做进阶题，最后练系统设计
4. **项目准备**：至少准备 1-2 个 Agent 相关项目经验，能讲清楚架构、挑战和成果
5. **关注前沿**：定期阅读论文和博客，了解最新进展

### 面试常见考察维度
| 维度 | 权重 | 准备重点 |
|------|------|---------|
| 基础知识 | 30% | LLM 原理、Agent 概念、Prompt |
| 工程能力 | 30% | RAG 构建、框架使用、系统设计 |
| 问题解决 | 20% | 幻觉处理、成本优化、可靠性 |
| 项目经验 | 20% | 实际项目、挑战、指标、成果 |
