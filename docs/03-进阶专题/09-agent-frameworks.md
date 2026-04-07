# 主流 Agent 框架

> 了解主流框架的设计理念和适用场景，面试中经常被问到框架选型问题。
> **最后更新：2026 年 4 月**

## 1. 框架全景图（2026）

```
Agent 框架生态
├── 厂商官方 SDK（2025-2026 新趋势）⭐
│   ├── OpenAI Agents SDK        ← GPT 模型 + 内置工具
│   ├── Anthropic Claude Agents SDK ← 深度 MCP 集成
│   └── Google ADK               ← 集成 A2A + MCP
├── 单 Agent 框架
│   ├── LangChain / LangGraph (126K+ Stars)
│   ├── LlamaIndex
│   └── Semantic Kernel (Microsoft)
├── 多 Agent 框架
│   ├── AutoGen (Microsoft)
│   ├── CrewAI (原生 MCP 支持)
│   └── MetaGPT
├── 轻量级/新兴框架
│   ├── Smolagents (HuggingFace)
│   ├── Agno (原 Phidata)
│   └── PydanticAI
└── 工作流编排
    ├── Dify
    └── Coze

MCP 支持情况：
  ✅ Claude Agents SDK (最深) | ✅ CrewAI | ✅ LangGraph
  ✅ Google ADK | ✅ OpenAI SDK (通过插件)
```

## 2. LangChain / LangGraph

### 2.1 LangChain

**定位**：最流行的 LLM 应用开发框架

**核心抽象**：
| 组件 | 说明 |
|------|------|
| **LLM/ChatModel** | 统一的模型接口 |
| **Prompt Template** | 模板化的提示管理 |
| **Chain** | 组件的链式组合 |
| **Tool** | 工具定义和调用 |
| **Memory** | 对话记忆管理 |
| **Retriever** | 检索器（RAG） |
| **Agent** | ReAct 等 Agent 实现 |

**代码示例**：
```python
from langchain.agents import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# 定义工具
tools = [
    Tool(name="search", func=search_fn, description="搜索信息"),
    Tool(name="calculator", func=calc_fn, description="数学计算"),
]

# 创建 Agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_react_agent(llm, tools, prompt)
```

**优缺点**：
- ✅ 生态最丰富，集成多
- ✅ 文档和社区活跃
- ❌ 抽象层级多，学习曲线陡
- ❌ 过度封装，调试困难

### 2.2 LangGraph

**定位**：基于图的 Agent 工作流编排（LangChain 团队出品）

**核心概念**：
- **State**：全局状态，Agent 的共享数据
- **Node**：处理节点（LLM 调用、工具执行等）
- **Edge**：节点之间的转移条件
- **Graph**：由节点和边组成的工作流图

```python
from langgraph.graph import StateGraph

# 定义状态
class AgentState(TypedDict):
    messages: list
    next_action: str

# 构建图
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge("agent", "tools")
graph.add_conditional_edges("tools", should_continue)
```

**优势**：
- 可视化工作流
- 支持循环和条件分支
- 内置 Human-in-the-loop
- 支持持久化状态（checkpoint）

## 3. LlamaIndex

**定位**：以数据为中心的 LLM 框架，RAG 能力最强

**核心抽象**：
| 组件 | 说明 |
|------|------|
| **Document** | 文档加载和解析 |
| **Index** | 索引构建（向量、树、关键词等） |
| **Query Engine** | 查询引擎 |
| **Agent** | 基于查询引擎的 Agent |

**适用场景**：
- 企业知识库问答
- 文档分析和摘要
- 结构化数据查询

**vs LangChain**：
| 维度 | LangChain | LlamaIndex |
|------|-----------|------------|
| 侧重点 | 通用 Agent 框架 | 数据/RAG 框架 |
| Agent 能力 | 强 | 中 |
| RAG 能力 | 中 | 强 |
| 数据连接器 | 少 | 多（160+） |

## 4. AutoGen (Microsoft)

**定位**：微软出品的多 Agent 对话框架

**核心特点**：
- Agent 之间通过**对话**协作
- 内置 **Human Proxy**，支持人类参与
- 支持代码执行和自动调试

```python
from autogen import AssistantAgent, UserProxyAgent

# 创建 Agent
assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding"})

# 发起对话
user_proxy.initiate_chat(assistant, message="写一个排序算法")
```

**适用场景**：
- 编程任务（自动写代码+测试）
- 需要多角色讨论的任务
- 需要人机协作的场景

## 5. CrewAI

**定位**：基于角色扮演的多 Agent 协作框架

**核心概念**：
| 概念 | 说明 |
|------|------|
| **Agent** | 有角色、目标、背景的智能体 |
| **Task** | 分配给 Agent 的任务 |
| **Crew** | Agent 团队 |
| **Process** | 执行流程（顺序/层级） |

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="研究员",
    goal="收集最新的AI Agent技术信息",
    backstory="你是一位AI领域的资深研究员",
    tools=[search_tool]
)

writer = Agent(
    role="作者",
    goal="撰写高质量的技术文章",
    backstory="你是一位技术写作专家"
)

# 定义任务
research_task = Task(description="调研RAG最新进展", agent=researcher)
write_task = Task(description="撰写调研报告", agent=writer)

# 组建团队
crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])
result = crew.kickoff()
```

**优缺点**：
- ✅ API 设计直觉，上手快
- ✅ 角色扮演范式自然
- ❌ 底层定制灵活性不如 LangGraph

## 6. MetaGPT

**定位**：模拟软件公司的多 Agent 框架

**核心理念**：用 SOP（标准操作流程）约束多 Agent 协作

**内置角色**：
- ProductManager → Architect → ProjectManager → Engineer → QA

**适用场景**：软件开发自动化

## 7. 其他值得关注的框架

### Smolagents (HuggingFace)
- 轻量级，代码简洁
- 支持代码执行作为 Agent 行动方式（Code Agent）
- 与 HuggingFace 生态深度集成

### Agno (原 Phidata)
- 高性能 Agent 框架
- 原生支持多模态
- 内置记忆和知识管理

### PydanticAI
- 类型安全的 Agent 框架
- Pydantic 团队出品
- 强调结构化输出

### Dify / Coze
- 低代码/无代码 Agent 构建平台
- 可视化工作流编排
- 适合非开发者快速搭建

## 8. 框架选型指南

| 需求场景 | 推荐框架 |
|---------|---------|
| 快速原型，通用 Agent | LangChain |
| 复杂工作流，需要精细控制 | LangGraph |
| RAG 为主的应用 | LlamaIndex |
| 多 Agent 对话协作 | AutoGen |
| 基于角色的多 Agent | CrewAI |
| 软件开发自动化 | MetaGPT |
| 轻量级、代码优先 | Smolagents |
| 企业低代码平台 | Dify |

## 9. 面试高频问题

### Q1: 对比 LangChain 和 LlamaIndex 的适用场景。
**要点**：LangChain 是通用 Agent 框架，适合需要工具调用和复杂流程的场景；LlamaIndex 以数据为中心，RAG 能力更强，适合知识库问答。两者可以结合使用。

### Q2: 什么是 LangGraph？它解决了什么问题？
**要点**：LangGraph 用有向图定义 Agent 工作流，支持循环、条件分支、持久化状态。解决了传统 Chain 无法表达复杂流程（如循环和条件判断）的问题。

### Q3: 如何选择 Agent 框架？
**要点**：根据 ① 任务复杂度 ② 是否需要多 Agent ③ RAG 需求 ④ 定制化程度 ⑤ 团队技术栈 ⑥ 社区和维护状态 来选择。简单场景甚至不需要框架。

### Q4: 使用 Agent 框架有什么缺点？
**要点**：① 引入额外抽象和复杂性 ② 版本更新频繁，API 不稳定 ③ 过度封装导致调试困难 ④ 性能开销 ⑤ 学习成本。简单场景直接调用 API 可能更好。

### Q5: 描述一个你用过的 Agent 框架的项目经验。
**准备思路**：选择一个框架，说清楚 ① 项目背景和目标 ② 为什么选这个框架 ③ 架构设计（Agent 角色、工具、流程）④ 遇到的挑战和解决方案 ⑤ 效果和指标。
