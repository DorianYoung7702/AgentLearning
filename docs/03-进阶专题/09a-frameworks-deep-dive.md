# Agent 框架实战深入：LangChain / LangGraph / OpenClaw

> 本文是 `09-agent-frameworks.md` 的实战补充，提供三大框架的**完整代码示例、架构解析和最佳实践**。
> **最后更新：2026 年 4 月**

---

## 1. LangChain 实战

### 1.1 核心架构

```
LangChain 架构分层（v0.3+）：

┌─────────────────────────────────────┐
│           langchain                 │  ← 高层 Agent / Chain 抽象
├─────────────────────────────────────┤
│         langchain-core              │  ← 核心接口：Runnable / Tool / Prompt
├─────────────────────────────────────┤
│   langchain-openai / langchain-anthropic ...│  ← 模型提供商集成
├─────────────────────────────────────┤
│   langchain-community               │  ← 社区工具和集成
└─────────────────────────────────────┘
```

### 1.2 LCEL（LangChain Expression Language）

LangChain v0.3+ 的核心范式是 **LCEL**——用管道符组合组件：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LCEL 管道：Prompt → LLM → Parser
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个AI Agent专家，用中文回答。"),
    ("human", "{question}")
])
llm = ChatOpenAI(model="gpt-4o", temperature=0)
parser = StrOutputParser()

# 用 | 管道符组合（Runnable 接口）
chain = prompt | llm | parser

# 调用
result = chain.invoke({"question": "什么是 ReAct 范式？"})
print(result)
```

### 1.3 Tool 定义与 Function Calling

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 方式 1：@tool 装饰器（推荐）
@tool
def search_web(query: str) -> str:
    """搜索互联网获取最新信息。当用户询问实时信息时使用此工具。"""
    # 实际调用搜索 API
    return f"搜索结果：{query} 的最新信息..."

@tool
def calculate(expression: str) -> str:
    """执行数学计算。输入数学表达式，返回计算结果。"""
    return str(eval(expression))

# 绑定工具到 LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools([search_web, calculate])

# 调用：LLM 自动决定是否使用工具
response = llm_with_tools.invoke("2024年诺贝尔物理学奖得主是谁？")
print(response.tool_calls)  # [{'name': 'search_web', 'args': {'query': '...'}}]
```

### 1.4 ReAct Agent 完整实现

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# 定义工具
@tool
def search(query: str) -> str:
    """搜索最新信息"""
    return f"找到关于 {query} 的信息：..."

@tool
def calculator(expr: str) -> str:
    """数学计算"""
    return str(eval(expr))

# Prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的AI助手，可以使用工具回答问题。"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 创建 Agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_tool_calling_agent(llm, [search, calculator], prompt)

# AgentExecutor 负责执行循环
executor = AgentExecutor(
    agent=agent,
    tools=[search, calculator],
    verbose=True,          # 打印每一步
    max_iterations=5,      # 最大迭代次数
    handle_parsing_errors=True
)

# 运行
result = executor.invoke({"input": "GPT-5 的定价是多少？换算成人民币每百万 Token 多少钱？"})
print(result["output"])
```

### 1.5 RAG Chain 实现

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 文档分割
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents(["你的文档内容..."])

# 2. 向量化存储
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. RAG Prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据以下上下文回答问题。如果无法从上下文找到答案，说不知道。\n\n上下文：{context}"),
    ("human", "{question}")
])

# 4. LCEL 管道
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | ChatOpenAI(model="gpt-4o", temperature=0)
    | StrOutputParser()
)

# 5. 查询
answer = rag_chain.invoke("MCP 协议是什么？")
```

### 1.6 LangChain 最佳实践

```
✅ 推荐：
  - 使用 LCEL 管道而非遗留的 LLMChain
  - 用 @tool 装饰器定义工具（类型安全 + 自动生成 Schema）
  - 用 langchain-core 接口编程，减少对具体实现的依赖
  - 配合 LangSmith 做追踪和评估

❌ 避免：
  - 直接使用 langchain.agents.initialize_agent（已废弃）
  - 过度嵌套 Chain，难以调试
  - 不做错误处理和重试
```

---

## 2. LangGraph 实战

### 2.1 核心概念

```
LangGraph = 有向图 + 状态机 + 持久化

与 LangChain 的关系：
  LangChain = 组件库（LLM、Tool、Prompt 等）
  LangGraph = 编排引擎（用图定义 Agent 工作流）
  → LangGraph 使用 LangChain 的组件，但用图代替 Chain

核心原语：
  State    → 全局共享状态（TypedDict / Pydantic）
  Node     → 处理函数（接收 State，返回更新）
  Edge     → 节点间的转移（固定 / 条件）
  Graph    → 编译后的可执行工作流
  Checkpointer → 状态持久化（中断/恢复/回放）
```

### 2.2 基础 ReAct Agent

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 1. 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 消息自动追加

# 2. 定义工具
@tool
def search(query: str) -> str:
    """搜索最新信息"""
    return f"关于 {query} 的结果..."

# 3. 绑定工具到 LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools([search])

# 4. 定义节点函数
def agent_node(state: AgentState):
    """Agent 决策节点"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: AgentState):
    """工具执行节点"""
    last_message = state["messages"][-1]
    results = []
    for call in last_message.tool_calls:
        result = search.invoke(call["args"])
        results.append({"role": "tool", "content": result, "tool_call_id": call["id"]})
    return {"messages": results}

# 5. 定义路由（条件边）
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"      # 有工具调用 → 去执行工具
    return END              # 无工具调用 → 结束

# 6. 构建图
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")  # 工具执行完回到 agent

# 7. 编译
app = graph.compile()

# 8. 运行
result = app.invoke({"messages": [{"role": "user", "content": "GPT-5 的最新定价？"}]})
print(result["messages"][-1].content)
```

### 2.3 Human-in-the-Loop（中断 & 确认）

```python
from langgraph.checkpoint.memory import MemorySaver

# 编译时加入 Checkpointer
memory = MemorySaver()
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["tools"]   # 在执行工具前中断
)

# 运行到工具节点前会暂停
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke({"messages": [{"role": "user", "content": "帮我发一封邮件"}]}, config)

# 此时 Agent 已决定调用邮件工具，但暂停等人确认
print("Agent 想执行：", result["messages"][-1].tool_calls)
# → 人工确认后继续
result = app.invoke(None, config)  # 传 None 表示继续
```

### 2.4 多 Agent 编排

```python
from langgraph.graph import StateGraph, END

class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_agent: str

# 研究 Agent
def researcher(state):
    response = researcher_llm.invoke(state["messages"])
    return {"messages": [response], "current_agent": "researcher"}

# 写作 Agent
def writer(state):
    response = writer_llm.invoke(state["messages"])
    return {"messages": [response], "current_agent": "writer"}

# 路由
def router(state):
    last = state["messages"][-1].content
    if "需要更多信息" in last:
        return "researcher"
    elif "开始写作" in last:
        return "writer"
    return END

# 构建多 Agent 图
graph = StateGraph(MultiAgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", router)
graph.add_edge("researcher", "supervisor")
graph.add_edge("writer", "supervisor")
```

### 2.5 LangGraph 最佳实践

```
架构原则：
  1. State 设计要精简 — 只放必要数据，大数据用外部存储
  2. Node 函数保持纯粹 — 一个 Node 做一件事
  3. 善用 Conditional Edges — 替代复杂 if/else 嵌套
  4. Checkpointer 必开 — 生产环境用 PostgresSaver
  5. 加入 max_iterations — 防止死循环

常用模式：
  ReAct         → agent ↔ tools 循环
  Supervisor    → supervisor → worker agents 分发
  Plan-Execute  → planner → executor → evaluator 循环
  Map-Reduce    → 并行处理 → 汇总结果
```

---

## 3. OpenClaw 实战

### 3.1 概述

**OpenClaw** 🦞 是 2025 年底发布的开源个人 AI Agent 框架，2026 年初爆火（GitHub 100K+ Stars）。

```
OpenClaw 定位：
  不是开发框架，而是一个 **可直接使用的 Agent 产品/平台**

核心理念：
  AI Agent 不应该只是聊天 → 而是能真正执行任务
  = LLM + 系统级工具访问 + 多渠道接入 + Skills 插件

对比：
  LangChain/LangGraph = 开发者用来构建 Agent 的框架
  OpenClaw             = 可直接使用的个人 AI Agent（同时可扩展）
```

### 3.2 架构

```
OpenClaw 架构：

  ┌──────────────────────────────────────────────────┐
  │  多渠道接入                                         │
  │  WhatsApp / Telegram / Slack / Discord / WeChat   │
  │  iMessage / Teams / Signal / WebChat / ...        │
  └─────────────────────┬────────────────────────────┘
                        │
                        ▼
  ┌──────────────────────────────────────────────────┐
  │  Gateway（控制平面）                                │
  │  - WebSocket 控制中心 (ws://127.0.0.1:18789)      │
  │  - 多 Agent 路由                                   │
  │  - 会话管理 / 事件分发                              │
  └─────────────────────┬────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │  Agent 1  │  │  Agent 2  │  │  Agent N  │
  │  (工作区)  │  │  (工作区)  │  │  (工作区)  │
  └─────┬────┘  └──────────┘  └──────────┘
        │
  ┌─────┼─────────┐
  ▼     ▼         ▼
 Tools  Browser   Nodes
        (CDP)     (macOS/iOS/Android)
```

### 3.3 核心特性

| 特性 | 说明 |
|------|------|
| **Local-first** | 本地运行，数据不出设备 |
| **多渠道接入** | 20+ 消息渠道（WhatsApp/Telegram/Slack/微信等） |
| **多 Agent 路由** | 不同渠道/用户路由到不同 Agent 工作区 |
| **Skills 插件系统** | 100+ 内置技能，可自定义扩展 |
| **浏览器控制** | Chrome CDP 协议控制浏览器 |
| **Voice Mode** | 语音唤醒 + 持续对话 |
| **Nodes** | 跨设备协同（macOS/iOS/Android） |
| **Live Canvas** | Agent 驱动的可视化工作区 |

### 3.4 安装与快速上手

```bash
# 推荐安装方式
npx openclaw@latest onboard

# 或全局安装
npm install -g openclaw
openclaw onboard     # 引导式配置

# 配置模型（支持多模型）
# 支持：Claude / GPT / Gemini / DeepSeek / Ollama（本地）
```

### 3.5 Skills 系统

Skills 是 OpenClaw 的核心扩展机制，类似于 MCP Server 的概念：

```
Skills 类型：
  - Bundled Skills   → 内置（文件操作、Shell、浏览器等）
  - Managed Skills   → 官方维护的扩展（GitHub、Google 等）
  - Workspace Skills → 用户自定义

内置能力（部分）：
  📂 文件读写
  💻 Shell 命令执行
  🌐 浏览器控制（CDP）
  📧 邮件收发（Gmail Hooks）
  📅 日历管理
  🔍 网页搜索
  📷 截屏/录屏
  📍 位置获取（移动端）
```

### 3.6 Agent 间通信（多 Agent）

```
OpenClaw 支持多 Agent 配置：

1. Gateway 配置多个 Agent 工作区
2. 每个工作区有独立的 System Prompt + Skills + 模型
3. 通过 sessions_* tools 实现 Agent 间通信
4. 不同渠道/用户可路由到不同 Agent

示例场景：
  WhatsApp 消息 → 个人助手 Agent
  Slack 消息     → 工作助手 Agent
  Telegram 消息  → 技术问答 Agent
```

### 3.7 安全模型

```
OpenClaw 安全设计：
  - Gateway 绑定 loopback（127.0.0.1）
  - 远程访问通过 Tailscale VPN
  - 每个渠道独立的 DM 访问控制
  - 工具执行沙箱隔离
  - 可审计的操作日志

⚠️ 注意：OpenClaw 拥有系统级权限（文件、Shell、浏览器）
  → 生产使用须严格配置权限范围
  → 不应暴露给不受信的用户
```

### 3.8 OpenClaw vs 其他方案

| 维度 | OpenClaw | LangGraph | ChatGPT |
|------|----------|-----------|---------|
| 定位 | 个人 AI Agent 产品 | Agent 编排框架 | 云端对话助手 |
| 运行位置 | 本地 | 服务端 | 云端 |
| 系统访问 | ✅ 文件/Shell/浏览器 | ❌ 需自行集成 | ❌ 沙箱受限 |
| 多渠道 | ✅ 20+ 消息渠道 | ❌ 需自行开发 | ❌ 仅 Web/App |
| 可编程性 | 中（Skills 扩展） | 高（完全自定义） | 低（GPTs 配置） |
| 适用者 | 个人/极客用户 | 开发者/企业 | 普通用户 |

---

## 4. 三大框架对比总结

### 4.1 选型决策树

```
你要做什么？
│
├─ 构建企业级 Agent 应用（需要代码完全控制）
│  └─ → LangGraph + LangChain
│
├─ 个人自动化（文件/邮件/浏览器/消息）
│  └─ → OpenClaw
│
├─ 快速原型验证
│  └─ → LangChain（LCEL 管道）
│
├─ 多 Agent 角色协作
│  └─ → LangGraph（Supervisor 模式）或 CrewAI
│
└─ 只需要 RAG 问答
   └─ → LangChain RAG Chain 或 LlamaIndex
```

### 4.2 横向对比

| 维度 | LangChain | LangGraph | OpenClaw |
|------|-----------|-----------|----------|
| **定位** | LLM 应用组件库 | Agent 编排引擎 | 个人 AI Agent |
| **语言** | Python / JS | Python / JS | TypeScript |
| **Stars** | 100K+ | 包含在 LangChain | 100K+ |
| **学习曲线** | 中 | 中高 | 低（直接使用） |
| **Agent 循环** | AgentExecutor | Graph + State | 内置 Agent 循环 |
| **工具集成** | @tool 装饰器 | 同 LangChain | Skills 插件 |
| **MCP 支持** | ✅ | ✅ | 通过 Skills |
| **持久化** | 需自行实现 | ✅ Checkpointer | ✅ 内置 |
| **多 Agent** | 需要 LangGraph | ✅ 原生 | ✅ 多 Agent 路由 |
| **可观测性** | LangSmith | LangSmith | Gateway 日志 |
| **生产就绪** | 中 | 高 | 中（偏个人使用） |

---

## 5. 面试高频问题

### Q1: LangChain 的 LCEL 是什么？相比旧 API 有什么优势？
**要点**：LCEL（LangChain Expression Language）用管道符 `|` 组合 Runnable 组件。优势：① 统一的 `invoke/stream/batch` 接口 ② 自动支持流式输出 ③ 支持并行（RunnableParallel）④ 类型安全 ⑤ 替代了旧的 LLMChain/SequentialChain 等。

### Q2: LangGraph 的 State 和 Checkpointer 如何协作？
**要点**：State 是图的全局共享数据，每个 Node 接收 State 并返回更新。Checkpointer 在每个 Node 执行后自动保存 State 快照，支持：① 中断/恢复（Human-in-the-loop）② 回放历史状态 ③ 故障恢复 ④ 分支（从历史状态 fork 新执行）。生产推荐 PostgresSaver。

### Q3: LangGraph 如何实现 Human-in-the-Loop？
**要点**：通过 `interrupt_before` / `interrupt_after` 参数指定中断点。编译时绑定 Checkpointer，Agent 运行到指定节点自动暂停。人工审核后调用 `app.invoke(None, config)` 继续执行。实际应用：敏感操作（发邮件、删文件）前要求人工确认。

### Q4: OpenClaw 的架构特点是什么？和传统 Agent 框架有什么区别？
**要点**：OpenClaw 是**可直接使用的 Agent 产品**而非开发框架。核心架构：Gateway（WebSocket 控制平面）+ 多 Agent 路由 + Skills 插件 + 多渠道接入。与 LangGraph 的区别：LangGraph 面向开发者构建自定义 Agent，OpenClaw 面向终端用户即装即用。

### Q5: 如何在 LangGraph 中实现多 Agent 协作？
**要点**：Supervisor 模式最常用 — 一个 Supervisor Node 根据任务分配给 Worker Agent Nodes，每个 Worker 完成后回到 Supervisor。通过 Conditional Edges 实现路由。State 中记录当前 Agent、对话历史和任务进度。

### Q6: 为什么 OpenClaw 能在 2026 年爆火？对 Agent 行业有什么启示？
**要点**：① Local-first 解决隐私顾虑 ② Skills 插件降低扩展门槛 ③ 多渠道接入（WhatsApp/Telegram 等）覆盖真实场景 ④ 从"能聊天"到"能做事"的范式转变。启示：Agent 从开发者工具走向大众产品，系统级工具访问和多渠道接入成为关键差异化。
