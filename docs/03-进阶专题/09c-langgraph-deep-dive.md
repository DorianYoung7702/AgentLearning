# LangGraph 深度解析

> LangGraph 是 2026 年构建生产级 Agent 的**首选编排引擎**。理解其设计哲学，面试中能讲出比"状态图"更深的东西。
> **最后更新：2026 年 4 月**

---

## 1. LangGraph 的定位与设计哲学

### 1.1 为什么需要 LangGraph？

LangChain 的 AgentExecutor 只能表达一种流程——**线性循环**（Think → Act → Observe → Think ...）。但真实的 Agent 需要更复杂的流程：

| 场景 | AgentExecutor 能做？ | LangGraph 能做？ |
|------|---------------------|-----------------|
| 简单 ReAct 循环 | ✅ | ✅ |
| 条件分支（按任务类型走不同路径） | ❌ | ✅ |
| 并行执行多个子任务 | ❌ | ✅ |
| 中途暂停等人工确认 | ❌ | ✅ |
| 从失败点恢复继续 | ❌ | ✅ |
| 多 Agent 协作（Supervisor 分发） | ❌ | ✅ |
| 回溯到之前的状态重试 | ❌ | ✅ |
| 嵌套子图（Sub-graph） | ❌ | ✅ |

**核心洞察**：Agent 的执行流程本质上是一个**有限状态机（FSM）**——当前在哪个状态、根据条件跳转到下一个状态。LangGraph 用**图（Graph）** 来显式表达这个状态机。

### 1.2 设计哲学

```
LangGraph 的三个核心原则：

1. 显式优于隐式
   → 工作流的每个步骤、每个分支都显式写在图里
   → 不藏逻辑在 Prompt 或黑盒循环里
   → 对比 AgentExecutor：所有逻辑隐藏在内部循环中

2. 状态是一等公民
   → Agent 的所有数据都放在 State 对象中
   → 每个 Node 读取 State、返回 State 更新
   → 可持久化、可回放、可分支

3. 控制流与数据流分离
   → 图的结构（Node + Edge）定义控制流
   → State 承载数据流
   → 两者解耦 → 容易理解、测试和修改
```

### 1.3 LangGraph vs LangChain 的关系

```
LangChain = 零件工厂
  提供 LLM、Tool、Prompt、Retriever 等标准化零件

LangGraph = 装配车间
  用图把零件编排成完整的 Agent 工作流

实际使用时二者配合：
  - 用 LangChain 的 ChatOpenAI 调用模型
  - 用 LangChain 的 @tool 定义工具
  - 用 LangGraph 的 StateGraph 编排流程
  - 用 LangGraph 的 Checkpointer 做持久化
```

---

## 2. 核心原语深入

### 2.1 State（状态）

State 是 LangGraph 的**核心数据结构**，所有 Node 共享同一个 State。

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话消息
    current_plan: str                        # 当前计划
    iteration_count: int                     # 迭代次数
```

**关键概念：Reducer 函数**

```
State 的字段有两种更新方式：

1. 覆盖（默认）
   current_plan: str
   → Node 返回 {"current_plan": "新计划"} → 直接覆盖旧值

2. 追加（通过 Reducer）
   messages: Annotated[list, add_messages]
   → Node 返回 {"messages": [new_msg]} → 追加到已有列表

add_messages 是内置的 Reducer：
  - 新消息追加到列表末尾
  - 如果消息 ID 相同，则替换（用于更新工具结果）
  - 支持 RemoveMessage 删除特定消息
```

**State 设计原则**：

```
✅ 好的 State 设计：
  - 只放 Node 之间需要共享的数据
  - 用类型注解明确每个字段的含义
  - 大数据（文档、图片）放外部存储，State 只存引用

❌ 差的 State 设计：
  - 把所有中间变量都塞进 State
  - 没有 Reducer 导致消息被覆盖而非追加
  - State 过大导致序列化/持久化开销高
```

### 2.2 Node（节点）

Node 是**处理函数**，接收 State 作为输入，返回 State 的部分更新。

```python
def agent_node(state: AgentState) -> dict:
    """Agent 决策节点"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # 只返回需要更新的字段
```

**Node 的设计原则**：

```
1. 单一职责
   一个 Node 做一件事：调用 LLM / 执行工具 / 检索文档 / 路由决策
   不要在一个 Node 里又调 LLM 又执行工具

2. 幂等性
   同样的输入应该产生同样的输出
   便于重试和测试

3. 无副作用（尽量）
   副作用（发邮件、写数据库）集中在专门的 Node
   方便 Human-in-the-Loop 审核
```

**Node 的类型**：

| 类型 | 实现方式 | 适用场景 |
|------|---------|---------|
| **普通函数** | `def node(state) -> dict` | 大多数场景 |
| **异步函数** | `async def node(state) -> dict` | IO 密集场景 |
| **子图（Sub-graph）** | 另一个编译后的 Graph | 复杂流程模块化 |
| **预构建 Node** | `ToolNode`、`create_react_agent` | 快速搭建 |

### 2.3 Edge（边）

Edge 定义 Node 之间的**转移规则**。

```
三种 Edge 类型：

1. 固定边（Normal Edge）
   graph.add_edge("tools", "agent")
   → tools 执行完后一定到 agent

2. 条件边（Conditional Edge）
   graph.add_conditional_edges("agent", route_fn, {"tools": "tools", END: END})
   → 根据 route_fn 的返回值决定下一步

3. 入口边（Entry Point）
   graph.set_entry_point("agent")
   → 图的起始节点
```

**条件边的路由函数**：

```python
def route_fn(state: AgentState) -> str:
    """路由函数：根据 State 决定下一步"""
    last_message = state["messages"][-1]
    
    # 有工具调用 → 去执行工具
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # 迭代次数超限 → 强制结束
    if state.get("iteration_count", 0) > 10:
        return END
    
    # 否则结束
    return END
```

**路由函数的设计要点**：
- 返回值必须是 Node 名称或 `END`
- 在 `add_conditional_edges` 中用字典映射所有可能的返回值
- 路由函数应该是**纯函数**——只读取 State，不修改
- 命名要有语义：`should_continue`、`route_by_task_type`

---

## 3. 图的编译与执行

### 3.1 编译过程

```python
# 构建图
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

# 编译
app = graph.compile(checkpointer=memory)
```

**`compile()` 背后做了什么？**

```
1. 验证图结构
   - 是否所有 Edge 引用的 Node 都存在
   - 是否有不可达的 Node
   - 是否有 Entry Point

2. 构建执行引擎
   - 解析 Node 之间的依赖关系
   - 识别可并行执行的 Node
   - 绑定 Checkpointer（如果有）

3. 返回 CompiledGraph
   - 实现 Runnable 接口（invoke/stream/batch）
   - 可像 LangChain 组件一样使用
```

### 3.2 执行流程（以 ReAct 为例）

```
app.invoke({"messages": [HumanMessage("北京天气")]})

执行过程：

Step 1: 进入 "agent" 节点
  State: {messages: [HumanMessage("北京天气")]}
  → LLM 返回 AIMessage(tool_calls=[{name: "weather", args: {city: "北京"}}])
  → State 更新: {messages: [..., AIMessage(tool_calls=[...])]}
  → Checkpoint 保存 ✓

Step 2: 路由判断 (should_continue)
  → tool_calls 非空 → 返回 "tools"

Step 3: 进入 "tools" 节点
  → 执行 weather("北京") → "晴，25℃"
  → State 更新: {messages: [..., ToolMessage("晴，25℃")]}
  → Checkpoint 保存 ✓

Step 4: 固定边 → 回到 "agent" 节点
  → LLM 返回 AIMessage(content="北京今天晴朗，25℃")
  → State 更新: {messages: [..., AIMessage("北京今天晴朗，25℃")]}
  → Checkpoint 保存 ✓

Step 5: 路由判断 (should_continue)
  → tool_calls 为空 → 返回 END

Step 6: 结束，返回最终 State
```

### 3.3 流式执行

```python
# stream() 返回每一步的增量更新
for event in app.stream({"messages": [HumanMessage("北京天气")]}):
    for node_name, output in event.items():
        print(f"[{node_name}] {output}")

# 输出：
# [agent] {"messages": [AIMessage(tool_calls=[...])]}
# [tools] {"messages": [ToolMessage("晴，25℃")]}
# [agent] {"messages": [AIMessage("北京今天晴朗，25℃")]}
```

**流式的两个层级**：

| 层级 | API | 粒度 |
|------|-----|------|
| **Node 级别** | `app.stream()` | 每个 Node 完成后输出一次 |
| **Token 级别** | `app.astream_events()` | LLM 的每个 Token 实时输出 |

生产环境通常用 `astream_events` 实现打字机效果：

```python
async for event in app.astream_events(input, version="v2"):
    if event["event"] == "on_chat_model_stream":
        token = event["data"]["chunk"].content
        print(token, end="", flush=True)
```

---

## 4. Checkpointer（持久化）

### 4.1 为什么 Checkpointer 是 LangGraph 最重要的特性？

```
没有 Checkpointer 的 Agent = 无状态函数
  - 执行完就丢失所有中间状态
  - 无法中断和恢复
  - 无法多轮对话（每次都从头开始）

有 Checkpointer 的 Agent = 有状态服务
  - 每个 Node 执行后自动保存 State 快照
  - 可以从任意历史状态恢复
  - 天然支持多轮对话
  - 支持 Human-in-the-Loop
```

### 4.2 Checkpointer 的内部机制

```
每次 Node 执行后：

  ┌──────────────────────────────────┐
  │ Checkpoint                        │
  │                                   │
  │ thread_id:  "user-123"           │ ← 对话线程 ID
  │ checkpoint_id: "ckpt-456"        │ ← 快照 ID
  │ parent_id:  "ckpt-455"          │ ← 上一个快照
  │ state: { messages: [...], ... }  │ ← 完整 State
  │ metadata: { step: 3, node: "tools" } │
  │ timestamp: 2026-04-08T13:00:00  │
  └──────────────────────────────────┘
```

**Checkpoint 形成一棵树**：

```
                    ckpt-1 (entry)
                        |
                    ckpt-2 (agent)
                        |
                    ckpt-3 (tools)   ← interrupt_before 暂停
                       / \
           (人工确认继续)  (人工拒绝，修改输入)
                     /       \
                ckpt-4      ckpt-4' (分支)
                    |
                ckpt-5 (END)
```

### 4.3 可用的 Checkpointer 实现

| Checkpointer | 存储 | 适用场景 |
|--------------|------|---------|
| `MemorySaver` | 内存 | 开发调试（重启即丢失） |
| `SqliteSaver` | SQLite | 单机生产 |
| `PostgresSaver` | PostgreSQL | **多实例生产部署（推荐）** |
| `RedisSaver` | Redis | 需要低延迟的场景 |

**生产推荐 PostgresSaver**：

```python
from langgraph.checkpoint.postgres import PostgresSaver

# 连接 PostgreSQL
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/agents"
)

app = graph.compile(checkpointer=checkpointer)
```

### 4.4 thread_id 的含义

```python
config = {"configurable": {"thread_id": "user-123"}}
```

**thread_id 是 LangGraph 隔离对话的关键**：
- 不同 thread_id = 不同的对话上下文
- 同一个 thread_id 的多次 invoke = 多轮对话
- Checkpointer 按 thread_id 存储和检索 State

```
thread_id 的典型映射：
  一个用户的一次会话 = 一个 thread_id
  同一用户的多个会话 = 多个 thread_id

命名规范：
  "user-{user_id}-session-{session_id}"
```

---

## 5. Human-in-the-Loop（HITL）

### 5.1 为什么 Agent 需要 HITL？

```
Agent 有能力执行有副作用的操作：
  - 发送邮件
  - 删除文件
  - 执行数据库写操作
  - 调用付费 API
  - 提交代码

这些操作一旦执行就不可逆 → 需要人工确认
```

### 5.2 三种中断模式

```python
# 模式 1：执行前中断（最常用）
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["tools"]  # 在 tools 节点执行前暂停
)

# 模式 2：执行后中断
app = graph.compile(
    checkpointer=memory,
    interrupt_after=["agent"]   # 在 agent 节点执行后暂停（审核 LLM 的决策）
)

# 模式 3：动态中断（Node 内部决定）
from langgraph.types import interrupt

def tool_node(state):
    for tool_call in state["messages"][-1].tool_calls:
        if tool_call["name"] in DANGEROUS_TOOLS:
            # 只有危险工具才中断
            human_response = interrupt(
                {"question": f"确认执行 {tool_call['name']}?", "tool_call": tool_call}
            )
            if human_response != "approved":
                return {"messages": [ToolMessage("用户拒绝执行", tool_call_id=tool_call["id"])]}
    # 执行工具...
```

### 5.3 HITL 完整流程

```
Step 1: 用户发起请求
  app.invoke({"messages": [HumanMessage("帮我删除 temp 文件夹")]}, config)

Step 2: Agent 决定调用 delete_folder 工具
  → 到达 interrupt_before=["tools"] → 暂停
  → Checkpoint 保存当前 State

Step 3: 前端展示确认界面
  "Agent 要执行 delete_folder(path='/temp')，是否确认？"

Step 4a: 用户确认 → 继续执行
  app.invoke(None, config)  # 传 None = 从上次中断点继续
  → tools 节点执行 delete_folder → 返回结果 → agent 生成回复

Step 4b: 用户拒绝 → 修改后继续
  # 修改 State（移除工具调用，添加提示消息）
  app.update_state(config, {
      "messages": [HumanMessage("不要删除，改为列出文件")]
  })
  app.invoke(None, config)  # 从修改后的 State 继续
```

### 5.4 Time Travel（时间旅行）

Checkpointer 保存了每一步的 State → 可以回溯到任意历史状态：

```python
# 获取所有历史 Checkpoint
history = list(app.get_state_history(config))
# → [StateSnapshot(step=5), StateSnapshot(step=4), ..., StateSnapshot(step=1)]

# 回溯到 Step 2 的状态
old_state = history[3]  # step=2

# 从历史状态 fork 一个新分支
app.invoke(None, {"configurable": {
    "thread_id": config["configurable"]["thread_id"],
    "checkpoint_id": old_state.config["configurable"]["checkpoint_id"]
}})
```

**用途**：
- 调试：Agent 在第 3 步做了错误决策 → 回溯到第 2 步 → 换个 Prompt 重试
- A/B 测试：从同一个 Checkpoint 分支，比较不同策略的结果
- 错误恢复：Agent 执行到一半崩溃 → 从最后一个 Checkpoint 恢复

---

## 6. 常用 Agent 模式

### 6.1 ReAct 模式

```
最基础的模式：Agent ↔ Tools 循环

   ┌──────┐   有 tool_calls   ┌───────┐
   │ agent │ ───────────────→ │ tools │
   └──┬───┘                   └───┬───┘
      │                           │
      │     ← 固定边 ←            │
      │                           │
      └─── 无 tool_calls → END

适用：单 Agent + 工具调用的场景（80% 的需求）
```

LangGraph 提供预构建的 ReAct Agent：

```python
from langgraph.prebuilt import create_react_agent

# 一行代码创建 ReAct Agent
agent = create_react_agent(model=llm, tools=[search, calculator])

# 等价于手动构建 agent → should_continue → tools → agent 的图
result = agent.invoke({"messages": [HumanMessage("...")]})
```

### 6.2 Plan-Execute 模式

```
先规划再执行，适合复杂多步任务：

   ┌──────────┐
   │ planner  │ → 生成任务列表
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │ executor │ → 执行当前任务
   └────┬─────┘
        │
        ▼
   ┌──────────┐    未完成
   │ evaluator│ ──────────→ executor
   └────┬─────┘
        │ 全部完成
        ▼
       END
```

```python
class PlanExecuteState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: list[str]           # 任务列表
    current_task_index: int   # 当前执行到第几个
    results: list[str]        # 每个任务的结果

def planner(state):
    """LLM 生成执行计划"""
    response = llm.invoke([
        SystemMessage("把用户需求分解为多个具体步骤，返回 JSON 列表"),
        *state["messages"]
    ])
    plan = json.loads(response.content)
    return {"plan": plan, "current_task_index": 0}

def executor(state):
    """执行当前任务"""
    task = state["plan"][state["current_task_index"]]
    result = agent_with_tools.invoke({"messages": [HumanMessage(task)]})
    return {
        "results": state.get("results", []) + [result],
        "current_task_index": state["current_task_index"] + 1
    }

def should_continue(state):
    if state["current_task_index"] >= len(state["plan"]):
        return "summarizer"  # 全部完成 → 汇总
    return "executor"        # 继续执行下一个任务
```

### 6.3 Supervisor 模式（多 Agent）

```
一个 Supervisor 分配任务给多个 Worker Agent：

                  ┌────────────┐
            ┌────→│ researcher │────┐
            │     └────────────┘    │
   ┌────────┴──┐                ┌───▼──────┐
   │ supervisor │←───────────────│  (返回)   │
   └────┬──────┘                └──────────┘
        │     ┌──────────┐
        └────→│  writer  │────→ (返回 supervisor)
              └──────────┘

Supervisor 的 System Prompt：
  "你是一个项目经理。根据当前任务状态，决定下一步：
   - 如果需要调研信息，交给 researcher
   - 如果需要撰写内容，交给 writer
   - 如果任务完成，回复 FINISH"
```

```python
def supervisor(state):
    response = llm.invoke([
        SystemMessage(SUPERVISOR_PROMPT),
        *state["messages"]
    ])
    # LLM 返回下一个 Agent 的名称
    return {"next_agent": response.content.strip()}

def route_to_agent(state):
    next_agent = state["next_agent"]
    if next_agent == "FINISH":
        return END
    return next_agent  # "researcher" 或 "writer"

graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", route_to_agent)
graph.add_edge("researcher", "supervisor")
graph.add_edge("writer", "supervisor")
```

### 6.4 Map-Reduce 模式

```
并行处理多个子任务，最后汇总：

   ┌─────────┐
   │ splitter │ → 拆分为 N 个子任务
   └────┬────┘
        │
   ┌────┼────┬────┐
   ▼    ▼    ▼    ▼    （并行执行）
  [T1] [T2] [T3] [T4]
   │    │    │    │
   └────┼────┴────┘
        ▼
   ┌──────────┐
   │ reducer  │ → 汇总所有结果
   └──────────┘

适用：批量文档分析、多角度调研、并行检索
```

```python
from langgraph.constants import Send

def splitter(state):
    """拆分任务，并行分发"""
    tasks = state["tasks"]  # ["分析文档A", "分析文档B", "分析文档C"]
    # Send 实现并行分发
    return [Send("worker", {"task": t, "results": state["results"]}) for t in tasks]

graph.add_conditional_edges("splitter", splitter)
```

### 6.5 模式对比

| 模式 | 复杂度 | 适用场景 | 关键特征 |
|------|--------|---------|---------|
| **ReAct** | 低 | 单 Agent + 工具 | 最简单，80% 场景够用 |
| **Plan-Execute** | 中 | 复杂多步任务 | 先规划后执行，可追踪进度 |
| **Supervisor** | 中高 | 多 Agent 分工 | 中央调度，角色明确 |
| **Map-Reduce** | 中 | 批量/并行 | 分而治之，效率高 |
| **Hierarchical** | 高 | 大型多 Agent 系统 | 多层 Supervisor 嵌套 |

---

## 7. Sub-graph（子图）

### 7.1 为什么需要子图？

复杂 Agent 的图可能有几十个 Node → 难以维护。子图是**模块化**的手段。

```
主图：
   ┌────────────┐      ┌────────────┐
   │  router    │─────→│ RAG 子图    │ ← 封装了检索+生成逻辑
   │            │      └────────────┘
   │            │      ┌────────────┐
   │            │─────→│ Agent 子图  │ ← 封装了工具调用逻辑
   └────────────┘      └────────────┘
```

```python
# 定义 RAG 子图
rag_graph = StateGraph(RAGState)
rag_graph.add_node("retriever", retriever_node)
rag_graph.add_node("generator", generator_node)
rag_graph.add_edge("retriever", "generator")
rag_graph.set_entry_point("retriever")
rag_subgraph = rag_graph.compile()

# 在主图中使用子图
main_graph = StateGraph(MainState)
main_graph.add_node("router", router_node)
main_graph.add_node("rag", rag_subgraph)      # 子图作为 Node
main_graph.add_node("agent", agent_subgraph)   # 另一个子图
```

**子图的 State 映射**：子图可以有自己的 State 类型，LangGraph 会自动映射同名字段。需要手动映射时用 `state_schema` 参数。

---

## 8. 生产部署

### 8.1 LangGraph Platform（官方托管）

```
LangGraph Platform 提供：
  - 一键部署 LangGraph Agent（Docker / Cloud）
  - 内置 API Server（REST + WebSocket）
  - 内置 Cron 任务
  - 内置 LangSmith 集成
  - 水平扩展

部署方式：
  1. LangGraph Cloud（SaaS，最简单）
  2. Self-hosted（Docker，自建）
  3. Standalone（Python Server，最灵活）
```

### 8.2 自建部署架构

```
生产级 LangGraph 部署：

  ┌──────────┐     ┌──────────────────┐
  │  前端     │────→│  API Gateway      │
  └──────────┘     │  (限流/认证/路由)  │
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────┐
                   │  LangGraph Server │ ← 多实例水平扩展
                   │  (Compiled Graph)  │
                   └────────┬─────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
      ┌──────────┐  ┌──────────┐  ┌──────────┐
      │PostgreSQL│  │ Redis    │  │ LangSmith│
      │(Checkpoint)│ │(Cache)   │  │(Tracing) │
      └──────────┘  └──────────┘  └──────────┘
```

### 8.3 性能优化

```
1. Checkpoint 优化
   - 使用 PostgresSaver 而非 MemorySaver
   - 定期清理过期 Checkpoint
   - State 不放大对象（文档内容存外部存储）

2. 并发优化
   - 多个 thread_id 可并行处理
   - ToolNode 支持并行执行多个工具
   - 用异步 Node（async def）提高 IO 并发

3. 成本优化
   - max_iterations 防止死循环烧 Token
   - 简单任务用 mini 模型，复杂任务用旗舰模型
   - 缓存 LLM 响应（相同输入不重复调用）

4. 可靠性
   - Checkpointer 确保故障后可恢复
   - with_retry 对 LLM 调用加重试
   - 超时机制防止工具调用卡死
```

---

## 9. LangGraph 的常见坑

### 9.1 State 更新不生效

```python
# ❌ 错误：直接修改 State
def bad_node(state):
    state["messages"].append(new_msg)  # 不会触发 Reducer！
    return state

# ✅ 正确：返回增量更新
def good_node(state):
    return {"messages": [new_msg]}  # Reducer 会正确追加
```

### 9.2 条件边映射不全

```python
# ❌ 错误：route_fn 返回了映射中不存在的值
graph.add_conditional_edges("agent", route_fn, {"tools": "tools"})
# 如果 route_fn 返回 END → KeyError！

# ✅ 正确：映射所有可能的返回值
graph.add_conditional_edges("agent", route_fn, {"tools": "tools", END: END})
```

### 9.3 忘记设置 Checkpointer 导致 HITL 失败

```python
# ❌ 没有 Checkpointer → interrupt_before 不生效
app = graph.compile(interrupt_before=["tools"])  # 无效！

# ✅ 必须搭配 Checkpointer
app = graph.compile(checkpointer=MemorySaver(), interrupt_before=["tools"])
```

### 9.4 死循环

```python
# ❌ agent → tools → agent → tools → ... 无限循环
# 原因：LLM 每次都返回 tool_calls

# ✅ 解决方案
# 方案 1：State 中加迭代计数
def should_continue(state):
    if state["iteration_count"] > 5:
        return END  # 强制结束
    ...

# 方案 2：create_react_agent 的 recursion_limit 参数
agent = create_react_agent(model, tools, recursion_limit=10)
```

---

## 10. 面试深度问题

### Q1: 解释 LangGraph 的 State + Reducer 机制，为什么这么设计？
**深度回答**：State 是全局共享数据，Node 返回部分更新而非完整 State。Reducer 控制更新方式——`add_messages` 追加消息而非覆盖。设计动机：① **并发安全**——多个 Node 可以同时更新 State 的不同字段 ② **历史追踪**——追加而非覆盖让消息历史完整保留 ③ **Checkpoint 友好**——每次只存增量，减少存储开销。类比 Redux（前端状态管理）：State = Store，Node = Reducer，Edge = Action Dispatch。

### Q2: Checkpointer 是如何实现 Human-in-the-Loop 的？
**深度回答**：① 在 `interrupt_before` 指定的节点前，图执行暂停 ② Checkpointer 将当前 State（包括所有消息和中间变量）持久化存储 ③ 返回控制权给调用方（前端展示确认界面）④ 用户确认后，调用 `invoke(None, config)`——LangGraph 从 Checkpointer 读取最后的 State，从中断点继续执行。本质是**协程**的思想——保存执行上下文，让出控制权，之后恢复。

### Q3: 对比 ReAct、Plan-Execute 和 Supervisor 三种模式，什么时候用哪种？
**深度回答**：
- **ReAct**：单 Agent + 工具，适合明确的问答/查询任务。优点简单，缺点不能拆分复杂任务。
- **Plan-Execute**：单 Agent 先规划后执行，适合多步骤复杂任务（如"帮我写一篇调研报告"）。优点可追踪进度、可中途调整计划，缺点规划本身可能不准确。
- **Supervisor**：多 Agent 分工，适合需要不同专业能力的任务（如研究+写作+审核）。优点角色清晰、可并行，缺点 Supervisor 的路由决策是额外的 LLM 调用。
选择原则：**能用简单模式就不用复杂模式**——ReAct 能解决就不用 Supervisor。

### Q4: LangGraph 的图执行和传统 DAG（如 Airflow）有什么区别？
**深度回答**：① 传统 DAG 是**无环**的（Directed Acyclic Graph），不支持循环；LangGraph 支持**有环图**（agent → tools → agent 循环），这对 Agent 至关重要 ② DAG 通常是**预定义的静态流程**；LangGraph 通过条件边实现**动态路由**（LLM 决定下一步）③ DAG 没有共享状态；LangGraph 的 State 是全局共享的 ④ LangGraph 的 Checkpointer 支持中断/恢复，DAG 通常只支持重试。

### Q5: 如何测试一个 LangGraph Agent？
**深度回答**：分三层测试——① **Node 单元测试**：每个 Node 是纯函数，mock State 输入，验证输出（最容易测试）② **子图集成测试**：编译子图，用固定输入验证输出和路由（mock LLM 返回固定 tool_calls）③ **端到端测试**：用 LangSmith Dataset，创建测试用例（输入 + 期望输出），运行 Agent 并用 LLM-as-Judge 评分。关键：**测 State 转换和路由逻辑，不测 LLM 输出内容**（因为 LLM 不确定性太高）。

### Q6: 生产环境部署 LangGraph 需要注意什么？
**深度回答**：① **Checkpointer 选型**：PostgresSaver（支持多实例共享）而非 MemorySaver ② **并发隔离**：通过 thread_id 隔离不同用户的 State ③ **防死循环**：`recursion_limit` + State 中的迭代计数 ④ **超时控制**：LLM 调用和工具调用都要设超时 ⑤ **可观测性**：集成 LangSmith 或 LangFuse 追踪每次执行 ⑥ **成本控制**：Token 预算 + 模型降级（简单判断用 mini，推理用旗舰）⑦ **安全**：HITL 审核敏感操作、工具执行沙箱。
