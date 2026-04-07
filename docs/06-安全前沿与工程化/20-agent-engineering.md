# Agent 工程化最佳实践

> 本篇专注于 Agent 工程和架构方向的落地细节，是区分"会用框架"和"能搭系统"的关键。

## 1. Agent 可观测性（Observability）

### 1.1 为什么 Agent 的可观测性特别难？

| 传统服务 | Agent 服务 |
|---------|-----------|
| 输入输出确定 | 输入输出非确定 |
| 调用链固定 | 调用链动态（取决于 LLM 决策） |
| 延迟可预测 | 延迟波动大（1s ~ 60s+） |
| 错误明确 | "错误"可能是幻觉、不完整、不相关 |
| 请求独立 | 多步执行，步骤间有因果关系 |

### 1.2 Agent Tracing（全链路追踪）

```
一次 Agent 交互的追踪结构：

Trace (一次完整交互)
 ├── Span: 用户输入处理
 │    └── 输入文本、Token 数
 ├── Span: 记忆检索
 │    ├── 查询向量
 │    ├── 检索结果 (top-k)
 │    └── 延迟: 50ms
 ├── Span: LLM 调用 #1 (规划)
 │    ├── 模型: gpt-4o
 │    ├── 输入 Token: 2,500
 │    ├── 输出 Token: 150
 │    ├── 延迟: 1,200ms
 │    ├── Temperature: 0
 │    └── 输出内容 (推理 + 工具调用决策)
 ├── Span: 工具调用 - search_docs
 │    ├── 参数: {"query": "..."}
 │    ├── 返回结果
 │    └── 延迟: 300ms
 ├── Span: LLM 调用 #2 (生成回答)
 │    ├── 输入 Token: 4,200
 │    ├── 输出 Token: 500
 │    └── 延迟: 2,100ms
 └── Span: 输出过滤 (Guardrails)
      ├── 通过
      └── 延迟: 20ms

总计: 输入 6,700 Token, 输出 650 Token, 延迟 3,670ms, 成本 $0.023
```

### 1.3 可观测性工具链

| 工具 | 类型 | 特点 | 适用场景 |
|------|------|------|---------|
| **LangSmith** | SaaS | LangChain 官方，集成深 | LangChain/LangGraph 项目 |
| **LangFuse** | 开源/SaaS | 开源可自建，集成广 | 通用，注重隐私 |
| **Phoenix (Arize)** | 开源 | 评估+追踪一体 | 需要评估能力 |
| **Helicone** | SaaS | 代理层，无侵入 | 快速接入，成本分析 |
| **OpenLLMetry** | 开源 | 基于 OpenTelemetry | 已有 OTel 基础设施 |

### 1.4 关键监控指标

```
业务指标 (Business Metrics)：
  - 任务成功率 (success_rate)
  - 用户满意度 (CSAT / thumbs up/down)
  - 自动解决率 (vs 转人工)

质量指标 (Quality Metrics)：
  - 幻觉率 (hallucination_rate)
  - 工具调用准确率 (tool_accuracy)
  - 答案相关性 (relevance_score)
  - 输出格式正确率 (format_accuracy)

效率指标 (Efficiency Metrics)：
  - 平均步骤数 (avg_steps)
  - 平均 Token 消耗 (avg_tokens)
  - P50/P95/P99 延迟
  - Token 单位成本

稳定性指标 (Reliability Metrics)：
  - LLM API 错误率
  - 工具调用失败率
  - 超时率
  - 重试率
```

### 1.5 告警策略

```
Critical 告警：
  - 成功率 < 80% (5分钟窗口)
  - 错误率 > 10% (5分钟窗口)

Warning 告警：
  - P95 延迟 > 30s (10分钟窗口)
  - 平均 Token/任务 > 20000 (1小时窗口)
  - 小时成本 > $50 (1小时窗口)

Info 告警：
  - 工具失败率 > 5% (30分钟窗口)
```

## 2. Agent 测试工程

### 2.1 测试金字塔

```
            /  \        E2E 测试（模拟真实用户场景）
           /    \       
          /------\      集成测试（Agent + 工具 + LLM 联调）
         /        \     
        /----------\    单元测试（工具函数、解析器、记忆管理）
       /            \   
      /--------------\  静态检查（Prompt Lint）
```

### 2.2 各层测试策略

**层级1：Prompt Lint（静态检查）**

```python
def lint_system_prompt(prompt: str) -> list:
    issues = []
    if len(prompt) > 10000:
        issues.append("System prompt 过长，可能影响注意力")
    if "json" in prompt.lower() and "```" not in prompt:
        issues.append("要求 JSON 输出但未提供格式示例")
    if not any(word in prompt for word in ["不要", "禁止", "不允许"]):
        issues.append("缺少负面约束指令")
    return issues
```

**层级2：单元测试**

```python
def test_search_tool():
    result = search_docs("测试查询")
    assert isinstance(result, list)
    assert len(result) <= 10
    assert all("content" in r for r in result)

def test_json_parser():
    assert robust_json_parse('{"a": 1}') == {"a": 1}
    assert robust_json_parse('```json\n{"a": 1}\n```') == {"a": 1}
    assert robust_json_parse('无效内容') is None
```

**层级3：集成测试（Mock LLM）**

```python
class MockLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
    def chat(self, messages):
        return next(self.responses)

def test_agent_tool_calling():
    mock_llm = MockLLM([
        {"tool_calls": [{"name": "search", "args": {"query": "test"}}]},
        {"content": "根据搜索结果，答案是..."}
    ])
    agent = Agent(llm=mock_llm, tools=[search_tool])
    result = agent.run("测试问题")
    assert result.success
    assert len(result.steps) == 2
```

**层级4：E2E 测试（真实 LLM）**

```python
test_cases = [
    {
        "input": "查询我上个月的订单",
        "expected_tools": ["query_orders"],
        "expected_keywords": ["订单"],
        "max_steps": 3,
        "max_latency_s": 10,
    },
]

def run_e2e_test(case):
    result = agent.run(case["input"])
    actual_tools = [s.tool for s in result.steps if s.type == "tool_call"]
    assert set(case["expected_tools"]).issubset(set(actual_tools))
    for kw in case["expected_keywords"]:
        assert kw in result.content
    assert len(result.steps) <= case["max_steps"]
```

### 2.3 回归测试流程

```
每次变更（Prompt/模型/工具/流程）后：

git push
  -> Prompt Lint (< 10s)
  -> Unit Tests (< 1min)
  -> Integration Tests with Mock LLM (< 5min)
  -> E2E Core Cases with real LLM (< 30min)
  -> 指标对比 (退化 > 阈值 -> 阻断)
  -> 人工审核
  -> 灰度 10% -> 观察 -> 全量
```

### 2.4 测试数据管理

```
测试数据集分层：
  - Golden Set (50条)      <- 核心场景，人工标注
  - Regression Set (200条)  <- 历史 bug 对应用例
  - Edge Case Set (100条)   <- 边界和异常情况
  - Adversarial Set (50条)  <- 安全攻击用例
  - Random Sample           <- 定期从线上采样更新
```

## 3. Prompt 工程化管理

### 3.1 Prompt 版本管理

```
方案1：Git 管理（推荐）

prompts/
  system_prompt.md
  tool_descriptions.yaml
  few_shot_examples.json
  CHANGELOG.md

每次修改走 PR/MR 流程，需要 Review。

方案2：Prompt Registry（数据库管理）

{
    "prompt_id": "agent_v2.3",
    "content": "...",
    "version": "2.3",
    "author": "zhangsan",
    "metrics": {"success_rate": 0.92},
    "status": "active"
}
```

### 3.2 Prompt A/B 测试

```
流量分配：
  90% -> Prompt v2.2 (当前版本)
  10% -> Prompt v2.3 (测试版本)

对比指标：任务成功率、Token 消耗、用户满意度
统计要求：至少 1000 次请求，p-value < 0.05
核心指标不退化 -> 切换到新版本
```

### 3.3 Prompt 模块化

```python
system_prompt = compose_prompt(
    role=load_prompt("roles/customer_service.md"),
    tools=load_prompt("tools/order_tools.yaml"),
    rules=load_prompt("rules/safety_rules.md"),
    format=load_prompt("formats/json_response.md"),
    examples=load_prompt("examples/order_query.json"),
)
```

## 4. Agent 状态管理与持久化

### 4.1 为什么需要状态持久化？

- **长时间运行**：任务可能执行数分钟到数小时
- **断点恢复**：服务重启后能继续执行
- **Human-in-the-loop**：暂停等待人工审核后继续
- **调试回放**：重现某个状态点的执行过程

### 4.2 状态模型设计

```python
@dataclass
class AgentState:
    session_id: str
    task_id: str
    status: str = "running"  # running | paused | completed | failed
    
    messages: list = field(default_factory=list)
    memory_summary: str = ""
    current_step: int = 0
    plan: list = field(default_factory=list)
    completed_steps: list = field(default_factory=list)
    intermediate_results: dict = field(default_factory=dict)
    
    total_tokens: int = 0
    total_cost: float = 0.0
    
    def checkpoint(self) -> str:
        """序列化为 JSON 用于持久化"""
        return json.dumps(asdict(self), default=str)
    
    @classmethod
    def restore(cls, data: str) -> "AgentState":
        return cls(**json.loads(data))
```

### 4.3 Checkpoint 策略

```
何时创建 Checkpoint：
  - 每次 LLM 调用后
  - 每次工具调用后
  - 进入等待状态时（Human-in-the-loop）
  - 定时（每 30 秒）

存储选择：
  - Redis       <- 高频读写，短期存储
  - PostgreSQL  <- 持久化，支持查询
  - S3/OSS      <- 大量历史状态归档

清理策略：
  - 保留最近 N 个 Checkpoint
  - 任务完成后保留 7 天
  - 失败任务保留 30 天（便于分析）
```

### 4.4 断点恢复流程

```
Agent 崩溃/重启
  -> 从存储加载最近的 Checkpoint
  -> 恢复 AgentState
  -> 检查 pending_tool_calls
      有未完成的工具调用 -> 重新执行
      无 -> 从 current_step 继续
  -> 继续执行 plan 中的后续步骤
```

## 5. Streaming 架构

### 5.1 为什么 Agent 需要 Streaming？

```
非 Streaming：用户发送请求 -> 等待 10 秒 -> 一次性返回
Streaming：
  -> [0.3s] "正在思考..."
  -> [1.0s] "正在搜索相关文档..."
  -> [2.0s] "找到 3 篇相关文档"
  -> [3.0s] 逐字输出回答...
  -> [5.0s] 完成
```

### 5.2 分层 Streaming 设计

```
Layer 1: LLM Token Streaming
  LLM 逐 Token 返回，技术：SSE

Layer 2: Agent Step Streaming
  每个步骤（思考、工具调用、结果）实时推送

Layer 3: 进度状态推送
  任务整体进度、当前阶段

消息格式（SSE）：
  {"type": "step", "step_type": "thinking", "content": "分析中..."}
  {"type": "step", "step_type": "tool_call", "content": "搜索知识库..."}
  {"type": "token", "content": "根据"}
  {"type": "token", "content": "搜索"}
  {"type": "done"}
```

### 5.3 SSE 服务端实现

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

async def agent_stream(user_message: str):
    yield f"data: {json.dumps({'type': 'step', 'content': '分析中...'})}\n\n"
    
    result = await search_docs(user_message)
    yield f"data: {json.dumps({'type': 'step', 'content': f'找到{len(result)}篇文档'})}\n\n"
    
    async for token in llm_stream(messages):
        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
    
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

@app.get("/agent/stream")
async def stream_endpoint(message: str):
    return StreamingResponse(agent_stream(message), media_type="text/event-stream")
```

### 5.4 客户端消费（JavaScript）

```javascript
const es = new EventSource(`/agent/stream?message=${encodeURIComponent(msg)}`);
es.onmessage = (event) => {
    const data = JSON.parse(event.data);
    switch (data.type) {
        case 'step':   showStatus(data.content); break;
        case 'token':  appendToAnswer(data.content); break;
        case 'done':   es.close(); break;
        case 'error':  showError(data.content); es.close(); break;
    }
};
```

## 6. Agent 编排引擎设计

### 6.1 核心抽象

```python
from abc import ABC, abstractmethod

class Node(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def execute(self, state: dict) -> dict:
        pass

class Edge:
    def __init__(self, source: str, target: str, condition=None):
        self.source = source
        self.target = target
        self.condition = condition

class Workflow:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.entry_point = ""
    
    def add_node(self, node: Node):
        self.nodes[node.name] = node
    
    def add_edge(self, source, target, condition=None):
        self.edges.append(Edge(source, target, condition))
    
    async def run(self, initial_state: dict) -> dict:
        state = initial_state.copy()
        current = self.entry_point
        visit_count = {}
        
        while current and current != "__end__":
            visit_count[current] = visit_count.get(current, 0) + 1
            if visit_count[current] > 20:
                raise RuntimeError(f"疑似死循环: {current}")
            
            state = await self.nodes[current].execute(state)
            current = self._next(current, state)
        return state
    
    def _next(self, current, state):
        for edge in self.edges:
            if edge.source != current:
                continue
            if edge.condition is None or edge.condition(state):
                return edge.target
        return None
```

### 6.2 使用示例

```python
workflow = Workflow()
workflow.add_node(LLMNode("think"))
workflow.add_node(ToolNode("act"))
workflow.add_node(LLMNode("respond"))
workflow.set_entry("think")

workflow.add_edge("think", "act", lambda s: s.get("needs_tool"))
workflow.add_edge("think", "respond", lambda s: not s.get("needs_tool"))
workflow.add_edge("act", "think")
workflow.add_edge("respond", "__end__")
```

## 7. 成本工程

### 7.1 成本模型

```python
PRICING = {  # $/1M Token
    "gpt-4o":      {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "qwen-72b":    {"input": 0.8, "output": 0.8},
}

# 单次 Agent 任务成本估算
# 假设 3 步，每步输入 3000 Token，输出 500 Token
# 注意上下文递增：步骤越后输入越多
# gpt-4o: ~$0.02-0.05/任务
# gpt-4o-mini: ~$0.001-0.003/任务
```

### 7.2 语义缓存（Semantic Cache）

```python
class SemanticCache:
    def __init__(self, vector_store, threshold=0.95):
        self.store = vector_store
        self.threshold = threshold
    
    def get(self, query: str):
        results = self.store.search(query, top_k=1)
        if results and results[0].score >= self.threshold:
            return results[0].cached_response
        return None
    
    def set(self, query: str, response: str):
        self.store.add(query, metadata={"response": response})
```

### 7.3 Token 预算管理

```python
class TokenBudget:
    def __init__(self, max_tokens=50000, max_cost=0.5):
        self.max_tokens = max_tokens
        self.max_cost = max_cost
        self.used_tokens = 0
        self.used_cost = 0.0
    
    def can_proceed(self) -> bool:
        return self.used_tokens < self.max_tokens and self.used_cost < self.max_cost
    
    def record(self, tokens, cost):
        self.used_tokens += tokens
        self.used_cost += cost
```

## 8. 生产架构参考

### 8.1 中型 Agent 平台架构

```
负载均衡器
    |
  Agent Service (多副本)
    |
共享基础设施:
  - LLM Gateway (模型路由 + 限流 + 负载均衡 + 降级)
  - Vector DB (Milvus/Weaviate)
  - State Store (Redis)
  - Message Queue (Kafka/RabbitMQ - 异步任务)
  - Relational DB (PostgreSQL - 配置/审计)
  - Observability (LangFuse + Prometheus + Grafana)
```

### 8.2 LLM Gateway 设计

```
LLM Gateway 核心职责：

1. 模型路由
   简单任务 -> gpt-4o-mini / qwen-7b
   复杂任务 -> gpt-4o / claude-3.5

2. 负载均衡
   多个 API Key 轮询
   多个模型提供商之间分流

3. 限流
   全局 RPM/TPM 限制
   租户级别配额

4. 降级
   主模型超时 -> 自动切换备用模型
   所有 API 不可用 -> 返回降级响应

5. 缓存
   语义缓存减少重复调用

6. 统计
   记录每次调用的 Token、延迟、成本
```

### 8.3 异步 Agent 架构

```
适用于长时间运行的 Agent 任务（如数据分析、报告生成）：

用户提交任务
  -> API 返回 task_id（立即响应）
  -> 任务进入消息队列
  -> Worker 消费任务，执行 Agent
  -> 过程中持续写 Checkpoint 和进度
  -> 用户可随时查询进度（轮询或 WebSocket）
  -> 完成后通知用户（WebSocket / 回调 / 邮件）
```

## 9. 面试高频问题

### Q1: 如何设计 Agent 的可观测性系统？
**要点**：基于 Trace-Span 模型追踪每次交互的完整链路（LLM 调用、工具调用、Token/延迟/成本）。使用 LangFuse/LangSmith 等工具。监控四类指标：业务/质量/效率/稳定性。设置分级告警。

### Q2: Agent 系统如何做测试？
**要点**：四层测试金字塔 — Prompt Lint（静态检查）→ 单元测试（工具/解析器）→ 集成测试（Mock LLM + Agent 流程）→ E2E（真实 LLM）。分层测试数据集。每次变更做回归测试。

### Q3: 如何管理 Agent 的 Prompt？
**要点**：Git 版本管理 + PR Review + A/B 测试 + 灰度发布。Prompt 模块化（角色/工具/规则/格式分开维护）。建立 Prompt 评估基准，每次变更对比指标。

### Q4: Agent 的状态管理和断点恢复怎么做？
**要点**：定义 AgentState 数据结构，关键节点创建 Checkpoint（LLM 调用后、工具调用后）。存储到 Redis/PG。崩溃后加载最近 Checkpoint 恢复执行。Human-in-the-loop 场景暂停/恢复。

### Q5: 如何设计 Agent 的 Streaming 架构？
**要点**：三层 Streaming — Token 级（LLM 逐字输出）、Step 级（思考/工具调用/结果）、进度级（整体状态）。基于 SSE 实现。前端根据消息类型分别渲染。

### Q6: 如何设计一个 LLM Gateway？
**要点**：五大职责 — 模型路由（按任务复杂度选模型）、负载均衡（多 Key/多厂商）、限流（RPM/TPM）、降级（主备切换）、缓存（语义缓存）。统一记录调用指标。

### Q7: 如何做 Agent 的成本控制？
**要点**：① 模型分级路由 ② 语义缓存 ③ Token 预算上限 ④ 上下文压缩 ⑤ Prompt 精简 ⑥ 提前终止 ⑦ 成本监控和告警。建立成本模型，预估和追踪每任务成本。

### Q8: 设计一个 Agent 编排引擎需要考虑哪些问题？
**要点**：核心抽象（Node/Edge/Workflow）、状态传递、条件分支、循环检测（最大访问次数）、并行执行、错误处理、持久化 Checkpoint、可视化。参考 LangGraph 的设计理念。
