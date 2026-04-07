# Agentic Design Patterns

> Andrew Ng 总结的 Agentic AI 设计模式，以及业界实践中的 Agent 工作流模式。

## 1. 四大 Agentic 设计模式（Andrew Ng）

### 1.1 Reflection（反思）

```
Agent 检查并改进自己的输出。

流程：
  生成初始输出
    ↓
  自我审查（或用另一个 LLM 审查）
    ↓
  发现问题 → 修改 → 再审查
    ↓
  满意 → 输出最终结果

示例（代码生成）：
  LLM 写代码 → LLM 审查代码发现 bug → 修复 → 再审查 → 通过
```

**实现方式**：
- 自我反思：同一个 LLM 审查自己的输出
- 交叉反思：用另一个 LLM 或角色审查
- 工具辅助反思：运行测试/linter 提供反馈

### 1.2 Tool Use（工具使用）

```
LLM 判断是否需要工具，并调用工具完成任务。

这是 Agent 最基础的模式，已在 04-tool-use.md 详细讲解。

关键点：
  - 工具描述质量决定调用正确性
  - 支持并行调用独立工具
  - 错误处理和重试机制
```

### 1.3 Planning（规划）

```
Agent 将复杂任务分解为子任务并编排执行。

已在 06-planning-reasoning.md 详细讲解。

关键变体：
  - ReAct：逐步规划 + 执行
  - Plan-and-Execute：先整体规划再执行
  - 动态重规划：根据中间结果调整
```

### 1.4 Multi-Agent（多智能体）

```
多个 Agent 协作完成任务。

已在 08-multi-agent.md 详细讲解。

关键模式：
  - 辩论模式：多 Agent 从不同角度讨论
  - 分工模式：不同 Agent 负责不同子任务
  - 验证模式：一个 Agent 工作，另一个审查
```

## 2. Agent 工作流模式（实践总结）

### 2.1 Prompt Chaining（提示链）

```
将复杂任务拆分为固定的多步骤，每步一个 LLM 调用。

[Step 1: 提取关键信息]
  → output1
[Step 2: 基于 output1 分析]
  → output2
[Step 3: 基于 output2 生成报告]
  → final_output

特点：
  - 流程固定，不涉及动态决策
  - 每步可用不同 Prompt/模型
  - 可靠性高，易于调试
  - 适合结构化、确定性任务
```

**示例：内容审核**
```
输入文本 → [分类器: 识别内容类型] 
         → [情感分析: 判断情绪] 
         → [合规检查: 检测违规]
         → [生成审核报告]
```

### 2.2 Routing（路由）

```
根据输入特征将请求分发到不同的处理流程。

用户输入 → [Router]
              ├── 意图A → 处理流程A
              ├── 意图B → 处理流程B
              └── 意图C → 处理流程C

Router 实现方式：
  - LLM 分类
  - 规则匹配
  - 专用分类模型
  - 语义相似度匹配
```

**示例：客服系统**
```
用户消息 → [意图分类]
             ├── "查询订单" → 订单查询 Agent
             ├── "退货" → 退货处理 Agent
             └── "咨询产品" → 知识库问答 Agent
```

### 2.3 Parallelization（并行化）

```
将独立的子任务并行处理后汇总。

模式A - 分段并行：
  大文档 → 拆分为多段 → 每段并行处理 → 合并结果

模式B - 投票并行：
  同一问题 → 多个 LLM 并行回答 → 投票/合并 → 最终答案

模式C - 多角度并行：
  问题 → 从多个角度并行分析 → 综合分析
```

### 2.4 Orchestrator-Workers（编排-执行）

```
一个编排器动态分配任务给多个执行器。

[Orchestrator]
  1. 分析任务
  2. 分解为子任务
  3. 分配给合适的 Worker
  4. 收集结果
  5. 整合输出

          ┌→ [Worker A] → 结果A ─┐
[Orch] ──┼→ [Worker B] → 结果B ──┼→ [Orch] → 最终结果
          └→ [Worker C] → 结果C ─┘

与 Prompt Chaining 区别：
  Chaining: 步骤固定、预定义
  Orchestrator: 步骤动态、由 LLM 决定
```

### 2.5 Evaluator-Optimizer（评估-优化循环）

```
一个 LLM 生成，另一个 LLM 评估，循环优化直到满意。

[Generator] → 输出 → [Evaluator] → 评分/反馈
     ↑                              │
     └──────── 基于反馈改进 ─────────┘

停止条件：
  - 评分达到阈值
  - 达到最大迭代次数
  - 评估认为"足够好"

示例：
  写文章 → 审查 → "建议加入数据支撑" → 改进 → 审查 → "通过"
```

### 2.6 Human-in-the-Loop（人机协作）

```
关键节点引入人工判断。

自动执行 → [检查点] → 需要人工？
                         ├── 否 → 继续自动
                         └── 是 → 暂停，等待人工
                                    ├── 人工审核 → 继续
                                    ├── 人工修改 → 基于修改继续
                                    └── 人工拒绝 → 终止/重试

触发条件：
  - 置信度低于阈值
  - 高风险操作（删除、支付等）
  - 首次执行新类型任务
  - 用户配置要求
```

## 3. Agent Workflow 编排

### 3.1 状态机模式

```python
# 定义 Agent 为状态机
states = {
    "INIT": 初始状态,
    "PLANNING": 规划中,
    "EXECUTING": 执行中,
    "WAITING_TOOL": 等待工具结果,
    "REFLECTING": 反思中,
    "WAITING_HUMAN": 等待人工,
    "COMPLETED": 完成,
    "FAILED": 失败
}

transitions = {
    "INIT" → "PLANNING",
    "PLANNING" → "EXECUTING",
    "EXECUTING" → "WAITING_TOOL" | "REFLECTING" | "COMPLETED",
    "WAITING_TOOL" → "EXECUTING",
    "REFLECTING" → "PLANNING" | "COMPLETED",
    ...
}
```

### 3.2 DAG（有向无环图）模式

```
LangGraph 等框架使用的核心模式：

  Start → [Node A] → [Condition]
                         ├── True → [Node B] → End
                         └── False → [Node C] → [Node A] (循环)

优点：
  - 可视化
  - 支持并行、分支、循环
  - 可持久化状态
```

### 3.3 事件驱动模式

```
Agent 响应事件而非按序执行：

EventBus:
  on("user_message") → handle_message()
  on("tool_result") → process_result()
  on("timeout") → handle_timeout()
  on("error") → handle_error()

优点：
  - 解耦
  - 灵活
  - 适合异步场景
```

## 4. 模式选择指南

| 任务特征 | 推荐模式 |
|---------|---------|
| 固定流程、多步处理 | Prompt Chaining |
| 分类分发 | Routing |
| 独立子任务 | Parallelization |
| 复杂动态任务 | Orchestrator-Workers |
| 需要迭代优化 | Evaluator-Optimizer |
| 需要质量保证 | Reflection |
| 需要安全审核 | Human-in-the-Loop |
| 以上组合 | 混合模式 |

## 5. 面试高频问题

### Q1: 描述 Andrew Ng 提出的四大 Agentic 设计模式。
**要点**：Reflection（自我审查改进）、Tool Use（工具调用扩展能力）、Planning（任务分解编排）、Multi-Agent（多智能体协作）。这四个模式可以组合使用。

### Q2: Prompt Chaining 和 Orchestrator-Workers 有什么区别？
**要点**：Chaining 步骤固定预定义，执行路径不变；Orchestrator 由 LLM 动态决定任务分解和分配。Chaining 更可靠易调试，Orchestrator 更灵活但更复杂。

### Q3: 什么场景适合 Human-in-the-Loop？如何设计？
**要点**：高风险操作（支付、删除）、低置信度决策、首次执行新任务类型。设计：定义触发条件 → Agent 暂停并呈现上下文 → 人工审核/修改 → 基于结果继续或终止。注意超时处理。

### Q4: 如何设计一个可靠的 Evaluator-Optimizer 循环？
**要点**：① Generator 和 Evaluator 用不同 Prompt/角色 ② Evaluator 给出具体改进建议（非仅打分）③ 设置最大迭代次数（防止无限循环）④ 定义明确的通过标准 ⑤ 记录每轮改进以追踪。

### Q5: 实际项目中如何组合多种 Agent 模式？
**要点**：以一个具体场景回答。例如：客服系统 = Routing（意图分发）+ Tool Use（查订单）+ Reflection（回答质量检查）+ Human-in-the-Loop（复杂投诉转人工）。
