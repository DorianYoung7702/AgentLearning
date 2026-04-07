# MCP 协议与 Agent 互操作

> MCP 已成为 Agent 连接工具的**事实标准**，是面试必考内容。
> **最后更新：2026 年 4 月**

**MCP 大事记**：
- 2024 Q4：Anthropic 发布 MCP 协议
- 2025 Q3：MCP Registry 上线（收录近 2000 个 MCP Server）
- 2025 Q4：MCP 加入 **Linux Foundation**，成为多公司开放标准
- 2025-11：发布最新 Spec 版本，新增 **Tasks 原语**（实验性）
- 2026 Q1：发布 2026 路线图（传输扩展、Agent 通信、企业就绪、治理成熟）

## 1. 什么是 MCP？

### 1.1 定义

**MCP（Model Context Protocol）** 是 Anthropic 于 2024 年底提出、现由 Linux Foundation 托管的开放标准协议，旨在标准化 **LLM 应用与外部数据源/工具之间的通信方式**。

**类比理解**：
```
USB 之于硬件设备 = MCP 之于 AI 工具/数据源

没有 USB：每个设备需要专用接口
有了 USB：统一接口，即插即用

没有 MCP：每个工具需要专门集成代码
有了 MCP：统一协议，任何 MCP Client 可连接任何 MCP Server
```

### 1.2 解决了什么问题？

**之前**（N×M 问题）：
```
每个 LLM 应用需要为每个工具写专门的集成代码

App1 ─── 集成代码A ──→ Tool A
App1 ─── 集成代码B ──→ Tool B
App2 ─── 集成代码A' ─→ Tool A   (重复开发)
App2 ─── 集成代码B' ─→ Tool B   (重复开发)

N 个应用 × M 个工具 = N×M 个集成
```

**MCP 之后**（N+M 问题）：
```
App1 ──┐                ┌──→ MCP Server A (Tool A)
App2 ──┤── MCP 协议 ────┤──→ MCP Server B (Tool B)
App3 ──┘                └──→ MCP Server C (Tool C)

N 个应用 + M 个服务 = N+M 个实现
```

## 2. MCP 架构

### 2.1 核心组件

```
┌──────────────┐     MCP 协议     ┌──────────────┐
│  MCP Client  │ ←──────────────→ │  MCP Server  │
│  (LLM 应用)  │                  │  (工具/数据)  │
└──────┬───────┘                  └──────┬───────┘
       │                                 │
       │                          ┌──────┴───────┐
  ┌────┴─────┐                    │ 本地资源      │
  │ MCP Host │                    │ (文件/数据库)  │
  │(IDE/App) │                    │ 或            │
  └──────────┘                    │ 远程服务      │
                                  │ (API/SaaS)    │
                                  └──────────────┘
```

| 组件 | 说明 | 示例 |
|------|------|------|
| **MCP Host** | 运行 MCP Client 的宿主应用 | Cursor、Windsurf、Claude Desktop |
| **MCP Client** | 负责与 MCP Server 通信 | 内置在 Host 中 |
| **MCP Server** | 暴露工具和数据的服务端 | GitHub MCP Server、数据库 MCP Server |

### 2.2 MCP 提供的能力

| 能力 | 说明 | 方向 |
|------|------|------|
| **Tools（工具）** | 可调用的函数 | Server → Client（LLM 调用） |
| **Resources（资源）** | 可读取的数据 | Server → Client |
| **Prompts（提示模板）** | 预定义的 Prompt 模板 | Server → Client |
| **Sampling（采样）** | 请求 LLM 生成内容 | Client → Server |

### 2.3 通信协议

**传输层**：
| 传输方式 | 说明 | 适用场景 |
|---------|------|---------|
| **stdio** | 标准输入输出 | 本地 MCP Server |
| **HTTP + SSE** | HTTP 请求 + Server-Sent Events | 远程 MCP Server |
| **Streamable HTTP** | 可流式的 HTTP | 新版本推荐 |

**消息格式**：基于 **JSON-RPC 2.0**
```json
// Client → Server (请求)
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search_docs",
    "arguments": {"query": "MCP 协议"}
  }
}

// Server → Client (响应)
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [{"type": "text", "text": "搜索结果..."}]
  }
}
```

## 3. MCP Server 开发

### 3.1 Python SDK 示例

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("my-server")

# 定义工具
@server.tool()
async def search_docs(query: str) -> list[TextContent]:
    """在知识库中搜索文档"""
    results = await do_search(query)
    return [TextContent(type="text", text=str(results))]

# 定义资源
@server.resource("docs://{doc_id}")
async def get_document(doc_id: str) -> str:
    """获取指定文档内容"""
    return await load_document(doc_id)

# 启动服务
if __name__ == "__main__":
    server.run()
```

### 3.2 MCP Server 设计原则

1. **工具描述清晰**：让 LLM 能正确理解和调用
2. **参数类型明确**：JSON Schema 定义参数
3. **错误处理友好**：返回有意义的错误信息
4. **安全性**：输入验证、权限控制
5. **幂等性**：查询类工具保证幂等

## 4. MCP 生态

### 4.1 已支持 MCP 的平台

| 平台 | 类型 | 说明 |
|------|------|------|
| Claude Desktop | Host | Anthropic 官方 |
| Cursor | Host | AI 编程 IDE |
| Windsurf | Host | AI 编程 IDE |
| VS Code (Copilot) | Host | 微软 |
| Cline | Host | VS Code 插件 |
| Continue | Host | 开源 AI 编程 |

### 4.2 常用 MCP Server

| Server | 功能 |
|--------|------|
| GitHub MCP | 仓库管理、PR、Issues |
| Filesystem MCP | 文件读写操作 |
| PostgreSQL MCP | 数据库查询 |
| Slack MCP | 消息发送和读取 |
| Google Drive MCP | 文档管理 |
| Brave Search MCP | Web 搜索 |

## 5. Google A2A 协议

### 5.1 什么是 A2A？

**A2A（Agent-to-Agent）** 是 Google 于 2025 年提出的 Agent 间通信协议。

**MCP vs A2A**：
```
MCP:  LLM 应用 ←→ 工具/数据    (应用与工具之间)
A2A:  Agent   ←→ Agent          (Agent 与 Agent 之间)
```

### 5.2 A2A 核心概念

| 概念 | 说明 |
|------|------|
| **Agent Card** | Agent 的名片，描述能力和接口 |
| **Task** | Agent 之间协作的任务单元 |
| **Message** | Agent 之间交换的消息 |
| **Artifact** | 任务产生的输出物 |

### 5.3 A2A 工作流程

```
1. Agent A 发现 Agent B（通过 Agent Card）
2. Agent A 创建 Task，发送给 Agent B
3. Agent B 处理 Task，返回结果或请求更多信息
4. 循环直到任务完成
5. Agent B 返回 Artifact（最终产出）
```

### 5.4 MCP + A2A 互补关系

```
┌─────────────────────────────────┐
│         Multi-Agent System       │
│                                  │
│  Agent A ←── A2A ──→ Agent B    │  ← Agent 间协作用 A2A
│    │                    │        │
│    │ MCP                │ MCP    │  ← Agent 与工具用 MCP
│    ▼                    ▼        │
│  Tools/Data          Tools/Data  │
└─────────────────────────────────┘
```

## 6. 面试高频问题

### Q1: 什么是 MCP？它解决了什么问题？
**要点**：MCP 是标准化 LLM 应用与工具/数据源通信的开放协议。解决了 N×M 集成问题，实现了"即插即用"的工具接入。类比 USB 之于硬件。

### Q2: MCP 的架构组成是什么？
**要点**：三层 — Host（宿主应用）、Client（协议客户端）、Server（工具/数据提供方）。通信基于 JSON-RPC 2.0，传输层支持 stdio（本地）和 HTTP+SSE（远程）。

### Q3: MCP 提供了哪些能力类型？
**要点**：Tools（可调用函数）、Resources（可读数据）、Prompts（提示模板）、Sampling（请求 LLM 生成）。最常用的是 Tools 和 Resources。

### Q4: MCP 和 Function Calling 有什么区别？
**要点**：Function Calling 是 LLM 生成工具调用的能力（模型层面）；MCP 是工具如何暴露和被调用的标准协议（通信层面）。MCP Server 暴露 Tools，LLM 通过 Function Calling 决定调用哪个 Tool，MCP Client 通过 MCP 协议执行调用。

### Q5: 对比 MCP 和 A2A 协议。
**要点**：MCP 解决 Agent 与工具/数据的连接（垂直集成）；A2A 解决 Agent 与 Agent 的协作（水平协作）。两者互补：Agent 通过 MCP 使用工具，通过 A2A 与其他 Agent 协作。

### Q6: 如何设计一个 MCP Server？
**要点**：① 确定要暴露的工具和资源 ② 设计清晰的工具描述和参数 Schema ③ 实现工具逻辑 ④ 错误处理和输入验证 ⑤ 选择传输方式（本地用 stdio，远程用 HTTP）⑥ 安全性考虑（认证、授权）。
