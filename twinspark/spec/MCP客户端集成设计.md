# TwinSparkBot MCP 客户端集成详细设计

## 概述

为 TwinSparkBot 添加 MCP 客户端能力，让 Agent 能够连接外部 MCP 服务器、发现并调用其提供的工具。设计遵循项目"极简重写"定位，采用分层隔离 + 最小侵入策略。

## 架构设计

```
用户输入 → Agent.run()
             ↓
       _build_turn_messages() (记忆 + 技能)
             ↓
       _generate(messages)
             ↓
       LLMClient.chat_raw(messages, tools=schemas)
             ↓
       ┌─ 有 tool_calls ──────────────────────┐
       │  追加 assistant 消息                    │
       │  并发执行: asyncio.gather(             │
       │    tool.run(**args) for each call      │
       │  )                                     │
       │  追加 tool result 消息                  │
       │  continue → 下一轮生成                  │
       └─ 无 tool_calls → break 返回文本 ───────┘
             ↓
       返回最终回复

工具来源:
  ToolRegistry ← MCPTool 实例 (来自外部 MCP 服务器)
       ↑
  MCPManager.discover_and_register()
       ↑
  MCPClient (stdio) ──── 外部 MCP Server (npx/python进程)
  MCPClient (sse)   ──── 外部 MCP Server (HTTP服务)
```

## 目录结构（新增文件）

```
twinspark/
├── twinspark/
│   ├── tools/
│   │   ├── mcp/                    # MCP 客户端子包
│   │   │   ├── __init__.py         # 包入口，导出核心类
│   │   │   ├── client.py           # MCP 连接客户端（stdio/SSE）
│   │   │   ├── adapter.py          # MCP工具 → Tool 适配器
│   │   │   └── manager.py          # 多服务器生命周期管理
│   │   ├── base.py                 # 未修改
│   │   ├── registry.py             # 未修改
│   │   └── __init__.py             # 未修改
│   ├── core/
│   │   ├── agent.py                # 修改：补齐工具循环
│   │   └── llm.py                  # 修改：新增 chat_raw() 方法
│   └── config.py                   # 修改：新增 MCP 配置字段
├── mcp-servers.yaml.example        # 配置示例文件
├── tests/
│   ├── test_mcp_client.py          # 客户端单元测试
│   ├── test_mcp_adapter.py         # 适配器单元测试
│   └── test_agent_tools.py         # Agent工具循环集成测试
└── pyproject.toml                  # 修改：添加 mcp 依赖
```

## 核心模块说明

### MCPClient（client.py）

单个 MCP 服务器的异步连接客户端，管理连接生命周期。

- 支持 stdio 和 SSE 两种传输方式
- 使用 AsyncExitStack 管理连接上下文
- 内置 3 次指数退避重试（1s/2s/4s）
- 超时保护（默认 60s 连接，300s 调用）

### MCPTool（adapter.py）

将 MCP 远程工具适配为 TwinSpark Tool 接口。

- 继承 Tool 基类，is_async = True
- 命名格式：`{server_name}:{tool_name}`
- run() 方法委托给 MCPClient.call_tool()

### MCPManager（manager.py）

多服务器生命周期管理器。

- 逐个连接服务器，单个失败不影响其他（降级运行）
- 自动发现工具并注册到 ToolRegistry
- 提供 start()/stop() 生命周期方法

### Agent 工具循环（agent.py）

在 _generate() 方法中实现完整的工具调用循环：

1. 有工具时使用 chat_raw() 获取完整消息
2. 检测 tool_calls → 追加 assistant 消息
3. asyncio.gather() 并发执行工具
4. 追加 tool result 消息
5. continue 让模型基于结果响应
6. max_tool_rounds 防止无限循环

## 配置方式

### 环境变量

```env
TWINSPARK_MCP_ENABLED=true
TWINSPARK_MCP_CONFIG_PATH=~/.twinspark/mcp-servers.yaml
```

### YAML 配置文件

```yaml
servers:
  filesystem:
    transport: stdio
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    timeout: 120

  fetch:
    transport: sse
    url: "http://localhost:8080/sse"
    timeout: 60
```

## 设计原则

1. **中文注释**：所有模块文档字符串、类说明、关键逻辑均使用中文注释
2. **类型标注**：所有公开方法必须有完整类型标注
3. **错误隔离**：MCP 连接/调用失败不中断 Agent 主流程
4. **向后兼容**：不修改 Tool、ToolRegistry 的现有接口
5. **异步优先**：所有 MCP 操作使用 async/await，不引入额外线程

## 风险缓解

| 风险 | 缓解措施 |
|------|---------|
| MCP SDK 版本变动 | 锁定 mcp>=1.26.0,<2.0 |
| 工具名冲突 | 强制 server:tool 命名空间 |
| 连接超时阻塞启动 | 每服务器独立 try/except |
| LLM 不返回 tool_calls | 无 schemas 时走原有文本路径 |
| 无限工具循环 | max_tool_rounds=5 上限 |
| 流式模式不支持工具 | 第一版仅 non-stream 支持 |

## 后续扩展方向

- 流式模式工具调用支持
- 熔断器模式（连续失败自动禁用）
- OAuth 认证支持
- HTTP Streamable 传输方式
- 动态工具刷新（tools/list_changed 通知）
- MCP 状态诊断 HTTP 端点
