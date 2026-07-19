# 钉钉 Stream 模式集成方案

## 1. 背景

### 1.1 问题陈述

TwinSpark 钉钉机器人当前采用 **HTTP Webhook 模式**接收消息，要求：
- 公网可达的 HTTPS 端点
- 有效的 SSL 证书
- **ICP 备案**（中国大陆强制要求）

由于 `cogniflux.me` 域名部署在阿里云中国大陆 ECS（47.110.57.8）但**未做 ICP 备案**，
运营商/阿里云在传输层拦截国内流量，导致：
- 钉钉验证服务器无法回调 Webhook 地址
- 国内客户端无法建立 TLS 连接（Connection reset by peer）
- 仅海外测试工具（如 SSLShopper）可正常连接

**结论**：在不做 ICP 备案的前提下，HTTP Webhook 模式在中国大陆 ECS 上**无法工作**。

### 1.2 解决方案选择

| 方案 | 时间成本 | 可行性 |
|------|----------|--------|
| ICP 备案 | 1-3 周审批 | 需要企业实体 + 域名实名 |
| 迁移 ECS 至香港/新加坡 | 1-2 天 | 增加延迟，需重新部署 |
| **切换 Stream 模式** | **1-2 天** | **零基础设施成本，官方推荐** |

选择 **Stream 模式**：钉钉官方推荐，无需公网 URL/ICP 备案，本地即可开发调试。

---

## 2. Stream 模式原理

### 2.1 架构对比

```
Webhook 模式（当前）：
  用户 @机器人 → 钉钉 → POST → 我方 HTTPS 端点 → 处理 → sessionWebhook 回复

Stream 模式（目标）：
  用户 @机器人 → 钉钉 ←WebSocket← 我方应用（主动连接）→ 处理 → SDK 内置回复
```

### 2.2 连接流程

1. **注册凭证**：应用 POST `/v1.0/gateway/connections/open`，携带 clientId + clientSecret
2. **建立 WebSocket**：连接返回的 endpoint URL（附带 ticket 参数）
3. **持久通信**：60 秒心跳保活，消息通过 WebSocket 帧推送
4. **自动重连**：网络断开后 SDK 自动指数退避重连

### 2.3 认证机制

- **传输层**：ticket 验证（SDK 自动管理）
- **无需应用层签名验证**：与 Webhook 的 HMAC-SHA256 不同

### 2.4 消息格式

```json
{
  "specVersion": "1.0",
  "type": "CALLBACK",
  "headers": {
    "appId": "xxx",
    "connectionId": "xxx",
    "contentType": "application/json",
    "messageId": "xxx",
    "time": 1234567890000,
    "topic": "/v1.0/im/bot/messages/get"
  },
  "data": "{\"conversationId\":\"cidXXX\",\"senderNick\":\"张三\",\"text\":{\"content\":\"hello\"},\"msgtype\":\"text\",\"senderId\":\"xxx\",...}"
}
```

---

## 3. 技术方案

### 3.1 依赖

```
dingtalk-stream >= 0.24.0
```

官方 Python SDK，PyPI 可安装，支持 Python 3.6+。

### 3.2 新建模块：`twinspark/dingtalk_stream.py`

职责：
- 封装 `DingTalkStreamClient` 生命周期管理
- 实现 `ChatbotHandler` 子类处理消息
- 集成 Agent 调用 + AI 卡片流式推送
- 提供 FastAPI lifespan 集成入口

核心类：

```python
class TwinSparkStreamHandler(ChatbotHandler):
    """处理钉钉 Stream 推送的机器人消息"""

    async def process(self, callback: CallbackMessage) -> tuple[str, str]:
        # 1. 解析消息（conversationId, text, senderNick 等）
        # 2. 判断消息模式：card / text
        # 3. card 模式：复用 DingTalkCardClient + stream_to_card()
        # 4. text 模式：agent.run() + reply_text()
        # 5. 返回 ACK
```

```python
class DingTalkStreamManager:
    """管理 Stream 客户端生命周期"""

    async def start(self, agent: Agent) -> None:
        # 创建 DingTalkStreamClient，注册 handler，启动后台任务

    async def stop(self) -> None:
        # 优雅关闭 WebSocket 连接
```

### 3.3 配置变更：`twinspark/config.py`

新增字段（复用现有 `dingtalk_app_key` 和 `dingtalk_app_secret`）：

```python
# 连接模式选择
dingtalk_mode: str = Field("disabled", alias="DINGTALK_MODE")
# 可选值: "disabled" | "stream" | "webhook"
# - disabled: 不启用钉钉集成
# - stream:   Stream 模式（推荐，无需公网）
# - webhook:  Webhook 模式（需公网 + ICP 备案）
```

**注意**：`dingtalk_app_key` 和 `dingtalk_app_secret` 已存在，Stream 模式直接复用。

### 3.4 API 层变更：`twinspark/api.py`

修改 `lifespan` 函数：

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    agent = build_agent()
    skill_loader = build_skill_loader()
    app.state.agent = agent
    app.state.skill_loader = skill_loader

    # 条件启动钉钉 Stream 模式
    stream_manager = None
    config = get_config()
    if config.dingtalk_mode == "stream":
        from twinspark.dingtalk_stream import DingTalkStreamManager
        stream_manager = DingTalkStreamManager(config)
        await stream_manager.start(agent)
        logger.info("DingTalk Stream client started")

    try:
        yield
    finally:
        if stream_manager:
            await stream_manager.stop()
        await agent.aclose()
```

### 3.5 代码复用清单

| 现有模块 | 复用情况 | 说明 |
|----------|----------|------|
| `DingTalkTokenManager` | ✅ 100% | 卡片 API 的 token 管理，与消息接收模式无关 |
| `DingTalkCardClient` | ✅ 100% | 卡片创建/更新/完成，HTTP API 不变 |
| `stream_to_card()` | ✅ 100% | 流式写入卡片，逻辑通用 |
| `process_dingtalk_card_stream()` | ✅ 90% | 核心逻辑复用，入口参数微调 |
| `verify_dingtalk_signature()` | ❌ 移除 | Stream 模式无需应用层签名 |
| `send_dingtalk_reply()` | ❌ 替换 | 改用 SDK 内置 `reply_text()` |
| `DingTalkCallbackBody` | ❌ 替换 | 改用 SDK `ChatbotMessage` |
| `/v1/dingtalk/webhook` 端点 | ⚠️ 保留 | webhook 模式兼容，stream 模式不使用 |

### 3.6 AI 卡片兼容性

**关键结论：Stream 模式完全兼容 AI 卡片 API**

- `createAndDeliver`（创建卡片）：通过 HTTP API 调用，不依赖 Webhook
- `card/streaming`（流式更新）：通过 HTTP API 调用，不依赖 Webhook
- 现有的打字机效果流式卡片逻辑**零修改**

消息处理流程：
```
Stream 收到消息
  → 解析 conversationId
  → DingTalkCardClient.create_card(conversationId)
  → agent.run_stream() + stream_to_card()
  → 完成
```

### 3.7 并发与连接管理

- **消息并发**：SDK 内部为每条消息创建独立协程任务，天然并发
- **重连策略**：SDK 内置指数退避（1s → 30s）；应用层外包 while True 兜底
- **心跳**：60 秒间隔，3 秒超时触发重连
- **优雅关闭**：lifespan finally 中 cancel stream task

---

## 4. 环境变量配置

```bash
# .env 配置示例

# 钉钉集成模式: disabled / stream / webhook
DINGTALK_MODE=stream

# 钉钉应用凭证（Stream 和 Webhook 共用）
DINGTALK_APP_KEY=dingXXXXXXXXXX
DINGTALK_APP_SECRET=your_app_secret

# AI 卡片配置（可选，Stream/Webhook 通用）
DINGTALK_CARD_TEMPLATE_ID=your_template_id
DINGTALK_MESSAGE_MODE=card          # card / text
DINGTALK_STREAM_INTERVAL_MS=500     # 流式推送间隔
```

---

## 5. 钉钉开放平台配置步骤

1. 登录 https://open.dingtalk.com → 应用开发 → 企业内部开发
2. 选择已有机器人应用
3. 进入「开发配置」→「消息接收模式」
4. 选择 **Stream 模式**（而非 HTTP 模式）
5. **无需填写消息接收地址**
6. 保存后直接发布

---

## 6. 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `twinspark/dingtalk_stream.py` | 新建 | Stream 模式核心模块 |
| `twinspark/config.py` | 修改 | 新增 `dingtalk_mode` 字段，废弃旧开关 |
| `twinspark/api.py` | 修改 | lifespan 集成 Stream 启动/关闭 |
| `twinspark/dingtalk.py` | 保留 | Webhook 模式代码保留做兼容 |
| `twinspark/dingtalk_card.py` | 不变 | 卡片 API 完全复用 |
| `.env.example` | 修改 | 更新配置模板 |
| `tests/test_dingtalk_stream.py` | 新建 | Stream 模块测试 |
| `pyproject.toml` | 修改 | 添加 dingtalk-stream 依赖 |

---

## 7. 测试计划

1. **单元测试**：mock SDK，验证 handler 消息解析、Agent 调用、错误处理
2. **集成测试**：mock WebSocket，验证连接管理、重连、心跳
3. **ECS 真实测试**：部署后在钉钉群 @机器人，验证端到端流程
4. **卡片测试**：确认 AI 卡片创建 + 流式更新在 Stream 模式下正常工作

---

## 8. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| SDK 版本更新 breaking change | 锁定 `>=0.24.0,<1.0` |
| WebSocket 长连接被中间件断开 | SDK 自动重连 + 应用层兜底 |
| LLM 处理超时影响 ACK | process() 中先 ACK 再异步处理 |
| dingtalk-stream 与 asyncio 事件循环冲突 | SDK 支持 asyncio，在 lifespan 中正确集成 |

---

## 9. 里程碑

- **M1**：新建 `dingtalk_stream.py`，实现纯文本消息收发
- **M2**：集成 AI 卡片流式推送
- **M3**：更新配置 + .env.example + 测试
- **M4**：ECS 部署验证 + 推送远端
