# 钉钉 AI 卡片流式输出升级方案

## Summary

在现有钉钉纯文本集成基础上，新增 AI 卡片流式输出能力。收到消息后立即创建一张 AI 卡片（显示"思考中..."），然后通过 `agent.run_stream()` 逐块生成回复，每 500ms 将累积文本通过钉钉卡片流式更新 API 推送，实现打字机效果。

**核心数据流（卡片模式）：**
```
用户 @机器人 → 钉钉 POST /v1/dingtalk/webhook
  → 验证签名 → 立即返回 HTTP 200
  → 后台: 获取 access_token（缓存）
  → 后台: 创建 AI 卡片实例 → 获得 cardInstanceId
  → 后台: agent.run_stream() 逐块输出
  → 后台: 每 500ms 批量更新卡片内容（PUT streaming API）
  → 后台: 流式结束 → 最终更新完整内容 + 标记完成
```

## 关键设计决策

| 决策点 | 选择 | 原因 |
|--------|------|------|
| 接入模式 | HTTP 模式（非 Stream/WebSocket） | 复用现有 webhook 架构，无需重构消息接收层 |
| 向后兼容 | 配置开关 `DINGTALK_MESSAGE_MODE`，默认 "text" | 不破坏现有纯文本功能，用户可选升级 |
| 流式更新频率 | 500ms 缓冲窗口 | 钉钉限流约 3-5 次/秒，500ms 安全且视觉流畅 |
| Token 管理 | 内存缓存 + asyncio.Lock | access_token 有效 2 小时，避免重复获取和并发刷新 |
| 失败策略 | 自动降级为纯文本 | 卡片 API 任何错误都回退到已有的 send_dingtalk_reply |
| 模块结构 | 新建 `dingtalk_card.py` 单独文件 | 与现有 `dingtalk.py` 职责分离，互不影响 |

## 文件变更清单

| 文件 | 操作 | 变更量 | 说明 |
|------|------|--------|------|
| `twinspark/dingtalk_card.py` | 新建 | ~220 行 | 卡片 API 客户端（token + 创建 + 流式更新） |
| `twinspark/config.py` | 修改 | +8 行 | 新增卡片相关可选配置 |
| `twinspark/dingtalk.py` | 修改 | +50 行 | 新增卡片模式处理函数 |
| `twinspark/api.py` | 修改 | +15 行 | webhook 端点添加模式分支 |
| `.env.example` | 修改 | +6 行 | 新增卡片配置示例 |
| `tests/test_dingtalk_card.py` | 新建 | ~150 行 | 卡片模块测试 |

**总计：~370 行新增 + ~73 行修改**

## 实现步骤

### Step 1: 扩展配置（`config.py`）

在 Config 类中现有钉钉字段之后添加：

```python
# DingTalk AI Card (optional, requires app_key)
dingtalk_app_key: str = Field("", alias="DINGTALK_APP_KEY")
dingtalk_card_template_id: str = Field("", alias="DINGTALK_CARD_TEMPLATE_ID")
dingtalk_message_mode: str = Field("text", alias="DINGTALK_MESSAGE_MODE")
    # "text" = 纯文本（现有行为）
    # "card" = AI卡片流式
    # "auto" = 优先卡片，失败回退文本
dingtalk_stream_interval_ms: int = Field(500, alias="DINGTALK_STREAM_INTERVAL_MS")
```

**要点：**
- 所有字段可选，默认值保持现有行为（"text"模式）
- `dingtalk_app_key` 是调用卡片 API 所需（与现有 `dingtalk_app_secret` 配合获取 access_token）
- `dingtalk_card_template_id` 为卡片模板 ID（需在钉钉开放平台预先创建）

### Step 2: 创建卡片 API 客户端（`dingtalk_card.py`）

新建 `twinspark/twinspark/dingtalk_card.py`，约 220 行，包含：

**2.1 Access Token 管理**
```python
class DingTalkTokenManager:
    """管理钉钉 access_token 的获取和缓存"""
    
    def __init__(self, app_key: str, app_secret: str):
        self._app_key = app_key
        self._app_secret = app_secret
        self._token: str = ""
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()
    
    async def get_token(self) -> str:
        """获取有效的 access_token，过期自动刷新"""
        if time.time() < self._expires_at - 300:  # 提前5分钟刷新
            return self._token
        async with self._lock:  # 防止并发刷新
            if time.time() < self._expires_at - 300:
                return self._token
            # POST https://api.dingtalk.com/v1.0/oauth2/accessToken
            # body: {"appKey": ..., "appSecret": ...}
            # response: {"accessToken": "...", "expireIn": 7200}
            ...
```

**关键细节：**
- asyncio.Lock 防止并发请求导致重复刷新（thundering herd）
- 提前 5 分钟刷新避免边界过期
- token 有效期 7200 秒（2 小时）

**2.2 卡片实例管理**
```python
class DingTalkCardClient:
    """钉钉 AI 卡片 API 客户端"""
    
    def __init__(self, token_manager: DingTalkTokenManager, template_id: str):
        self._token_mgr = token_manager
        self._template_id = template_id
    
    async def create_card(
        self, conversation_id: str, content: str = "正在思考..."
    ) -> str:
        """创建卡片实例，返回 cardInstanceId
        
        API: POST https://api.dingtalk.com/v1.0/card/instances/createAndDeliver
        """
        ...
    
    async def streaming_update(self, card_instance_id: str, content: str) -> None:
        """流式更新卡片内容（覆盖模式）
        
        API: PUT https://api.dingtalk.com/v1.0/card/streaming
        """
        ...
    
    async def finish_card(self, card_instance_id: str, final_content: str) -> None:
        """标记卡片流式输出完成，发送最终内容"""
        ...
```

**关键细节：**
- 创建卡片时传入 `conversationId`（钉钉用于投放到对应群聊）
- 流式更新使用覆盖模式（`isReplaceAll: true`）— Markdown 内容必须全量替换
- finish 时发送完整内容确保最终展示正确
- 所有 API 调用包含 2 次重试 + 错误日志

**2.3 流式缓冲处理**
```python
async def stream_to_card(
    card_client: DingTalkCardClient,
    card_instance_id: str,
    token_stream: AsyncIterator[str],
    flush_interval: float = 0.5,
) -> str:
    """将 Agent 流式输出通过缓冲推送到卡片
    
    每 flush_interval 秒将累积文本推送一次，返回完整回复文本。
    """
    buffer = []
    full_text = []
    last_flush = time.time()
    
    async for delta in token_stream:
        buffer.append(delta)
        full_text.append(delta)
        
        now = time.time()
        if now - last_flush >= flush_interval:
            content = "".join(full_text)
            await card_client.streaming_update(card_instance_id, content)
            buffer.clear()
            last_flush = now
    
    # 最终刷新
    final_content = "".join(full_text)
    await card_client.finish_card(card_instance_id, final_content)
    return final_content
```

**关键细节：**
- 500ms 刷新窗口：平衡流畅度和 API 调用频率
- 全量替换模式：每次推送截止到当前的完整文本（非增量）
- 返回完整文本用于后续持久化到会话历史

### Step 3: 扩展钉钉集成模块（`dingtalk.py`）

在现有 `send_dingtalk_reply()` 之后新增卡片模式处理函数：

```python
async def process_dingtalk_card_stream(
    agent: "Agent",
    user_message: str,
    conversation_id: str,
    sender_staff_id: str = "",
    session_webhook: str = "",
) -> None:
    """AI 卡片流式模式处理（后台任务）
    
    1. 创建卡片（显示"思考中..."）
    2. 调用 agent.run_stream() 逐块生成
    3. 通过 stream_to_card() 定时推送更新
    4. 异常时降级发送纯文本
    """
    from twinspark.config import get_config
    from twinspark.dingtalk_card import (
        DingTalkTokenManager, DingTalkCardClient, stream_to_card
    )
    
    config = get_config()
    
    try:
        # 初始化卡片客户端
        token_mgr = DingTalkTokenManager(config.dingtalk_app_key, config.dingtalk_app_secret)
        card_client = DingTalkCardClient(token_mgr, config.dingtalk_card_template_id)
        
        # 创建卡片
        card_id = await card_client.create_card(conversation_id)
        
        # 流式生成 + 推送
        stream = agent.run_stream(user_message, session_id=conversation_id)
        await stream_to_card(
            card_client, card_id, stream,
            flush_interval=config.dingtalk_stream_interval_ms / 1000.0,
        )
        
    except Exception:
        logger.exception("Card streaming failed, falling back to text")
        # 降级：用纯文本发送完整回复
        try:
            reply = await agent.run(user_message, session_id=conversation_id)
            await send_dingtalk_reply(session_webhook, reply, sender_staff_id)
        except Exception:
            logger.exception("Text fallback also failed")
            await send_dingtalk_reply(session_webhook, "处理消息时出错，请稍后再试。")
```

**关键细节：**
- 卡片失败时自动降级为纯文本（使用现有 `send_dingtalk_reply`）
- `agent.run_stream()` 内部已处理消息持久化（finally 块）
- 卡片降级时用 `agent.run()` 重新生成（避免重复持久化问题）

### Step 4: 修改 webhook 端点（`api.py`）

修改现有 `dingtalk_webhook` 函数中的后台处理逻辑：

```python
# 4. 后台异步处理（立即返回 200）
async def _process():
    try:
        if config.dingtalk_message_mode in ("card", "auto") and config.dingtalk_app_key:
            await process_dingtalk_card_stream(
                agent, user_message, callback.conversationId,
                callback.senderStaffId, callback.sessionWebhook,
            )
        else:
            reply = await agent.run(user_message, session_id=callback.conversationId)
            await send_dingtalk_reply(
                callback.sessionWebhook, reply, callback.senderStaffId
            )
    except Exception:
        logger.exception("DingTalk background processing failed")
        await send_dingtalk_reply(
            callback.sessionWebhook, "处理消息时出错，请稍后再试。"
        )
```

**变更最小化：**
- 仅替换 `_process()` 内部逻辑，添加一个条件分支
- 导入 `process_dingtalk_card_stream` from dingtalk 模块
- 现有签名验证、请求解析、立即返回逻辑完全不变

### Step 5: 更新 `.env.example`

追加：
```bash
# --- DingTalk AI Card (optional, enables streaming) ---
# DINGTALK_APP_KEY=your-app-key
# DINGTALK_CARD_TEMPLATE_ID=your-card-template-id
# DINGTALK_MESSAGE_MODE=text          # text | card | auto
# DINGTALK_STREAM_INTERVAL_MS=500     # card update interval
```

### Step 6: 编写测试（`tests/test_dingtalk_card.py`）

覆盖：
1. **TokenManager**: token 获取、缓存命中、过期刷新、并发锁
2. **CardClient**: 创建卡片成功/失败、流式更新、finish
3. **stream_to_card**: 缓冲刷新时机、完整文本返回
4. **端点集成**: card 模式下 webhook 正确路由、降级回退

## 钉钉开放平台配置步骤

### 1. 获取应用凭证
1. 登录 [钉钉开发者后台](https://open.dingtalk.com)
2. 进入已有的机器人应用（或新建）
3. 在「凭证与基础信息」获取 `AppKey` 和 `AppSecret`

### 2. 创建 AI 卡片模板
1. 进入「卡片平台」→「创建卡片模板」
2. 选择「AI 卡片」类型
3. 模板中定义一个 Markdown 类型的 `content` 变量
4. 保存并发布，获取 `模板ID`

### 3. 配置权限
在应用权限管理中开通：
- `Card.Instance.Write` — 创建卡片实例
- `Card.Streaming.Write` — 流式更新卡片

### 4. 部署配置
```bash
# ECS 上
vi /home/TwinSparkBot/twinspark/.env

# 新增：
DINGTALK_APP_KEY=dingxxxxxxxx
DINGTALK_CARD_TEMPLATE_ID=your-template-id
DINGTALK_MESSAGE_MODE=card
```

### 5. 重启服务
```bash
cd /home/TwinSparkBot/twinspark
source venv/bin/activate
uvicorn twinspark.api:app --host 0.0.0.0 --port 8000
```

## 风险与缓解

| 风险 | 严重性 | 缓解措施 |
|------|--------|---------|
| 卡片 API 返回错误（模板ID无效等） | 高 | `auto` 模式自动降级纯文本；详细错误日志 |
| access_token 刷新失败 | 中 | 重试 2 次；失败后本次降级文本，下次重试刷新 |
| 流式更新触发钉钉限流（429） | 中 | 500ms 间隔已保守；遇 429 暂停 1s 后继续；极端情况降级 |
| 卡片模板未创建/未发布 | 高 | 启动时日志警告（config 有值但 API 失败）；文档明确步骤 |
| 并发用户竞争 token 刷新 | 低 | asyncio.Lock 已防止 thundering herd |
| 长回复（>5000字）卡片渲染异常 | 低 | 监控内容长度；超长时截断 + 提示"回复过长" |
| 现有纯文本模式被意外破坏 | 低 | 默认 `DINGTALK_MESSAGE_MODE=text`，不改变任何现有行为 |

## 被排除的替代方案

| 替代方案 | 排除原因 |
|---------|---------|
| Stream 模式（WebSocket 长连接） | 需要重构整个消息接收架构，与现有 webhook 方案不兼容；适合未来独立迭代 |
| 将 card_client 合并到 dingtalk.py | 职责混乱：签名验证+文本发送 vs token管理+卡片API 是不同关注点 |
| 增量更新（非覆盖模式） | 钉钉 Markdown 字段不支持增量追加，必须全量替换 |
| 固定每 N 个 token 更新 | 不如基于时间窗口稳定——token 输出速率波动大，时间窗口保证视觉一致性 |
| 使用 Redis 缓存 access_token | 过度设计：单进程部署内存缓存足够；多实例时再引入 |

## 依赖关系

```
Step 1 (config.py) ← 无前置
    ↓
Step 2 (dingtalk_card.py) ← 依赖 Step 1
    ↓
Step 3 (dingtalk.py 扩展) ← 依赖 Step 2
    ↓
Step 4 (api.py 修改) ← 依赖 Step 3
    ↓
Step 5 (.env.example) ← 可并行
    ↓
Step 6 (tests) ← 依赖 Step 1-4
```

关键路径 Step 1 → 2 → 3 → 4，预计 1-2 小时完成实现。

## 注意事项

- **卡片模板 ID 是必须的**：需要用户手动在钉钉开放平台创建 AI 卡片模板
- **AppKey 与 AppSecret 的区别**：现有 `DINGTALK_APP_SECRET` 用于签名验证；新增 `DINGTALK_APP_KEY` 配合它一起获取 access_token
- **agent.run_stream() 的持久化**：流式模式下 Agent 内部的 finally 块已保证消息持久化，无需额外处理
- **降级时的重复消息风险**：卡片模式失败降级时，如果 run_stream 已部分执行（消息已持久化），降级用 run() 会导致重复。需要在降级路径中跳过 agent 调用，直接发送错误提示