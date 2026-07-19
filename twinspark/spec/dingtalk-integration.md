# 钉钉企业内部机器人对接方案

## Summary

在 TwinSpark 现有 FastAPI 服务中新增 `/v1/dingtalk/webhook` 端点，接收钉钉回调后**立即返回 HTTP 200**（满足钉钉 2-3 秒超时要求），然后通过 `asyncio.create_task()` 在后台调用 `agent.run()` 生成回复，最后通过 `sessionWebhook` 异步发送回钉钉群。

**核心数据流：**
```
用户 @机器人 → 钉钉服务器 POST /v1/dingtalk/webhook
  → 验证签名 → 立即返回 {"code": 0}
  → 后台: agent.run(message, session_id=conversationId)
  → 后台: httpx.post(sessionWebhook, reply)
  → 钉钉群展示回复
```

## 关键设计决策

| 决策点 | 选择 | 原因 |
|--------|------|------|
| 响应模式 | 立即返回 + asyncio.create_task 后台处理 | 钉钉要求 2-3 秒内返回 HTTP 200，LLM 生成需 10-30 秒 |
| 模块结构 | 单文件 `twinspark/dingtalk.py` | 项目处于早期，逻辑量小（~150 行），避免过度抽象 |
| 会话映射 | conversationId → session_id | 同群共享上下文，符合群聊交互预期 |
| 签名验证 | 可关闭（开发模式） | 提供 `DINGTALK_VERIFY_SIGNATURE=false` 便于本地调试 |
| Agent 调用 | agent.run()（非流式） | 钉钉不支持流式消息，需完整回复后一次性发送 |

## 文件变更清单

| 文件 | 操作 | 变更量 | 说明 |
|------|------|--------|------|
| `twinspark/twinspark/dingtalk.py` | 新建 | ~150 行 | 钉钉协议全部逻辑（模型、签名、发送） |
| `twinspark/twinspark/config.py` | 修改 | +5 行 | 添加 2 个可选配置字段 |
| `twinspark/twinspark/api.py` | 修改 | +30 行 | 添加 webhook 端点 + lifespan 日志 |
| `twinspark/.env.example` | 修改 | +4 行 | 添加钉钉配置示例 |
| `twinspark/tests/test_dingtalk.py` | 新建 | ~100 行 | 签名验证 + 端点测试 |

**总计：~250 行新增 + ~35 行修改**

## 实现步骤

### Step 1: 扩展配置（`twinspark/config.py`）

在 `Config` 类中（第 48-52 行之后）添加两个可选字段：

```python
# DingTalk integration (optional)
dingtalk_app_secret: str = Field("", alias="DINGTALK_APP_SECRET")
dingtalk_verify_signature: bool = Field(True, alias="DINGTALK_VERIFY_SIGNATURE")
```

**要点：**
- 均为可选（有默认值），不影响现有启动流程
- `extra="ignore"` 已在 model_config 中设置，新环境变量不会报错
- `dingtalk_app_secret` 为空时，webhook 端点返回 503（未配置）

### Step 2: 创建钉钉模块（`twinspark/dingtalk.py`）

单文件包含三部分功能：

**2.1 Pydantic 数据模型**
```python
class DingTalkCallbackBody(BaseModel):
    """钉钉 webhook 回调请求体"""
    conversationId: str
    senderNick: str
    senderStaffId: str = ""
    text: dict  # {"content": "用户消息"}
    msgtype: str = "text"
    sessionWebhook: str
    sessionWebhookExpiredTime: int = 0
    # 其他字段设为 Optional，钉钉可能不总是发送全部字段
```

**2.2 签名验证**
```python
def verify_dingtalk_signature(timestamp: str, sign: str, app_secret: str) -> bool:
    """HMAC-SHA256 签名验证
    
    算法：Base64(HMAC-SHA256(timestamp + "\\n" + secret, secret))
    """
    sign_str = f"{timestamp}\n{app_secret}"
    hmac_code = hmac.new(
        app_secret.encode("utf-8"),
        sign_str.encode("utf-8"),
        hashlib.sha256
    ).digest()
    calculated = base64.b64encode(hmac_code).decode("utf-8")
    return hmac.compare_digest(calculated, sign)
```

**关键细节：**
- 使用 `hmac.compare_digest()` 防止时序攻击
- 时间戳有效期检查（1 小时内）
- 标准库 `hmac` + `hashlib` + `base64`，无需新依赖

**2.3 异步回复发送**
```python
async def send_dingtalk_reply(
    session_webhook: str, reply_text: str, sender_staff_id: str = ""
) -> None:
    """通过 sessionWebhook 异步发送回复到钉钉"""
    payload = {
        "msgtype": "text",
        "text": {"content": reply_text},
    }
    if sender_staff_id:
        payload["at"] = {"atUserIds": [sender_staff_id], "isAtAll": False}

    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(3):  # 简单重试
            try:
                resp = await client.post(session_webhook, json=payload)
                resp.raise_for_status()
                return
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (410, 403):
                    # sessionWebhook 已过期，不再重试
                    logger.warning("sessionWebhook expired: %s", e)
                    return
                if attempt == 2:
                    logger.error("Failed to send reply after 3 attempts: %s", e)
            except httpx.RequestError as e:
                if attempt == 2:
                    logger.error("Network error sending reply: %s", e)
            await asyncio.sleep(1 * (attempt + 1))  # 简单退避
```

**关键细节：**
- 3 次重试 + 线性退避
- 410/403 直接放弃（webhook 过期）
- 5xx/网络错误才重试
- 使用 httpx（项目已有依赖）

### Step 3: 添加 API 端点（`twinspark/api.py`）

在文件末尾（第 357 行之后）添加钉钉 webhook 端点：

```python
@app.post("/v1/dingtalk/webhook")
async def dingtalk_webhook(request: Request, agent: Agent = Depends(get_agent)):
    """钉钉企业内部机器人回调端点"""
    from twinspark.config import get_config
    from twinspark.dingtalk import (
        DingTalkCallbackBody, verify_dingtalk_signature, send_dingtalk_reply
    )

    config = get_config()
    if not config.dingtalk_app_secret:
        raise HTTPException(status_code=503, detail="DingTalk not configured")

    # 1. 签名验证
    if config.dingtalk_verify_signature:
        timestamp = request.headers.get("Timestamp", "")
        sign = request.headers.get("Sign", "")
        if not verify_dingtalk_signature(timestamp, sign, config.dingtalk_app_secret):
            raise HTTPException(status_code=401, detail="Invalid signature")

    # 2. 解析请求体
    body = await request.json()
    callback = DingTalkCallbackBody(**body)

    # 3. 提取用户消息（去除可能的 @前缀 空格）
    user_message = callback.text.get("content", "").strip()
    if not user_message:
        return {"code": 0, "msg": "empty message"}

    # 4. 后台异步处理（立即返回 200）
    async def _process():
        try:
            reply = await agent.run(user_message, session_id=callback.conversationId)
            await send_dingtalk_reply(
                callback.sessionWebhook, reply, callback.senderStaffId
            )
        except Exception:
            logger.exception("DingTalk background processing failed")
            await send_dingtalk_reply(
                callback.sessionWebhook, "处理消息时出错，请稍后再试。"
            )

    asyncio.create_task(_process())

    # 5. 立即返回，告知钉钉"已收到"
    return {"code": 0, "msg": "ok"}
```

**关键细节：**
- `asyncio.create_task()` 确保立即返回，不阻塞
- 复用现有 `get_agent` 依赖注入
- 后台任务内包含完整错误处理，失败时发送友好提示
- 使用 `conversationId` 作为 session_id（群聊共享上下文）

### Step 4: 更新环境变量模板（`.env.example`）

追加：
```bash
# --- DingTalk Bot (optional) ---
# DINGTALK_APP_SECRET=your-dingtalk-robot-app-secret
# DINGTALK_VERIFY_SIGNATURE=true
```

### Step 5: 编写测试（`tests/test_dingtalk.py`）

覆盖三个维度：
1. **签名验证**：正确签名通过、错误签名拒绝、过期时间戳拒绝
2. **端点行为**：未配置返回 503、签名错误返回 401、正常请求返回 200
3. **消息处理**：Agent 被正确调用、session_id 使用 conversationId

## 钉钉开放平台配置步骤

### 1. 创建机器人
1. 登录 [钉钉开发者后台](https://open.dingtalk.com)
2. 进入「应用开发」→「企业内部开发」→「机器人」
3. 创建机器人应用，记录 AppSecret

### 2. 配置消息接收
1. 在机器人设置页找到「消息接收模式」选择「HTTP」
2. 填写消息接收地址：`http://<ECS公网IP>:8000/v1/dingtalk/webhook`
3. 保存并验证连通性

### 3. 部署配置
```bash
# 在 ECS 上编辑 .env
cd /home/TwinSparkBot/twinspark
vi .env

# 添加：
DINGTALK_APP_SECRET=你从开放平台获取的AppSecret
DINGTALK_VERIFY_SIGNATURE=true
```

### 4. 重启服务
```bash
source venv/bin/activate
uvicorn twinspark.api:app --host 0.0.0.0 --port 8000
```

### 5. 安全组配置
确保 ECS 安全组入站规则允许 8000 端口（或使用 Nginx 反代到 80/443）。

### 6. 发布上线
在钉钉开发者后台发布机器人版本，然后在群聊中添加该机器人。

## 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| LLM 超时（>60s）导致无回复 | LLMClient 已有 300s 读超时 + 3 次重试；极端情况 catch 异常发送错误提示 |
| sessionWebhook 过期（用户发消息后几小时才处理） | 正常场景不会发生（create_task 立即执行）；过期时 410 直接放弃 |
| 签名验证算法实现错误 | 单元测试用已知向量验证；提供 VERIFY_SIGNATURE=false 开关调试 |
| 高并发消息淹没 Agent | MemoryStore 已有 RLock + WAL 保护；asyncio 天然并发安全 |
| ECS 重启时后台任务丢失 | asyncio.create_task 是进程内的，重启会丢失进行中的任务；但消息已入 session 历史，用户重发即可 |

## 被排除的替代方案

| 替代方案 | 排除原因 |
|---------|---------|
| asyncio.Queue + Consumer 工作线程 | 过度设计：单 ECS 部署无需队列；create_task 已满足需求且零额外复杂度 |
| 多文件子包 (integrations/dingtalk/) | 过度抽象：当前逻辑仅 ~150 行，单文件更清晰；未来需要时再拆分 |
| 同步等待 Agent 回复再返回 HTTP 200 | 不可行：LLM 生成 10-30 秒，远超钉钉 2-3 秒超时限制 |
| Redis 持久化队列 | 过早优化：增加外部依赖和运维成本；MVP 阶段进程内 create_task 足够 |
| 流式生成 + 分块发送多条消息 | 钉钉不支持消息编辑/追加；分多条消息发送体验差且有频率限制 |

## 依赖关系

```
Step 1 (config.py) ← 无前置依赖
    ↓
Step 2 (dingtalk.py) ← 依赖 Step 1（导入 config）
    ↓
Step 3 (api.py) ← 依赖 Step 2（导入 dingtalk 模块）
    ↓
Step 4 (.env.example) ← 可与 Step 1-3 并行
    ↓
Step 5 (tests) ← 依赖 Step 1-3 完成
```

Step 1 → Step 2 → Step 3 为关键路径，约 30 分钟可完成实现。
