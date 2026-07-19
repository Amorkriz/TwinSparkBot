# TwinSpark 极简 Agent 重写计划

## 目标与范围

从 `hermes-agent/`(6205 文件)抽取核心概念,**从零极简重写**一个干净的最小可运行 Agent(约 3500 行核心代码),放在工作区新子目录 `twinspark/`,与 `hermes-agent/` 平级共存。

**保留能力**:基础 LLM 流式多轮对话 · 记忆/持久化 · 被动技能 · CLI/TUI + HTTP API 双入口。

**明确砍掉**:所有消息网关(Telegram/Discord/Slack 等)、除百炼外的全部 provider、六种终端后端、cron、子代理委派、学习图渲染、curator 后台维护、桌面应用等。

## 关键设计决策(已与用户确认)

- **单一 Provider**:官方 `openai` SDK,`base_url=https://dashscope.aliyuncs.com/compatible-mode/v1`,默认模型 `qwen-plus`(可配置 `qwen-max` 等)。硬编码,不做 provider 插件系统。
- **被动技能**:检索相关 `SKILL.md` 注入系统提示上下文,**不涉及工具执行、不做 /learn 自学习闭环**。
- **工具**:迁移一个干净的工具**基类 + 极简注册表**(保证架构可扩展),**先不实现任何具体工具**。
- **流式输出**:CLI 打字机效果 + HTTP API 的 SSE(`text/event-stream`)。
- **记忆**:SQLite + FTS5,启用 WAL 模式。

## 目录结构(新建)

```
twinspark/
├── twinspark/
│   ├── __init__.py
│   ├── config.py            # 配置:API key、base_url、model、db/skills 路径(env + .env)
│   ├── core/
│   │   ├── llm.py           # DashScope 客户端封装 + 流式(async generator)
│   │   ├── conversation.py  # 消息构建、上下文组装、单轮/多轮循环
│   │   └── agent.py         # Agent 主类:协调 LLM + 记忆 + 技能
│   ├── memory/
│   │   └── store.py         # SQLite:sessions/messages/facts + FTS5 + WAL
│   ├── skills/
│   │   ├── loader.py        # 扫描/解析 SKILL.md(YAML 头 + Markdown)
│   │   └── retriever.py     # 按查询匹配相关技能,返回注入文本(被动)
│   ├── tools/
│   │   ├── base.py          # Tool 基类抽象(从 hermes ToolEntry 精简迁移)
│   │   └── registry.py      # 极简注册表 + register 装饰器(不含 AST 扫描)
│   ├── cli.py               # prompt_toolkit 异步 REPL,流式渲染
│   └── api.py               # FastAPI 应用,SSE 流式端点
├── tests/
│   ├── test_memory.py
│   ├── test_skills.py
│   ├── test_llm.py          # 用 mock/monkeypatch,不依赖真实 API
│   └── test_api.py
├── pyproject.toml
├── .env.example             # DASHSCOPE_API_KEY=...
└── README.md
```

## hermes 参考文件(仅参考,不整块移植)

| 目的 | hermes 参考文件 |
|---|---|
| 对话循环骨架 | `agent/conversation_loop.py`、`agent/chat_completion_helpers.py`(LLM 调用/错误分类) |
| 记忆表结构与 FTS5 触发器 | `plugins/memory/holographic/store.py:16-76`(facts 表 + FTS5 + 触发器,精简掉 entities/hrr_vector/memory_banks) |
| 记忆检索 | `plugins/memory/holographic/retrieval.py`(FTS5 查询 + trust_score 排序,去掉 HRR/实体) |
| 工具基类 | `tools/registry.py:78-107`(ToolEntry 字段:name/schema/handler/is_async/description),精简成干净 base |
| 技能格式与创作标准 | `agent/learn_prompt.py:30-96`(SKILL.md YAML 头 + 章节标准) |
| CLI REPL | `cli.py:58-150`(prompt_toolkit 设置) |

## 任务分解与依赖

### Task 1:项目骨架与配置(无依赖)
- 建 `twinspark/` 目录树、`pyproject.toml`、`.env.example`、`README.md`。
- 依赖锁定:`openai`、`fastapi`、`uvicorn[standard]`、`pydantic`、`pyyaml`、`prompt-toolkit`、`httpx`。Python 3.11。
- `config.py`:从环境变量/`.env` 读取 `DASHSCOPE_API_KEY`、`base_url`、`model`、数据库路径(默认 `~/.twinspark/state.db`)、技能目录(默认 `~/.twinspark/skills/`)。

### Task 2:LLM 客户端 + 流式(依赖 Task 1)
- `core/llm.py`:封装 `openai.OpenAI/AsyncOpenAI`,`base_url` 指向百炼。
- 提供 `chat(messages) -> str` 与 `stream(messages) -> AsyncIterator[str]` 两种接口。
- 复用单一 client 实例(连接池),`httpx` 超时治理(read 超时放宽以支持长流)。
- 错误分类参考 `chat_completion_helpers.py`,做最小重试(429/5xx)。

### Task 3:记忆存储(依赖 Task 1,可与 Task 2 并行)
- `memory/store.py`,SQLite + WAL。表:
  - `sessions(session_id, created_at, updated_at, metadata)`
  - `messages(msg_id, session_id, role, content, created_at)`
  - `facts(fact_id, content UNIQUE, tags, trust_score, retrieval_count, created_at, updated_at)` + `facts_fts`(FTS5)+ 三个同步触发器(参考 store.py:48-66)。
- 接口:`add_message`、`get_history(session_id)`、`add_fact`、`recall(query, limit)`(FTS5 MATCH + trust_score DESC 排序)、`search_history(query)`。
- 并发:`sqlite3` 连接加锁 / WAL;为未来向量检索预留 `retriever` 抽象。

### Task 4:工具基类框架(依赖 Task 1,可并行)
- `tools/base.py`:定义 `Tool` 抽象(字段 name/description/parameters(JSON schema)/is_async,方法 `run(**kwargs)`)。
- `tools/registry.py`:极简 `ToolRegistry` + `@register` 装饰器 + `get_openai_schemas()`(输出 OpenAI function schema)。**不含** hermes 的 AST 扫描、TTL 缓存、check_fn 探测。
- **不实现任何具体工具**,仅留 base + 空注册表 + 单元测试验证注册/取 schema。

### Task 5:被动技能系统(依赖 Task 1、Task 3)
- `skills/loader.py`:扫描 `skills/<category>/<name>/SKILL.md`,解析 YAML 前言(name/description/tags)+ 正文。
- `skills/retriever.py`:按用户查询做关键词匹配(可复用 memory 的 FTS5 或简单打分),返回 top-N 技能正文文本用于注入。
- 附一个示例技能目录 + 一个示例 SKILL.md 供演示。

### Task 6:Agent 核心与对话循环(依赖 Task 2、3、5)
- `core/conversation.py`:组装系统提示(基础人格 + 注入的记忆 + 注入的技能)+ 历史消息 + 当前输入。
- `core/agent.py`:`Agent` 类,方法 `run(user_msg, session_id, stream=True)`:
  1. `memory.recall(user_msg)` 前拉取事实
  2. `skills.retriever` 匹配相关技能
  3. 组装消息 → `llm.stream()` 流式产出
  4. 落库 `memory.add_message`(user+assistant)
- 工具循环在本版本为**空实现占位**(检测到 tool_call 时预留分派点,但因无工具,正常路径只输出文本)。

### Task 7:CLI/TUI 入口(依赖 Task 6)
- `cli.py`:`prompt_toolkit` 异步 REPL,`session.prompt_async()`,流式打印助手输出。
- 命令:`/new`(新会话)、`/skills`(列表)、`/memory <query>`(检索)、`/quit`。历史持久化到 `~/.twinspark/history`。
- 入口 `python -m twinspark.cli` 或 console_script `twinspark`。

### Task 8:HTTP API 入口(依赖 Task 6,可与 Task 7 并行)
- `api.py`:FastAPI 应用。端点:
  - `POST /v1/chat`(非流式,返回完整回复)
  - `POST /v1/chat/stream`(SSE,`StreamingResponse` + `text/event-stream`)
  - `GET /v1/sessions/{id}/messages`、`GET /v1/memory/search`、`GET /v1/skills`
  - `GET /health`
- 内存维护 `session_id → Agent 上下文`;`uvicorn` 启动(`twinspark.api:app`)。

### Task 9:测试与验证(依赖 Task 2-8)
- 单测:记忆 CRUD/FTS5 检索、技能加载解析、工具注册取 schema、API 端点(LLM 用 monkeypatch mock,不消耗真实额度)。
- 集成冒烟:配置真实 `DASHSCOPE_API_KEY` 后,CLI 单轮对话 + API `/v1/chat/stream` SSE 流式验证(需用户提供 key,或标记为可选/跳过)。

## 依赖关系图

```
Task1 ─┬─ Task2(LLM) ──┐
       ├─ Task3(记忆) ──┼─ Task6(Agent核心) ─┬─ Task7(CLI) ─┐
       ├─ Task4(工具基类)┤                    └─ Task8(API) ─┴─ Task9(测试)
       └─ Task5(技能) ───┘  (Task5 依赖 Task3)
```
Task2/3/4 可在 Task1 后并行;Task7/8 可在 Task6 后并行。

## 遵循的约定

- 全异步优先(`async def` + `AsyncOpenAI`),CLI 用 `asyncio` 驱动 `prompt_toolkit`。
- 配置集中在 `config.py`,敏感信息只走环境变量/`.env`,**不硬编码 API key**。
- SQLite 启用 `PRAGMA journal_mode=WAL`;文件写入用原子写模式。
- 技能文件格式对齐 hermes(YAML 前言 + Markdown),便于未来兼容 agentskills.io。

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| 百炼对 OpenAI 参数兼容性差异(如 `stream_options`、部分采样参数) | Task2 早期做真实冒烟;对不支持参数做条件下发;错误分类兜底 |
| SQLite 多并发(API workers)"database is locked" | 启用 WAL;单连接串行化写;必要时后续引入连接池/队列 |
| 流式中断/超时导致回复不完整 | `httpx` read 超时放宽;SSE 客户端断开时优雅收尾并落库已生成部分 |
| 被动技能注入过多撑爆上下文 | retriever 限制 top-N + 字符预算;超限截断 |
| FTS5 事实量增大后检索变慢 | 建 trust_score 索引 + limit 分页;预留向量检索接口 |

## 被否决的替代方案

- **整块移植 hermes 模块(Jack 方案,~1.7-2 万行)**:虽新代码量小,但会把 conversation_loop(5316 行)的重耦合、大量私有属性访问、fallback 链一并带入,与用户明确选择的"极简重写"目标冲突,长期维护成本高。**否决**,仅采纳其"哪些文件值得参考 + 耦合分析"的洞见。
- **保留 provider 插件系统**:声明式 ProviderProfile 虽优雅,但单一百炼场景下属于过度设计。**否决**,改为硬编码单 provider。
- **MVP 完全不做工具**:用户要求迁移工具基类以备扩展,故保留 base + registry 抽象,仅不实现具体工具。
- **记忆保留实体解析 + HRR 向量 + 信任评分完整体系**:复杂度高、依赖 NumPy。**否决**,首版只做 FTS5 + trust_score 排序,预留扩展接口。