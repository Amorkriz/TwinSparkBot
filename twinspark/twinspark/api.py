"""HTTP API for TwinSpark (FastAPI + Server-Sent Events).

This module exposes the :class:`~twinspark.core.agent.Agent` over HTTP so the
bot can be driven by any client. It is intentionally thin: all conversational
logic lives in the agent core; the API only translates HTTP requests into
``agent`` calls and shapes the responses.

Concurrency model
------------------
A **single process-level** :class:`Agent` is created on startup and shared by
every request. The agent's collaborators are already concurrency-safe:

* :class:`~twinspark.memory.store.MemoryStore` uses one connection guarded by a
  lock and runs in SQLite WAL mode, so concurrent reads/writes are safe.
* :class:`~twinspark.core.llm.LLMClient` reuses one async client / connection
  pool.

Different conversations are isolated purely by ``session_id``: every request
passes its own id to ``agent.run(..., session_id=...)`` /
``agent.run_stream(..., session_id=...)`` which the agent honours as a per-call
override, so histories never bleed across sessions.

Lifecycle
---------
The agent (and a :class:`~twinspark.skills.loader.SkillLoader`) are created in
the FastAPI ``lifespan`` startup and torn down (``await agent.aclose()``) on
shutdown. If ``DASHSCOPE_API_KEY`` is missing, startup fails fast with a clear
log message.

Run it with::

    uvicorn twinspark.api:app --host 0.0.0.0 --port 8000

Endpoints
---------
* ``GET  /health``                              -> liveness probe.
* ``POST /v1/chat``                             -> non-streaming reply.
* ``POST /v1/chat/stream``                      -> streaming reply (SSE).
* ``GET  /v1/sessions/{session_id}/messages``   -> session history.
* ``GET  /v1/memory/search``                    -> recall durable facts.
* ``GET  /v1/skills``                           -> list available skills.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from twinspark.core.agent import Agent
from twinspark.skills.loader import SkillLoader

logger = logging.getLogger(__name__)

__all__ = ["app", "build_agent", "build_skill_loader"]


# --------------------------------------------------------------------------- #
# Resource factories (kept as module-level functions so tests can monkeypatch  #
# them to inject fakes without a real DASHSCOPE_API_KEY).                       #
# --------------------------------------------------------------------------- #
def build_agent() -> Agent:
    """Construct the process-level :class:`Agent`.

    Validates configuration eagerly so a missing ``DASHSCOPE_API_KEY`` surfaces
    as a clear startup error rather than a confusing failure on the first
    request.

    Returns:
        A fully wired :class:`Agent` using real collaborators.

    Raises:
        pydantic.ValidationError: If ``DASHSCOPE_API_KEY`` is not configured.
    """
    # Import lazily so tests that monkeypatch this factory never trigger config
    # loading (and therefore never require a real API key).
    from twinspark.config import get_config

    get_config()  # raises early with a clear message if the key is missing
    return Agent()


def build_skill_loader() -> SkillLoader:
    """Construct the process-level :class:`SkillLoader`."""
    return SkillLoader()


# --------------------------------------------------------------------------- #
# Pydantic request / response models                                           #
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    """Request body for the chat endpoints."""

    message: str = Field(..., description="The user's input message.")
    session_id: Optional[str] = Field(
        None,
        description=(
            "Conversation id. When omitted the server generates one and "
            "returns it so the client can continue the conversation."
        ),
    )
    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Optional sampling temperature forwarded to the model.",
    )


class ChatResponse(BaseModel):
    """Response body for the non-streaming chat endpoint."""

    session_id: str = Field(..., description="The session id used for the turn.")
    reply: str = Field(..., description="The assistant's full reply text.")


class MessageItem(BaseModel):
    """A single persisted message in a session's history."""

    role: str
    content: str
    created_at: Optional[str] = None


class SessionMessagesResponse(BaseModel):
    """Response body for the session-history endpoint."""

    session_id: str
    messages: list[MessageItem]


class MemorySearchResponse(BaseModel):
    """Response body for the memory-search endpoint."""

    query: str
    results: list[dict[str, Any]]


class SkillItem(BaseModel):
    """A lightweight skill summary."""

    name: str
    description: str = ""
    category: str = ""
    tags: list[str] = Field(default_factory=list)


class SkillsResponse(BaseModel):
    """Response body for the skills-listing endpoint."""

    skills: list[SkillItem]


class HealthResponse(BaseModel):
    """Response body for the health probe."""

    status: str


# --------------------------------------------------------------------------- #
# Lifespan: create shared resources on startup, release them on shutdown        #
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage the lifetime of the shared agent and skill loader."""
    try:
        agent = build_agent()
    except Exception:  # noqa: BLE001 - re-raised after a clear log line
        logger.error(
            "Failed to start TwinSpark API: could not construct the agent. "
            "Ensure DASHSCOPE_API_KEY is set (export it or add it to a .env "
            "file). See twinspark/config.py for all supported settings.",
            exc_info=True,
        )
        raise

    skill_loader = build_skill_loader()

    app.state.agent = agent
    app.state.skill_loader = skill_loader
    logger.info("TwinSpark API started; agent and skill loader ready.")

    try:
        yield
    finally:
        try:
            await agent.aclose()
        except Exception:  # noqa: BLE001 - best-effort teardown
            logger.warning("Error while closing the agent", exc_info=True)
        logger.info("TwinSpark API shut down; resources released.")


app = FastAPI(
    title="TwinSpark API",
    version="0.1.0",
    description="HTTP interface for the TwinSpark agent (chat, memory, skills).",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Dependencies                                                                 #
# --------------------------------------------------------------------------- #
def get_agent(request: Request) -> Agent:
    """Return the process-level agent stored on app state."""
    agent: Optional[Agent] = getattr(request.app.state, "agent", None)
    if agent is None:  # pragma: no cover - only if lifespan did not run
        raise HTTPException(status_code=503, detail="Agent is not ready.")
    return agent


def get_skill_loader(request: Request) -> SkillLoader:
    """Return the process-level skill loader stored on app state."""
    loader: Optional[SkillLoader] = getattr(request.app.state, "skill_loader", None)
    if loader is None:  # pragma: no cover - only if lifespan did not run
        raise HTTPException(status_code=503, detail="Skill loader is not ready.")
    return loader


def _llm_kwargs(req: ChatRequest) -> dict[str, Any]:
    """Extract optional LLM parameters from the request (omitting ``None``)."""
    kwargs: dict[str, Any] = {}
    if req.temperature is not None:
        kwargs["temperature"] = req.temperature
    return kwargs


# --------------------------------------------------------------------------- #
# Endpoints                                                                    #
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="ok")


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    agent: Agent = Depends(get_agent),
) -> ChatResponse:
    """Run a single non-streaming turn and return the full reply.

    A missing ``session_id`` is generated server-side and echoed back so the
    client can keep the conversation going.
    """
    session_id = req.session_id or uuid.uuid4().hex
    reply = await agent.run(
        req.message, session_id=session_id, **_llm_kwargs(req)
    )
    return ChatResponse(session_id=session_id, reply=reply)


def _sse(data: str, event: Optional[str] = None) -> str:
    """Format a single Server-Sent Event frame.

    Newlines inside ``data`` are split into multiple ``data:`` lines so the
    frame stays valid per the SSE spec.
    """
    lines: list[str] = []
    if event is not None:
        lines.append(f"event: {event}")
    for chunk_line in data.split("\n"):
        lines.append(f"data: {chunk_line}")
    return "\n".join(lines) + "\n\n"


@app.post("/v1/chat/stream")
async def chat_stream(
    req: ChatRequest,
    agent: Agent = Depends(get_agent),
) -> StreamingResponse:
    """Stream a single turn as Server-Sent Events.

    The stream begins with a ``session`` event carrying the (possibly
    generated) ``session_id``, followed by one ``data:`` frame per text delta,
    and ends with a ``done`` event (``data: [DONE]``).

    If the client disconnects mid-stream the async generator is cancelled; we
    close the underlying agent generator so its ``finally`` block persists the
    partial reply, then exit quietly.
    """
    session_id = req.session_id or uuid.uuid4().hex
    llm_kwargs = _llm_kwargs(req)

    async def event_source() -> AsyncIterator[str]:
        agen = agent.run_stream(req.message, session_id=session_id, **llm_kwargs)
        try:
            # Announce the session id up front so streaming clients can persist
            # it for follow-up turns.
            yield _sse(session_id, event="session")
            async for delta in agen:
                if delta:
                    yield _sse(delta)
            yield _sse("[DONE]", event="done")
        except asyncio.CancelledError:
            # Client disconnected: stop quietly. Closing ``agen`` below runs the
            # agent's finally-block which persists whatever was generated.
            logger.info(
                "Client disconnected during stream (session=%s)", session_id
            )
            raise
        finally:
            # Ensure the agent's streaming generator is finalized so its
            # try/finally persists the partial (or full) reply exactly once.
            await agen.aclose()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/v1/sessions/{session_id}/messages",
    response_model=SessionMessagesResponse,
)
async def get_session_messages(
    session_id: str,
    agent: Agent = Depends(get_agent),
) -> SessionMessagesResponse:
    """Return a session's persisted message history (oldest first)."""
    history = agent.memory.get_history(session_id)
    messages = [MessageItem(**row) for row in history]
    return SessionMessagesResponse(session_id=session_id, messages=messages)


@app.get("/v1/memory/search", response_model=MemorySearchResponse)
async def memory_search(
    q: str = Query(..., description="Natural-language search query."),
    limit: int = Query(5, ge=1, le=50, description="Maximum facts to return."),
    agent: Agent = Depends(get_agent),
) -> MemorySearchResponse:
    """Full-text search durable facts via the agent's memory store."""
    results = agent.memory.recall(q, limit=limit)
    return MemorySearchResponse(query=q, results=results)


@app.get("/v1/skills", response_model=SkillsResponse)
async def list_skills(
    loader: SkillLoader = Depends(get_skill_loader),
) -> SkillsResponse:
    """List the skills available to the agent."""
    skills = [SkillItem(**item) for item in loader.list_skills()]
    return SkillsResponse(skills=skills)


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

    # 3. 提取用户消息
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

    # 5. 立即返回
    return {"code": 0, "msg": "ok"}
