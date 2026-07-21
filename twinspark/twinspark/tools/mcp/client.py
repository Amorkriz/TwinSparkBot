"""MCP 客户端 —— 管理与单个 MCP 服务器的异步连接。

本模块实现了 MCPClient 类，负责：
1. 通过 stdio 或 SSE 传输方式连接到 MCP 服务器
2. 使用 AsyncExitStack 管理连接上下文的生命周期
3. 提供工具列表获取和工具调用的异步接口
4. 内置重试机制（3次，指数退避 1s/2s/4s）和超时保护
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)

# 重试配置常量
_MAX_RETRIES: int = 3
_BASE_BACKOFF_SECONDS: float = 1.0


class MCPClient:
    """单个 MCP 服务器的异步连接客户端。

    管理与一个外部 MCP 服务器的连接生命周期，
    支持 stdio 和 SSE 两种传输方式。

    使用 contextlib.AsyncExitStack 持有传输层和会话的上下文，
    使得连接在 connect() 后保持活跃，直到显式调用 disconnect()。

    Attributes:
        name: 服务器标识名称，用于日志和工具命名前缀。
    """

    def __init__(
        self,
        name: str,
        transport: str,
        command: str = "",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str = "",
        timeout: int = 60,
    ) -> None:
        """初始化 MCP 客户端。

        Args:
            name: 服务器标识名称，用于日志输出和工具名称前缀。
            transport: 传输方式，支持 "stdio" 或 "sse"。
            command: stdio 模式下要启动的服务器命令。
            args: stdio 模式下传给命令的参数列表。
            env: stdio 模式下附加的环境变量。
            url: SSE 模式下服务器的 URL 地址。
            timeout: 连接和调用的超时时间（秒），默认 60 秒。

        Raises:
            ValueError: transport 不是 "stdio" 或 "sse" 时抛出。
        """
        if transport not in ("stdio", "sse"):
            raise ValueError(f"不支持的传输方式: {transport!r}，仅支持 'stdio' 或 'sse'")

        self.name: str = name
        self._transport: str = transport
        self._command: str = command
        self._args: list[str] = args or []
        self._env: dict[str, str] | None = env
        self._url: str = url
        self._timeout: int = timeout

        # 运行时状态（连接后填充）
        self._exit_stack: contextlib.AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        """建立与 MCP 服务器的连接。

        内置重试机制：最多 3 次尝试，使用指数退避（1s/2s/4s）。
        连接成功后会调用 session.initialize() 完成 MCP 协议握手。

        Raises:
            MCPConnectionError: 所有重试均失败后抛出（通过上层捕获处理）。
        """
        # 如果已有旧连接，先清理
        if self._exit_stack is not None:
            await self.disconnect()

        last_error: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with asyncio.timeout(self._timeout):
                    await self._do_connect()
                logger.info("MCP 服务器 [%s] 连接成功（第 %d 次尝试）", self.name, attempt)
                return
            except Exception as exc:
                last_error = exc
                # 连接失败时清理可能残留的 exit_stack
                if self._exit_stack is not None:
                    try:
                        await self._exit_stack.aclose()
                    except Exception:
                        pass
                    self._exit_stack = None
                    self._session = None

                if attempt < _MAX_RETRIES:
                    backoff = _BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    logger.warning(
                        "MCP 服务器 [%s] 第 %d 次连接失败，%0.1fs 后重试: %s",
                        self.name, attempt, backoff, exc,
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        "MCP 服务器 [%s] 连接失败，已耗尽全部 %d 次重试: %s",
                        self.name, _MAX_RETRIES, exc,
                    )

        # 延迟导入避免循环引用
        from twinspark.tools.mcp import MCPConnectionError
        raise MCPConnectionError(
            server_name=self.name,
            message=f"经过 {_MAX_RETRIES} 次尝试仍无法连接",
            cause=last_error,
        )

    async def _do_connect(self) -> None:
        """执行实际的连接逻辑（不含重试）。

        根据传输方式创建对应的 transport context 和 session context，
        使用 AsyncExitStack 将它们的生命周期托管起来。
        """
        self._exit_stack = contextlib.AsyncExitStack()

        if self._transport == "stdio":
            server_params = StdioServerParameters(
                command=self._command,
                args=self._args,
                env=self._env,
            )
            # 进入 stdio transport 上下文
            transport_ctx = stdio_client(server_params)
            read, write = await self._exit_stack.enter_async_context(transport_ctx)
        else:
            # SSE 传输
            transport_ctx = sse_client(self._url)
            read, write = await self._exit_stack.enter_async_context(transport_ctx)

        # 进入 ClientSession 上下文
        session = ClientSession(read, write)
        self._session = await self._exit_stack.enter_async_context(session)

        # 执行 MCP 协议握手
        await self._session.initialize()

    async def disconnect(self) -> None:
        """优雅关闭与 MCP 服务器的连接，释放所有资源。

        多次调用是安全的（幂等操作）。
        即使关闭过程中发生异常也不会向外传播。
        """
        if self._exit_stack is None:
            return

        try:
            await self._exit_stack.aclose()
            logger.info("MCP 服务器 [%s] 连接已关闭", self.name)
        except Exception as exc:
            logger.warning("MCP 服务器 [%s] 关闭连接时发生异常: %s", self.name, exc)
        finally:
            self._exit_stack = None
            self._session = None

    async def list_tools(self) -> list[dict[str, Any]]:
        """从 MCP 服务器获取可用工具列表。

        Returns:
            工具描述字典列表，每个元素格式为：
            {"name": "...", "description": "...", "inputSchema": {...}}

        Raises:
            MCPConnectionError: 客户端未连接时抛出。
            MCPToolError: 获取工具列表失败时抛出。
        """
        self._ensure_connected()

        try:
            async with asyncio.timeout(self._timeout):
                result = await self._session.list_tools()  # type: ignore[union-attr]

            # 将 MCP SDK 返回的工具对象转换为标准字典格式
            tools: list[dict[str, Any]] = []
            for tool in result.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema if hasattr(tool, "inputSchema") else {"type": "object", "properties": {}},
                })
            logger.debug("MCP 服务器 [%s] 发现 %d 个工具", self.name, len(tools))
            return tools

        except asyncio.TimeoutError as exc:
            from twinspark.tools.mcp import MCPConnectionError
            raise MCPConnectionError(
                server_name=self.name,
                message="获取工具列表超时",
                cause=exc,
            )
        except Exception as exc:
            from twinspark.tools.mcp import MCPToolError
            raise MCPToolError(
                tool_name="list_tools",
                server_name=self.name,
                message=f"获取工具列表失败: {exc}",
                cause=exc,
            )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """调用 MCP 服务器上的指定工具。

        Args:
            name: 远程工具名称（不含服务器前缀）。
            arguments: 工具调用参数字典。

        Returns:
            序列化后的结果字符串。简单文本直接返回，
            复杂对象通过 JSON 序列化后返回。

        Raises:
            MCPToolError: 工具调用失败时抛出。
        """
        self._ensure_connected()

        try:
            async with asyncio.timeout(self._timeout):
                result = await self._session.call_tool(name, arguments)  # type: ignore[union-attr]

            # 将结果统一转为字符串
            return self._serialize_result(result)

        except asyncio.TimeoutError as exc:
            from twinspark.tools.mcp import MCPToolError
            raise MCPToolError(
                tool_name=name,
                server_name=self.name,
                message=f"调用超时（{self._timeout}s）",
                cause=exc,
            )
        except Exception as exc:
            # 避免重复包装自定义异常
            from twinspark.tools.mcp import MCPToolError
            if isinstance(exc, MCPToolError):
                raise
            raise MCPToolError(
                tool_name=name,
                server_name=self.name,
                message=f"调用失败: {exc}",
                cause=exc,
            )

    @property
    def is_connected(self) -> bool:
        """当前是否处于已连接状态。"""
        return self._session is not None and self._exit_stack is not None

    def _ensure_connected(self) -> None:
        """检查连接状态，未连接时抛出异常。"""
        if not self.is_connected:
            from twinspark.tools.mcp import MCPConnectionError
            raise MCPConnectionError(
                server_name=self.name,
                message="客户端未连接，请先调用 connect()",
            )

    @staticmethod
    def _serialize_result(result: Any) -> str:
        """将 MCP 调用结果序列化为字符串。

        处理策略：
        - 如果结果包含 content 列表，提取文本内容拼接返回
        - 字符串直接返回
        - 其他类型 JSON 序列化
        """
        # MCP SDK 返回的 CallToolResult 通常有 content 属性
        if hasattr(result, "content") and isinstance(result.content, list):
            text_parts: list[str] = []
            for block in result.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
                else:
                    # 非文本块序列化为 JSON
                    text_parts.append(json.dumps(block.__dict__, ensure_ascii=False, default=str))
            return "\n".join(text_parts) if text_parts else ""

        if isinstance(result, str):
            return result

        # 兜底：JSON 序列化
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)

    def __repr__(self) -> str:
        status = "已连接" if self.is_connected else "未连接"
        return f"MCPClient(name={self.name!r}, transport={self._transport!r}, status={status})"
