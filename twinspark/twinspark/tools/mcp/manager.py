"""MCP 多服务器生命周期管理器。

本模块实现了 MCPManager 类，负责：
- 根据配置列表批量创建和管理 MCPClient 实例
- 并行连接多个 MCP 服务器（单个失败不影响其他）
- 从所有已连接服务器发现远程工具
- 将发现的工具批量注册到 TwinSpark ToolRegistry

设计原则：
- 降级运行：单个服务器连接失败只记录警告，不影响其他服务器
- 优雅关闭：stop() 确保所有连接被正确释放
- 幂等操作：重复调用 start/stop 是安全的
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from twinspark.tools.mcp.client import MCPClient
from twinspark.tools.mcp.adapter import MCPTool, create_mcp_tools

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """MCP 服务器连接配置。

    用于描述一个 MCP 服务器的连接参数，
    支持 stdio 和 SSE 两种传输方式。

    Attributes:
        name: 服务器标识名称（唯一），用于日志和工具命名前缀。
        transport: 传输方式，"stdio" 或 "sse"。
        command: stdio 模式下的启动命令。
        args: stdio 模式下的命令参数。
        env: stdio 模式下的附加环境变量。
        url: SSE 模式下的服务器 URL。
        timeout: 连接和调用超时时间（秒），默认 60。
    """

    name: str
    transport: str  # "stdio" | "sse"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    timeout: int = 60


class MCPManager:
    """MCP 多服务器生命周期管理器。

    负责管理多个 MCPClient 实例的连接和关闭，
    从所有已连接服务器发现工具并注册到 ToolRegistry。

    典型使用方式：
        ```python
        configs = [MCPServerConfig(name="fs", transport="stdio", command="npx", args=["-y", "@mcp/fs"])]
        manager = MCPManager(configs)
        await manager.start(registry)
        # ... 使用工具 ...
        await manager.stop()
        ```

    设计特点：
    - 单个服务器连接失败不影响其他服务器（降级运行）
    - 所有异常被内部捕获并记录，不会向上传播
    """

    def __init__(self, configs: list[MCPServerConfig]) -> None:
        """初始化 MCP 管理器。

        Args:
            configs: MCP 服务器配置列表。每个配置描述一个待连接的服务器。
        """
        self._configs: list[MCPServerConfig] = configs
        self._clients: dict[str, MCPClient] = {}
        self._tools: list[MCPTool] = []
        self._started: bool = False

    async def start(self, registry: Any) -> None:
        """启动所有 MCP 服务器连接，发现工具并注册。

        按配置列表依次连接各 MCP 服务器，从已连接的服务器获取工具列表，
        创建对应的 MCPTool 适配器并注册到 ToolRegistry。

        单个服务器连接失败不影响其他服务器（降级运行模式）。

        Args:
            registry: TwinSpark ToolRegistry 实例，用于注册发现的工具。
        """
        if self._started:
            logger.warning("MCPManager 已处于启动状态，跳过重复启动")
            return

        logger.info("MCPManager 开始启动，共 %d 个服务器配置", len(self._configs))

        for config in self._configs:
            # 逐个连接服务器（出错不中断）
            client = await self._connect_server(config)
            if client is None:
                continue

            self._clients[config.name] = client

            # 从已连接服务器发现工具
            tools = await self._discover_tools(client, config.name)
            if not tools:
                continue

            # 注册工具到 ToolRegistry
            self._register_tools(tools, registry)

        self._started = True
        logger.info(
            "MCPManager 启动完成: %d/%d 个服务器连接成功，共注册 %d 个工具",
            len(self._clients), len(self._configs), len(self._tools),
        )

    async def stop(self) -> None:
        """优雅关闭所有 MCP 服务器连接。

        逐个断开已连接的服务器，释放所有资源。
        单个服务器断开失败不影响其他服务器的关闭。
        多次调用是安全的（幂等操作）。
        """
        if not self._started:
            return

        logger.info("MCPManager 开始关闭，共 %d 个活跃连接", len(self._clients))

        for name, client in list(self._clients.items()):
            try:
                await client.disconnect()
                logger.debug("MCP 服务器 [%s] 已断开", name)
            except Exception as exc:
                logger.warning("MCP 服务器 [%s] 断开时发生异常: %s", name, exc)

        self._clients.clear()
        self._tools.clear()
        self._started = False
        logger.info("MCPManager 已完全关闭")

    @property
    def connected_servers(self) -> list[str]:
        """当前已连接的服务器名称列表。"""
        return [name for name, client in self._clients.items() if client.is_connected]

    @property
    def available_tools(self) -> list[str]:
        """当前所有可用工具的名称列表（格式: 服务器名:工具名）。"""
        return [tool.name for tool in self._tools]

    async def _connect_server(self, config: MCPServerConfig) -> MCPClient | None:
        """连接单个 MCP 服务器，失败返回 None。

        Args:
            config: 服务器连接配置。

        Returns:
            连接成功返回 MCPClient 实例，失败返回 None。
        """
        client = MCPClient(
            name=config.name,
            transport=config.transport,
            command=config.command,
            args=config.args,
            env=config.env if config.env else None,
            url=config.url,
            timeout=config.timeout,
        )

        try:
            await client.connect()
            return client
        except Exception as exc:
            logger.error(
                "MCP 服务器 [%s] 连接失败，该服务器将被跳过: %s",
                config.name, exc,
            )
            return None

    async def _discover_tools(self, client: MCPClient, server_name: str) -> list[MCPTool]:
        """从已连接的服务器发现并创建工具适配器。

        Args:
            client: 已连接的 MCPClient 实例。
            server_name: 服务器标识名称。

        Returns:
            MCPTool 适配器列表，获取失败时返回空列表。
        """
        try:
            tool_schemas = await client.list_tools()
            if not tool_schemas:
                logger.info("MCP 服务器 [%s] 未暴露任何工具", server_name)
                return []
            return create_mcp_tools(server_name, tool_schemas, client)
        except Exception as exc:
            logger.error(
                "MCP 服务器 [%s] 获取工具列表失败: %s",
                server_name, exc,
            )
            return []

    def _register_tools(self, tools: list[MCPTool], registry: Any) -> None:
        """将工具适配器注册到 ToolRegistry。

        单个工具注册失败（如名称冲突）不影响其他工具。

        Args:
            tools: 待注册的 MCPTool 列表。
            registry: TwinSpark ToolRegistry 实例。
        """
        for tool in tools:
            try:
                registry.register(tool)
                self._tools.append(tool)
                logger.debug("工具 [%s] 已注册到 ToolRegistry", tool.name)
            except Exception as exc:
                logger.warning(
                    "工具 [%s] 注册失败（可能名称冲突）: %s",
                    tool.name, exc,
                )

    def __repr__(self) -> str:
        return (
            f"MCPManager(servers={len(self._clients)}/{len(self._configs)}, "
            f"tools={len(self._tools)}, started={self._started})"
        )
