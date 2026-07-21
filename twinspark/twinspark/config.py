"""Centralized configuration for TwinSpark.

Loads settings from environment variables and an optional ``.env`` file.
No secrets are ever hard-coded; the DashScope API key must be provided by the
environment (or a local ``.env`` file that is git-ignored).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load a local .env file (if present) before Settings reads the environment.
# ``override=False`` keeps any values already exported in the shell.
load_dotenv(override=False)


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"
DEFAULT_DB_PATH = "~/.twinspark/state.db"
DEFAULT_SKILLS_DIR = "~/.twinspark/skills/"


class MCPServerConfig(BaseModel):
    """单个 MCP 服务器的连接配置。

    支持两种传输方式：
    - stdio: 通过子进程标准输入输出通信，需指定 command 和 args。
    - sse:   通过 Server-Sent Events HTTP 连接通信，需指定 url。

    Attributes:
        name: 服务器唯一标识名称，用于日志和工具命名空间。
        transport: 传输方式，仅允许 "stdio" 或 "sse"。
        command: stdio 模式下要执行的可执行文件路径。
        args: 传递给可执行文件的命令行参数列表。
        env: 启动子进程时附加的环境变量（合并到当前环境）。
        url: SSE 模式下服务器的 HTTP(S) 端点 URL。
        timeout: 连接超时时间（秒），超时后视为不可达。
    """

    name: str
    transport: str  # "stdio" 或 "sse"
    command: str = ""
    args: list[str] = []
    env: dict[str, str] = {}
    url: str = ""
    timeout: int = 60


class Config(BaseSettings):
    """Runtime configuration for TwinSpark.

    Attributes:
        dashscope_api_key: DashScope (Bailian) API key. Required, no default.
        base_url: OpenAI-compatible endpoint base URL.
        model: Default model name to use for completions.
        db_path: Path to the local SQLite state database (expanded).
        skills_dir: Directory holding user skills (expanded).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Allow a field named ``model`` without colliding with pydantic internals.
        protected_namespaces=(),
    )

    dashscope_api_key: str = Field(..., alias="DASHSCOPE_API_KEY")
    base_url: str = Field(DEFAULT_BASE_URL, alias="TWINSPARK_BASE_URL")
    model: str = Field(DEFAULT_MODEL, alias="TWINSPARK_MODEL")
    db_path: Path = Field(Path(DEFAULT_DB_PATH), alias="TWINSPARK_DB_PATH")
    skills_dir: Path = Field(Path(DEFAULT_SKILLS_DIR), alias="TWINSPARK_SKILLS_DIR")

    # DingTalk integration (optional)
    dingtalk_mode: str = Field("disabled", alias="DINGTALK_MODE")
    # Possible values: "disabled" | "stream" | "webhook"
    # - disabled: DingTalk integration off
    # - stream:   Stream mode (recommended, no public URL needed)
    # - webhook:  Webhook mode (requires public HTTPS + ICP filing)
    dingtalk_app_secret: str = Field("", alias="DINGTALK_APP_SECRET")
    dingtalk_verify_signature: bool = Field(True, alias="DINGTALK_VERIFY_SIGNATURE")

    # DingTalk AI Card (optional, requires app_key + app_secret)
    dingtalk_app_key: str = Field("", alias="DINGTALK_APP_KEY")
    dingtalk_card_template_id: str = Field("", alias="DINGTALK_CARD_TEMPLATE_ID")
    dingtalk_message_mode: str = Field("text", alias="DINGTALK_MESSAGE_MODE")
    dingtalk_stream_interval_ms: int = Field(500, alias="DINGTALK_STREAM_INTERVAL_MS")

    # MCP（Model Context Protocol）集成配置
    mcp_enabled: bool = Field(False, alias="TWINSPARK_MCP_ENABLED")
    mcp_config_path: Path = Field(
        Path("~/.twinspark/mcp-servers.yaml"),
        alias="TWINSPARK_MCP_CONFIG_PATH",
    )

    @field_validator("db_path", "skills_dir", "mcp_config_path", mode="before")
    @classmethod
    def _expand_paths(cls, value: object) -> Path:
        """Expand ``~`` and environment variables into absolute paths."""
        raw = os.path.expandvars(str(value))
        return Path(raw).expanduser()

    def ensure_directories(self) -> None:
        """Ensure parent directories for configured paths exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the process-wide configuration singleton.

    On first call this reads the environment / ``.env`` file, expands paths,
    and creates any missing parent directories. Subsequent calls return the
    cached instance.

    Raises:
        pydantic.ValidationError: If ``DASHSCOPE_API_KEY`` is not set.
    """
    config = Config()
    config.ensure_directories()
    return config


def load_mcp_servers(config_path: Path) -> list[MCPServerConfig]:
    """从 YAML 配置文件加载 MCP 服务器列表。

    配置文件格式示例（~/.twinspark/mcp-servers.yaml）：

        servers:
          filesystem:
            transport: stdio
            command: npx
            args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
          web-search:
            transport: sse
            url: http://localhost:8080/sse
            timeout: 30

    Args:
        config_path: YAML 配置文件的绝对路径（已展开 ~）。

    Returns:
        解析后的 MCPServerConfig 列表；文件不存在或无 servers 键时返回空列表。
    """
    # 文件不存在时静默返回空列表，避免未配置 MCP 时报错
    if not config_path.exists():
        return []

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    servers_dict: dict[str, Any] = raw.get("servers", {})
    if not isinstance(servers_dict, dict):
        return []

    result: list[MCPServerConfig] = []
    for name, server_cfg in servers_dict.items():
        if not isinstance(server_cfg, dict):
            continue
        # 将字典键名注入为 name 字段
        server_cfg["name"] = name
        result.append(MCPServerConfig(**server_cfg))
    return result
