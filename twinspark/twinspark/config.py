"""Centralized configuration for TwinSpark.

Loads settings from environment variables and an optional ``.env`` file.
No secrets are ever hard-coded; the DashScope API key must be provided by the
environment (or a local ``.env`` file that is git-ignored).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load a local .env file (if present) before Settings reads the environment.
# ``override=False`` keeps any values already exported in the shell.
load_dotenv(override=False)


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"
DEFAULT_DB_PATH = "~/.twinspark/state.db"
DEFAULT_SKILLS_DIR = "~/.twinspark/skills/"


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
    dingtalk_app_secret: str = Field("", alias="DINGTALK_APP_SECRET")
    dingtalk_verify_signature: bool = Field(True, alias="DINGTALK_VERIFY_SIGNATURE")

    # DingTalk AI Card (optional, requires app_key + app_secret)
    dingtalk_app_key: str = Field("", alias="DINGTALK_APP_KEY")
    dingtalk_card_template_id: str = Field("", alias="DINGTALK_CARD_TEMPLATE_ID")
    dingtalk_message_mode: str = Field("text", alias="DINGTALK_MESSAGE_MODE")
    dingtalk_stream_interval_ms: int = Field(500, alias="DINGTALK_STREAM_INTERVAL_MS")

    @field_validator("db_path", "skills_dir", mode="before")
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
