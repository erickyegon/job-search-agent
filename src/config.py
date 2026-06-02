"""Configuration management — loads env vars and YAML configs."""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

CONFIG_DIR = Path(__file__).parent.parent / "config"


@dataclass
class Config:
    # API keys (from environment)
    anthropic_api_key: str = ""
    resend_api_key: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Paths
    db_path: str = "jobs.db"
    config_dir: Path = CONFIG_DIR

    # Runtime
    dry_run: bool = False
    notify_email: bool = True
    notify_telegram: bool = True
    recipient_email: str = "keyegon@gmail.com"

    # Claude model
    claude_model: str = "claude-haiku-4-5-20251001"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            resend_api_key=os.environ.get("RESEND_API_KEY", ""),
            telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
            db_path=os.environ.get("DB_PATH", "jobs.db"),
            dry_run=os.environ.get("DRY_RUN", "false").lower() == "true",
            recipient_email=os.environ.get("RECIPIENT_EMAIL", "keyegon@gmail.com"),
        )

    def validate(self) -> list[str]:
        errors = []
        if not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required")
        if self.notify_email and not self.resend_api_key:
            errors.append("RESEND_API_KEY is required for email notifications")
        if self.notify_telegram and not self.telegram_bot_token:
            errors.append("TELEGRAM_BOT_TOKEN is required for Telegram notifications")
        if self.notify_telegram and not self.telegram_chat_id:
            errors.append("TELEGRAM_CHAT_ID is required for Telegram notifications")
        return errors


def load_yaml(filename: str) -> dict:
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_profile() -> dict:
    return load_yaml("profile.yaml")


def load_search_queries() -> dict:
    return load_yaml("search_queries.yaml")


def load_trusted_domains() -> dict:
    data = load_yaml("trusted_domains.yaml")
    # Flatten all domain lists into a single set
    all_domains = set()
    for key in ("ats_domains", "job_board_domains", "company_domains"):
        all_domains.update(data.get(key, []))
    return {"all": all_domains, **data}
