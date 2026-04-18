"""Configuration settings for Kaala."""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Application settings."""

    database_path: str = "kaala.db"
    default_model: str = "GEMINI-2-FLASH-LITE"
    poll_interval: int = 60
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls(
            database_path=os.getenv("KAALA_DB_PATH", "kaala.db"),
            default_model=os.getenv("KAALA_DEFAULT_MODEL", "GEMINI-2-FLASH-LITE"),
            poll_interval=int(os.getenv("KAALA_POLL_INTERVAL", "60")),
            host=os.getenv("KAALA_HOST", "127.0.0.1"),
            port=int(os.getenv("KAALA_PORT", "8000")),
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings