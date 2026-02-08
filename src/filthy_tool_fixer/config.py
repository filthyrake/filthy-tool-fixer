"""Application configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "FILTHY_"}

    # Proxy server
    host: str = "0.0.0.0"
    port: int = 8079

    # Primary backend (fast model)
    backend_url: str = "http://localhost:11434"
    backend_type: str = "ollama"

    # Escalation backend (quality model)
    escalation_backend_url: str = "http://localhost:11435"
    escalation_backend_type: str = "ollama"

    # Timeouts (seconds)
    request_timeout: float = 45.0
    backend_timeout: float = 120.0
    escalation_timeout: float = 180.0

    # Concurrency
    max_concurrent_tool_requests: int = 3

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "console"

    # Profiles directory
    profiles_dir: str = "profiles"


settings = Settings()
