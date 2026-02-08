"""ModelProfile dataclass for model-specific configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EscalationConfig:
    enabled: bool = False
    model: str = ""
    backend_url: str = ""
    timeout: float = 180.0


@dataclass
class ToolCallingConfig:
    system_suffix: str = ""
    temperature_override: float | None = 0.0
    strip_thinking: bool = False
    think_tag_pattern: str = "<think>.*?</think>"  # Regex pattern for thinking tags
    keep_alive: str = "5m"
    condense_tools: bool = False
    condense_system_prompt: bool = False
    max_system_tokens: int = 0  # 0 = no limit


@dataclass
class ModelProfile:
    pattern: str = "*"
    max_retries: int = 3
    request_timeout: float = 45.0
    backend_timeout: float = 120.0
    backend_url: str = ""  # Override primary backend URL (empty = use default)
    tool_calling: ToolCallingConfig = field(default_factory=ToolCallingConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
