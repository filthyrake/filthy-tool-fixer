"""TOML profile loading with model name matching."""

from __future__ import annotations

import tomllib
from fnmatch import fnmatch
from pathlib import Path

from filthyllm.logging import get_logger
from filthyllm.profiles.types import EscalationConfig, ModelProfile, ToolCallingConfig

log = get_logger(__name__)


class ProfileLoader:
    def __init__(self, profiles_dir: str | Path) -> None:
        self._profiles: list[tuple[str, ModelProfile]] = []
        self._default: ModelProfile = ModelProfile()
        self._load(Path(profiles_dir))

    def _load(self, directory: Path) -> None:
        if not directory.is_dir():
            log.warning("profiles_dir_missing", path=str(directory))
            return

        for path in sorted(directory.glob("*.toml")):
            try:
                with open(path, "rb") as f:
                    data = tomllib.load(f)
                profile = self._parse(data)
                if path.stem == "_default":
                    self._default = profile
                    log.info("profile_loaded", name="_default")
                else:
                    self._profiles.append((profile.pattern, profile))
                    log.info("profile_loaded", name=path.stem, pattern=profile.pattern)
            except Exception:
                log.exception("profile_load_error", path=str(path))

    def _parse(self, data: dict) -> ModelProfile:
        model_data = data.get("model", {})
        tc_data = data.get("tool_calling", {})
        esc_data = data.get("escalation", {})

        return ModelProfile(
            pattern=model_data.get("pattern", "*"),
            max_retries=model_data.get("max_retries", 3),
            request_timeout=model_data.get("request_timeout", 45.0),
            backend_timeout=model_data.get("backend_timeout", 120.0),
            backend_url=model_data.get("backend_url", ""),
            tool_calling=ToolCallingConfig(
                system_suffix=tc_data.get("system_suffix", ""),
                temperature_override=tc_data.get("temperature_override", 0.0),
                strip_thinking=tc_data.get("strip_thinking", False),
                think_tag_pattern=tc_data.get("think_tag_pattern", "<think>.*?</think>"),
                keep_alive=tc_data.get("keep_alive", "5m"),
                condense_tools=tc_data.get("condense_tools", False),
                condense_system_prompt=tc_data.get("condense_system_prompt", False),
                max_system_tokens=tc_data.get("max_system_tokens", 0),
            ),
            escalation=EscalationConfig(
                enabled=esc_data.get("enabled", False),
                model=esc_data.get("model", ""),
                backend_url=esc_data.get("backend_url", ""),
                timeout=esc_data.get("timeout", 180.0),
            ),
        )

    def match(self, model_name: str) -> ModelProfile:
        for pattern, profile in self._profiles:
            if fnmatch(model_name, pattern):
                log.debug("profile_matched", model=model_name, pattern=pattern)
                return profile
        log.debug("profile_default", model=model_name)
        return self._default
