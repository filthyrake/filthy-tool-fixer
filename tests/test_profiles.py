"""Tests for profile loading and matching."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from filthy_tool_fixer.profiles.loader import ProfileLoader


class TestProfileLoading:
    def test_load_from_project_profiles(self):
        loader = ProfileLoader("profiles")
        profile = loader.match("qwen3:30b-a3b")
        assert profile.pattern == "qwen3:*"
        assert profile.max_retries == 3
        assert profile.tool_calling.strip_thinking is True
        assert profile.escalation.enabled is True
        assert profile.escalation.model == "qwen3:235b-a22b"

    def test_default_fallback(self):
        loader = ProfileLoader("profiles")
        profile = loader.match("llama3:70b")
        assert profile.pattern == "*"
        assert profile.escalation.enabled is False

    def test_fnmatch_pattern(self):
        loader = ProfileLoader("profiles")
        # 235B matches the more specific 235B profile
        assert loader.match("qwen3:235b-a22b").pattern == "qwen3:235b*"
        # Other qwen3 models match the general qwen3 profile
        assert loader.match("qwen3:0.5b").pattern == "qwen3:*"
        assert loader.match("qwen3:30b-a3b").pattern == "qwen3:*"

    def test_235b_profile_routes_to_escalation_backend(self):
        loader = ProfileLoader("profiles")
        profile = loader.match("qwen3:235b-a22b")
        assert profile.backend_url == "http://localhost:11435"
        assert profile.backend_timeout == 600.0
        assert profile.max_retries == 1
        assert profile.escalation.enabled is False

    def test_30b_uses_default_backend(self):
        loader = ProfileLoader("profiles")
        profile = loader.match("qwen3:30b-a3b")
        assert profile.backend_url == ""  # Uses default primary
        assert profile.escalation.enabled is True

    def test_missing_profiles_dir(self):
        loader = ProfileLoader("/nonexistent/path")
        # Should return defaults without crashing
        profile = loader.match("anything")
        assert profile.max_retries == 3

    def test_custom_toml(self, tmp_path):
        toml_content = """
[model]
pattern = "custom:*"
max_retries = 5
request_timeout = 30.0
backend_timeout = 60.0

[tool_calling]
system_suffix = "Custom suffix"
temperature_override = 0.5
strip_thinking = false
keep_alive = "10m"

[escalation]
enabled = false
"""
        (tmp_path / "custom.toml").write_text(toml_content)
        # Also create a _default.toml so the loader has a fallback
        (tmp_path / "_default.toml").write_text("""
[model]
pattern = "*"
max_retries = 1
""")

        loader = ProfileLoader(str(tmp_path))
        p = loader.match("custom:7b")
        assert p.max_retries == 5
        assert p.request_timeout == 30.0
        assert p.tool_calling.system_suffix == "Custom suffix"
        assert p.tool_calling.temperature_override == 0.5

        # Non-matching falls to default
        d = loader.match("other-model")
        assert d.max_retries == 1


class TestGLMProfiles:
    """Tests for GLM-4.7 model profiles."""

    def test_glm_flash_matches(self):
        loader = ProfileLoader("profiles")
        profile = loader.match("glm-4.7-flash:latest")
        assert profile.pattern == "glm-4.7-flash:*"

    def test_glm_flash_settings(self):
        loader = ProfileLoader("profiles")
        profile = loader.match("glm-4.7-flash:latest")
        assert profile.max_retries == 3
        assert profile.backend_url == ""  # GPU, uses default primary
        assert profile.tool_calling.strip_thinking is True
        assert profile.tool_calling.condense_tools is True
        assert profile.tool_calling.num_ctx == 65536
        assert profile.tool_calling.accept_text_after_tool_use is True

    def test_glm_flash_no_escalation(self):
        """GLM-4.7 quality tier is cloud-only on Ollama, so no escalation."""
        loader = ProfileLoader("profiles")
        profile = loader.match("glm-4.7-flash:latest")
        assert profile.escalation.enabled is False


class TestDevstralProfiles:
    """Tests for Devstral Small 2 model profiles."""

    def test_devstral_matches(self):
        loader = ProfileLoader("profiles")
        profile = loader.match("devstral-small-2:latest")
        assert profile.pattern == "devstral-small-2:*"

    def test_devstral_settings(self):
        loader = ProfileLoader("profiles")
        profile = loader.match("devstral-small-2:latest")
        assert profile.max_retries == 3
        assert profile.backend_url == ""  # GPU, uses default primary
        assert profile.tool_calling.strip_thinking is False
        assert profile.tool_calling.condense_tools is True
        assert profile.tool_calling.num_ctx == 65536
        assert profile.tool_calling.accept_text_after_tool_use is True

    def test_devstral_no_escalation(self):
        """No larger Mistral model available on the server."""
        loader = ProfileLoader("profiles")
        profile = loader.match("devstral-small-2:latest")
        assert profile.escalation.enabled is False
