"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from filthy_tool_fixer.models import FunctionDefinition, ToolDefinition
from filthy_tool_fixer.profiles.types import EscalationConfig, ModelProfile, ToolCallingConfig


@pytest.fixture
def search_tool() -> ToolDefinition:
    return ToolDefinition(
        type="function",
        function=FunctionDefinition(
            name="search_files",
            description="Search for files matching a pattern",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results to return"},
                    "include_hidden": {"type": "boolean", "description": "Include hidden files"},
                },
                "required": ["query"],
            },
        ),
    )


@pytest.fixture
def write_tool() -> ToolDefinition:
    return ToolDefinition(
        type="function",
        function=FunctionDefinition(
            name="write_file",
            description="Write content to a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        ),
    )


@pytest.fixture
def sample_tools(search_tool, write_tool) -> list[ToolDefinition]:
    return [search_tool, write_tool]


@pytest.fixture
def default_profile() -> ModelProfile:
    return ModelProfile(
        max_retries=3,
        request_timeout=45.0,
        backend_timeout=120.0,
        tool_calling=ToolCallingConfig(
            system_suffix="Use tools directly.",
            temperature_override=0.0,
            strip_thinking=True,
            keep_alive="5m",
        ),
        escalation=EscalationConfig(
            enabled=True,
            model="qwen3:235b-a22b",
            backend_url="http://localhost:11435",
            timeout=180.0,
        ),
    )
