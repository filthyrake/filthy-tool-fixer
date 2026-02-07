"""Tests for proxy orchestrator — request enhancement, routing, think tag stripping."""

from __future__ import annotations

import pytest

from filthyllm.models import ChatCompletionResponse, ChatMessage
from filthyllm.profiles.types import EscalationConfig, ModelProfile, ToolCallingConfig
from filthyllm.proxy import ProxyOrchestrator

from tests.helpers import MockBackend, make_request, make_response, make_tool_call


class TestRequestEnhancement:
    @pytest.fixture
    def orchestrator(self):
        backend = MockBackend([make_response(content="hello")])
        return ProxyOrchestrator(primary_backend=backend), backend

    @pytest.mark.asyncio
    async def test_system_suffix_injected(self, sample_tools):
        good_response = make_response(
            tool_calls=[make_tool_call("search_files", {"query": "test"})]
        )
        backend = MockBackend([good_response])
        orch = ProxyOrchestrator(primary_backend=backend)

        profile = ModelProfile(
            tool_calling=ToolCallingConfig(system_suffix="Call tools directly."),
            escalation=EscalationConfig(enabled=False),
        )

        request = make_request(
            tools=sample_tools,
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Find files"),
            ],
        )

        await orch.handle_request(request, profile)

        # The backend should have received the enhanced system message
        sent = backend.requests[0]
        assert "Call tools directly." in sent.messages[0].content

    @pytest.mark.asyncio
    async def test_system_suffix_creates_system_message_if_missing(self, sample_tools):
        good_response = make_response(
            tool_calls=[make_tool_call("search_files", {"query": "test"})]
        )
        backend = MockBackend([good_response])
        orch = ProxyOrchestrator(primary_backend=backend)

        profile = ModelProfile(
            tool_calling=ToolCallingConfig(system_suffix="Call tools directly."),
            escalation=EscalationConfig(enabled=False),
        )

        request = make_request(
            tools=sample_tools,
            messages=[ChatMessage(role="user", content="Find files")],
        )

        await orch.handle_request(request, profile)
        sent = backend.requests[0]
        assert sent.messages[0].role == "system"
        assert sent.messages[0].content == "Call tools directly."

    @pytest.mark.asyncio
    async def test_temperature_override(self, sample_tools):
        good_response = make_response(
            tool_calls=[make_tool_call("search_files", {"query": "test"})]
        )
        backend = MockBackend([good_response])
        orch = ProxyOrchestrator(primary_backend=backend)

        profile = ModelProfile(
            tool_calling=ToolCallingConfig(temperature_override=0.1),
            escalation=EscalationConfig(enabled=False),
        )

        request = make_request(tools=sample_tools)
        request.temperature = 0.7  # Client set 0.7, but profile overrides to 0.1

        await orch.handle_request(request, profile)
        sent = backend.requests[0]
        assert sent.temperature == 0.1

    @pytest.mark.asyncio
    async def test_no_enhancement_without_tools(self):
        backend = MockBackend([make_response(content="hello")])
        orch = ProxyOrchestrator(primary_backend=backend)

        profile = ModelProfile(
            tool_calling=ToolCallingConfig(
                system_suffix="This should NOT be added",
                temperature_override=0.0,
            ),
        )

        request = make_request(tools=None)  # No tools
        request.temperature = 0.7

        await orch.handle_request(request, profile)
        sent = backend.requests[0]
        # Temperature should NOT be overridden for non-tool requests
        assert sent.temperature == 0.7


class TestThinkTagStripping:
    @pytest.mark.asyncio
    async def test_strips_think_tags(self):
        response = make_response(
            content="<think>Let me think about this...</think>Here is my answer."
        )
        backend = MockBackend([response])
        orch = ProxyOrchestrator(primary_backend=backend)

        profile = ModelProfile(
            tool_calling=ToolCallingConfig(strip_thinking=True),
        )

        request = make_request(tools=None)
        result = await orch.handle_request(request, profile)
        assert "<think>" not in result.choices[0].message.content
        assert result.choices[0].message.content == "Here is my answer."

    @pytest.mark.asyncio
    async def test_no_stripping_when_disabled(self):
        response = make_response(
            content="<think>Thinking</think>Answer"
        )
        backend = MockBackend([response])
        orch = ProxyOrchestrator(primary_backend=backend)

        profile = ModelProfile(
            tool_calling=ToolCallingConfig(strip_thinking=False),
        )

        request = make_request(tools=None)
        result = await orch.handle_request(request, profile)
        assert "<think>" in result.choices[0].message.content


class TestNonToolPassthrough:
    @pytest.mark.asyncio
    async def test_non_streaming_passthrough(self):
        backend = MockBackend([make_response(content="Hello!")])
        orch = ProxyOrchestrator(primary_backend=backend)

        profile = ModelProfile()
        request = make_request(tools=None)

        result = await orch.handle_request(request, profile)
        assert isinstance(result, ChatCompletionResponse)
        assert result.choices[0].message.content == "Hello!"

    @pytest.mark.asyncio
    async def test_streaming_passthrough(self):
        backend = MockBackend([])
        orch = ProxyOrchestrator(primary_backend=backend)

        profile = ModelProfile()
        request = make_request(tools=None)
        request.stream = True

        result = await orch.handle_request(request, profile)
        # Should return an async iterator for streaming
        assert hasattr(result, "__aiter__")


class TestBackendRouting:
    @pytest.mark.asyncio
    async def test_profile_routes_to_correct_backend(self, sample_tools):
        """When profile has backend_url matching escalation, use that backend."""
        primary = MockBackend([make_response(content="primary")])
        escalation = MockBackend([
            make_response(tool_calls=[make_tool_call("search_files", {"query": "test"})])
        ])
        orch = ProxyOrchestrator(
            primary_backend=primary,
            escalation_backend=escalation,
            primary_url="http://localhost:11434",
            escalation_url="http://localhost:11435",
        )

        profile = ModelProfile(
            backend_url="http://localhost:11435",
            escalation=EscalationConfig(enabled=False),
        )
        request = make_request(tools=sample_tools)

        await orch.handle_request(request, profile)
        # Request should have gone to escalation backend, not primary
        assert len(escalation.requests) == 1
        assert len(primary.requests) == 0

    @pytest.mark.asyncio
    async def test_default_backend_when_no_override(self, sample_tools):
        """When profile has no backend_url, use primary backend."""
        primary = MockBackend([
            make_response(tool_calls=[make_tool_call("search_files", {"query": "test"})])
        ])
        escalation = MockBackend([])
        orch = ProxyOrchestrator(
            primary_backend=primary,
            escalation_backend=escalation,
            primary_url="http://localhost:11434",
            escalation_url="http://localhost:11435",
        )

        profile = ModelProfile(escalation=EscalationConfig(enabled=False))
        request = make_request(tools=sample_tools)

        await orch.handle_request(request, profile)
        assert len(primary.requests) == 1
        assert len(escalation.requests) == 0

    @pytest.mark.asyncio
    async def test_passthrough_uses_routed_backend(self):
        """Non-tool requests also respect backend routing."""
        primary = MockBackend([])
        escalation = MockBackend([make_response(content="from escalation")])
        orch = ProxyOrchestrator(
            primary_backend=primary,
            escalation_backend=escalation,
            primary_url="http://localhost:11434",
            escalation_url="http://localhost:11435",
        )

        profile = ModelProfile(backend_url="http://localhost:11435")
        request = make_request(tools=None)

        result = await orch.handle_request(request, profile)
        assert result.choices[0].message.content == "from escalation"
        assert len(escalation.requests) == 1
        assert len(primary.requests) == 0


class TestModelNamePreservation:
    @pytest.mark.asyncio
    async def test_original_model_name_in_response(self, sample_tools):
        good_response = make_response(
            tool_calls=[make_tool_call("search_files", {"query": "test"})],
            model="qwen3:30b-a3b",
        )
        backend = MockBackend([good_response])
        orch = ProxyOrchestrator(primary_backend=backend)

        profile = ModelProfile(escalation=EscalationConfig(enabled=False))
        request = make_request(tools=sample_tools, model="my-custom-alias")

        response, headers = await orch.handle_request(request, profile)
        assert response.model == "my-custom-alias"
