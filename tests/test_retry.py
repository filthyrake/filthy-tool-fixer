"""Tests for retry loop, feedback, and escalation."""

from __future__ import annotations

import time

import pytest

from filthy_tool_fixer.models import ChatMessage, ToolDefinition
from filthy_tool_fixer.profiles.types import EscalationConfig, ModelProfile, ToolCallingConfig
from filthy_tool_fixer.retry.feedback import build_feedback_message
from filthy_tool_fixer.retry.loop import RetryLoop
from filthy_tool_fixer.validation.schema import ValidationError, ValidationResult

from tests.helpers import MockBackend, make_request, make_response, make_tool_call


class TestFeedbackMessage:
    def test_single_error_feedback(self):
        result = ValidationResult(
            valid=False,
            errors=[
                ValidationError(
                    tool_call_index=0,
                    error_type="unknown_tool",
                    message="Tool 'serch_files' does not exist. Did you mean 'search_files'?",
                    severity=3,
                )
            ],
        )
        msg = build_feedback_message(result)
        assert msg.role == "user"
        assert "serch_files" in msg.content
        assert "search_files" in msg.content

    def test_multiple_errors_feedback(self):
        result = ValidationResult(
            valid=False,
            errors=[
                ValidationError(0, "unknown_tool", "Tool 'x' does not exist.", 3),
                ValidationError(0, "extra_field", "Extra param 'y'.", 1),
            ],
        )
        msg = build_feedback_message(result)
        assert "2 errors" in msg.content
        assert "Most critical" in msg.content

    def test_raises_for_valid_result(self):
        with pytest.raises(ValueError):
            build_feedback_message(ValidationResult(valid=True))


class TestRetryLoop:
    @pytest.fixture
    def profile(self):
        return ModelProfile(
            max_retries=2,
            request_timeout=45.0,
            backend_timeout=120.0,
            tool_calling=ToolCallingConfig(),
            escalation=EscalationConfig(enabled=False),
        )

    @pytest.fixture
    def profile_with_escalation(self):
        return ModelProfile(
            max_retries=1,
            request_timeout=45.0,
            backend_timeout=120.0,
            tool_calling=ToolCallingConfig(),
            escalation=EscalationConfig(
                enabled=True,
                model="qwen3:235b-a22b",
                backend_url="http://localhost:11435",
                timeout=180.0,
            ),
        )

    @pytest.mark.asyncio
    async def test_valid_on_first_try(self, sample_tools, profile):
        good_response = make_response(
            tool_calls=[make_tool_call("search_files", {"query": "test"})]
        )
        backend = MockBackend([good_response])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=sample_tools)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        assert backend.call_count == 1
        assert "X-FilthyToolFixer-Degraded" not in headers

    @pytest.mark.asyncio
    async def test_retry_on_invalid_then_succeed(self, sample_tools, profile):
        bad_response = make_response(
            tool_calls=[make_tool_call("serch_files", {"query": "test"})]  # typo
        )
        good_response = make_response(
            tool_calls=[make_tool_call("search_files", {"query": "test"})]
        )
        backend = MockBackend([bad_response, good_response])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=sample_tools)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        assert backend.call_count == 2
        assert "X-FilthyToolFixer-Degraded" not in headers
        # Second request should contain feedback
        second_req = backend.requests[1]
        assert any("does not exist" in (m.content or "") for m in second_req.messages)

    @pytest.mark.asyncio
    async def test_text_response_nudge_then_passthrough(self, sample_tools, profile):
        # Model responds with text twice — first gets nudged, second passes through
        text1 = make_response(content="I would search for files...")
        text2 = make_response(content="Here is the answer directly.")
        backend = MockBackend([text1, text2])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=sample_tools)  # tool_choice defaults to None (optional)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        assert backend.call_count == 2  # One nudge, then accept text
        assert response.choices[0].message.content == "Here is the answer directly."

    @pytest.mark.asyncio
    async def test_text_response_nudge_triggers_tool_call(self, sample_tools, profile):
        # Model narrates first, then uses tools after nudge
        narration = make_response(content="I could search for that...")
        good_response = make_response(
            tool_calls=[make_tool_call("search_files", {"query": "test"})]
        )
        backend = MockBackend([narration, good_response])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=sample_tools)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        assert backend.call_count == 2
        assert response.choices[0].message.tool_calls is not None

    @pytest.mark.asyncio
    async def test_narration_feedback_when_tools_required(self, sample_tools, profile):
        # Model narrates instead of calling tools, but tool_choice=required
        narration = make_response(content="I would search for files using search_files...")
        good_response = make_response(
            tool_calls=[make_tool_call("search_files", {"query": "test"})]
        )
        backend = MockBackend([narration, good_response])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=sample_tools)
        request.tool_choice = "required"
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        assert backend.call_count == 2
        # Check narration feedback was sent
        second_req = backend.requests[1]
        assert any("described what you would do" in (m.content or "") for m in second_req.messages)

    @pytest.mark.asyncio
    async def test_duplicate_detection_breaks_loop(self, sample_tools, profile):
        # Same bad response every time → should stop early
        bad_response = make_response(
            tool_calls=[make_tool_call("serch_files", {"query": "test"})]
        )
        backend = MockBackend([bad_response, bad_response, bad_response])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=sample_tools)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        # Should stop after detecting duplicate, not exhaust all retries
        assert backend.call_count == 2
        assert headers.get("X-FilthyToolFixer-Degraded") == "true"

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self, sample_tools, profile):
        bad1 = make_response(tool_calls=[make_tool_call("bad1", {"query": "test"})])
        bad2 = make_response(tool_calls=[make_tool_call("bad2", {"query": "test"})])
        bad3 = make_response(tool_calls=[make_tool_call("bad3", {"query": "test"})])
        backend = MockBackend([bad1, bad2, bad3])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=sample_tools)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        assert backend.call_count == 3  # initial + 2 retries
        assert headers.get("X-FilthyToolFixer-Degraded") == "true"


class TestEscalation:
    @pytest.mark.asyncio
    async def test_escalation_on_primary_failure(self, sample_tools):
        profile = ModelProfile(
            max_retries=0,  # no retries, go straight to escalation
            request_timeout=45.0,
            backend_timeout=120.0,
            tool_calling=ToolCallingConfig(),
            escalation=EscalationConfig(
                enabled=True,
                model="qwen3:235b-a22b",
                backend_url="http://localhost:11435",
                timeout=180.0,
            ),
        )

        bad_response = make_response(
            tool_calls=[make_tool_call("serch_files", {"query": "test"})]
        )
        good_response = make_response(
            tool_calls=[make_tool_call("search_files", {"query": "test"})]
        )

        primary = MockBackend([bad_response])
        escalation = MockBackend([good_response])
        loop = RetryLoop(backend=primary, escalation_backend=escalation, profile=profile)

        request = make_request(tools=sample_tools)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        assert primary.call_count == 1
        assert escalation.call_count == 1
        assert headers.get("X-FilthyToolFixer-Escalated") == "true"
        assert headers.get("X-FilthyToolFixer-Model") == "qwen3:235b-a22b"

    @pytest.mark.asyncio
    async def test_escalation_also_fails(self, sample_tools):
        profile = ModelProfile(
            max_retries=0,
            request_timeout=45.0,
            backend_timeout=120.0,
            tool_calling=ToolCallingConfig(),
            escalation=EscalationConfig(
                enabled=True,
                model="qwen3:235b-a22b",
                backend_url="http://localhost:11435",
                timeout=180.0,
            ),
        )

        bad_response = make_response(
            tool_calls=[make_tool_call("fake_tool", {"query": "test"})]
        )

        primary = MockBackend([bad_response])
        escalation = MockBackend([bad_response])
        loop = RetryLoop(backend=primary, escalation_backend=escalation, profile=profile)

        request = make_request(tools=sample_tools)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        assert headers.get("X-FilthyToolFixer-Degraded") == "true"

    @pytest.mark.asyncio
    async def test_no_escalation_when_disabled(self, sample_tools):
        profile = ModelProfile(
            max_retries=0,
            request_timeout=45.0,
            backend_timeout=120.0,
            tool_calling=ToolCallingConfig(),
            escalation=EscalationConfig(enabled=False),
        )

        bad_response = make_response(
            tool_calls=[make_tool_call("fake_tool", {"query": "test"})]
        )

        primary = MockBackend([bad_response])
        escalation = MockBackend([])
        loop = RetryLoop(backend=primary, escalation_backend=escalation, profile=profile)

        request = make_request(tools=sample_tools)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        assert escalation.call_count == 0
        assert headers.get("X-FilthyToolFixer-Degraded") == "true"


class TestBestAttemptScoring:
    @pytest.mark.asyncio
    async def test_best_attempt_selected(self, sample_tools):
        """When all attempts fail, the best one should be returned."""
        profile = ModelProfile(
            max_retries=1,
            request_timeout=45.0,
            backend_timeout=120.0,
            tool_calling=ToolCallingConfig(),
            escalation=EscalationConfig(enabled=False),
        )

        # First: has valid JSON in tool calls but wrong tool name (score 5)
        better = make_response(
            tool_calls=[make_tool_call("fake_tool", {"query": "test"})]
        )
        # Second: narrates (score 1)
        worse = make_response(content="I would do something")

        primary = MockBackend([better, worse])
        loop = RetryLoop(backend=primary, profile=profile)

        request = make_request(tools=sample_tools)
        request.tool_choice = "required"
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        # Should return the better response (with tool calls)
        assert response.choices[0].message.tool_calls is not None
