"""Tests for retry loop, feedback, and escalation."""

from __future__ import annotations

import time

import pytest

from filthy_tool_fixer.models import FunctionDefinition, ToolDefinition
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
        # Model responds with text on every attempt — retries all exhausted,
        # then accepts text since tools are optional
        text1 = make_response(content="I would search for files...")
        text2 = make_response(content="Let me explain...")
        text3 = make_response(content="Here is the answer directly.")
        text4 = make_response(content="Final answer.")
        backend = MockBackend([text1, text2, text3, text4])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=sample_tools)  # tool_choice defaults to None (optional)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        # Should use all retries (max_retries=2 → 3 attempts) before accepting text
        assert backend.call_count == 3
        assert response.choices[0].message.content == "Here is the answer directly."

    @pytest.mark.asyncio
    async def test_text_accepted_when_prior_tool_results(self, sample_tools):
        # If profile opts in and conversation already has tool results,
        # a text response is the synthesized answer — accept immediately
        profile = ModelProfile(
            max_retries=2,
            request_timeout=45.0,
            backend_timeout=120.0,
            tool_calling=ToolCallingConfig(accept_text_after_tool_use=True),
            escalation=EscalationConfig(enabled=False),
        )
        text_answer = make_response(content="Based on my analysis, the project uses FastAPI.")
        backend = MockBackend([text_answer])
        loop = RetryLoop(backend=backend, profile=profile)

        # Simulate a conversation with prior tool use
        from filthy_tool_fixer.models import ChatMessage
        messages = [
            ChatMessage(role="user", content="What framework does this project use?"),
            ChatMessage(role="assistant", content="", tool_calls=[
                make_tool_call("read", {"file_path": "pyproject.toml"})
            ]),
            ChatMessage(role="tool", content="[tool result: dependencies include fastapi]"),
        ]
        request = make_request(tools=sample_tools, messages=messages)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        # Should accept text on first attempt — no nudging
        assert backend.call_count == 1
        assert response.choices[0].message.content == "Based on my analysis, the project uses FastAPI."
        assert "X-FilthyToolFixer-Degraded" not in headers

    @pytest.mark.asyncio
    async def test_text_still_nudged_without_prior_tool_results(self, sample_tools):
        # Even with opt-in, without prior tool results text should still be nudged
        profile = ModelProfile(
            max_retries=2,
            request_timeout=45.0,
            backend_timeout=120.0,
            tool_calling=ToolCallingConfig(accept_text_after_tool_use=True),
            escalation=EscalationConfig(enabled=False),
        )
        text1 = make_response(content="I would search for files...")
        good = make_response(tool_calls=[make_tool_call("search_files", {"query": "test"})])
        backend = MockBackend([text1, good])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=sample_tools)  # No tool results in history
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        # Should nudge and get tool call on attempt 2
        assert backend.call_count == 2
        assert response.choices[0].message.tool_calls is not None

    @pytest.mark.asyncio
    async def test_text_nudged_when_profile_does_not_opt_in(self, sample_tools, profile):
        # Default profile (accept_text_after_tool_use=False) should still nudge
        text1 = make_response(content="Here's what I found...")
        good = make_response(tool_calls=[make_tool_call("search_files", {"query": "test"})])
        backend = MockBackend([text1, good])
        loop = RetryLoop(backend=backend, profile=profile)

        from filthy_tool_fixer.models import ChatMessage
        messages = [
            ChatMessage(role="user", content="Find tests"),
            ChatMessage(role="assistant", content="", tool_calls=[
                make_tool_call("read", {"file_path": "pyproject.toml"})
            ]),
            ChatMessage(role="tool", content="[result]"),
        ]
        request = make_request(tools=sample_tools, messages=messages)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        # Profile doesn't opt in — should nudge despite prior tool results
        assert backend.call_count == 2
        assert response.choices[0].message.tool_calls is not None

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
        assert any("MUST call a tool" in (m.content or "") for m in second_req.messages)

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


class TestToolCallRescue:
    @pytest.fixture
    def profile(self):
        return ModelProfile(
            max_retries=2,
            request_timeout=45.0,
            backend_timeout=120.0,
            tool_calling=ToolCallingConfig(),
            escalation=EscalationConfig(enabled=False),
        )

    @pytest.mark.asyncio
    async def test_rescue_narrated_tool_call(self, sample_tools, profile):
        # Model narrates a tool call as JSON text — should be rescued
        narrated = make_response(
            content='{"type": "function", "name": "search_files", "parameters": {"query": "test"}}'
        )
        backend = MockBackend([narrated])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=sample_tools)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        # Should rescue the tool call and validate it successfully
        assert backend.call_count == 1
        assert response.choices[0].message.tool_calls is not None
        assert response.choices[0].message.tool_calls[0].function.name == "search_files"

    @pytest.mark.asyncio
    async def test_rescue_with_code_fence(self, sample_tools, profile):
        narrated = make_response(
            content='```json\n{"name": "search_files", "arguments": {"query": "hello"}}\n```'
        )
        backend = MockBackend([narrated])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=sample_tools)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        assert backend.call_count == 1
        assert response.choices[0].message.tool_calls[0].function.name == "search_files"

    @pytest.mark.asyncio
    async def test_rescue_invalid_still_retries(self, sample_tools, profile):
        # Narrated tool call with wrong name — rescued but fails validation, triggers retry
        narrated = make_response(
            content='{"name": "nonexistent_tool", "parameters": {"x": 1}}'
        )
        good = make_response(
            tool_calls=[make_tool_call("search_files", {"query": "test"})]
        )
        backend = MockBackend([narrated, good])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=sample_tools)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        # Narrated call had unknown tool name — not rescued (name not in tool set)
        # So it falls through to narration nudge, then second attempt succeeds
        assert backend.call_count == 2

    @pytest.mark.asyncio
    async def test_rescue_llama_native_format(self, sample_tools, profile):
        # Llama models emit tool calls as <|python_start|>func(args)<|python_end|>
        narrated = make_response(
            content='<|python_start|>search_files(query="test files")<|python_end|>'
        )
        backend = MockBackend([narrated])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=sample_tools)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        assert backend.call_count == 1
        tc = response.choices[0].message.tool_calls[0]
        assert tc.function.name == "search_files"
        assert '"query"' in tc.function.arguments
        assert "test files" in tc.function.arguments

    @pytest.mark.asyncio
    async def test_rescue_pythonic_single_call(self, sample_tools, profile):
        # Llama 4 pythonic format with one call
        narrated = make_response(
            content='[search_files(query="hello")]'
        )
        backend = MockBackend([narrated])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=sample_tools)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        assert backend.call_count == 1
        tc = response.choices[0].message.tool_calls[0]
        assert tc.function.name == "search_files"
        assert '"hello"' in tc.function.arguments

    @pytest.mark.asyncio
    async def test_rescue_pythonic_three_calls(self, sample_tools, profile):
        # Llama 4 pythonic format with 3 calls — tests the regex handles N calls
        narrated = make_response(
            content='[search_files(query="a"), write_file(path="/tmp/x", content="y"), search_files(query="b")]'
        )
        backend = MockBackend([narrated])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=sample_tools)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        assert backend.call_count == 1
        tcs = response.choices[0].message.tool_calls
        assert len(tcs) == 3
        assert tcs[0].function.name == "search_files"
        assert tcs[1].function.name == "write_file"
        assert tcs[2].function.name == "search_files"

    @pytest.mark.asyncio
    async def test_rescue_narrated_empty_args(self, sample_tools, profile):
        # Tool call with empty arguments dict — should not be clobbered by falsy check
        narrated = make_response(
            content='{"name": "search_files", "arguments": {}, "parameters": {"query": "wrong"}}'
        )
        backend = MockBackend([narrated])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=sample_tools)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        # Should use "arguments" (empty dict), not fall through to "parameters"
        tc = response.choices[0].message.tool_calls[0]
        assert tc.function.arguments == "{}"

    @pytest.mark.asyncio
    async def test_rescue_embedded_json_in_mixed_text(self, sample_tools, profile):
        # Maverick pattern: text explanation + JSON tool call at the end
        narrated = make_response(
            content=(
                "The project appears to be Python-based. Let me check the dependencies.\n\n"
                '{"name": "search_files", "parameters": {"query": "requirements"}}'
            )
        )
        backend = MockBackend([narrated])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=sample_tools)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        # Should rescue the embedded tool call AND keep the text
        assert backend.call_count == 1
        assert response.choices[0].message.tool_calls is not None
        assert response.choices[0].message.tool_calls[0].function.name == "search_files"
        assert "Python-based" in response.choices[0].message.content
        # JSON blob should be removed from content
        assert '{"name"' not in response.choices[0].message.content

    @pytest.mark.asyncio
    async def test_no_rescue_for_plain_text(self, sample_tools, profile):
        # Regular text response — should NOT be rescued
        text1 = make_response(content="I can help you with that!")
        text2 = make_response(content="Sure, let me explain.")
        text3 = make_response(content="Here's your answer.")
        backend = MockBackend([text1, text2, text3])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=sample_tools)
        response, headers = await loop.execute(
            request=request, tools=sample_tools, budget_remaining=45.0, start_time=time.monotonic()
        )

        # No rescue possible — should exhaust retries then passthrough
        assert backend.call_count == 3
        assert response.choices[0].message.tool_calls is None


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


class TestParamNameRepair:
    """Test auto-repair of near-miss parameter names."""

    @pytest.fixture
    def question_tool(self):
        """A tool with a plural param name — mimics opencode's question tool."""
        return ToolDefinition(
            type="function",
            function=FunctionDefinition(
                name="question",
                description="Ask the user a question",
                parameters={
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "description": "List of questions",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["questions"],
                },
            ),
        )

    @pytest.fixture
    def tools_with_question(self, sample_tools, question_tool):
        return sample_tools + [question_tool]

    @pytest.fixture
    def profile(self):
        return ModelProfile(
            max_retries=2,
            request_timeout=45.0,
            backend_timeout=120.0,
            tool_calling=ToolCallingConfig(),
            escalation=EscalationConfig(enabled=False),
        )

    @pytest.mark.asyncio
    async def test_repair_singular_to_plural(self, tools_with_question, profile):
        """'question' param should be repaired to 'questions'."""
        # Model calls question tool with singular param name
        response = make_response(
            tool_calls=[make_tool_call("question", {"question": ["What stack?"]})]
        )
        backend = MockBackend([response])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=tools_with_question)
        result, headers = await loop.execute(
            request=request, tools=tools_with_question, budget_remaining=45.0, start_time=time.monotonic()
        )

        # Should succeed on first attempt after auto-repair
        assert backend.call_count == 1
        assert "X-FilthyToolFixer-Degraded" not in headers

    @pytest.mark.asyncio
    async def test_no_repair_for_correct_params(self, tools_with_question, profile):
        """Correct param names should not be touched."""
        response = make_response(
            tool_calls=[make_tool_call("question", {"questions": ["What stack?"]})]
        )
        backend = MockBackend([response])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=tools_with_question)
        result, headers = await loop.execute(
            request=request, tools=tools_with_question, budget_remaining=45.0, start_time=time.monotonic()
        )

        assert backend.call_count == 1
        assert "X-FilthyToolFixer-Degraded" not in headers

    @pytest.mark.asyncio
    async def test_no_repair_for_distant_names(self, tools_with_question, profile):
        """Param names that are too different should not be repaired."""
        response = make_response(
            tool_calls=[make_tool_call("question", {"xyz_unrelated": ["What stack?"]})]
        )
        # Second attempt also bad
        response2 = make_response(
            tool_calls=[make_tool_call("question", {"xyz_unrelated": ["What stack?"]})]
        )
        backend = MockBackend([response, response2])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=tools_with_question)
        result, headers = await loop.execute(
            request=request, tools=tools_with_question, budget_remaining=45.0, start_time=time.monotonic()
        )

        # Should fail — distant name not auto-repaired
        assert headers.get("X-FilthyToolFixer-Degraded") == "true"

    @pytest.mark.asyncio
    async def test_repair_rescued_tool_call(self, tools_with_question, profile):
        """Rescued tool calls should also get param names repaired."""
        narrated = make_response(
            content='{"name": "question", "parameters": {"question": ["What is the stack?"]}}'
        )
        backend = MockBackend([narrated])
        loop = RetryLoop(backend=backend, profile=profile)

        request = make_request(tools=tools_with_question)
        result, headers = await loop.execute(
            request=request, tools=tools_with_question, budget_remaining=45.0, start_time=time.monotonic()
        )

        # Rescued from text + param name repaired = success on first attempt
        assert backend.call_count == 1
        tc = result.choices[0].message.tool_calls[0]
        assert tc.function.name == "question"
        assert '"questions"' in tc.function.arguments
