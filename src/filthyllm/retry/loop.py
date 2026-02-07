"""Retry orchestrator with timeout budget, degradation detection, and model escalation."""

from __future__ import annotations

import json
import time
from typing import Any

from filthyllm.backends.base import BackendAdapter
from filthyllm.logging import get_logger
from filthyllm.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ToolDefinition,
)
from filthyllm.profiles.types import ModelProfile
from filthyllm.retry.feedback import build_feedback_message
from filthyllm.validation.schema import ValidationResult, validate_tool_calls

log = get_logger(__name__)


class RetryLoop:
    def __init__(
        self,
        backend: BackendAdapter,
        escalation_backend: BackendAdapter | None = None,
        profile: ModelProfile | None = None,
    ) -> None:
        self._backend = backend
        self._escalation_backend = escalation_backend
        self._profile = profile or ModelProfile()

    async def execute(
        self,
        request: ChatCompletionRequest,
        tools: list[ToolDefinition],
        budget_remaining: float,
        start_time: float,
    ) -> tuple[ChatCompletionResponse, dict[str, str]]:
        """Execute the retry loop with optional escalation.

        Returns (response, extra_headers) where extra_headers contains
        metadata like X-FilthyLLM-Model and X-FilthyLLM-Degraded.
        """
        extra_headers: dict[str, str] = {}

        # Only force tool calls if tool_choice requires it
        tool_choice = request.tool_choice
        must_call_tools = tool_choice == "required" or isinstance(tool_choice, dict)

        # Force non-streaming for the internal retry loop
        buffered_request = request.model_copy(update={"stream": False})

        best_response: ChatCompletionResponse | None = None
        best_score = -1
        last_errors_key: str | None = None

        max_retries = self._profile.max_retries

        for attempt in range(max_retries + 1):
            elapsed = time.monotonic() - start_time
            remaining = budget_remaining - elapsed

            # Don't start a retry if insufficient time remains
            if attempt > 0 and remaining < 10.0:
                log.info("retry_budget_exhausted", attempt=attempt, remaining=remaining)
                break

            # Use the lesser of remaining budget and backend timeout
            effective_timeout = min(remaining, self._profile.backend_timeout)

            try:
                response = await self._backend.chat_completion(
                    buffered_request,
                    timeout=effective_timeout,
                    keep_alive=self._profile.tool_calling.keep_alive,
                )
            except Exception:
                log.exception("backend_request_failed", attempt=attempt)
                break

            # Score and track best response
            score = self._score_response(response, tools)
            if score > best_score:
                best_score = score
                best_response = response

            # Extract tool calls and validate
            tool_calls = self._extract_tool_calls(response)
            if not tool_calls:
                finish = ""
                if response.choices:
                    finish = response.choices[0].finish_reason or ""

                # If tools are optional and model chose text on a retry,
                # accept it — we already nudged once
                if not must_call_tools and finish == "stop" and attempt > 0:
                    log.info(
                        "text_response_passthrough",
                        attempt=attempt,
                        finish_reason=finish,
                    )
                    extra_headers["X-FilthyLLM-Attempts"] = str(attempt + 1)
                    return response, extra_headers

                log.info(
                    "no_tool_calls_in_response",
                    attempt=attempt,
                    finish_reason=finish,
                    must_call_tools=must_call_tools,
                )
                # Nudge the model once to use tools, then accept text
                if attempt < max_retries:
                    buffered_request = self._append_narration_feedback(
                        buffered_request, response
                    )
                    continue
                break

            result = validate_tool_calls(tool_calls, tools)
            if result.valid:
                log.info("tool_calls_valid", attempt=attempt)
                extra_headers["X-FilthyLLM-Attempts"] = str(attempt + 1)
                return response, extra_headers

            # Check for duplicate invalid response (same errors = model stuck)
            errors_key = self._errors_key(result)
            if errors_key == last_errors_key:
                log.info("duplicate_errors_detected", attempt=attempt)
                break
            last_errors_key = errors_key

            # Check for degradation (more errors than before)
            log.info(
                "validation_failed",
                attempt=attempt,
                error_count=len(result.errors),
                worst=result.worst_error.message if result.worst_error else "",
            )

            if attempt < max_retries:
                buffered_request = self._append_feedback(
                    buffered_request, response, result
                )

        # Primary model failed — try escalation
        if (
            self._escalation_backend
            and self._profile.escalation.enabled
            and self._profile.escalation.model
        ):
            log.info("escalating_to_quality_model", model=self._profile.escalation.model)
            escalation_response = await self._try_escalation(
                request, tools, best_response, start_time
            )
            if escalation_response:
                extra_headers["X-FilthyLLM-Model"] = self._profile.escalation.model
                extra_headers["X-FilthyLLM-Escalated"] = "true"
                return escalation_response, extra_headers

        # Total failure — return best attempt with degraded header
        extra_headers["X-FilthyLLM-Degraded"] = "true"
        if best_response:
            return best_response, extra_headers

        # No response at all (e.g. all attempts timed out)
        log.error("no_response_obtained")
        from filthyllm.models import ChatMessage, Choice, Usage

        fallback = ChatCompletionResponse(
            id="error",
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content="I was unable to process this request. The backend did not return a response in time.",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(),
        )
        return fallback, extra_headers

    async def _try_escalation(
        self,
        original_request: ChatCompletionRequest,
        tools: list[ToolDefinition],
        failed_response: ChatCompletionResponse | None,
        start_time: float,
    ) -> ChatCompletionResponse | None:
        """Attempt escalation to the quality model with synthesized error context."""
        if self._escalation_backend is None:
            log.error("escalation_backend_not_configured")
            return None

        elapsed = time.monotonic() - start_time
        escalation_budget = self._profile.escalation.timeout
        if elapsed > escalation_budget:
            log.info("escalation_budget_exceeded", elapsed=elapsed)
            return None

        # Build escalation request with synthesized context
        messages = list(original_request.messages)

        if failed_response:
            # Add a brief summary of what went wrong, not the full failed conversation
            summary = self._build_escalation_summary(failed_response, tools)
            messages.append(ChatMessage(role="user", content=summary))

        escalation_request = original_request.model_copy(
            update={
                "model": self._profile.escalation.model,
                "messages": messages,
                "stream": False,
            }
        )

        try:
            response = await self._escalation_backend.chat_completion(
                escalation_request,
                timeout=escalation_budget - elapsed,
                keep_alive=self._profile.tool_calling.keep_alive,
            )
        except Exception:
            log.exception("escalation_request_failed")
            return None

        tool_calls = self._extract_tool_calls(response)
        if not tool_calls:
            return None

        result = validate_tool_calls(tool_calls, tools)
        if result.valid:
            log.info("escalation_succeeded")
            return response

        log.info("escalation_also_failed", error_count=len(result.errors))
        return None

    def _build_escalation_summary(
        self,
        failed_response: ChatCompletionResponse,
        tools: list[ToolDefinition],
    ) -> str:
        """Synthesize a brief error summary for the escalation model."""
        tool_calls = self._extract_tool_calls(failed_response)
        if not tool_calls:
            return (
                "A previous model attempted to respond but did not make any tool calls. "
                "Please respond with the appropriate tool call(s)."
            )

        result = validate_tool_calls(tool_calls, tools)
        if result.valid:
            return ""

        error_msgs = [e.message for e in result.errors[:3]]
        return (
            "A previous model attempted to call tools but failed with these errors: "
            + "; ".join(error_msgs)
            + ". Please make the correct tool call(s)."
        )

    def _extract_tool_calls(self, response: ChatCompletionResponse):
        """Extract tool calls from the first choice."""
        if not response.choices:
            return []
        choice = response.choices[0]
        if choice.message and choice.message.tool_calls:
            return choice.message.tool_calls
        return []

    def _score_response(
        self, response: ChatCompletionResponse, tools: list[ToolDefinition]
    ) -> int:
        """Score a response for 'best attempt' selection.

        valid > parseable JSON > has tool_calls field > has content > nothing
        """
        tool_calls = self._extract_tool_calls(response)
        if not tool_calls:
            # Has some content at least?
            if response.choices and response.choices[0].message:
                return 1
            return 0

        result = validate_tool_calls(tool_calls, tools)
        if result.valid:
            return 10

        # Has tool calls but they're invalid — check if args are valid JSON
        all_parseable = True
        for tc in tool_calls:
            try:
                json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                all_parseable = False
                break

        return 5 if all_parseable else 3

    def _errors_key(self, result: ValidationResult) -> str:
        """Generate a hashable key from errors for duplicate detection."""
        return "|".join(
            sorted(f"{e.error_type}:{e.tool_call_index}:{e.message}" for e in result.errors)
        )

    def _append_feedback(
        self,
        request: ChatCompletionRequest,
        response: ChatCompletionResponse,
        result: ValidationResult,
    ) -> ChatCompletionRequest:
        """Append the model's failed response and error feedback to the conversation."""
        messages = list(request.messages)

        # Add the model's failed response
        if response.choices and response.choices[0].message:
            messages.append(response.choices[0].message)

        # Add feedback
        feedback = build_feedback_message(result)
        messages.append(feedback)

        return request.model_copy(update={"messages": messages})

    def _append_narration_feedback(
        self,
        request: ChatCompletionRequest,
        response: ChatCompletionResponse,
    ) -> ChatCompletionRequest:
        """Feedback for when the model narrates instead of calling tools."""
        messages = list(request.messages)

        if response.choices and response.choices[0].message:
            messages.append(response.choices[0].message)

        messages.append(
            ChatMessage(
                role="user",
                content=(
                    "You described what you would do instead of actually calling the tool. "
                    "Please respond with the actual tool call, not a description of it."
                ),
            )
        )

        return request.model_copy(update={"messages": messages})
