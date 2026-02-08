"""Retry orchestrator with timeout budget, degradation detection, and model escalation."""

from __future__ import annotations

import json
import re
import time
import uuid
from difflib import get_close_matches
from filthy_tool_fixer.backends.base import BackendAdapter
from filthy_tool_fixer.logging import get_logger
from filthy_tool_fixer.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    FunctionCall,
    ToolCall,
    ToolDefinition,
)
from filthy_tool_fixer.profiles.types import ModelProfile
from filthy_tool_fixer.retry.feedback import build_feedback_message
from filthy_tool_fixer.validation.schema import ValidationResult, validate_tool_calls

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
        metadata like X-FilthyToolFixer-Model and X-FilthyToolFixer-Degraded.
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
        ever_attempted_tools = False  # Track if model ever tried tool calls

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

            # Extract tool calls and validate
            tool_calls = self._extract_tool_calls(response)

            # Rescue tool calls narrated as JSON text in the content field
            if not tool_calls:
                rescued = self._rescue_tool_calls_from_text(response, tools)
                if rescued:
                    tool_calls = rescued
                    # Patch the response so downstream sees proper tool_calls
                    response.choices[0].message.tool_calls = rescued
                    response.choices[0].message.content = ""
                    response.choices[0].finish_reason = "tool_calls"
                    log.info(
                        "tool_calls_rescued_from_text",
                        attempt=attempt,
                        count=len(rescued),
                        names=[tc.function.name for tc in rescued],
                        args=[tc.function.arguments[:200] for tc in rescued],
                    )

            if tool_calls:
                ever_attempted_tools = True
                # Auto-repair near-miss parameter names before validation
                self._repair_tool_calls(tool_calls, tools)

            # Score AFTER rescue+repair so rescued responses rank correctly
            score = self._score_response(response, tools)
            if score > best_score:
                best_score = score
                best_response = response

            if not tool_calls:
                finish = ""
                if response.choices:
                    finish = response.choices[0].finish_reason or ""

                # If the conversation already has tool results, the model
                # has been successfully using tools. A text response now
                # is the synthesized answer — accept it immediately.
                has_prior_tool_results = any(
                    m.role == "tool" for m in request.messages
                )

                log.info(
                    "no_tool_calls_in_response",
                    attempt=attempt,
                    finish_reason=finish,
                    must_call_tools=must_call_tools,
                    has_prior_tool_results=has_prior_tool_results,
                )

                if has_prior_tool_results and not must_call_tools and finish == "stop":
                    log.info(
                        "text_response_accepted",
                        attempt=attempt,
                        reason="prior_tool_results_in_conversation",
                    )
                    extra_headers["X-FilthyToolFixer-Attempts"] = str(attempt + 1)
                    return response, extra_headers

                # Keep nudging until retries exhausted — give the model
                # every chance to engage with tools before falling through
                # to escalation
                if attempt < max_retries:
                    buffered_request = self._append_narration_feedback(
                        buffered_request, response, tools
                    )
                    continue

                # All retries spent without a tool call. If tools are
                # optional AND the model never even attempted tools, accept
                # text (it may genuinely not need tools). But if the model
                # tried tools at any point, it clearly needed them — fall
                # through to escalation instead.
                if not must_call_tools and finish == "stop" and not ever_attempted_tools:
                    log.info(
                        "text_response_passthrough",
                        attempt=attempt,
                        finish_reason=finish,
                    )
                    extra_headers["X-FilthyToolFixer-Attempts"] = str(attempt + 1)
                    return response, extra_headers

                break

            result = validate_tool_calls(tool_calls, tools)
            if result.valid:
                # Check for semantic misuse (e.g., glob without wildcards)
                semantic = self._check_semantic_issues(tool_calls)
                if semantic and attempt < max_retries:
                    log.info("semantic_issue", attempt=attempt, feedback=semantic)
                    buffered_request = self._append_semantic_feedback(
                        buffered_request, response, semantic
                    )
                    continue
                log.info("tool_calls_valid", attempt=attempt)
                extra_headers["X-FilthyToolFixer-Attempts"] = str(attempt + 1)
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
                errors=[e.message for e in result.errors[:5]],
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
                extra_headers["X-FilthyToolFixer-Model"] = self._profile.escalation.model
                extra_headers["X-FilthyToolFixer-Escalated"] = "true"
                return escalation_response, extra_headers

        # Total failure — return best attempt with degraded header
        extra_headers["X-FilthyToolFixer-Degraded"] = "true"
        if best_response:
            return best_response, extra_headers

        # No response at all (e.g. all attempts timed out)
        log.error("no_response_obtained")
        from filthy_tool_fixer.models import ChatMessage, Choice, Usage

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

        # Rescue tool calls from text in escalation too
        if not tool_calls:
            rescued = self._rescue_tool_calls_from_text(response, tools)
            if rescued:
                tool_calls = rescued
                response.choices[0].message.tool_calls = rescued
                response.choices[0].message.content = ""
                response.choices[0].finish_reason = "tool_calls"
                log.info("escalation_tool_calls_rescued", count=len(rescued))

        if not tool_calls:
            return None

        # Auto-repair near-miss parameter names
        self._repair_tool_calls(tool_calls, tools)

        result = validate_tool_calls(tool_calls, tools)
        if result.valid:
            log.info("escalation_succeeded")
            return response

        log.info(
            "escalation_also_failed",
            error_count=len(result.errors),
            errors=[e.message for e in result.errors[:5]],
        )
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

    # Llama 3 native: <|python_start|>func(args)<|python_end|>
    # Llama 4 native: <|python_tag|>func.call(args)<|eom_id|>
    _LLAMA_TOOL_RE = re.compile(
        r"<\|python_(?:start|tag)\|>\s*(\w+)(?:\.call)?\((.*?)\)\s*(?:<\|python_end\|>|<\|eom_id\|>)?",
        re.DOTALL,
    )

    # Llama 4 pythonic: [func_name(param="val", param2=val2)]
    # May contain multiple calls: [func1(a=1), func2(b=2), func3(c=3)]
    _PYTHONIC_BRACKET_RE = re.compile(r"\[([^\]]+)\]", re.DOTALL)
    _PYTHONIC_CALL_RE = re.compile(r"(\w+)\(([^)]*)\)", re.DOTALL)

    def _rescue_tool_calls_from_text(
        self,
        response: ChatCompletionResponse,
        tools: list[ToolDefinition],
    ) -> list[ToolCall] | None:
        """Try to extract tool calls from text content.

        Models sometimes emit tool calls in non-standard formats instead of
        using the tool_calls response field. This detects common patterns:
        - Llama native: <|python_start|>func(args)<|python_end|>
        - JSON narration: {"name": "func", "parameters": {...}}
        - Code-fenced JSON
        """
        if not response.choices or not response.choices[0].message:
            return None
        content = (response.choices[0].message.content or "").strip()
        if not content:
            return None

        tool_names = {t.function.name for t in tools}

        # Try Llama native format: <|python_start|>/<|python_tag|> tags
        llama_matches = self._LLAMA_TOOL_RE.findall(content)
        if llama_matches:
            rescued = self._parse_llama_native_calls(llama_matches, tool_names)
            if rescued:
                log.debug("rescued_llama_native", count=len(rescued))
                return rescued

        # Try Llama 4 pythonic format: [func_name(param="val")]
        bracket_match = self._PYTHONIC_BRACKET_RE.search(content)
        if bracket_match:
            inner = bracket_match.group(1)
            pythonic_matches = self._PYTHONIC_CALL_RE.findall(inner)
            rescued = self._parse_pythonic_calls(pythonic_matches, tool_names)
            if rescued:
                log.debug("rescued_pythonic", count=len(rescued))
                return rescued

        # Try JSON narration (plain or code-fenced)
        json_str = content
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
        if fence_match:
            json_str = fence_match.group(1).strip()

        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return None

        candidates = data if isinstance(data, list) else [data]

        rescued: list[ToolCall] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            tc = self._parse_narrated_tool_call(item, tool_names)
            if tc:
                rescued.append(tc)

        return rescued if rescued else None

    def _parse_llama_native_calls(
        self,
        matches: list[tuple[str, str]],
        tool_names: set[str],
    ) -> list[ToolCall] | None:
        """Parse Llama-style function calls: func_name(key="val", key2=val2)."""
        rescued: list[ToolCall] = []
        for func_name, args_str in matches:
            if func_name not in tool_names:
                continue

            # Parse the keyword arguments using ast.literal_eval on a dict
            args = self._parse_python_kwargs(args_str)
            if args is None:
                continue

            call_id = f"call_{uuid.uuid4().hex[:12]}"
            rescued.append(ToolCall(
                id=call_id,
                type="function",
                function=FunctionCall(name=func_name, arguments=json.dumps(args)),
            ))

        return rescued if rescued else None

    def _parse_pythonic_calls(
        self,
        matches: list[tuple[str, str]],
        tool_names: set[str],
    ) -> list[ToolCall] | None:
        """Parse Llama 4 pythonic format: [func_name(param="val", param2=val2)].

        Each match is a (name, args_str) tuple from _PYTHONIC_CALL_RE.findall().
        Handles any number of calls within brackets.
        """
        rescued: list[ToolCall] = []
        for func_name, args_str in matches:
            if not func_name or func_name not in tool_names:
                continue

            args = self._parse_python_kwargs(args_str)
            if args is None:
                continue

            call_id = f"call_{uuid.uuid4().hex[:12]}"
            rescued.append(ToolCall(
                id=call_id,
                type="function",
                function=FunctionCall(name=func_name, arguments=json.dumps(args)),
            ))

        return rescued if rescued else None

    def _parse_python_kwargs(self, args_str: str) -> dict | None:
        """Parse Python-style keyword arguments into a dict.

        Handles: key="value", key=123, key=True, key=[1,2,3]
        """
        import ast

        args_str = args_str.strip()
        if not args_str:
            return {}

        # Wrap in dict() call syntax for ast parsing: dict(key="val") → {"key": "val"}
        try:
            tree = ast.parse(f"dict({args_str})", mode="eval")
            # Extract keyword arguments from the Call node
            call_node = tree.body
            if not isinstance(call_node, ast.Call):
                return None

            result = {}
            for kw in call_node.keywords:
                if kw.arg is None:
                    continue
                result[kw.arg] = ast.literal_eval(kw.value)
            return result
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return None

    def _parse_narrated_tool_call(
        self, item: dict, tool_names: set[str]
    ) -> ToolCall | None:
        """Parse a single narrated tool call dict into a ToolCall.

        Handles common patterns:
          {"name": "...", "arguments": {...}}
          {"name": "...", "parameters": {...}}
          {"type": "function", "name": "...", "parameters": {...}}
          {"function": {"name": "...", "arguments": {...}}}
        """
        name = None
        args = None

        # Pattern: {"function": {"name": ..., "arguments": ...}}
        if "function" in item and isinstance(item["function"], dict):
            inner = item["function"]
            name = inner.get("name")
            args = inner.get("arguments")
            if args is None:
                args = inner.get("parameters")
        else:
            # Pattern: {"name": ..., "arguments"|"parameters": ...}
            name = item.get("name")
            args = item.get("arguments")
            if args is None:
                args = item.get("parameters")

        if not name or name not in tool_names:
            return None

        # Serialize args to JSON string if they're a dict/list
        if isinstance(args, (dict, list)):
            args_str = json.dumps(args)
        elif isinstance(args, str):
            args_str = args
        else:
            args_str = "{}"

        call_id = f"call_{uuid.uuid4().hex[:12]}"
        return ToolCall(
            id=call_id,
            type="function",
            function=FunctionCall(name=name, arguments=args_str),
        )

    def _repair_tool_calls(
        self,
        tool_calls: list[ToolCall],
        tools: list[ToolDefinition],
    ) -> bool:
        """Auto-repair near-miss parameter names in tool calls.

        Models often get parameter names almost-right (e.g. 'question' instead
        of 'questions'). This fuzzy-matches and corrects them in-place.
        Returns True if any repairs were made.
        """
        tool_map = {t.function.name: t for t in tools}
        repaired = False
        log.debug("repair_starting", call_count=len(tool_calls), tool_names=[tc.function.name for tc in tool_calls])

        for tc in tool_calls:
            tool_def = tool_map.get(tc.function.name)
            if not tool_def or not tool_def.function.parameters:
                continue

            schema = tool_def.function.parameters
            properties = schema.get("properties", {})
            if not properties:
                continue

            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                continue

            if not isinstance(args, dict):
                log.debug("repair_skip_non_dict", tool=tc.function.name, args_type=type(args).__name__)
                continue

            valid_names = list(properties.keys())
            new_args = {}
            changed = False
            log.debug("repair_checking", tool=tc.function.name, arg_keys=list(args.keys()), valid=valid_names)

            for key, value in args.items():
                if key in properties:
                    new_args[key] = value
                else:
                    # Fuzzy match against valid parameter names
                    matches = get_close_matches(key, valid_names, n=1, cutoff=0.8)
                    if matches:
                        new_args[matches[0]] = value
                        changed = True
                        log.info(
                            "param_name_repaired",
                            tool=tc.function.name,
                            original=key,
                            corrected=matches[0],
                        )
                    else:
                        new_args[key] = value  # Keep as-is, validation will catch it

            if changed:
                tc.function.arguments = json.dumps(new_args)
                repaired = True

        return repaired

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
        tools: list[ToolDefinition] | None = None,
    ) -> ChatCompletionRequest:
        """Feedback for when the model narrates instead of calling tools."""
        messages = list(request.messages)

        if response.choices and response.choices[0].message:
            messages.append(response.choices[0].message)

        # Use tools from the request (which may be filtered) for guidance
        tool_list = request.tools or tools or []
        names = [t.function.name for t in tool_list]

        # Build actionable guidance with tool-specific hints
        hints = []
        if "read" in names:
            hints.append("'read' to view file contents")
        if "glob" in names:
            hints.append("'glob' to find files by pattern")
        if "grep" in names:
            hints.append("'grep' to search file contents")
        if "bash" in names:
            hints.append("'bash' to run commands")

        hint_text = ""
        if hints:
            hint_text = " Use " + ", ".join(hints) + "."

        messages.append(
            ChatMessage(
                role="user",
                content=(
                    "You MUST call a tool. Do NOT respond with text."
                    f"{hint_text}"
                    f" Available tools: {', '.join(names)}."
                    " Pick the most appropriate tool and call it now."
                ),
            )
        )

        return request.model_copy(update={"messages": messages})

    def _check_semantic_issues(self, tool_calls: list[ToolCall]) -> str | None:
        """Check for common semantic misuse patterns in valid tool calls.

        Returns a feedback string if issues found, None if OK.
        """
        for tc in tool_calls:
            if tc.function.name == "glob":
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    continue
                pattern = args.get("pattern", "")
                if isinstance(pattern, str) and not any(c in pattern for c in "*?["):
                    return (
                        f"WRONG. glob finds files by NAME pattern — it needs wildcards "
                        f"(*, **, ?). '{pattern}' has none. "
                        f"To search file CONTENTS, use grep with pattern='{pattern}'. "
                        "To list all files, use glob with pattern='*'. "
                        "To read a specific file, use read with file_path='/path/to/file'."
                    )
        return None

    def _append_semantic_feedback(
        self,
        request: ChatCompletionRequest,
        response: ChatCompletionResponse,
        feedback: str,
    ) -> ChatCompletionRequest:
        """Append semantic issue feedback to guide the model toward correct tool use."""
        messages = list(request.messages)
        if response.choices and response.choices[0].message:
            messages.append(response.choices[0].message)
        messages.append(ChatMessage(role="user", content=feedback))
        return request.model_copy(update={"messages": messages})
