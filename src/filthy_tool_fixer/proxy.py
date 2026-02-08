"""Core proxy orchestrator — routes requests through validation, retry, and escalation."""

from __future__ import annotations

import asyncio
import re
import time
from typing import AsyncIterator

from filthy_tool_fixer.backends.base import BackendAdapter
from filthy_tool_fixer.config import settings
from filthy_tool_fixer.logging import get_logger
from filthy_tool_fixer.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    FunctionDefinition,
    ToolDefinition,
)
from filthy_tool_fixer.profiles.types import ModelProfile
from filthy_tool_fixer.retry.loop import RetryLoop

log = get_logger(__name__)

# Cache compiled think-tag patterns per profile
_think_re_cache: dict[str, re.Pattern] = {}

# Regex patterns for condensing tool descriptions
_EXAMPLE_BLOCK_RE = re.compile(r"<example>.*?</example>", re.DOTALL)
_USAGE_NOTES_RE = re.compile(r"(?:^|\n)Usage notes:.*?(?=\n[A-Z#]|\Z)", re.DOTALL)
_IMPORTANT_BLOCK_RE = re.compile(r"(?:^|\n)IMPORTANT:.*?(?=\n[A-Z#\-]|\Z)", re.DOTALL)
_SECTION_HEADER_RE = re.compile(r"\n#[^\n]*\n")


def _condense_description(desc: str) -> str:
    """Strip verbose examples and usage notes from a tool description.

    Keeps the first paragraph (core description) and essential usage info.
    """
    if not desc or len(desc) < 500:
        return desc

    # Remove <example> blocks
    desc = _EXAMPLE_BLOCK_RE.sub("", desc)

    # Take first meaningful paragraph as the core description
    lines = desc.strip().split("\n")
    core_lines = []
    for line in lines:
        stripped = line.strip()
        # Stop at verbose sections
        if stripped.startswith(("Usage notes:", "When to use", "When NOT to use",
                                "## When to Use", "## When NOT", "## Examples",
                                "## Task States", "# Committing", "# Creating pull",
                                "# Other common", "# Code style", "# Proactive",
                                "# Following conventions", "# Doing tasks")):
            break
        # Skip empty lines after we have content
        if not stripped and core_lines and not core_lines[-1].strip():
            continue
        core_lines.append(line)
        # Stop after reasonable length
        if len("\n".join(core_lines)) > 300:
            break

    result = "\n".join(core_lines).strip()

    # If we ended up with nothing useful, just truncate the original
    if len(result) < 20:
        result = desc[:300].rsplit(".", 1)[0] + "."

    return result


def _condense_system_prompt(prompt: str, max_tokens: int = 0) -> str:
    """Strip verbose sections from a client system prompt.

    Removes example blocks, commit/PR instructions, and other sections
    that are irrelevant for tool-calling decisions.
    """
    if not prompt or len(prompt) < 1000:
        return prompt

    # Remove <example> blocks
    result = _EXAMPLE_BLOCK_RE.sub("", prompt)

    # Remove large instruction sections that don't help with tool selection
    sections_to_strip = [
        # Git/commit instructions (massive, irrelevant to tool selection)
        re.compile(r"# Committing changes with git.*?(?=\n# |\Z)", re.DOTALL),
        # PR creation instructions
        re.compile(r"# Creating pull requests.*?(?=\n# |\Z)", re.DOTALL),
        # Other common operations
        re.compile(r"# Other common operations.*?(?=\n# |\Z)", re.DOTALL),
    ]

    for pattern in sections_to_strip:
        result = pattern.sub("", result)

    # Collapse multiple blank lines
    result = re.sub(r"\n{3,}", "\n\n", result)

    # Hard truncate if max_tokens set (rough: 1 token ≈ 4 chars)
    if max_tokens > 0:
        max_chars = max_tokens * 4
        if len(result) > max_chars:
            result = result[:max_chars].rsplit("\n", 1)[0]

    return result.strip()


def _condense_tools(tools: list[ToolDefinition]) -> list[ToolDefinition]:
    """Create condensed copies of tool definitions with shorter descriptions."""
    condensed = []
    for tool in tools:
        new_desc = _condense_description(tool.function.description)
        condensed.append(ToolDefinition(
            type=tool.type,
            function=FunctionDefinition(
                name=tool.function.name,
                description=new_desc,
                parameters=tool.function.parameters,
            ),
        ))
    return condensed


class ProxyOrchestrator:
    def __init__(
        self,
        primary_backend: BackendAdapter,
        escalation_backend: BackendAdapter | None = None,
        *,
        primary_url: str = "",
        escalation_url: str = "",
    ) -> None:
        self._primary = primary_backend
        self._escalation = escalation_backend
        self._tool_semaphore = asyncio.Semaphore(settings.max_concurrent_tool_requests)

        # Map backend URLs to adapters for per-profile routing
        self._backends_by_url: dict[str, BackendAdapter] = {}
        if primary_url:
            self._backends_by_url[primary_url.rstrip("/")] = primary_backend
        if escalation_url and escalation_backend:
            self._backends_by_url[escalation_url.rstrip("/")] = escalation_backend

    def _select_backend(self, profile: ModelProfile) -> BackendAdapter:
        """Select the backend adapter based on profile's backend_url override."""
        if profile.backend_url:
            url = profile.backend_url.rstrip("/")
            backend = self._backends_by_url.get(url)
            if backend:
                log.debug("backend_override", url=url)
                return backend
            log.warning("backend_url_not_found", url=url, available=list(self._backends_by_url.keys()))
        return self._primary

    def _has_tools(self, request: ChatCompletionRequest) -> bool:
        return bool(request.tools)

    def _enhance_request(
        self, request: ChatCompletionRequest, profile: ModelProfile
    ) -> ChatCompletionRequest:
        """Apply profile-driven enhancements to the request."""
        messages = list(request.messages)
        tc = profile.tool_calling

        # Condense the client's system prompt if enabled
        if self._has_tools(request) and tc.condense_system_prompt:
            if messages and messages[0].role == "system":
                original = messages[0].content or ""
                condensed = _condense_system_prompt(original, tc.max_system_tokens)
                if len(condensed) < len(original):
                    log.debug(
                        "system_prompt_condensed",
                        original_len=len(original),
                        condensed_len=len(condensed),
                    )
                messages[0] = ChatMessage(role="system", content=condensed)

        # Inject system suffix for tool-calling requests (prepend for priority)
        if self._has_tools(request) and tc.system_suffix:
            if messages and messages[0].role == "system":
                original = messages[0].content or ""
                messages[0] = ChatMessage(
                    role="system",
                    content=tc.system_suffix + "\n\n" + original,
                )
            else:
                messages.insert(
                    0,
                    ChatMessage(role="system", content=tc.system_suffix),
                )

        # Temperature override for tool-calling requests
        temperature = request.temperature
        if self._has_tools(request) and tc.temperature_override is not None:
            temperature = tc.temperature_override

        updates: dict = {"messages": messages, "temperature": temperature}

        # Override tool_choice if profile specifies it
        if self._has_tools(request) and tc.tool_choice_override:
            updates["tool_choice"] = tc.tool_choice_override

        # Set Ollama context window size if configured (merge with existing options)
        if self._has_tools(request) and tc.num_ctx > 0:
            existing_options = getattr(request, "options", None) or {}
            updates["options"] = {**existing_options, "num_ctx": tc.num_ctx}

        # Exclude tools the model can't handle well
        effective_tools = request.tools
        if self._has_tools(request) and tc.exclude_tools and request.tools:
            excluded = set(tc.exclude_tools)
            effective_tools = [t for t in request.tools if t.function.name not in excluded]
            if len(effective_tools) < len(request.tools):
                log.debug(
                    "tools_excluded",
                    excluded=tc.exclude_tools,
                    remaining=[t.function.name for t in effective_tools],
                )
            updates["tools"] = effective_tools

        # Condense verbose tool descriptions to reduce context size
        if self._has_tools(request) and tc.condense_tools and effective_tools:
            updates["tools"] = _condense_tools(effective_tools)

        return request.model_copy(update=updates)

    def _strip_thinking(self, response: ChatCompletionResponse, pattern: str) -> ChatCompletionResponse:
        """Remove thinking blocks from response content using the profile's pattern."""
        if not pattern:
            return response
        if pattern not in _think_re_cache:
            _think_re_cache[pattern] = re.compile(pattern, re.DOTALL)
        compiled = _think_re_cache[pattern]
        for choice in response.choices:
            if choice.message and choice.message.content:
                choice.message.content = compiled.sub("", choice.message.content).strip()
        return response

    async def handle_request(
        self,
        request: ChatCompletionRequest,
        profile: ModelProfile,
    ) -> ChatCompletionResponse | AsyncIterator[bytes]:
        """Main entry point — route request through the appropriate pipeline."""
        enhanced = self._enhance_request(request, profile)
        backend = self._select_backend(profile)

        if not self._has_tools(request):
            # No tools — pure passthrough (streaming or non-streaming)
            if request.stream:
                return backend.chat_completion_stream(
                    enhanced,
                    timeout=profile.backend_timeout,
                    keep_alive=profile.tool_calling.keep_alive,
                )
            resp = await backend.chat_completion(
                enhanced,
                timeout=profile.backend_timeout,
                keep_alive=profile.tool_calling.keep_alive,
            )
            if profile.tool_calling.strip_thinking:
                resp = self._strip_thinking(resp, profile.tool_calling.think_tag_pattern)
            return resp

        # Tool-calling path — always buffer for validation
        async with self._tool_semaphore:
            return await self._handle_tool_request(enhanced, request, profile)

    async def _handle_tool_request(
        self,
        enhanced: ChatCompletionRequest,
        original: ChatCompletionRequest,
        profile: ModelProfile,
    ) -> tuple[ChatCompletionResponse, dict[str, str]]:
        """Handle a tool-calling request with validation, retry, and escalation."""
        start = time.monotonic()
        budget = profile.request_timeout
        backend = self._select_backend(profile)

        retry_loop = RetryLoop(
            backend=backend,
            escalation_backend=self._escalation,
            profile=profile,
        )

        response, extra_headers = await retry_loop.execute(
            request=enhanced,
            tools=enhanced.tools or [],
            budget_remaining=budget,
            start_time=start,
        )

        if profile.tool_calling.strip_thinking:
            response = self._strip_thinking(response, profile.tool_calling.think_tag_pattern)

        # Restore the originally-requested model name in the response body
        response.model = original.model

        return response, extra_headers
