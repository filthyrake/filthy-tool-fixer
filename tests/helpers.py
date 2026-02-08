"""Test helpers — factory functions and mock backend."""

from __future__ import annotations

import json
from typing import AsyncIterator

from filthy_tool_fixer.backends.base import BackendAdapter
from filthy_tool_fixer.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    FunctionCall,
    ToolCall,
    ToolDefinition,
    Usage,
)


def make_tool_call(
    name: str = "search_files",
    arguments: str | dict = '{"query": "test"}',
    call_id: str = "call_1",
) -> ToolCall:
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments)
    return ToolCall(
        id=call_id,
        type="function",
        function=FunctionCall(name=name, arguments=arguments),
    )


def make_response(
    tool_calls: list[ToolCall] | None = None,
    content: str | None = None,
    model: str = "qwen3:30b-a3b",
) -> ChatCompletionResponse:
    message = ChatMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
    )
    return ChatCompletionResponse(
        id="test-id",
        model=model,
        choices=[Choice(index=0, message=message, finish_reason="stop")],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def make_request(
    tools: list[ToolDefinition] | None = None,
    model: str = "qwen3:30b-a3b",
    messages: list[ChatMessage] | None = None,
) -> ChatCompletionRequest:
    if messages is None:
        messages = [ChatMessage(role="user", content="Search for Python files")]
    return ChatCompletionRequest(
        model=model,
        messages=messages,
        tools=tools,
    )


class MockBackend(BackendAdapter):
    """A mock backend that returns pre-configured responses."""

    def __init__(self, responses: list[ChatCompletionResponse] | None = None) -> None:
        self._responses = list(responses or [])
        self._call_count = 0
        self._requests: list[ChatCompletionRequest] = []

    async def startup(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        timeout: float | None = None,
        keep_alive: str | None = None,
    ) -> ChatCompletionResponse:
        self._requests.append(request)
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = self._responses[-1] if self._responses else make_response(content="fallback")
        self._call_count += 1
        return resp

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        timeout: float | None = None,
        keep_alive: str | None = None,
    ) -> AsyncIterator[bytes]:
        yield b'data: {"choices": [{"delta": {"content": "hello"}}]}\n'
        yield b"data: [DONE]\n"

    async def health_check(self) -> bool:
        return True

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def requests(self) -> list[ChatCompletionRequest]:
        return self._requests
