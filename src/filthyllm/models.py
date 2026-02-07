"""Pydantic v2 models for OpenAI-compatible chat completion API types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --- Request types ---


class FunctionDefinition(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    type: str = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    id: str = ""
    index: int | None = None
    type: str = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    # Pass through any extra fields the client sends
    model_config = {"extra": "allow"}


# --- Response types ---


class ChoiceDelta(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage | None = None
    delta: ChoiceDelta | None = None
    finish_reason: str | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[Choice] = Field(default_factory=list)
    usage: Usage | None = None
    # Extra headers to attach (not serialized in JSON body)
    model_config = {"extra": "allow"}
