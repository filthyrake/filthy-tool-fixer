"""Backend adapter abstract base class."""

from __future__ import annotations

import abc
from typing import AsyncIterator

from filthyllm.models import ChatCompletionRequest, ChatCompletionResponse


class BackendAdapter(abc.ABC):
    @abc.abstractmethod
    async def startup(self) -> None:
        """Initialize connections."""

    @abc.abstractmethod
    async def shutdown(self) -> None:
        """Close connections."""

    @abc.abstractmethod
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        timeout: float | None = None,
        keep_alive: str | None = None,
    ) -> ChatCompletionResponse:
        """Send a non-streaming chat completion request."""

    @abc.abstractmethod
    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        timeout: float | None = None,
        keep_alive: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Send a streaming chat completion request, yielding raw SSE bytes."""

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Return True if backend is reachable."""
