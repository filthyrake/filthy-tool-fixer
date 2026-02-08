"""vLLM backend adapter with constrained decoding support."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from filthy_tool_fixer.backends.base import BackendAdapter
from filthy_tool_fixer.logging import get_logger
from filthy_tool_fixer.models import ChatCompletionRequest, ChatCompletionResponse, ToolDefinition

log = get_logger(__name__)

_CHAT_PATH = "/v1/chat/completions"


def build_tool_call_schema(tools: list[ToolDefinition]) -> dict[str, Any]:
    """Generate a JSON schema from tool definitions for constrained decoding.

    Produces a schema that forces the model to output a valid tool call
    matching one of the provided tool definitions.
    """
    if not tools:
        return {}

    tool_schemas = []
    for tool in tools:
        func = tool.function
        # Each tool call has name + arguments matching the tool's parameter schema
        tool_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "const": func.name},
                "arguments": func.parameters if func.parameters else {"type": "object"},
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
        tool_schemas.append(tool_schema)

    if len(tool_schemas) == 1:
        return tool_schemas[0]

    return {"oneOf": tool_schemas}


class VLLMAdapter(BackendAdapter):
    def __init__(self, base_url: str, default_timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_timeout = default_timeout
        self._client: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._default_timeout, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        log.info("vllm_adapter_started", base_url=self._base_url)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            log.info("vllm_adapter_stopped", base_url=self._base_url)

    def _build_payload(
        self,
        request: ChatCompletionRequest,
        keep_alive: str | None = None,
    ) -> dict[str, Any]:
        payload = request.model_dump(exclude_none=True)

        # Inject guided_json for constrained decoding when tools are present
        if request.tools:
            schema = build_tool_call_schema(request.tools)
            if schema:
                payload["extra_body"] = {"guided_json": json.dumps(schema)}
                log.debug("vllm_constrained_decoding", schema_keys=list(schema.keys()))

        return payload

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        timeout: float | None = None,
        keep_alive: str | None = None,
    ) -> ChatCompletionResponse:
        assert self._client is not None
        payload = self._build_payload(request, keep_alive)
        payload["stream"] = False

        effective_timeout = timeout or self._default_timeout
        resp = await self._client.post(
            _CHAT_PATH,
            json=payload,
            timeout=httpx.Timeout(effective_timeout, connect=10.0),
        )
        resp.raise_for_status()
        return ChatCompletionResponse.model_validate(resp.json())

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        timeout: float | None = None,
        keep_alive: str | None = None,
    ) -> AsyncIterator[bytes]:
        assert self._client is not None
        payload = self._build_payload(request, keep_alive)
        payload["stream"] = True

        effective_timeout = timeout or self._default_timeout
        async with self._client.stream(
            "POST",
            _CHAT_PATH,
            json=payload,
            timeout=httpx.Timeout(effective_timeout, connect=10.0),
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    yield (line + "\n").encode()

    async def health_check(self) -> bool:
        try:
            assert self._client is not None
            resp = await self._client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
