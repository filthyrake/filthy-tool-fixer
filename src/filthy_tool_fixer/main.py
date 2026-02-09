"""FastAPI application — routes, lifespan, health checks."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from filthy_tool_fixer.backends.ollama import OllamaAdapter
from filthy_tool_fixer.config import settings
from filthy_tool_fixer.logging import (
    generate_request_id,
    get_logger,
    request_id_var,
    setup_logging,
)
from filthy_tool_fixer.models import ChatCompletionRequest
from filthy_tool_fixer.profiles.loader import ProfileLoader
from filthy_tool_fixer.proxy import ProxyOrchestrator

log = get_logger(__name__)


class _RequestCounter:
    """In-flight request counter with drain support for graceful shutdown.

    All methods are synchronous — safe under asyncio's cooperative model
    since no await can interleave between the read-modify-write steps.
    """

    def __init__(self) -> None:
        self.count = 0
        self.drain_event = asyncio.Event()
        self.drain_event.set()  # Initially drained

    def enter(self) -> None:
        self.count += 1
        self.drain_event.clear()

    def exit(self) -> None:
        self.count -= 1
        if self.count == 0:
            self.drain_event.set()


# Module-level references set during lifespan
_orchestrator: ProxyOrchestrator | None = None
_profile_loader: ProfileLoader | None = None
_primary_backend: OllamaAdapter | None = None
_escalation_backend: OllamaAdapter | None = None
_requests = _RequestCounter()

_SHUTDOWN_DRAIN_TIMEOUT = 15.0  # Max seconds to wait for in-flight requests
_STREAM_CHUNK_TIMEOUT = 60.0  # Max seconds to wait between stream chunks


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _orchestrator, _profile_loader, _primary_backend, _escalation_backend

    setup_logging()
    log.info(
        "starting",
        backend_url=settings.backend_url,
        escalation_url=settings.escalation_backend_url,
    )

    # Load profiles
    _profile_loader = ProfileLoader(settings.profiles_dir)

    # Initialize backends (tolerate startup failures for graceful degradation)
    _primary_backend = OllamaAdapter(
        base_url=settings.backend_url,
        default_timeout=settings.backend_timeout,
    )
    try:
        await _primary_backend.startup()
    except Exception:
        log.exception("primary_backend_startup_failed")

    _escalation_backend = OllamaAdapter(
        base_url=settings.escalation_backend_url,
        default_timeout=settings.escalation_timeout,
    )
    try:
        await _escalation_backend.startup()
    except Exception:
        log.exception("escalation_backend_startup_failed")

    _orchestrator = ProxyOrchestrator(
        primary_backend=_primary_backend,
        escalation_backend=_escalation_backend,
        primary_url=settings.backend_url,
        escalation_url=settings.escalation_backend_url,
    )

    log.info("ready", port=settings.port)
    yield

    # Wait for in-flight requests to drain before closing backends
    if _requests.count > 0:
        log.info("shutdown_draining", in_flight=_requests.count)
        try:
            await asyncio.wait_for(
                _requests.drain_event.wait(), timeout=_SHUTDOWN_DRAIN_TIMEOUT
            )
        except asyncio.TimeoutError:
            log.warning("shutdown_drain_timeout", in_flight=_requests.count)

    # Shutdown backends with timeout to prevent hanging
    for backend, name in [(_primary_backend, "primary"), (_escalation_backend, "escalation")]:
        try:
            await asyncio.wait_for(backend.shutdown(), timeout=5.0)
        except asyncio.TimeoutError:
            log.warning(f"{name}_backend_shutdown_timeout")
        except Exception:
            log.exception(f"{name}_backend_shutdown_error")

    log.info("stopped")


app = FastAPI(title="Filthy Tool Fixer", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/health/ready")
async def health_ready() -> JSONResponse:
    if _primary_backend is None:
        return JSONResponse(status_code=503, content={"status": "not_ready", "error": "not initialized"})
    primary_ok = await _primary_backend.health_check()

    escalation_ok = False
    if _escalation_backend:
        escalation_ok = await _escalation_backend.health_check()

    ready = primary_ok  # Only primary is required
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "primary_backend": "ok" if primary_ok else "unavailable",
            "escalation_backend": "ok" if escalation_ok else "unavailable",
        },
    )


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    """Proxy the models list from the primary backend."""
    if _primary_backend is None or _primary_backend._client is None:
        return JSONResponse(status_code=503, content={"error": "Backend not initialized"})
    try:
        resp = await _primary_backend._client.get("/v1/models", timeout=10.0)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception:
        log.exception("models_list_failed")
        return JSONResponse(
            content={"error": "Failed to fetch models from backend"},
            status_code=502,
        )


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request):
    if _orchestrator is None or _profile_loader is None:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "Proxy not initialized", "type": "proxy_error"}},
        )

    rid = generate_request_id()
    request_id_var.set(rid)

    start = time.monotonic()

    try:
        body_bytes = await request.body()
        if len(body_bytes) > 10_000_000:  # 10MB
            return JSONResponse(
                status_code=413,
                content={"error": {"message": "Request body too large", "type": "invalid_request_error"}},
            )
        body = json.loads(body_bytes)
        chat_request = ChatCompletionRequest.model_validate(body)
    except Exception:
        log.exception("invalid_request")
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Invalid request body", "type": "invalid_request_error"}},
        )

    if len(chat_request.messages) > 200:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Too many messages ({len(chat_request.messages)}), max 200", "type": "invalid_request_error"}},
        )

    profile = _profile_loader.match(chat_request.model)
    tool_names = [t.function.name for t in chat_request.tools] if chat_request.tools else []
    log.info(
        "request_received",
        model=chat_request.model,
        has_tools=bool(chat_request.tools),
        tool_names=tool_names,
        stream=chat_request.stream,
        message_count=len(chat_request.messages),
        last_role=chat_request.messages[-1].role if chat_request.messages else "",
    )

    _requests.enter()
    try:
        result = await asyncio.wait_for(
            _orchestrator.handle_request(chat_request, profile),
            timeout=profile.request_timeout,
        )
    except asyncio.TimeoutError:
        log.warning("request_hard_timeout", elapsed_ms=round((time.monotonic() - start) * 1000))
        return JSONResponse(
            status_code=504,
            content={"error": {"message": "Request timed out", "type": "proxy_timeout"}},
        )
    except asyncio.CancelledError:
        log.info("request_cancelled")
        raise
    except Exception:
        log.exception("request_failed")
        return JSONResponse(
            status_code=502,
            content={"error": {"message": "Backend request failed", "type": "proxy_error"}},
        )
    finally:
        _requests.exit()

    elapsed = time.monotonic() - start

    # Streaming response (non-tool passthrough)
    if hasattr(result, "__aiter__"):
        log.info("response_streaming", elapsed_ms=round(elapsed * 1000))
        return StreamingResponse(
            _timeout_stream(result),
            media_type="text/event-stream",
            headers={"X-FilthyToolFixer-Request-ID": rid},
        )

    # Buffered response (tool-calling path returns (response, headers) tuple)
    if isinstance(result, tuple):
        response, extra_headers = result
        headers = {"X-FilthyToolFixer-Request-ID": rid}
        headers.update(extra_headers)
        log.info(
            "response_complete",
            elapsed_ms=round(elapsed * 1000),
            extra_headers=extra_headers,
        )

        # If client requested streaming, synthesize SSE from buffered response
        if chat_request.stream:
            return StreamingResponse(
                _synthesize_sse(response),
                media_type="text/event-stream",
                headers=headers,
            )

        return JSONResponse(
            content=response.model_dump(exclude_none=True),
            headers=headers,
        )

    # Direct non-streaming response (non-tool passthrough)
    log.info("response_complete", elapsed_ms=round(elapsed * 1000))
    return JSONResponse(
        content=result.model_dump(exclude_none=True),
        headers={"X-FilthyToolFixer-Request-ID": rid},
    )


async def _timeout_stream(
    stream: AsyncIterator[bytes],
    chunk_timeout: float = _STREAM_CHUNK_TIMEOUT,
) -> AsyncIterator[bytes]:
    """Wrap a stream with per-chunk timeout to prevent hanging on stalled backends."""
    aiter = stream.__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(aiter.__anext__(), timeout=chunk_timeout)
            yield chunk
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError:
            log.warning("stream_chunk_timeout", timeout=chunk_timeout)
            error = {"error": {"message": "Backend stream stalled", "type": "proxy_timeout"}}
            yield f"data: {json.dumps(error)}\n\n".encode()
            yield b"data: [DONE]\n\n"
            break
        except asyncio.CancelledError:
            raise


async def _synthesize_sse(response) -> AsyncIterator[bytes]:
    """Synthesize SSE events from a buffered ChatCompletionResponse.

    Emits the response as a single chunk followed by [DONE].
    """
    data = response.model_dump(exclude_none=True)
    # Convert to streaming format
    data["object"] = "chat.completion.chunk"
    for choice in data.get("choices", []):
        if "message" in choice:
            delta = choice.pop("message")
            # Ensure content field is present (some clients expect it even as null)
            if "content" not in delta:
                delta["content"] = None
            # Streaming tool calls require an index field on each tool call
            for i, tc in enumerate(delta.get("tool_calls", [])):
                tc["index"] = i
            choice["delta"] = delta

    yield f"data: {json.dumps(data)}\n\n".encode()
    yield b"data: [DONE]\n\n"
