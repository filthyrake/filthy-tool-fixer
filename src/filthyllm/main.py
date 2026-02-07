"""FastAPI application — routes, lifespan, health checks."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from filthyllm.backends.ollama import OllamaAdapter
from filthyllm.config import settings
from filthyllm.logging import (
    generate_request_id,
    get_logger,
    request_id_var,
    setup_logging,
)
from filthyllm.models import ChatCompletionRequest
from filthyllm.profiles.loader import ProfileLoader
from filthyllm.proxy import ProxyOrchestrator

log = get_logger(__name__)

# Module-level references set during lifespan
_orchestrator: ProxyOrchestrator | None = None
_profile_loader: ProfileLoader | None = None
_primary_backend: OllamaAdapter | None = None
_escalation_backend: OllamaAdapter | None = None
_in_flight: int = 0
_drain_event: asyncio.Event | None = None

_SHUTDOWN_DRAIN_TIMEOUT = 15.0  # Max seconds to wait for in-flight requests


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _orchestrator, _profile_loader, _primary_backend, _escalation_backend, _drain_event

    setup_logging()
    log.info(
        "starting",
        backend_url=settings.backend_url,
        escalation_url=settings.escalation_backend_url,
    )

    _drain_event = asyncio.Event()
    _drain_event.set()  # Initially "drained" (no requests)

    # Load profiles
    _profile_loader = ProfileLoader(settings.profiles_dir)

    # Initialize backends
    _primary_backend = OllamaAdapter(
        base_url=settings.backend_url,
        default_timeout=settings.backend_timeout,
    )
    await _primary_backend.startup()

    _escalation_backend = OllamaAdapter(
        base_url=settings.escalation_backend_url,
        default_timeout=settings.escalation_timeout,
    )
    await _escalation_backend.startup()

    _orchestrator = ProxyOrchestrator(
        primary_backend=_primary_backend,
        escalation_backend=_escalation_backend,
        primary_url=settings.backend_url,
        escalation_url=settings.escalation_backend_url,
    )

    log.info("ready", port=settings.port)
    yield

    # Wait for in-flight requests to drain before closing backends
    if _in_flight > 0:
        log.info("shutdown_draining", in_flight=_in_flight)
        try:
            await asyncio.wait_for(_drain_event.wait(), timeout=_SHUTDOWN_DRAIN_TIMEOUT)
        except asyncio.TimeoutError:
            log.warning("shutdown_drain_timeout", in_flight=_in_flight)

    await _primary_backend.shutdown()
    await _escalation_backend.shutdown()
    log.info("stopped")


app = FastAPI(title="FilthyLLM", lifespan=lifespan)


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
    global _in_flight

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
        import json as _json
        body = _json.loads(body_bytes)
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

    _in_flight += 1
    if _drain_event is not None:
        _drain_event.clear()
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
        _in_flight -= 1
        if _in_flight == 0 and _drain_event is not None:
            _drain_event.set()

    elapsed = time.monotonic() - start

    # Streaming response (non-tool passthrough)
    if hasattr(result, "__aiter__"):
        log.info("response_streaming", elapsed_ms=round(elapsed * 1000))
        return StreamingResponse(
            result,
            media_type="text/event-stream",
            headers={"X-FilthyLLM-Request-ID": rid},
        )

    # Buffered response (tool-calling path returns (response, headers) tuple)
    if isinstance(result, tuple):
        response, extra_headers = result
        headers = {"X-FilthyLLM-Request-ID": rid}
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
        headers={"X-FilthyLLM-Request-ID": rid},
    )


async def _synthesize_sse(response) -> AsyncIterator[bytes]:
    """Synthesize SSE events from a buffered ChatCompletionResponse.

    Emits the response as a single chunk followed by [DONE].
    """
    import json

    data = response.model_dump(exclude_none=True)
    # Convert to streaming format
    data["object"] = "chat.completion.chunk"
    for choice in data.get("choices", []):
        if "message" in choice:
            delta = choice.pop("message")
            # Streaming tool calls require an index field on each tool call
            for i, tc in enumerate(delta.get("tool_calls", [])):
                tc["index"] = i
            choice["delta"] = delta

    yield f"data: {json.dumps(data)}\n\n".encode()
    yield b"data: [DONE]\n\n"
