# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, with dev deps)
source .venv/bin/activate
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_validation.py -v

# Run a single test by name
pytest tests/ -k "test_name_substring" -v

# Lint
ruff check src/ tests/

# Start the proxy server
./start.sh
```

## Architecture

Filthy Tool Fixer is an OpenAI-compatible proxy that intercepts tool-calling requests to local LLMs (Ollama/vLLM), validates the tool calls against JSON schemas, and retries with error feedback or escalates to a larger model on failure. Non-tool requests are pure streaming passthrough.

### Request flow

1. **`main.py`** — FastAPI app, routes, lifespan. Parses the request, matches a model profile, enforces hard timeout via `asyncio.wait_for`, then delegates to `ProxyOrchestrator`.

2. **`proxy.py` (`ProxyOrchestrator`)** — Enhances the request per profile (system prompt injection, context condensing, temperature override, tool description condensing). Non-tool requests go directly to the backend. Tool requests are buffered (never streamed internally) and routed through the retry loop under a concurrency semaphore.

3. **`retry/loop.py` (`RetryLoop`)** — Manages the validation-retry-escalation cycle. On each attempt: call backend → extract tool calls → validate → if invalid, append the failed response + error feedback to messages and retry. Tracks "best response" by score. Detects duplicate errors (model stuck). If all retries fail, tries escalation to a quality model.

4. **`retry/feedback.py`** — Constructs concise error feedback messages for the LLM based on `ValidationResult`. Prioritizes the worst error.

5. **`validation/schema.py`** — Validates tool calls against definitions: tool name existence (with fuzzy "did you mean?"), JSON argument parsing, required fields, type checking via `jsonschema.Draft7Validator`, extra/hallucinated field detection. Returns `ValidationResult` with typed errors and severity scores.

6. **`backends/`** — `BackendAdapter` ABC with `OllamaAdapter` and `VLLMAdapter` implementations. Ollama is the primary backend. vLLM adds constrained decoding via `guided_json`.

7. **`profiles/`** — TOML-based per-model configuration. `ProfileLoader` loads from `profiles/` directory, matches model names via `fnmatch` patterns (more specific patterns checked first via alphabetical file ordering). `_default.toml` is the fallback.

### Key design decisions

- **Tool-calling requests are always buffered** (stream=False internally) even if the client requested streaming. After validation succeeds, a streaming response is synthesized via SSE in `_synthesize_sse`.
- **Return type convention**: `handle_request` returns either a `ChatCompletionResponse` (non-tool), an `AsyncIterator[bytes]` (streaming passthrough), or a `tuple[ChatCompletionResponse, dict[str, str]]` (tool-calling path with extra headers).
- **`X-FilthyToolFixer-*` headers** are the observability surface: Attempts, Escalated, Degraded, Model, Request-ID.
- **Config**: `pydantic-settings` with `FILTHY_` env prefix. Profile TOML sections: `[model]`, `[tool_calling]`, `[escalation]`.

## Testing conventions

- `pytest-asyncio` with `asyncio_mode = "auto"` — no need for `@pytest.mark.asyncio` decorators.
- **`conftest.py`**: fixtures only (shared tool definitions, sample profiles).
- **`tests/helpers.py`**: factory functions (`make_tool_call`, `make_response`, `make_request`) and `MockBackend` (a `BackendAdapter` that returns pre-configured responses and records calls). Import helpers from `tests.helpers`, not from conftest.
- `pytest-httpx` is available for HTTP-level testing.

## Package name

The package was renamed from `filthyllm` to `filthy_tool_fixer`. The import path is `filthy_tool_fixer` and the wheel package is `filthy-tool-fixer`.
