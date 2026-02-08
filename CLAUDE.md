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

Filthy Tool Fixer is an OpenAI-compatible proxy that intercepts tool-calling requests to local LLMs (Ollama/vLLM), rescues malformed tool calls from non-standard output formats, auto-repairs near-miss parameter names, validates against JSON schemas, retries with error feedback, and escalates to a larger model on failure. Non-tool requests are pure streaming passthrough.

### Request flow

1. **`main.py`** — FastAPI app, routes, lifespan. Parses the request, matches a model profile, enforces hard timeout via `asyncio.wait_for`, then delegates to `ProxyOrchestrator`. Also handles SSE synthesis (`_synthesize_sse`) for tool-calling responses, `/v1/models` passthrough, graceful shutdown drain (15s), request body size limit (10MB), and message count limit (200).

2. **`proxy.py` (`ProxyOrchestrator`)** — Enhances the request per profile:
   - System prompt injection and condensing (`condense_system_prompt`, `max_system_tokens`)
   - Tool description condensing (`condense_tools`)
   - Temperature override for tool-calling requests
   - `tool_choice_override` — force tool use (e.g. `"required"` for Scout)
   - `exclude_tools` — strip tools the model can't handle, with post-filter routing (if all tools removed, falls through to non-tool passthrough)
   - `num_ctx` — override Ollama context window (merged with existing options)
   - Per-profile backend routing via `backend_url` (selects which Ollama instance to use)

   Non-tool requests go directly to the backend. Tool requests are buffered (never streamed internally) and routed through the retry loop under a concurrency semaphore.

3. **`retry/loop.py` (`RetryLoop`)** — The core pipeline: **rescue → repair → validate → retry → escalate**.

   On each attempt:
   - Call backend, get response
   - **Rescue**: If no tool calls in response, attempt to extract them from text content via a layered rescue pipeline:
     1. Narrated JSON rescue — parses `{"name": "tool", "arguments": {...}}` from text, including code-fenced JSON
     2. Llama native format — `<|python_start|>func(args)<|python_end|>` and `<|python_tag|>` patterns
     3. Pythonic bracket format — `[func_name(param="val")]` (Llama 4 style)
     4. Embedded JSON rescue — extracts JSON tool call objects from mixed narrative text (the "Maverick pattern"), preserving surrounding text as content
   - **Repair**: Auto-fix near-miss parameter names via `difflib.get_close_matches` (0.8 cutoff)
   - **Validate**: Schema validation (name, args, types, required fields, extra fields)
   - **Semantic check**: Post-validation checks (e.g. glob without wildcards)
   - If invalid: append failed response + error feedback to messages and retry
   - **`accept_text_after_tool_use`**: When enabled and conversation already has `role="tool"` messages, text responses are accepted immediately instead of nudging for more tool calls (prevents infinite loops with models like Maverick and Qwen3-Coder)
   - Tracks "best response" by score, detects duplicate errors (model stuck)
   - If all retries fail, tries escalation (which also applies rescue and repair)

4. **`retry/feedback.py`** — Constructs concise error feedback messages for the LLM based on `ValidationResult`. Prioritizes the worst error.

5. **`validation/schema.py`** — Validates tool calls against definitions: tool name existence (with fuzzy "did you mean?"), JSON argument parsing, required fields, type checking via `jsonschema.Draft7Validator`, extra/hallucinated field detection. Returns `ValidationResult` with typed errors and severity scores.

6. **`backends/`** — `BackendAdapter` ABC with `OllamaAdapter` and `VLLMAdapter` implementations. Ollama is the primary backend. vLLM adds constrained decoding via `guided_json`.

7. **`profiles/`** — TOML-based per-model configuration. `ProfileLoader` loads from `profiles/` directory, matches model names via `fnmatch` patterns (more specific patterns checked first via alphabetical file ordering). `_default.toml` is the fallback. Current profiles: `_default`, `qwen3`, `qwen3-235b`, `qwen3-coder`, `qwen3-coder-480b`, `llama3.3`, `llama4-maverick`, `llama4-scout`.

### Key design decisions

- **Tool-calling requests are always buffered** (stream=False internally) even if the client requested streaming. After validation succeeds, `main.py` synthesizes a streaming response via SSE in `_synthesize_sse`.
- **Return type convention**: `handle_request` returns `ChatCompletionResponse | AsyncIterator[bytes]` for non-tool paths. The tool-calling path (`_handle_tool_request`) returns `tuple[ChatCompletionResponse, dict[str, str]]` with extra headers; `main.py` dispatches via `isinstance(result, tuple)`.
- **Rescue before validation** is a core strategy. The system handles four non-standard output formats (narrated JSON, Llama native tags, pythonic syntax, embedded JSON in text) before ever reaching the validator. This means models that can't produce proper tool call format can still work.
- **Parameter auto-repair** silently corrects near-miss parameter names (e.g. `file_path` → `filePath`) before validation. Deliberate trade-off: tolerance over strictness.
- **Semantic validation** adds a second layer beyond structural schema checks (currently: glob without wildcards).
- **`X-FilthyToolFixer-*` headers** are the observability surface: Attempts, Escalated, Degraded, Model, Request-ID.
- **Config**: `pydantic-settings` with `FILTHY_` env prefix. Profile TOML sections: `[model]`, `[tool_calling]`, `[escalation]`. Key newer `[tool_calling]` fields: `tool_choice_override`, `exclude_tools`, `num_ctx`, `accept_text_after_tool_use`.

## Testing conventions

- `pytest-asyncio` with `asyncio_mode = "auto"` in pyproject.toml.
- **`conftest.py`**: fixtures only (shared tool definitions, sample profiles).
- **`tests/helpers.py`**: factory functions (`make_tool_call`, `make_response`, `make_request`) and `MockBackend` (a `BackendAdapter` that returns pre-configured responses and records calls). Import helpers from `tests.helpers`, not from conftest.
- Tests organized into thematic classes: `TestFeedbackMessage`, `TestRetryLoop`, `TestToolCallRescue`, `TestEscalation`, `TestBestAttemptScoring`, `TestParamNameRepair`.
- Test files: `test_validation.py`, `test_profiles.py`, `test_proxy.py`, `test_retry.py`.
- `pytest-httpx` is available for HTTP-level testing.

## Package name

The package was renamed from `filthyllm` to `filthy_tool_fixer`. The import path is `filthy_tool_fixer` and the wheel package is `filthy-tool-fixer`.
