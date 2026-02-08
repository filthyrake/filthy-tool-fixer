# Development

## Setup

```bash
git clone https://github.com/filthyrake/filthy-tool-fixer.git
cd filthy-tool-fixer
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

Tests run locally without a server — everything is mocked.

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_validation.py -v

# Run with coverage
pytest tests/ --cov=filthy_tool_fixer --cov-report=term-missing
```

The test suite uses `pytest-asyncio` with `asyncio_mode = "auto"` (configured in `pyproject.toml`).

### Test Organization

```
tests/
  conftest.py         # Fixtures only
  helpers.py          # Helper functions and classes (imported by tests)
  test_validation.py  # Schema validation tests
  test_retry.py       # Retry loop tests
  test_proxy.py       # Proxy orchestrator tests
  test_profiles.py    # Profile loading tests
  ...
```

**Convention:** `conftest.py` contains only pytest fixtures. Helper functions and test utilities go in `tests/helpers.py`.

## Project Structure

```
src/filthy_tool_fixer/
  __init__.py
  main.py              # FastAPI app, routes, lifespan, health checks
  config.py            # Settings (env vars with FILTHY_ prefix)
  proxy.py             # Core orchestrator — condensing, injection, routing
  models.py            # Pydantic models (ChatCompletionRequest, etc.)
  logging.py           # Structured logging setup (structlog)
  backends/
    base.py            # Abstract BackendAdapter
    ollama.py          # Ollama adapter (httpx-based)
  retry/
    loop.py            # Retry loop — validation, rescue, repair, escalation
    feedback.py        # Error feedback message construction
  validation/
    schema.py          # Tool call validation against JSON Schema
  profiles/
    loader.py          # TOML profile loader with fnmatch matching
    types.py           # ModelProfile, ToolCallingConfig, EscalationConfig
profiles/
  _default.toml        # Global fallback profile
  qwen3.toml           # Qwen3 30B fast tier
  qwen3-235b.toml      # Qwen3 235B quality tier
  qwen3-coder.toml     # Qwen3-Coder 30B fast tier
  qwen3-coder-480b.toml # Qwen3-Coder 480B quality tier
  gpt-oss.toml         # GPT-OSS 20B fast tier
  gpt-oss-120b.toml    # GPT-OSS 120B quality tier
  llama4-maverick.toml # Llama 4 Maverick
  llama4-scout.toml    # Llama 4 Scout (not working)
  llama3.3.toml        # Llama 3.3 70B (untested)
```

## Key Patterns

### Request flow

1. `main.py` receives the request, parses it, matches a profile
2. `proxy.py` enhances the request (condense, inject, route)
3. If tools present: `retry/loop.py` runs the validation-retry-escalation cycle
4. Tool calls validated by `validation/schema.py`
5. Response returned with observability headers

### Proxy returns

The proxy's `handle_request` returns different types depending on the path:

- **Non-tool passthrough (streaming):** `AsyncIterator[bytes]` — streamed directly
- **Non-tool passthrough (buffered):** `ChatCompletionResponse` — serialized to JSON
- **Tool-calling:** `(ChatCompletionResponse, dict[str, str])` — response + extra headers

### Profile matching

Profiles load in `sorted()` alphabetical order. First `fnmatch` match wins. More specific patterns must sort before their catch-all variant (e.g., `gpt-oss-120b.toml` before `gpt-oss.toml`).

## Dependencies

Core:
- **FastAPI** + **uvicorn** — HTTP server
- **httpx** — Async HTTP client for Ollama
- **pydantic** v2 + **pydantic-settings** — Data models and config
- **structlog** — Structured logging
- **jsonschema** — Tool call argument validation

Dev:
- **pytest** + **pytest-asyncio** — Testing
- **pytest-httpx** — HTTP mocking for httpx
