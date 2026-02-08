# Configuration

## Environment Variables

All settings use the `FILTHY_` prefix. Set them in a `.env` file or export directly.

| Variable | Default | Description |
|----------|---------|-------------|
| `FILTHY_HOST` | `0.0.0.0` | Bind address |
| `FILTHY_PORT` | `8079` | Proxy listen port |
| `FILTHY_BACKEND_URL` | `http://localhost:11434` | Primary Ollama instance (fast models) |
| `FILTHY_BACKEND_TYPE` | `ollama` | Primary backend type |
| `FILTHY_ESCALATION_BACKEND_URL` | `http://localhost:11435` | Escalation Ollama instance (quality models) |
| `FILTHY_ESCALATION_BACKEND_TYPE` | `ollama` | Escalation backend type |
| `FILTHY_REQUEST_TIMEOUT` | `45.0` | Default total retry loop budget (seconds) |
| `FILTHY_BACKEND_TIMEOUT` | `120.0` | Default single backend request timeout (seconds) |
| `FILTHY_ESCALATION_TIMEOUT` | `180.0` | Default escalation request timeout (seconds) |
| `FILTHY_MAX_CONCURRENT_TOOL_REQUESTS` | `3` | Max concurrent tool-calling requests |
| `FILTHY_LOG_LEVEL` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `FILTHY_LOG_FORMAT` | `json` | Log format (`json` or `console`) |
| `FILTHY_PROFILES_DIR` | `profiles` | Path to TOML profiles directory |

### Timeout Hierarchy

Timeouts interact in a specific way:

```
request_timeout (total budget for all retries)
  |
  +-- backend_timeout (single call to Ollama, capped by remaining budget)
  |
  +-- backend_timeout (retry 1)
  |
  +-- escalation.timeout (escalation attempt budget)
```

The `request_timeout` is the hard ceiling enforced by `asyncio.wait_for()`. Each individual backend call uses `min(remaining_budget, backend_timeout)`. If less than 10 seconds remain, no retry is attempted.

## Model Profiles

TOML files in the `profiles/` directory configure per-model behavior. Profiles are loaded at startup in **alphabetical filename order**, and the first `fnmatch` pattern match wins.

### Profile Structure

Every profile has three sections:

```toml
[model]
# Model matching and timeout config

[tool_calling]
# Tool call enhancement, condensing, system prompt injection

[escalation]
# Model escalation on failure
```

### `[model]` Section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pattern` | string | `"*"` | `fnmatch` pattern for model name matching |
| `backend_url` | string | `""` | Override backend URL (empty = use primary) |
| `max_retries` | int | `3` | Max retry attempts on validation failure |
| `request_timeout` | float | `45.0` | Total request budget in seconds |
| `backend_timeout` | float | `120.0` | Single backend call timeout in seconds |

### `[tool_calling]` Section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_suffix` | string | `""` | System prompt text prepended to tool-calling requests |
| `temperature_override` | float | `0.0` | Override temperature for tool-calling requests |
| `strip_thinking` | bool | `false` | Remove `<think>` blocks from responses |
| `think_tag_pattern` | string | `"<think>.*?</think>"` | Regex pattern for thinking tags |
| `keep_alive` | string | `"5m"` | Ollama `keep_alive` parameter |
| `condense_tools` | bool | `false` | Strip verbose examples/notes from tool descriptions |
| `condense_system_prompt` | bool | `false` | Strip verbose sections from client system prompts |
| `max_system_tokens` | int | `0` | Hard cap on system prompt size (0 = no limit, ~4 chars/token) |
| `tool_choice_override` | string | `""` | Override `tool_choice` sent to backend (e.g., `"required"`) |
| `exclude_tools` | list | `[]` | Tool names to strip from request before sending to model |
| `num_ctx` | int | `0` | Override Ollama context window size (0 = model default) |
| `accept_text_after_tool_use` | bool | `false` | Accept text-only responses when conversation already has tool results |

### `[escalation]` Section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable escalation to quality model |
| `model` | string | `""` | Model name for escalation |
| `backend_url` | string | `""` | Backend URL for escalation model |
| `timeout` | float | `180.0` | Timeout budget for escalation attempt |

### Pattern Matching Order

Profiles are loaded alphabetically by filename. More specific patterns must sort before their catch-all:

```
gpt-oss-120b.toml    pattern = "gpt-oss:120b*"   <-- checked first
gpt-oss.toml         pattern = "gpt-oss:*"        <-- catches the rest
qwen3-235b.toml      pattern = "qwen3:235b*"      <-- checked first
qwen3.toml           pattern = "qwen3:*"           <-- catches the rest
_default.toml        pattern = "*"                  <-- global fallback
```

If no profile matches (shouldn't happen with `_default.toml`), the proxy uses hardcoded defaults.

### Example: Fast Model with Escalation

```toml
# profiles/qwen3.toml
[model]
pattern = "qwen3:*"
max_retries = 3
request_timeout = 120.0
backend_timeout = 120.0

[tool_calling]
system_suffix = "CRITICAL RULES FOR TOOL USE:\n1. You MUST use the provided tools..."
temperature_override = 0.0
strip_thinking = true
keep_alive = "30m"
condense_tools = true
condense_system_prompt = true
max_system_tokens = 800

[escalation]
enabled = true
model = "qwen3:235b-a22b"
backend_url = "http://localhost:11435"
timeout = 180.0
```

### Example: Quality Model (No Escalation)

```toml
# profiles/qwen3-235b.toml
[model]
pattern = "qwen3:235b*"
backend_url = "http://localhost:11435"
max_retries = 1
request_timeout = 900.0
backend_timeout = 600.0

[tool_calling]
system_suffix = "CRITICAL RULES FOR TOOL USE:..."
temperature_override = 0.0
strip_thinking = true
keep_alive = "30m"
condense_tools = true
condense_system_prompt = true
max_system_tokens = 800

[escalation]
enabled = false
```

### Creating a New Profile

1. Create a new `.toml` file in `profiles/`
2. Name it so it sorts correctly relative to other profiles for the same family
3. Set the `pattern` to match your model name(s)
4. Configure tool calling behavior based on the model's quirks
5. Set up escalation if a larger model is available
6. Restart the proxy to load the new profile

See [models.md](models.md) for per-model tuning recommendations.
