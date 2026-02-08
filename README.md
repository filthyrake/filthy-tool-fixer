# Filthy Tool Fixer

OpenAI-compatible proxy that makes local LLM tool calling actually work — through schema validation, retry with error feedback, model escalation, and context condensing.  This is *very much* a work in progress.

## Why

Local LLMs fail at tool calling not because of format issues (Ollama handles that) but because models pick wrong tools, hallucinate arguments, break on multi-turn, or narrate instead of calling. Filthy Tool Fixer sits between your client and Ollama, catches these failures, and fixes them automatically.

## How it works

```
Client (aider, OpenCode, etc.)
    │
    ▼
Filthy Tool Fixer Proxy (:8079)
    ├── Profile matching (model → TOML config)
    ├── Per-profile backend routing (model → correct Ollama instance)
    ├── Context condensing (strip verbose tool descriptions & system prompts)
    ├── System prompt injection ("call tools, don't describe")
    ├── Schema validation (name, args, types, no hallucinated fields)
    ├── Input limits (max tool calls, argument size, message count)
    ├── Retry with error feedback (configurable attempts per profile)
    ├── Model escalation (30B fails → 235B takes over)
    ├── Hard timeout enforcement (per-request budget)
    └── Think-tag stripping (<think> blocks removed)
    │                          │
    ▼                          ▼
Ollama :11434              Ollama :11435
(fast tier, GPU)           (quality tier, hybrid)
qwen3-coder:30b            qwen3-coder:480b
qwen3:30b-a3b              qwen3:235b-a22b
llama4:scout               llama4:maverick
llama3.3:70b
```

Non-tool requests are pure streaming passthrough. Tool-calling requests are buffered for validation.

## Quick start

```bash
# On the server (YOUR_SERVER_IP)
cd ~/filthy-tool-fixer
cp .env.example .env        # edit if needed
source .venv/bin/activate
./start.sh

# From any OpenAI-compatible client
curl http://YOUR_SERVER_IP:8079/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3:30b-a3b","messages":[{"role":"user","content":"What is the weather?"}],"tools":[...]}'
```

## Server setup

### Prerequisites

- Two Ollama instances on ports 11434 (fast/GPU) and 11435 (quality/hybrid)
- Python 3.11+

### Ollama instances

The primary instance runs on the default port. Create a second systemd unit for the quality model:

```ini
# /etc/systemd/system/ollama-quality.service
[Unit]
Description=Ollama Quality Model Service (port 11435)
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=0.0.0.0:11435"
Environment="OLLAMA_MODELS=/usr/share/ollama/.ollama/models"
Environment="OLLAMA_FLASH_ATTENTION=1"

[Install]
WantedBy=default.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ollama-quality
```

Pre-warm both models so they're loaded into memory:

```bash
# Load 30B onto GPU
curl http://localhost:11434/api/generate -d '{"model":"qwen3:30b-a3b","prompt":"","keep_alive":"30m"}'

# Load 235B into RAM (hybrid)
curl http://localhost:11435/api/generate -d '{"model":"qwen3:235b-a22b","prompt":"","keep_alive":"30m"}'
```

### Install and run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
./start.sh
```

Or via tmux for persistence:

```bash
tmux new-session -d -s filthy './start.sh'
```

### Firewall

```bash
sudo firewall-cmd --add-port=8079/tcp --permanent
sudo firewall-cmd --reload
```

## Configuration

All settings via environment variables (prefix `FILTHY_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `FILTHY_HOST` | `0.0.0.0` | Bind address |
| `FILTHY_PORT` | `8079` | Proxy port |
| `FILTHY_BACKEND_URL` | `http://localhost:11434` | Primary Ollama instance |
| `FILTHY_ESCALATION_BACKEND_URL` | `http://localhost:11435` | Quality model instance |
| `FILTHY_REQUEST_TIMEOUT` | `45.0` | Default total retry loop budget (seconds) |
| `FILTHY_BACKEND_TIMEOUT` | `120.0` | Default single backend request timeout |
| `FILTHY_ESCALATION_TIMEOUT` | `180.0` | Default escalation request timeout |
| `FILTHY_MAX_CONCURRENT_TOOL_REQUESTS` | `3` | Concurrency semaphore |
| `FILTHY_LOG_LEVEL` | `INFO` | Log level |
| `FILTHY_LOG_FORMAT` | `json` or `console` | Log format |
| `FILTHY_PROFILES_DIR` | `profiles` | TOML profiles directory |

## Model profiles

TOML files in `profiles/` configure per-model behavior. Models are matched by `fnmatch` pattern, with more specific patterns checked first (files are loaded in alphabetical order).

### Profile options

**`[model]`** — Model matching and timeout configuration:

| Field | Default | Description |
|-------|---------|-------------|
| `pattern` | `"*"` | fnmatch pattern for model name matching |
| `backend_url` | `""` | Override backend URL for this model (empty = use default primary) |
| `max_retries` | `3` | Maximum retry attempts on validation failure |
| `request_timeout` | `45.0` | Total request budget in seconds (hard-enforced) |
| `backend_timeout` | `120.0` | Single backend call timeout in seconds |

**`[tool_calling]`** — Tool call enhancement and condensing:

| Field | Default | Description |
|-------|---------|-------------|
| `system_suffix` | `""` | System prompt text prepended to tool-calling requests |
| `temperature_override` | `0.0` | Override temperature for tool-calling requests |
| `strip_thinking` | `false` | Remove `<think>` blocks from responses |
| `keep_alive` | `"5m"` | Ollama `keep_alive` parameter |
| `condense_tools` | `false` | Strip verbose examples/notes from tool descriptions |
| `condense_system_prompt` | `false` | Strip verbose sections from client system prompts |
| `max_system_tokens` | `0` | Hard cap on system prompt size (0 = no limit, ~4 chars/token) |
| `tool_choice_override` | `""` | Override `tool_choice` sent to backend (e.g. `"required"` to force tool use) |
| `exclude_tools` | `[]` | Tool names to strip from request before sending to model |
| `num_ctx` | `0` | Override Ollama context window size (0 = use model default) |
| `accept_text_after_tool_use` | `false` | Accept text-only responses when conversation already has tool results (prevents infinite tool-calling loops with models that narrate) |

**`[escalation]`** — Model escalation on failure:

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `false` | Enable escalation to quality model |
| `model` | `""` | Model name for escalation |
| `backend_url` | `""` | Backend URL for escalation model |
| `timeout` | `180.0` | Timeout budget for escalation attempt |

### Example: Fast model with escalation

```toml
# profiles/qwen3.toml
[model]
pattern = "qwen3:*"
max_retries = 3
request_timeout = 120.0
backend_timeout = 120.0

[tool_calling]
system_suffix = "CRITICAL RULES FOR TOOL USE:..."
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

### Example: Quality model with backend routing

```toml
# profiles/qwen3-235b.toml — routes to the correct Ollama instance
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

## Context condensing

Clients like OpenCode send massive payloads — 10K+ char system prompts and 28K+ chars of tool descriptions across 10+ tools. Small models choke on this context and make bad tool choices.

Filthy Tool Fixer can condense both:

- **Tool descriptions**: Strips `<example>` blocks, usage notes, verbose instructions. Keeps the core description paragraph. Typically reduces 28K → 4K chars (85% reduction).
- **System prompts**: Strips git commit instructions, PR creation steps, example blocks. Typically reduces 10K → 3K chars (70% reduction).
- **Hard token cap**: `max_system_tokens` enforces a hard character limit after condensing.

Enable per-profile with `condense_tools = true` and `condense_system_prompt = true`.

## Input limits

The proxy enforces limits to prevent resource exhaustion:

| Limit | Value | Description |
|-------|-------|-------------|
| Request body size | 10 MB | Maximum request body |
| Messages per request | 200 | Maximum conversation messages |
| Tool calls per response | 50 | Maximum tool calls from a single LLM response |
| Argument size | 1 MB | Maximum size of a single tool call's arguments |

## Response headers

Tool-calling responses include observability headers:

| Header | Description |
|--------|-------------|
| `X-FilthyToolFixer-Request-ID` | Unique request ID (correlates with logs) |
| `X-FilthyToolFixer-Attempts` | Number of attempts before success |
| `X-FilthyToolFixer-Escalated` | `true` if quality model was used |
| `X-FilthyToolFixer-Model` | Actual model used (when escalated) |
| `X-FilthyToolFixer-Degraded` | `true` if all attempts failed |

## Health checks

```bash
# Liveness
curl http://YOUR_SERVER_IP:8079/health
# {"status": "ok"}

# Readiness (checks both backends)
curl http://YOUR_SERVER_IP:8079/health/ready
# {"status": "ready", "primary_backend": "ok", "escalation_backend": "ok"}
```

## OpenCode integration

Create an `opencode.json` in your project root pointing at the proxy:

```json
{
  "providers": {
    "filthy-tool-fixer": {
      "type": "openai",
      "url": "http://YOUR_SERVER_IP:8079/v1/"
    }
  },
  "models": {
    "filthy-tool-fixer/qwen3-coder:30b": {
      "provider": "filthy-tool-fixer",
      "model": "qwen3-coder:30b",
      "maxTokens": 16384,
      "contextWindow": 65536
    },
    "filthy-tool-fixer/qwen3-coder:480b": {
      "provider": "filthy-tool-fixer",
      "model": "qwen3-coder:480b",
      "maxTokens": 16384,
      "contextWindow": 65536
    },
    "filthy-tool-fixer/qwen3:30b-a3b": {
      "provider": "filthy-tool-fixer",
      "model": "qwen3:30b-a3b",
      "maxTokens": 16384,
      "contextWindow": 32768
    },
    "filthy-tool-fixer/qwen3:235b-a22b": {
      "provider": "filthy-tool-fixer",
      "model": "qwen3:235b-a22b",
      "maxTokens": 16384,
      "contextWindow": 32768
    }
  },
  "agent": {
    "model": "filthy-tool-fixer/qwen3-coder:30b"
  }
}
```

Then add credentials and run:

```bash
# Add to ~/.local/share/opencode/auth.json:
# "filthy-tool-fixer": {"type": "api", "key": "not-needed"}

# Run with Qwen3-Coder 30B (recommended, ~1-15s per tool call)
opencode -m filthy-tool-fixer/qwen3-coder:30b

# Run with Qwen3 30B (fast, ~10-18s per tool call)
opencode -m filthy-tool-fixer/qwen3:30b-a3b

# Run with Qwen3 235B (quality, ~2-5 min per tool call)
opencode -m filthy-tool-fixer/qwen3:235b-a22b
```

## Tested models

Every model has its own personality when it comes to tool calling. Here's what we've found.

### Qwen3 30B-A3B (MoE, 3B active)

The workhorse. Fast, reliable, and takes direction well. Runs fully on GPU (~10-18s per tool call). Needs `strip_thinking = true` to clean up its internal monologue, and benefits heavily from system prompt nudging — without it, Qwen will happily describe what it *would* do instead of doing it. When it gets stuck (wrong tool name, hallucinated parameters), it responds well to error feedback and usually self-corrects within 2-3 retries. Escalates to the 235B when all else fails. Best bang-for-buck model for everyday coding tasks.

**Quirks**: Wraps reasoning in `<think>` tags. Occasionally hallucinates tool parameters that sound plausible but don't exist in the schema. Responds well to structured "CRITICAL RULES" system prompts.

### Qwen3 235B-A22B (MoE, 22B active)

The big gun. Runs hybrid CPU/GPU (~2-5 min per tool call depending on context), so you don't want it as your daily driver — but when the 30B can't figure it out, the 235B almost always can. Rarely needs retries (max_retries=1). Used primarily as the escalation target for the 30B.

**Quirks**: Same `<think>` tag habit as its smaller sibling. Surprisingly good at recovering from the 30B's mistakes when given the same context — it seems to understand what went wrong and corrects course.

### Qwen3-Coder 30B-A3B (MoE, 3.3B active)

The star of the show. Same MoE architecture as regular Qwen3 30B but purpose-built for code and agentic tool use. Runs fully on GPU (~1-15s per tool call). Does **not** use `<think>` tags — no stripping needed. Produces clean, native tool calls on the first attempt most of the time. When the conversation gets deep (10+ tool rounds), it occasionally responds with text first, but the system prompt nudge corrects it immediately on the second attempt. Zero validation failures, zero escalations needed in testing. 256K native context window.

In testing: 25+ consecutive requests at 100% success rate, 50 messages deep, sub-second responses on follow-up calls. This is the model to beat.

**Quirks**: Occasionally wants to narrate mid-conversation (just like regular Qwen3), but self-corrects after a single nudge. Without `accept_text_after_tool_use = true`, it will loop forever making tool calls instead of giving a final answer — a coding model that literally can't stop coding. No `<think>` tags, no embedded JSON blobs, no hallucinated parameters. Just clean tool calls. Escalates to the 480B if needed but hasn't needed to yet.

### Qwen3-Coder 480B-A35B (MoE, 35B active) — *limited validation*

The 480B quality-tier coder model. 290GB, runs hybrid CPU/GPU on port 11435. 256K native context. Tool call accuracy is **perfect** — every completion validated first attempt with zero retries. The catch is speed: ~1-2 minutes per turn on short contexts, and longer conversations (8+ messages) can exceed the 900s timeout. This is a hardware limitation (290GB model on 377GB RAM with only 24GB VRAM), not a model quality issue. With more GPU memory this model would fly.

**Quirks**: Same clean tool-calling behavior as the 30B — no `<think>` tags, no embedded JSON. Escalation is disabled since there's nothing bigger to escalate to. Best suited for short, high-stakes exchanges where quality matters more than speed.

### Llama 4 Maverick (400B MoE, 17B active)

The storyteller. Maverick *really* wants to explain itself. It will write a paragraph about what it's going to do, then tack a JSON tool call blob onto the end of its text response. The proxy's embedded rescue feature (`_rescue_embedded_tool_calls`) was built specifically for this model — it extracts the tool call from the text, preserves the narration as visible output, and executes the tool separately. Needs `accept_text_after_tool_use = true` or it will loop forever making tool calls without ever giving a final answer.

**Quirks**: Mixes text and tool calls in the same response. Occasionally sends empty arguments (`{}`) on the first try, then fixes them after validation feedback. Will search for `requirements.txt` three times before trying `pyproject.toml`. Charming but scatterbrained.

### Llama 4 Scout (109B MoE, 17B active) — *not working*

Does not produce usable tool calls even with aggressive proxy workarounds (`tool_choice_override = "required"`, `exclude_tools`, simplified system prompt, reduced `num_ctx`). Profile exists for experimentation but Scout is not recommended for tool-calling workloads. Use Maverick instead.

### Llama 3.3 70B (Dense) — *untested*

Profile exists but hasn't been validated yet. Configuration assumes it behaves similarly to the Llama 4 family (system prompt nudging, `accept_text_after_tool_use`, escalation to Maverick). If you test it, let us know how it goes.

## Development

```bash
# Run tests (Mac, no server needed — 71 tests)
source .venv/bin/activate
pytest tests/ -v

# Deploy to server (no rsync available, use tar+scp)
tar czf /tmp/filthy-tool-fixer.tar.gz --exclude='.venv' --exclude='__pycache__' --exclude='.git' --exclude='.env' --exclude='.claude' --exclude='opencode.json' -C ~ filthy-tool-fixer
scp /tmp/filthy-tool-fixer.tar.gz user@YOUR_SERVER_IP:~/
ssh user@YOUR_SERVER_IP "cd ~ && tar xzf filthy-tool-fixer.tar.gz && rm filthy-tool-fixer.tar.gz"
```

## Hardware

- **Server**: Intel Xeon Platinum 8160, NVIDIA A30 24GB, 377GB RAM
- **Qwen3-Coder 30B**: 19GB, runs fully on GPU (~1-15s per tool call)
- **Qwen3-Coder 480B**: 290GB, runs hybrid CPU/GPU (~1-2 min per turn, limited by RAM bandwidth)
- **Qwen3 30B**: 18GB, runs fully on GPU (~10-18s per tool call with validation)
- **Qwen3 235B**: 142GB, runs hybrid CPU/GPU (~2-5 min per tool call depending on context size)
