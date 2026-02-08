# Architecture

Filthy Tool Fixer is a FastAPI application that intercepts OpenAI-compatible chat completion requests, validates tool calls against their schemas, and retries with feedback when models make mistakes.

## Request Flow

```
                    Client Request
                         |
                    [Parse & Validate]
                         |
                    [Profile Matching]
                    model name -> TOML config
                         |
                    [Backend Selection]
                    profile.backend_url -> Ollama instance
                         |
                  +------+------+
                  |             |
            Has tools?    No tools
                  |             |
            [Enhance]    [Passthrough]
                  |        stream directly
                  |
        +---------+---------+
        |                   |
   [Condense Context]  [Inject System Prompt]
   strip verbose desc   "call tools, don't describe"
   truncate system msg
        |                   |
        +---------+---------+
                  |
            [Retry Loop]
                  |
        +---------+---------+
        |                   |
   [Send to Backend]   [Validate Response]
        |                   |
        |              Valid? ----yes----> Return
        |                   |
        |              [Rescue Tool Calls]
        |              extract from text/JSON
        |                   |
        |              [Auto-Repair]
        |              fuzzy-match params
        |                   |
        |              [Build Feedback]
        |              tell model what went wrong
        |                   |
        +---< retry <-------+
                  |
           Retries exhausted?
                  |
            [Escalation]
            switch to quality model
            send error summary
                  |
            Valid? ----yes----> Return with X-Escalated header
                  |
            [Degradation]
            return best attempt
            set X-Degraded header
```

## Key Components

### Profile Matching (`profiles/loader.py`)

TOML files in `profiles/` configure per-model behavior. On startup, all `.toml` files are loaded in **alphabetical order**. When a request arrives, the model name is matched against each profile's `pattern` field using `fnmatch`. First match wins.

This means more specific profiles must sort alphabetically before their catch-all:

```
gpt-oss-120b.toml   pattern = "gpt-oss:120b*"    <-- matches first
gpt-oss.toml         pattern = "gpt-oss:*"         <-- catches all others
```

The `_default.toml` profile matches `*` and provides fallback values for any model without a specific profile.

### Context Condensing (`proxy.py`)

Clients like OpenCode send massive payloads — 10K+ character system prompts and 28K+ characters of tool descriptions across 10+ tools. Small models choke on this.

**Tool condensing** (`condense_tools = true`):
- Strips `<example>` blocks, usage notes, verbose instructions
- Keeps the core description paragraph (first ~300 chars)
- Typical reduction: 28K -> 4K chars (85%)

**System prompt condensing** (`condense_system_prompt = true`):
- Strips git commit instructions, PR creation steps, example blocks
- Removes sections irrelevant to tool-calling decisions
- Typical reduction: 10K -> 3K chars (70%)

**Hard token cap** (`max_system_tokens`):
- Enforces a character limit after condensing (~4 chars/token)
- Truncates at the last clean line break

### System Prompt Injection (`proxy.py`)

The profile's `system_suffix` is **prepended** to the system message (before the client's prompt). This gives the nudge highest priority in the context. Different models need different nudges:

- **Qwen3**: Verbose "CRITICAL RULES FOR TOOL USE" with numbered instructions
- **Llama 4**: Simpler nudge with `accept_text_after_tool_use` to avoid infinite loops
- **GPT-OSS**: Minimal nudge — purpose-built for tool calling
- **Qwen3-Coder**: Rules emphasizing read-only vs write tools based on intent

### Schema Validation (`validation/schema.py`)

Every tool call is validated against the tool definitions provided in the request:

1. **Tool name** — Does the tool exist? If not, fuzzy-match suggests the closest name (severity 3)
2. **JSON parsing** — Are the arguments valid JSON? (severity 3)
3. **Extra fields** — Are there hallucinated parameters not in the schema? (severity 1)
4. **Required fields** — Are all required parameters present? (severity 2)
5. **Type validation** — Do argument types match the JSON Schema? Uses `jsonschema.Draft7Validator` (severity 2)

Severity levels determine which error is shown first in feedback (highest severity wins). For parallel tool calls, the entire batch is rejected if any single call fails.

### Tool Call Rescue (`retry/loop.py`)

Models don't always use the `tool_calls` response field. The retry loop detects and rescues tool calls from several non-standard formats:

| Format | Example | Source Model |
|--------|---------|-------------|
| **Llama native** | `<\|python_start\|>func(args)<\|python_end\|>` | Llama 3/4 |
| **Pythonic** | `[func_name(param="val")]` | Llama 4 |
| **JSON narration** | `{"name": "func", "parameters": {...}}` | Various |
| **Code-fenced JSON** | ` ```json {"name": "func"} ``` ` | Various |
| **Embedded mixed** | Text paragraph + JSON blob at the end | Llama 4 Maverick |

Rescued tool calls are patched into the response's `tool_calls` field and treated as normal from that point on.

### Auto-Repair (`retry/loop.py`)

Before validation, tool call arguments are fuzzy-matched against the schema:

```
Model sends: {"question": "What color?", "option": ["red", "blue"]}
Schema expects: {"questions": [...], "options": [...]}
Auto-repaired: {"questions": "What color?", "options": ["red", "blue"]}
```

Uses `difflib.get_close_matches` with a 0.8 cutoff. Only repairs parameter names — doesn't change values or types.

### Retry Loop (`retry/loop.py`)

The retry loop orchestrates the full validation-feedback-retry cycle:

1. **Budget tracking** — Each attempt deducts from a total time budget. Won't start a retry with less than 10 seconds remaining.
2. **Best-response scoring** — Tracks the best response across attempts: valid (10) > parseable JSON (5) > has tool_calls (3) > has content (1) > nothing (0).
3. **Duplicate detection** — If consecutive attempts produce the same errors, the model is stuck. Break out early.
4. **Narration feedback** — When the model responds with text instead of tool calls, sends specific nudges with tool-specific hints (e.g., "Use 'read' to view file contents").
5. **Semantic checking** — Catches misuse patterns even when the call is technically valid (e.g., `glob` without wildcards in the pattern).
6. **Text acceptance** — With `accept_text_after_tool_use = true`, accepts text responses when the conversation already has tool results (the model is giving its final answer).

### Model Escalation (`retry/loop.py`)

When all retries are exhausted:

1. Check if the profile has escalation enabled with a target model
2. Build a brief error summary from the best failed attempt
3. Send the original conversation + error summary to the quality model on the escalation backend
4. Validate the escalation response (rescue and auto-repair apply here too)
5. If valid, return with `X-FilthyToolFixer-Escalated: true`
6. If also fails, return the best attempt with `X-FilthyToolFixer-Degraded: true`

Escalation chains:

```
qwen3:30b       -> qwen3:235b
qwen3-coder:30b -> qwen3-coder:480b
gpt-oss:20b     -> gpt-oss:120b
```

### Backend Routing (`proxy.py`, `main.py`)

The proxy maintains two Ollama connections (primary + escalation). Profiles can route requests to either backend via `backend_url`:

- **Primary** (`localhost:11434`) — Fast models that fit on GPU
- **Escalation** (`localhost:11435`) — Quality models running hybrid CPU/GPU

The orchestrator maintains a `backends_by_url` map so profiles can freely route to either instance.

### SSE Synthesis (`main.py`)

Tool-calling requests are always buffered (non-streaming) internally for validation. If the client requested streaming, the proxy synthesizes SSE events from the buffered response after validation succeeds.

### Graceful Shutdown (`main.py`)

On shutdown, the proxy waits up to 15 seconds for in-flight requests to complete before closing backend connections. An `asyncio.Event` tracks the drain state.

## Concurrency

A semaphore (`FILTHY_MAX_CONCURRENT_TOOL_REQUESTS`, default 3) limits concurrent tool-calling requests. Non-tool requests are unlimited and stream through directly. This prevents resource exhaustion on the backend when multiple clients make tool-calling requests simultaneously.

## Observability

Every request gets a unique ID (`X-FilthyToolFixer-Request-ID`) that correlates across structured JSON logs. Tool-calling responses include additional headers:

| Header | Meaning |
|--------|---------|
| `X-FilthyToolFixer-Attempts` | Number of attempts before success |
| `X-FilthyToolFixer-Escalated` | `true` if the quality model was used |
| `X-FilthyToolFixer-Model` | Actual model name when escalated |
| `X-FilthyToolFixer-Degraded` | `true` if all attempts failed |

Logs use `structlog` in JSON format (configurable to console). Key events logged:

- `request_received` — Model, tool names, message count
- `system_prompt_condensed` — Before/after character counts
- `tool_calls_rescued_from_text` — Rescued format and tool names
- `param_name_repaired` — Original and corrected parameter names
- `validation_failed` — Error count and messages
- `escalating_to_quality_model` — Escalation target
- `response_complete` — Elapsed time and extra headers
