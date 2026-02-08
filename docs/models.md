# Supported Models

Every model has its own personality when it comes to tool calling. This guide covers what we've learned about each one and how the proxy is tuned to compensate.

## At a Glance

| Model | Tier | VRAM/RAM | Speed | Tool Accuracy | Profile |
|-------|------|----------|-------|---------------|---------|
| Qwen3-Coder 30B | Fast | 19GB GPU | 1-15s | Excellent | `qwen3-coder.toml` |
| Qwen3-Coder 480B | Quality | 290GB hybrid | 1-2 min | Perfect | `qwen3-coder-480b.toml` |
| Qwen3 30B | Fast | 18GB GPU | 10-18s | Good | `qwen3.toml` |
| Qwen3 235B | Quality | 142GB hybrid | 2-5 min | Very good | `qwen3-235b.toml` |
| GPT-OSS 20B | Fast | 12GB GPU | 1-4s | Good (simple), Weak (complex) | `gpt-oss.toml` |
| GPT-OSS 120B | Quality | 65GB hybrid | 39-56s | Excellent | `gpt-oss-120b.toml` |
| Llama 4 Maverick | Quality | large hybrid | varies | Moderate | `llama4-maverick.toml` |
| Llama 4 Scout | Fast | - | - | Not working | `llama4-scout.toml` |
| Llama 3.3 70B | Fast | - | - | Untested | `llama3.3.toml` |

---

## Qwen3 Family

### Qwen3-Coder 30B-A3B (MoE, 3.3B active)

**The star of the show.** Same MoE architecture as regular Qwen3 30B but purpose-built for code and agentic tool use. Runs fully on GPU (~1-15s per tool call).

**Why it's great:**
- Does **not** use `<think>` tags — no stripping needed
- Produces clean, native tool calls on the first attempt most of the time
- 256K native context window
- In testing: 25+ consecutive requests at 100% success rate, 50 messages deep

**Quirks:**
- Occasionally narrates mid-conversation (just like regular Qwen3), but self-corrects after a single nudge
- Without `accept_text_after_tool_use = true`, it will loop forever making tool calls instead of giving a final answer — a coding model that literally can't stop coding
- No `<think>` tags, no embedded JSON blobs, no hallucinated parameters

**Profile tuning:**
```toml
strip_thinking = false        # No think tags to strip
accept_text_after_tool_use = true  # Let it give final answers
condense_tools = true         # Reduce context pressure
```

### Qwen3-Coder 480B-A35B (MoE, 35B active)

The 480B quality-tier coder model. 290GB, runs hybrid CPU/GPU on port 11435. Tool call accuracy is **perfect** — every completion validated first attempt with zero retries.

**Limitations:** Speed. ~1-2 minutes per turn on short contexts, longer conversations (8+ messages) can exceed timeouts. This is a hardware limitation (290GB on 377GB RAM with 24GB VRAM), not a model quality issue.

**Quirks:**
- Same clean tool-calling behavior as the 30B
- Best suited for short, high-stakes exchanges where quality > speed
- Escalation disabled — nothing bigger to escalate to

### Qwen3 30B-A3B (MoE, 3B active)

**The workhorse.** Fast, reliable, takes direction well. Runs fully on GPU (~10-18s per tool call).

**Why it works:**
- Responds well to error feedback — usually self-corrects within 2-3 retries
- Benefits heavily from system prompt nudging
- Escalates to the 235B when all else fails

**Quirks:**
- Wraps reasoning in `<think>` tags — needs `strip_thinking = true`
- Occasionally hallucinates plausible-sounding parameters that don't exist in the schema
- Responds well to structured "CRITICAL RULES" system prompts
- Without a system nudge, will happily describe what it *would* do instead of doing it

**Profile tuning:**
```toml
strip_thinking = true          # Clean up internal monologue
condense_tools = true          # 85% reduction in tool description size
max_system_tokens = 800        # Hard cap on system prompt
```

### Qwen3 235B-A22B (MoE, 22B active)

**The big gun.** Runs hybrid CPU/GPU (~2-5 min per tool call). Don't use it as your daily driver — it's the escalation target when the 30B can't figure it out.

**Why it works:**
- Rarely needs retries (`max_retries = 1`)
- Surprisingly good at recovering from the 30B's mistakes when given the same context

**Quirks:**
- Same `<think>` tag habit as its smaller sibling
- Understands what went wrong and corrects course when given the 30B's error context

---

## GPT-OSS Family

### GPT-OSS 20B (MoE, 3.6B active)

OpenAI's open-weight model (August 2025). **Purpose-built for tool calling** with native OpenAI format. Ships pre-quantized in MXFP4. Blazing fast — 1-4 seconds per tool call on warm requests.

**Why it's great:**
- ~12GB, fits entirely on A30 GPU
- Matches o3-mini on TauBench
- Extremely fast: 1-4s per tool call after warm-up (first request ~30s for model load)
- Simple tool calls (read, glob, grep) work cleanly on first attempt
- Natural 20B -> 120B escalation chain

**Quirks:**
- **Complex nested schemas are its weakness.** Tools with deeply nested required fields (like OpenCode's `question` tool with its `questions[].header` and `questions[].options` structure) consistently fail — the model flattens nested fields to the top level or sends wrong types. Escalation to the 120B handles this.
- **Narration tendency in longer conversations.** Starting around message 12+, begins responding with text instead of tool calls on roughly half of requests. The proxy's narration feedback nudge corrects it on the next attempt (typically adds ~5s). Setting `accept_text_after_tool_use = true` lets it give final answers without looping.
- Weak on multilingual/Chinese tasks.

**Profile tuning:**
```toml
strip_thinking = false                # No think tags
condense_tools = true                 # Reduce context pressure
accept_text_after_tool_use = true     # Narrates mid-conversation, let it answer
max_system_tokens = 800               # Keep context tight
# Escalation to 120B covers complex schema failures
```

### GPT-OSS 120B (MoE, 5.1B active)

The quality-tier GPT-OSS model. ~65GB in MXFP4, runs hybrid CPU/GPU on port 11435. **Zero validation failures** in testing — every tool call valid on first attempt, including complex nested schemas that the 20B can't handle.

**Why it's great:**
- Matches o4-mini on TauBench
- 100% first-attempt tool call accuracy across all tested tools
- Handles complex nested schemas (like OpenCode's `question` tool) that break the 20B
- 39-56s per call once warm, first request ~120s (loading 65GB into memory)
- Escalation target for GPT-OSS 20B

**Quirks:**
- Same narration tendency as the 20B at deeper conversations (~message 10+). Responds with text instead of tool calls. With `accept_text_after_tool_use = true`, this is accepted as the final answer without a costly retry.
- Speeds up as context builds — went from 56s to 39s over the session as the model warmed up.

**Profile tuning:**
```toml
max_retries = 2                        # Quality model, fewer retries needed
request_timeout = 900.0                # Generous timeout for hybrid mode
accept_text_after_tool_use = true      # Narrates at deeper conversations
escalation.enabled = false             # Nothing bigger to escalate to
```

---

## Llama Family

### Llama 4 Maverick (400B MoE, 17B active)

**The storyteller.** Maverick *really* wants to explain itself. It will write a paragraph about what it's going to do, then tack a JSON tool call blob onto the end.

**How the proxy handles it:**
The `_rescue_embedded_tool_calls` feature was built specifically for Maverick. It:
1. Scans the text response for embedded JSON objects
2. Parses them as tool calls
3. Strips the JSON from the text, preserving the narration
4. Returns both the tool call and the cleaned text

**Quirks:**
- Mixes text and tool calls in the same response
- Occasionally sends empty arguments `{}` on the first try, then fixes after feedback
- Will search for `requirements.txt` three times before trying `pyproject.toml`
- Needs `accept_text_after_tool_use = true` or it loops forever

### Llama 4 Scout (109B MoE, 17B active)

**Not working.** Does not produce usable tool calls even with aggressive proxy workarounds (`tool_choice_override = "required"`, `exclude_tools`, simplified system prompt, reduced `num_ctx`).

The profile exists for experimentation. Use Maverick instead.

### Llama 3.3 70B (Dense)

**Untested.** Profile exists but hasn't been validated. Configuration assumes it behaves similarly to the Llama 4 family. If you test it, let us know.

---

## Escalation Chains

Each fast model has a corresponding quality model it escalates to:

```
Fast (GPU)                     Quality (Hybrid CPU/GPU)
──────────                     ───────────────────────
qwen3-coder:30b          ->    qwen3-coder:480b
qwen3:30b-a3b            ->    qwen3:235b-a22b
gpt-oss:20b              ->    gpt-oss:120b
llama4:scout             ->    llama4:maverick
llama3.3:70b             ->    llama4:maverick
```

Escalation adds an error summary from the failed attempt so the quality model understands what went wrong.

---

## Adding a New Model

1. **Pull the model** on the appropriate Ollama instance:
   ```bash
   # Fast model (GPU)
   ollama pull your-model:small

   # Quality model (hybrid)
   OLLAMA_HOST=localhost:11435 ollama pull your-model:large
   ```

2. **Create a profile** in `profiles/`:
   - Start by copying the profile of the most similar existing model
   - Set the `pattern` to match your model name
   - Name the file so it sorts correctly (more specific before general)

3. **Tune the profile:**
   - Does it use `<think>` tags? Set `strip_thinking = true`
   - Does it narrate instead of calling tools? Add a strong `system_suffix`
   - Does it mix text + tool calls? Set `accept_text_after_tool_use = true`
   - Does it struggle with large contexts? Enable `condense_tools` and `condense_system_prompt`
   - Does it have a quality counterpart? Configure `[escalation]`

4. **Test it:**
   ```bash
   curl http://localhost:8079/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"your-model:small","messages":[{"role":"user","content":"Read the file README.md"}],"tools":[{"type":"function","function":{"name":"read","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]}'
   ```

5. **Watch the logs** for validation errors, rescues, and retries. Adjust the profile accordingly.

---

## Model Research

See [MODEL_RESEARCH.md](../MODEL_RESEARCH.md) in the project root for candidates being evaluated for future support, including Mistral Small 3.2, Devstral, DeepSeek V3.1, and Command-R.
