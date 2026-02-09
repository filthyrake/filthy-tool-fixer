<div align="center">

# Filthy Tool Fixer

**Make local LLM tool calling actually work.**

Schema validation | Retry with feedback | Model escalation | Context condensing

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## The Problem

Local LLMs fail at tool calling — not because of format issues (Ollama handles that), but because models pick wrong tools, hallucinate arguments, narrate instead of calling, or break on multi-turn conversations.

## The Solution

Filthy Tool Fixer is an OpenAI-compatible proxy that sits between your client and Ollama. It catches tool-calling failures and fixes them automatically through validation, retry with error feedback, and model escalation.

```
Your Client (aider, OpenCode, etc.)
       |
       v
  Filthy Tool Fixer (:8079)
       |                |
       v                v
  Ollama :11434    Ollama :11435
  (fast, GPU)      (quality, hybrid)
```

## What It Does

- **Schema validation** — Catches wrong tools, hallucinated parameters, type errors, missing required fields
- **Retry with feedback** — Tells the model exactly what went wrong and asks it to try again
- **Tool call rescue** — Extracts tool calls from narrated text, embedded JSON, Llama native format
- **Auto-repair** — Fuzzy-matches near-miss parameter names (e.g., `question` -> `questions`)
- **Model escalation** — Fast model fails? Automatically escalates to the quality model
- **Context condensing** — Strips verbose tool descriptions and system prompts to fit small context windows
- **Per-model profiles** — TOML configs tuned for each model's quirks and capabilities
- **Observability headers** — Track attempts, escalations, and degradation per request

## Quick Start

```bash
# Clone and install
git clone https://github.com/filthyrake/filthy-tool-fixer.git
cd filthy-tool-fixer
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Configure
cp .env.example .env   # edit as needed

# Run
./start.sh
```

Point any OpenAI-compatible client at `http://YOUR_SERVER:8079/v1/` and go.

```bash
curl http://YOUR_SERVER:8079/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:30b-a3b",
    "messages": [{"role": "user", "content": "List files in the current directory"}],
    "tools": [...]
  }'
```

## Supported Models

| Model | Type | Speed | Notes |
|-------|------|-------|-------|
| **Qwen3-Coder 30B** | MoE (3.3B active) | 1-15s | Best overall. Clean tool calls, no think tags |
| **Qwen3-Coder 480B** | MoE (35B active) | 1-2 min | Perfect accuracy, limited by RAM bandwidth |
| **Qwen3 30B** | MoE (3B active) | 10-18s | Reliable workhorse, needs think-tag stripping |
| **Qwen3 235B** | MoE (22B active) | 2-5 min | Escalation target, rarely needs retries |
| **GPT-OSS 20B** | MoE (3.6B active) | 1-4s | Fast, simple tools great, complex schemas need escalation |
| **GPT-OSS 120B** | MoE (5.1B active) | 39-56s | Zero validation failures, handles complex schemas |
| **Llama 4 Maverick** | MoE (17B active) | varies | Narrates + calls, rescue extracts tool calls |
| **Llama 4 Scout** | MoE (17B active) | - | Not working for tool calling |
| **Llama 3.3 70B** | Dense | 60-100s | Working, needs retries for param names |

See [docs/models.md](docs/models.md) for detailed quirks and tuning notes per model.

## Documentation

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, first request, client setup |
| [Architecture](docs/architecture.md) | How the proxy works under the hood |
| [Configuration](docs/configuration.md) | Environment variables and profile options |
| [Models](docs/models.md) | Supported models, quirks, and tuning |
| [API Reference](docs/api.md) | Endpoints, headers, error codes |
| [Deployment](docs/deployment.md) | Server setup, Ollama instances, production tips |
| [Development](docs/development.md) | Running tests, contributing, deploying changes |

## Hardware

Our reference setup:

- **Server**: Intel Xeon Platinum 8160, NVIDIA A30 24GB, 377GB RAM
- **Fast models** (GPU): Qwen3-Coder 30B (19GB), Qwen3 30B (18GB), GPT-OSS 20B (12GB)
- **Quality models** (hybrid CPU/GPU): Qwen3-Coder 480B (290GB), Qwen3 235B (142GB), GPT-OSS 120B (65GB)

## License

MIT
