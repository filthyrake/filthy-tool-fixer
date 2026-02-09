# Getting Started

## Prerequisites

- Python 3.11+
- One or two [Ollama](https://ollama.com) instances running
- At least one model pulled (e.g., `ollama pull qwen3:30b-a3b`)

## Installation

```bash
git clone https://github.com/filthyrake/filthy-tool-fixer.git
cd filthy-tool-fixer
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configuration

Copy the example environment file and edit as needed:

```bash
cp .env.example .env
```

The defaults work for a standard local setup (Ollama on `localhost:11434`). See [configuration.md](configuration.md) for all options.

## Running

```bash
./start.sh
```

The proxy starts on port 8079 by default. Verify it's running:

```bash
# Liveness check
curl http://localhost:8079/health
# {"status": "ok"}

# Readiness check (verifies backend connectivity)
curl http://localhost:8079/health/ready
# {"status": "ready", "primary_backend": "ok", "escalation_backend": "ok"}
```

## Your First Request

Send a tool-calling request just like you would to the OpenAI API:

```bash
curl http://localhost:8079/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:30b-a3b",
    "messages": [
      {"role": "user", "content": "What files are in the current directory?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "list_files",
          "description": "List files in a directory",
          "parameters": {
            "type": "object",
            "properties": {
              "path": {"type": "string", "description": "Directory path"}
            },
            "required": ["path"]
          }
        }
      }
    ]
  }'
```

The proxy will:
1. Match the model to a TOML profile
2. Condense the request (if enabled in the profile)
3. Inject a system prompt nudge to encourage tool use
4. Forward to Ollama and validate the response
5. Retry with feedback if validation fails
6. Escalate to the quality model if retries are exhausted
7. Return the validated response with observability headers

Check the response headers for metadata:

```
X-FilthyToolFixer-Request-ID: ftf-abc123
X-FilthyToolFixer-Attempts: 1
```

## Client Integration

### OpenCode

Create `opencode.json` in your project root:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "filthy-tool-fixer": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Filthy Tool Fixer",
      "options": {
        "baseURL": "http://YOUR_SERVER:8079/v1"
      },
      "models": {
        "qwen3-coder:30b": {
          "name": "Qwen3-Coder 30B (Fast)",
          "attachment": true,
          "limit": { "context": 65536, "output": 16384 }
        },
        "devstral-small-2:latest": {
          "name": "Devstral Small 2 (Fast)",
          "attachment": true,
          "limit": { "context": 65536, "output": 8192 }
        },
        "qwen3:30b-a3b": {
          "name": "Qwen3 30B (Fast)",
          "attachment": true,
          "limit": { "context": 32768, "output": 8192 }
        }
      }
    }
  },
  "model": "filthy-tool-fixer/qwen3-coder:30b"
}
```

Then run:

```bash
opencode -m filthy-tool-fixer/qwen3-coder:30b
```

### aider

```bash
aider --openai-api-base http://YOUR_SERVER:8079/v1 --model openai/qwen3:30b-a3b
```

### Any OpenAI-compatible client

Point the `base_url` at `http://YOUR_SERVER:8079/v1/` and set the model name to match an Ollama model with a profile. No API key required (pass any string if the client requires one).

## Next Steps

- [Configuration](configuration.md) — Tune environment variables and profile options
- [Models](models.md) — Learn about each model's quirks and best use cases
- [Architecture](architecture.md) — Understand how the proxy works under the hood
