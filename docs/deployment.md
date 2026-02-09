# Deployment

## Server Requirements

- Linux server with Python 3.11+
- NVIDIA GPU with enough VRAM for your fast-tier model (A30 24GB works for 30B models)
- Enough system RAM for quality-tier models in hybrid mode
- Two Ollama instances: one for fast models (GPU), one for quality models (hybrid)

### Reference Hardware

Our setup: Intel Xeon Platinum 8160, NVIDIA A30 24GB, 377GB RAM.

| Model | Size | Runs on | Speed |
|-------|------|---------|-------|
| Qwen3-Coder 30B | 19GB | GPU (A30) | 1-15s |
| Qwen3 30B | 18GB | GPU (A30) | 10-18s |
| Devstral Small 2 24B | 15GB | GPU (A30) | 1-7s |
| GPT-OSS 20B | 12GB | GPU (A30) | 1-4s |
| Qwen3-Coder 480B | 290GB | Hybrid CPU/GPU | 1-2 min |
| Qwen3 235B | 142GB | Hybrid CPU/GPU | 2-5 min |
| GPT-OSS 120B | 65GB | Hybrid CPU/GPU | 39-56s |

## Ollama Setup

### Primary Instance (Fast Models)

The default Ollama installation runs on port 11434. This is your fast tier — models that fit on the GPU.

```bash
# Pull fast-tier models
ollama pull qwen3-coder:30b
ollama pull qwen3:30b-a3b
ollama pull gpt-oss:20b
ollama pull devstral-small-2
```

### Escalation Instance (Quality Models)

Create a second systemd unit for quality models on port 11435:

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

# Pull quality-tier models
OLLAMA_HOST=localhost:11435 ollama pull qwen3-coder:480b
OLLAMA_HOST=localhost:11435 ollama pull qwen3:235b-a22b
OLLAMA_HOST=localhost:11435 ollama pull gpt-oss:120b
```

### Pre-warming Models

Ollama loads models into memory on first request, which adds cold-start latency. Pre-warm them:

```bash
# Fast tier
curl http://localhost:11434/api/generate -d '{"model":"qwen3-coder:30b","prompt":"","keep_alive":"30m"}'
curl http://localhost:11434/api/generate -d '{"model":"qwen3:30b-a3b","prompt":"","keep_alive":"30m"}'
curl http://localhost:11434/api/generate -d '{"model":"gpt-oss:20b","prompt":"","keep_alive":"30m"}'
curl http://localhost:11434/api/generate -d '{"model":"devstral-small-2","prompt":"","keep_alive":"30m"}'

# Quality tier
curl http://localhost:11435/api/generate -d '{"model":"qwen3-coder:480b","prompt":"","keep_alive":"30m"}'
curl http://localhost:11435/api/generate -d '{"model":"qwen3:235b-a22b","prompt":"","keep_alive":"30m"}'
curl http://localhost:11435/api/generate -d '{"model":"gpt-oss:120b","prompt":"","keep_alive":"30m"}'
```

The `keep_alive` parameter controls how long the model stays loaded (default varies by Ollama version). Profiles set this per-model — most use `30m`.

## Installing the Proxy

```bash
cd ~
git clone https://github.com/filthyrake/filthy-tool-fixer.git
cd filthy-tool-fixer
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Edit `.env` if your Ollama instances are on different hosts or ports.

## Running

### Direct

```bash
source .venv/bin/activate
./start.sh
```

### tmux (persistent)

```bash
tmux new-session -d -s filthy './start.sh'
```

Reattach with `tmux attach -t filthy`.

### systemd

```ini
# /etc/systemd/system/filthy-tool-fixer.service
[Unit]
Description=Filthy Tool Fixer Proxy
After=network-online.target ollama.service ollama-quality.service

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/filthy-tool-fixer
ExecStart=/home/YOUR_USER/filthy-tool-fixer/.venv/bin/python -m uvicorn filthy_tool_fixer.main:app --host 0.0.0.0 --port 8079
Restart=always
RestartSec=5
Environment="FILTHY_LOG_FORMAT=json"

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now filthy-tool-fixer
```

## Firewall

Open the proxy port:

```bash
# firewalld
sudo firewall-cmd --add-port=8079/tcp --permanent
sudo firewall-cmd --reload

# ufw
sudo ufw allow 8079/tcp
```

## Deploying Updates

Our server doesn't have rsync, so we use tar+scp:

```bash
# On dev machine
tar czf /tmp/filthy-tool-fixer.tar.gz \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='.git' \
  --exclude='.env' \
  --exclude='.claude' \
  --exclude='opencode.json' \
  -C ~ filthy-tool-fixer

scp /tmp/filthy-tool-fixer.tar.gz user@YOUR_SERVER:~/

# On server
ssh user@YOUR_SERVER "cd ~ && tar xzf filthy-tool-fixer.tar.gz && rm filthy-tool-fixer.tar.gz"
```

Then restart the proxy (or the systemd service).

## Monitoring

### Health checks

```bash
# Liveness
curl http://YOUR_SERVER:8079/health

# Readiness (checks both Ollama backends)
curl http://YOUR_SERVER:8079/health/ready
```

### Logs

With `FILTHY_LOG_FORMAT=json` (default), logs are structured JSON:

```json
{"event": "request_received", "model": "qwen3:30b-a3b", "has_tools": true, "tool_names": ["read", "glob"], "message_count": 5}
{"event": "tool_calls_valid", "attempt": 0}
{"event": "response_complete", "elapsed_ms": 12340}
```

For human-readable logs during development, set `FILTHY_LOG_FORMAT=console`.

### Key log events to watch

| Event | Meaning |
|-------|---------|
| `tool_calls_valid` | Success on first or retry attempt |
| `validation_failed` | Tool call had schema errors (retrying) |
| `tool_calls_rescued_from_text` | Extracted tool call from narrated text |
| `param_name_repaired` | Fuzzy-matched a near-miss parameter name |
| `escalating_to_quality_model` | Fast model exhausted retries |
| `escalation_succeeded` | Quality model succeeded |
| `X-FilthyToolFixer-Degraded: true` | All attempts failed, returning best effort |
