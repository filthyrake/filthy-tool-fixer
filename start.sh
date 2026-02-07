#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env if present
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

HOST="${FILTHY_HOST:-0.0.0.0}"
PORT="${FILTHY_PORT:-8079}"

exec uvicorn filthyllm.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level warning \
    --no-access-log
