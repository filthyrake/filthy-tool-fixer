# API Reference

Filthy Tool Fixer exposes an OpenAI-compatible API. Any client that works with the OpenAI API should work with the proxy — just point `base_url` at the proxy and set the model name to an Ollama model.

## Endpoints

### `POST /v1/chat/completions`

The main endpoint. Accepts and returns the OpenAI chat completion format.

**Request body:**

```json
{
  "model": "qwen3:30b-a3b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "List files in the current directory"}
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
  ],
  "stream": false,
  "temperature": 0.7,
  "tool_choice": "auto"
}
```

**Behavior depends on whether `tools` are present:**

| Condition | Behavior |
|-----------|----------|
| No tools | Pure passthrough to Ollama (streaming or buffered) |
| Tools present | Buffered internally, validated, retried if needed |
| Tools + `stream: true` | Buffered for validation, then SSE synthesized from result |

**Response (non-streaming):**

```json
{
  "id": "chatcmpl-abc123",
  "model": "qwen3:30b-a3b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "list_files",
              "arguments": "{\"path\": \".\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 25,
    "total_tokens": 175
  }
}
```

**Response headers (tool-calling requests):**

| Header | Description |
|--------|-------------|
| `X-FilthyToolFixer-Request-ID` | Unique request ID (correlates with logs) |
| `X-FilthyToolFixer-Attempts` | Number of attempts before success |
| `X-FilthyToolFixer-Escalated` | `true` if the quality model was used |
| `X-FilthyToolFixer-Model` | Actual model name (when escalated) |
| `X-FilthyToolFixer-Degraded` | `true` if all attempts failed (best effort returned) |

### `GET /v1/models`

Proxies the model list from the primary Ollama backend.

```bash
curl http://localhost:8079/v1/models
```

Returns the same format as Ollama's `/v1/models` endpoint.

### `GET /health`

Liveness check. Always returns 200 if the proxy is running.

```json
{"status": "ok"}
```

### `GET /health/ready`

Readiness check. Verifies backend connectivity.

```json
{
  "status": "ready",
  "primary_backend": "ok",
  "escalation_backend": "ok"
}
```

Returns 503 if the primary backend is unreachable. The escalation backend being down is reported but does not affect readiness.

## Error Responses

All errors follow the OpenAI error format:

```json
{
  "error": {
    "message": "Description of what went wrong",
    "type": "error_type"
  }
}
```

| Status | Type | Cause |
|--------|------|-------|
| 400 | `invalid_request_error` | Malformed request body or too many messages (>200) |
| 413 | `invalid_request_error` | Request body exceeds 10MB |
| 502 | `proxy_error` | Backend request failed (Ollama error, network issue) |
| 503 | `proxy_error` | Proxy not initialized (startup in progress) |
| 504 | `proxy_timeout` | Request exceeded the profile's `request_timeout` budget |

## Input Limits

| Limit | Value | Description |
|-------|-------|-------------|
| Request body | 10 MB | Maximum raw request size |
| Messages | 200 | Maximum conversation messages per request |
| Tool calls | 50 | Maximum tool calls in a single LLM response |
| Argument size | 1 MB | Maximum size of a single tool call's arguments |

## Streaming

For non-tool requests, streaming is passed through directly to Ollama — chunks arrive as generated.

For tool-calling requests, streaming is handled differently:
1. The request is buffered internally (forced `stream: false` to Ollama)
2. Validation, retry, and escalation happen on the buffered response
3. If the client requested streaming, SSE events are synthesized from the final response
4. The response arrives as a single chunk followed by `data: [DONE]`

This means tool-calling requests don't get incremental streaming — the entire validated response arrives at once. This is a deliberate tradeoff: you can't validate a tool call until you have the complete response.
