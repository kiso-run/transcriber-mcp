# kiso-transcriber-mcp

Audio transcription exposed as a
[Model Context Protocol](https://modelcontextprotocol.io) server.

A thin cloud-only wrapper around an audio LLM. Two backends:

- **`openrouter`** (default) — direct OpenRouter calls. Single key:
  `OPENROUTER_API_KEY`. Useful for standalone usage.
- **`litellm`** — calls go through the consumer's local LiteLLM
  gateway at `LITELLM_BASE_URL`. Use this when audio traffic must flow
  through the same governance layer as other LLM calls (cost tracking,
  PII filter, residency tag, quota, fallback).

Audio is compressed locally to OGG Opus 32kbps mono 16kHz via ffmpeg
before upload — dramatically smaller payload, speech-optimised, no
loss of intelligibility. Local-STT engines are explicitly out of
scope; air-gapped consumers should fork the plugin.

Part of the [`kiso-run`](https://github.com/kiso-run) project.

## Install

```sh
uvx --from git+https://github.com/kiso-run/transcriber-mcp@v0.3.0 kiso-transcriber-mcp
```

System dependencies:

- `ffmpeg` + `ffprobe` (always required — local audio compression and
  duration detection)

## Required environment

| Variable                     | Required (when)                              | Purpose                                                                       |
|------------------------------|----------------------------------------------|-------------------------------------------------------------------------------|
| `KISO_TRANSCRIBER_BACKEND`   | optional (default `openrouter`)              | Backend selector: `openrouter` or `litellm`                                   |
| `OPENROUTER_API_KEY`         | required when backend = `openrouter`         | OpenRouter auth                                                               |
| `LITELLM_BASE_URL`           | required when backend = `litellm`            | URL of the consumer's LiteLLM gateway (e.g. `http://litellm:4000/v1`)         |
| `LITELLM_API_KEY`            | optional, used when backend = `litellm`      | Bearer token for the LiteLLM gateway, if the gateway requires auth            |
| `KISO_TRANSCRIBER_MODEL`     | optional (default `google/gemini-2.5-flash-lite`) | Model identifier — override when the consumer registers the audio model in LiteLLM under a different name |

## MCP client config

### Backend `openrouter` (default — single-key usage)

```json
{
  "mcpServers": {
    "transcriber": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/kiso-run/transcriber-mcp@v0.3.0",
        "kiso-transcriber-mcp"
      ],
      "env": { "OPENROUTER_API_KEY": "${env:OPENROUTER_API_KEY}" }
    }
  }
}
```

### Backend `litellm` (route through consumer's LiteLLM gateway)

```json
{
  "mcpServers": {
    "transcriber": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/kiso-run/transcriber-mcp@v0.3.0",
        "kiso-transcriber-mcp"
      ],
      "env": {
        "KISO_TRANSCRIBER_BACKEND": "litellm",
        "LITELLM_BASE_URL": "http://litellm:4000/v1",
        "LITELLM_API_KEY": "${env:LITELLM_API_KEY}",
        "KISO_TRANSCRIBER_MODEL": "audio"
      }
    }
  }
}
```

## Tools

### `transcribe_audio(file_path, language=None)`

Transcribe an audio file. Optional `language` ISO 639-1 hint
(`"en"`, `"it"`, …) improves accuracy. Returns `{success, text,
duration_sec, format, truncated, backend, stderr}`. Output truncated
at 50 000 chars with `truncated: true`.

### `audio_info(file_path)`

File metadata (no LLM call). Returns `{success, file_name,
size_bytes, format, duration_sec, estimated_chars, stderr}`.
`estimated_chars` is a rough projection (~750 chars per minute).

### `doctor()`

Reports runner health and active configuration:

```json
{
  "healthy": true,
  "issues": [],
  "backend": "openrouter"
}
```

Reports missing ffmpeg/ffprobe binaries and missing env vars per
selected backend.

## Supported formats

`ogg`, `mp3`, `m4a`, `wav`, `webm`, `flac`, `opus`, `aac`, `wma`,
`mp4`. Max audio length: 5 minutes (typical cloud audio-LLM cap);
split longer files first.

## Reliability

- Empty-response retry up to 2 attempts with 1s/2s backoff.
- Local ffmpeg compression to OGG Opus 32kbps mono 16kHz (silent
  fallback to the original file if ffmpeg fails).
- Output cap: 50 000 chars with `truncated: true` flag.
- HTTP timeout: 120 s per call.

## Development

```sh
uv sync
uv run pytest tests/ -q                          # unit only
OPENROUTER_API_KEY=... uv run pytest tests/ -q   # include live test
```

## License

MIT.
