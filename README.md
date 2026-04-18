# kiso-transcriber-mcp

Audio transcription exposed as a
[Model Context Protocol](https://modelcontextprotocol.io) server.
Uses Gemini 2.5 Flash Lite via OpenRouter; auto-compresses audio to
OGG Opus 32kbps mono 16kHz before transfer.

Part of the [`kiso-run`](https://github.com/kiso-run) project.

## Install

No PyPI. Consume via `uvx`:

```sh
uvx --from git+https://github.com/kiso-run/transcriber-mcp@v0.1.0 kiso-transcriber-mcp
```

## Required environment

| Variable             | Required | Purpose                       |
|----------------------|----------|-------------------------------|
| `OPENROUTER_API_KEY` | yes      | Gemini backend via OpenRouter |

Also requires `ffmpeg` + `ffprobe` on `PATH` (for audio compression
and duration detection).

## MCP client config

```json
{
  "mcpServers": {
    "transcriber": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/kiso-run/transcriber-mcp@v0.1.0",
        "kiso-transcriber-mcp"
      ],
      "env": { "OPENROUTER_API_KEY": "${env:OPENROUTER_API_KEY}" }
    }
  }
}
```

## Tools

### `transcribe_audio(file_path, language?)`

| Parameter  | Type   | Default | Notes                               |
|------------|--------|---------|-------------------------------------|
| `file_path`| string | required| Audio file path                     |
| `language` | string | —       | ISO 639-1 hint, e.g. `"en"`, `"it"` |

Supported formats: `ogg`, `mp3`, `m4a`, `wav`, `webm`, `flac`,
`opus`, `aac`, `wma`, `mp4`. Max duration: 5 minutes (split longer
files first).

Returns:

```json
{
  "success": true,
  "text": "transcribed text",
  "duration_sec": 34.2,
  "format": "ogg",
  "truncated": false,
  "stderr": ""
}
```

### `audio_info(file_path)`

Returns file metadata without transcribing:

```json
{
  "success": true,
  "file_name": "sample.ogg",
  "size_bytes": 9886,
  "format": "ogg",
  "duration_sec": 2.1,
  "estimated_chars": 27,
  "stderr": ""
}
```

### `doctor()`

```json
{ "healthy": true, "issues": [] }
```

Reports missing `ffmpeg` / `ffprobe` binaries or missing
`OPENROUTER_API_KEY`.

## Reliability

- **Empty-response retry**: up to 2 retries with 1s/2s backoff if
  Gemini returns an empty transcript.
- **Output cap**: transcripts truncated at 50 000 chars with a
  `truncated: true` flag on the return.
- **Compression fallback**: if `ffmpeg` compression fails, the
  original file is sent as-is.

## Development

```sh
uv sync
uv run pytest tests/ -q                    # unit only
OPENROUTER_API_KEY=... uv run pytest tests/ -q   # include live test
```

## License

MIT.
