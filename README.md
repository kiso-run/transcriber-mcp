# kiso-transcriber-mcp

Audio transcription exposed as a
[Model Context Protocol](https://modelcontextprotocol.io) server.

Pluggable backend, local-first by default:

- **`whisper-cpp`** (default) — local transcription via the
  whisper.cpp `whisper-cli` binary. No API key, no data egress, runs
  in the appliance. Excellent quality on clean speech (Italian
  natively supported); CPU-bound with the `small` model running near
  real-time on a 4-core CPU.
- **`gemini`** — Gemini 2.5 Flash Lite via OpenRouter. Useful for
  noisy multi-speaker scenes, language identification on unknown-
  language input, when one fast API call beats local CPU work.

Audio is auto-compressed to OGG Opus 32kbps mono 16kHz before
transcription on either backend (speech-optimised, dramatically
smaller file).

Part of the [`kiso-run`](https://github.com/kiso-run) project.

## Install

```sh
uvx --from git+https://github.com/kiso-run/transcriber-mcp@v0.2.0 kiso-transcriber-mcp
```

System dependencies (always required):

- `ffmpeg` + `ffprobe` (audio compression and duration detection)

Backend-specific dependencies:

- For `whisper-cpp` (default): the `whisper-cli` binary from
  [whisper.cpp](https://github.com/ggerganov/whisper.cpp), plus a
  ggml model file (typically `ggml-small.bin` ~150 MB; download from
  the whisper.cpp Hugging Face mirror).
- For `gemini`: only the env var (no extra binary).

## Required environment

| Variable                                | Required (when)                            | Purpose                                                                       |
|-----------------------------------------|--------------------------------------------|-------------------------------------------------------------------------------|
| `KISO_TRANSCRIBER_BACKEND`              | optional (default `whisper-cpp`)           | Backend selector: `whisper-cpp` or `gemini`                                   |
| `KISO_TRANSCRIBER_WHISPER_MODEL_PATH`   | required when backend = `whisper-cpp`      | Full path to a ggml model file, e.g. `/var/cache/whisper/ggml-small.bin`      |
| `KISO_TRANSCRIBER_WHISPER_BIN`          | optional (default `whisper-cli`)           | Path or name of the whisper.cpp binary                                        |
| `OPENROUTER_API_KEY`                    | required when backend = `gemini`           | Gemini backend via OpenRouter                                                 |

## MCP client config

### Backend `whisper-cpp` (default — local, no API key)

```json
{
  "mcpServers": {
    "transcriber": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/kiso-run/transcriber-mcp@v0.2.0",
        "kiso-transcriber-mcp"
      ],
      "env": {
        "KISO_TRANSCRIBER_WHISPER_MODEL_PATH": "/var/cache/whisper/ggml-small.bin"
      }
    }
  }
}
```

### Backend `gemini` (cloud, useful for noisy/multi-speaker audio)

```json
{
  "mcpServers": {
    "transcriber": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/kiso-run/transcriber-mcp@v0.2.0",
        "kiso-transcriber-mcp"
      ],
      "env": {
        "KISO_TRANSCRIBER_BACKEND": "gemini",
        "OPENROUTER_API_KEY": "${env:OPENROUTER_API_KEY}"
      }
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
`opus`, `aac`, `wma`, `mp4`.

Duration cap: **5 min on `gemini`** (Gemini-side input limit), **60
min on `whisper-cpp`** (local-side runaway guard). Split longer files
before calling.

Returns:

```json
{
  "success": true,
  "text": "transcribed text",
  "duration_sec": 34.2,
  "format": "ogg",
  "truncated": false,
  "backend": "whisper-cpp",
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

Reports runner health and active configuration:

```json
{
  "healthy": true,
  "issues": [],
  "backend": "whisper-cpp",
  "whisper_model_path": "/var/cache/whisper/ggml-small.bin"
}
```

For the `gemini` backend, omits `whisper_model_path` and reports
missing `OPENROUTER_API_KEY` if relevant. For both backends, reports
missing `ffmpeg` / `ffprobe` system binaries.

## When to use which backend

- **Default to `whisper-cpp`** for routine voice messages, internal
  meetings recorded on a phone, voice notes — anything where the
  audio is reasonably clean and the consumer wants zero data egress
  and zero per-call cost. Italian is supported natively at high
  quality with the `small` model.
- **Switch to `gemini`** when: audio is very noisy, multiple speakers
  overlap, the language is unknown ahead of time, or the user
  prioritises one fast API call over total cost. `gemini` is also a
  good fallback when whisper.cpp isn't installed in a target
  environment.
- The two are independent — a tenant can default to one and run a
  separate MCP server instance pointing at the other backend if both
  are needed concurrently.

## Whisper model size guidance

The model file pointed at by `KISO_TRANSCRIBER_WHISPER_MODEL_PATH`
controls speed/quality. Trade-offs:

| Model              | File size | Speed (CPU)      | Quality (IT/EN)                       |
|--------------------|-----------|------------------|---------------------------------------|
| `ggml-tiny.bin`    | ~75 MB    | very fast        | weak on Italian, OK English           |
| `ggml-base.bin`    | ~140 MB   | fast             | decent                                |
| `ggml-small.bin`   | ~460 MB   | near real-time   | **good — recommended default**        |
| `ggml-medium.bin`  | ~1.5 GB   | 2-3x real-time   | high                                  |
| `ggml-large-v3.bin`| ~3 GB     | 4-6x real-time   | best, recommended for critical audio  |

Download from the whisper.cpp Hugging Face mirror, e.g.:

```sh
mkdir -p /var/cache/whisper
curl -L \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin \
  -o /var/cache/whisper/ggml-small.bin
```

## Reliability

- **Whisper.cpp**: subprocess invocation with a 600 s timeout. Failures
  surface as `success=false` with the binary's stderr. The runner strips
  language tags (`[it]`) and timestamp markers from the output.
- **Gemini**: empty-response retry up to 2 attempts with 1s/2s backoff.
- **Output cap**: 50 000 chars with `truncated: true` flag, on both
  backends.
- **Compression fallback**: if `ffmpeg` compression fails for any
  reason, the original file is sent as-is.

## Migration from v0.1

v0.1 routed every call to Gemini and required `OPENROUTER_API_KEY`.
v0.2 changes the **default backend to `whisper-cpp`**.

If you depend on v0.1 behaviour, set
`KISO_TRANSCRIBER_BACKEND=gemini` in your client env. Otherwise no
action required — `transcribe_audio` calls continue to work and now
run locally on Whisper, free per call.

The duration cap also changes per backend: 5 min on `gemini`
(unchanged from v0.1), 60 min on `whisper-cpp`. If you regularly send
longer files, the new default lets you do that.

## Development

```sh
uv sync
uv run pytest tests/ -q                    # unit only
OPENROUTER_API_KEY=... uv run pytest tests/ -q   # include live test
```

## License

MIT.
