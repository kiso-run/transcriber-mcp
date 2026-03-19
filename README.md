# tool-transcriber

Transcribe audio files (voice messages, recordings, podcasts) to text using Gemini multimodal via OpenRouter.

## How it works

When a user sends a voice message via Discord/Telegram/etc., the connector saves it as an audio file in `uploads/`. The planner sees the audio file and uses this tool to transcribe it, then proceeds based on the transcribed text.

Audio is compressed to OGG Opus 32kbps mono 16kHz before sending to the API — optimized for speech, keeps costs minimal.

## Actions

| Action | Description | Required args |
|--------|-------------|---------------|
| `transcribe` | Convert audio to text (default) | `file_path` |
| `info` | Audio metadata (format, duration, size) | `file_path` |
| `list` | List audio files in uploads/ | none |

## Supported formats

OGG, MP3, M4A, WAV, WEBM, FLAC, OPUS, AAC, WMA, MP4

## Cost

Uses Gemini 2.5 Flash Lite via OpenRouter — 32 tokens/sec of audio at $0.15/1M tokens:

| Duration | Cost |
|----------|------|
| 15 sec (typical voice msg) | $0.00007 |
| 1 min | $0.0003 |
| 5 min (max) | $0.0014 |

## API key

Uses `KISO_LLM_API_KEY` — the same key kiso uses for all LLM calls. No extra configuration.

## System dependencies

- `ffmpeg` + `ffprobe` — audio compression and duration detection. Installed via `deps.sh`.

## Install

```bash
kiso tool install transcriber
```

## License

MIT
