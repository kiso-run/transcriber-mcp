# tool-transcriber

Transcribe audio files (voice messages, recordings, podcasts) to text using the Whisper API.

## How it works

When a user sends a voice message via Discord/Telegram/etc., the connector saves it as an `.ogg` file in `uploads/`. The planner sees the audio file and uses this tool to transcribe it, then proceeds with the user's request based on the transcribed text.

## Actions

| Action | Description | Required args |
|--------|-------------|---------------|
| `transcribe` | Convert audio to text (default) | `file_path` |
| `info` | Audio metadata (format, duration, size) | `file_path` |
| `list` | List audio files in uploads/ | none |

## Supported formats

OGG, MP3, M4A, WAV, WEBM, FLAC, OPUS, AAC, WMA, MP4

## Cost management

- Uses Whisper API (OpenAI `whisper-1`) — charges per minute of audio
- Short audio (<10 min): transcribed in one shot
- Hard cap at 60 minutes to prevent surprise costs
- Output truncated at 50K chars (~66 min of speech)
- Typical voice message (10-30s) costs fractions of a cent

## API key

The tool reuses the existing kiso LLM API key (`KISO_LLM_API_KEY`) by default. No extra configuration needed if kiso is already set up with OpenAI or OpenRouter.

To use a different provider:
```bash
kiso env set KISO_TOOL_TRANSCRIBER_API_KEY sk-your-key
kiso env set KISO_TOOL_TRANSCRIBER_BASE_URL https://api.openai.com/v1
```

## System dependencies

- `ffprobe` (from ffmpeg) — used for audio duration detection. Installed via `deps.sh`.

## Install

```bash
kiso tool install transcriber
```

## License

MIT
