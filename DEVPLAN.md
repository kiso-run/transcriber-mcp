# tool-transcriber — Development Plan

Audio transcription tool for kiso. Converts voice messages and audio files to text via Whisper API.

## Architecture

```
stdin (JSON) → run.py → resolve audio file → Whisper API → stdout (text)
```

- **Entry point**: `run.py` reads JSON from stdin, dispatches to action handler
- **Actions**: `transcribe` (default), `info`, `list`
- **API**: OpenAI Whisper (`whisper-1`) via OpenRouter or direct OpenAI
- **API key**: reuses `KISO_LLM_API_KEY` by default, override with `KISO_TOOL_TRANSCRIBER_API_KEY`
- **Cost guard**: hard cap at 60 min, output cap at 50K chars
- **System dep**: `ffprobe` (from ffmpeg) for duration detection

## Token / Cost Strategy

Whisper charges ~$0.006/min. Voice messages are typically 5-30s (~$0.003).
Long recordings need protection:

| Duration | Strategy | Est. cost |
|----------|----------|-----------|
| <10 min | Single API call | <$0.06 |
| 10-60 min | Single call (Whisper handles internally) | <$0.36 |
| >60 min | Transcribe first 60 min, return partial + note | $0.36 cap |

Output budget: 50K chars ≈ ~66 min of speech (150 words/min × 5 chars/word).
Transcription fits in output budget for any audio under the 60 min cap.

## M1 — Core implementation ✅

- [x] Project structure: kiso.toml, pyproject.toml, run.py, deps.sh, README, LICENSE
- [x] `transcribe` action: resolve file → call Whisper API → return text with header
- [x] `info` action: ffprobe duration + size + estimated transcript length
- [x] `list` action: enumerate audio files in uploads/ with duration
- [x] Path traversal guard (same as docreader)
- [x] API key fallback: KISO_TOOL_TRANSCRIBER_API_KEY → KISO_LLM_API_KEY
- [x] Cost guard: 60 min hard cap, 25 MB file size limit (Whisper API limit)
- [x] Output truncation at 50K chars with line-boundary cut
- [x] Configurable base URL (KISO_TOOL_TRANSCRIBER_BASE_URL)

## M2 — Unit tests

- [ ] Test `do_list` with audio files, mixed files, empty dir, no uploads/
- [ ] Test `do_info` with mocked ffprobe (duration, size, format)
- [ ] Test `do_transcribe` with mocked Whisper API response
- [ ] Test API key resolution (tool-specific → LLM fallback → error)
- [ ] Test file size guard (>25 MB rejected)
- [ ] Test path traversal guard
- [ ] Test output truncation on very long transcript
- [ ] Test `_format_duration` and `_format_size` helpers
- [ ] Functional test: stdin/stdout contract (list, transcribe, missing file)

## M3 — Static fixture files

- [ ] Generate small audio fixture with ffmpeg (`tests/fixtures/sample.ogg`, 2s silence + tone)
- [ ] `create_fixtures.sh` script to regenerate
- [ ] Use fixture in info/list tests (real ffprobe, no mock)

## M4 — Integration with kiso registry

- [ ] Verify `kiso tool install transcriber` works end-to-end (needs Docker + VPS)
- [ ] Live test: send voice message via Discord → transcription appears in response

## Known Issues

- Whisper API has 25 MB file size limit — very long recordings in high-quality formats may need compression
- No speaker diarization (who said what) — Whisper returns flat text
- Language auto-detection works but explicit hint improves accuracy significantly
