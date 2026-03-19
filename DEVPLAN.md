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

## M2 — Unit tests ✅

- [x] `do_list`: audio files, mixed (filters non-audio), empty dir, no uploads/, duration display
- [x] `do_info`: basic with duration, without duration, missing file
- [x] `do_transcribe`: success, with language, file too large, long audio cap (60 min), API error, output truncation, missing file_path
- [x] API key resolution: tool-specific, LLM fallback, priority, no key error
- [x] Path traversal guard
- [x] `_format_duration`, `_format_size`, `_get_duration` (ffprobe success/failure/not installed)
- [x] Functional: list via stdin, missing file exits 1, unknown action exits 1
- 34 tests, all passing

## M3 — Static fixture files ✅

- [x] `tests/fixtures/sample.ogg` — 2s 440 Hz tone (10 KB, Opus codec)
- [x] `tests/create_fixtures.sh` to regenerate
- [x] `TestFixtureIntegration` — info + list with real ffprobe on fixture
- [x] `test_real_ffprobe_on_fixture` — duration detection on real audio
- 37 tests total, all passing

## M4 — Integration with kiso registry (pending — needs VPS)

- [x] transcriber added to core registry.json
- [ ] Verify `kiso tool install transcriber` works end-to-end (needs Docker + VPS)
- [ ] Live test: send voice message via Discord → transcription appears in response

## Known Issues

- Whisper API has 25 MB file size limit — very long recordings in high-quality formats may need compression
- No speaker diarization (who said what) — Whisper returns flat text
- Language auto-detection works but explicit hint improves accuracy significantly
