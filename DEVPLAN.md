# tool-transcriber — Development Plan

Audio transcription tool for kiso. Converts voice messages and audio files to text using Gemini multimodal input via OpenRouter.

## Architecture

```
stdin (JSON) → run.py → resolve audio → ffmpeg compress → base64 → Gemini chat → stdout (text)
```

- **Entry point**: `run.py` reads JSON from stdin, dispatches to action handler
- **Actions**: `transcribe` (default), `info`, `list`
- **API**: Gemini 2.5 Flash via OpenRouter `/chat/completions` (audio as base64 inline content)
- **API key**: reuses `KISO_LLM_API_KEY` — same key as all other kiso LLM calls, zero extra config
- **Compression**: ffmpeg converts to OGG Opus mono 16kHz 32kbps before sending (speech-optimized)
- **Cost guard**: hard cap at 5 min audio, output cap at 50K chars
- **System dep**: `ffmpeg` / `ffprobe` for compression and duration detection

## Token / Cost Strategy

Gemini 2.5 Flash tokenizes audio at **32 tokens/second**. At $0.15/1M input tokens:

| Duration | Token audio | Cost | vs Whisper |
|----------|------------|------|-----------|
| 15 sec (typical voice msg) | 480 | **$0.00007** | 20x cheaper |
| 1 min | 1,920 | **$0.0003** | 20x cheaper |
| 5 min (hard cap) | 9,600 | **$0.0014** | 20x cheaper |

100 voice messages/day at 15s each = **$0.007/day**.

Compressed audio size: 5 min at 32kbps = ~1.2 MB → ~1.6 MB base64. Well within Gemini's input limits.

Output budget: 50K chars ≈ ~66 min of speech. Always fits within cap.

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

## M5 — Rewrite: Whisper API → Gemini multimodal via OpenRouter ✅

**Problem:** Whisper API is OpenAI-only — doesn't go through OpenRouter.
**Solution:** Gemini 2.5 Flash Lite via OpenRouter `/chat/completions`. 20x cheaper.

### Audio compression pipeline
- [x] `_compress_audio(path)` — ffmpeg → OGG Opus mono 16kHz 32kbps
- [x] Skip compression for small OGG files (<500 KB)
- [x] Temp file cleanup in finally block

### API rewrite
- [x] `_call_gemini_transcribe()` — base64 audio → chat completion with `input_audio` content part
- [x] Model: `google/gemini-2.5-flash-lite`
- [x] Language hint appended to system prompt
- [x] Base URL: OpenRouter default, configurable

### Cost guard update
- [x] Hard cap: 5 min (was 60 min) — duration check rejects with split suggestion
- [x] No file size check needed — compression keeps everything small

### Config simplification
- [x] API key: `KISO_LLM_API_KEY` only (removed `KISO_TOOL_TRANSCRIBER_API_KEY`)
- [x] Removed `[kiso.tool.env]` from kiso.toml
- [x] Updated README + kiso.toml description + version → 0.2.0

### Tests
- [x] All mocked responses updated to chat completion format
- [x] Compression tests: skip small OGG, compress large, compress non-OGG, ffmpeg failure fallback
- [x] 39 tests, all passing

### Validation
- [ ] `uv run pytest tests/ -q` passes
- [ ] Manual test: transcribe `/home/ymx1zq/Downloads/example.mp3` ("questa è una prova") → correct Italian text

## M6 — Security + robustness fixes (code review)

**Path traversal prefix attack (CRITICAL):**
- [ ] `run.py:_resolve_path()` — replace `str(resolved).startswith(str(ws_resolved))` with `resolved.relative_to(ws_resolved)`

**JSON input safety:**
- [ ] Wrap `json.load(sys.stdin)` in try-except JSONDecodeError — print clean error + exit 1

**MIME type when compression fails:**
- [ ] `_call_gemini_transcribe()` hardcodes `mime_type = "audio/ogg"` — if ffmpeg fails and original file is MP3/WAV, wrong MIME type is sent. Detect actual format from file extension.

**Empty API response handling:**
- [ ] `_call_gemini_transcribe()` — if `choices` is empty list, raise RuntimeError instead of returning empty string (distinguishes API failure from silence)

**Reduce max_tokens:**
- [ ] Change `max_tokens=4096` to `max_tokens=512` — voice messages rarely exceed 100 words

**Tests to add:**
- [ ] Path traversal lateral escape
- [ ] Malformed JSON stdin
- [ ] Empty choices array from API
- [ ] Malformed API response (missing message/content keys)
- [ ] Whitespace-only transcription
- [ ] `uv run pytest tests/ -q` passes

## Known Issues

- Gemini audio input: no speaker diarization (who said what) — returns flat text
- Compression to 32kbps is speech-optimized — music/complex audio may lose quality (acceptable for voice messages)
- Very noisy audio may produce lower quality transcription than Whisper (trade-off for cost + simplicity)
