# kiso-transcriber-mcp — Development Plan

## Status

**Legacy wrapper era — closed.** The `tool-transcriber` /
`wrapper-transcriber` subprocess-contract implementation has been
replaced by a Model Context Protocol server built on the official
`mcp` Python SDK.

**Current era: MCP server.** Tracked in `kiso-run/core` as M1509.

---

## v0.1 — MCP rewrite (2026-04-18)

- [x] Strip legacy wrapper files (`run.py`, `kiso.toml`, `deps.sh`,
      `validator.py`); preserve `tests/fixtures/sample.ogg`
- [x] New `pyproject.toml` with package name
      `kiso-transcriber-mcp`, entry point, MCP SDK dep
- [x] `src/kiso_transcriber_mcp/transcriber_runner.py` — ffmpeg
      compression, duration guard, Gemini 2.5 Flash Lite call via
      OpenRouter, empty-response retry
- [x] `src/kiso_transcriber_mcp/server.py` — FastMCP server with
      three tools: `transcribe_audio`, `audio_info`, `doctor`
- [x] 24 unit tests + 1 live test (fixture round-trip through
      OpenRouter), all green
- [x] README rewrite (MCP install, tools, env, client config)
- [ ] Cut `v0.1.0` tag on GitHub *(user action)*
- [ ] GitHub Actions CI *(deferred, not blocking no-gap invariant)*

**Design shifts from wrapper era**:

- **Single key**: only `OPENROUTER_API_KEY`. Dropped the
  `KISO_LLM_API_KEY` / `KISO_WRAPPER_TRANSCRIBER_BASE_URL`
  indirection — the OpenRouter base URL is hard-coded in the runner.
- **Dropped the `list` action**: an MCP server shouldn't list an
  arbitrary `uploads/` directory. File discovery is the caller's
  responsibility; the server only acts on an explicit `file_path`.
- **Structured return**: `{success, text, duration_sec, format,
  truncated, stderr}` instead of a plaintext header + body blob.
- **No sandboxed workspace**: the wrapper-era path traversal check
  was for the kiso session sandbox. MCP servers are invoked by
  trusted clients; path scoping is the caller's concern.

The content below is the original wrapper-era devplan, kept for
historical record.

---

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
- [x] `uv run pytest tests/ -q` passes — 44 tests
- [ ] Manual test: transcribe `/home/ymx1zq/Downloads/example.mp3` (needs VPS with OpenRouter API)

## M6 — Security + robustness fixes (code review) ✅

**Path traversal prefix attack (CRITICAL):**
- [x] `run.py:_resolve_path()` — replace `str(resolved).startswith(str(ws_resolved))` with `resolved.relative_to(ws_resolved)`

**JSON input safety:**
- [x] Wrap `json.load(sys.stdin)` in try-except JSONDecodeError — print clean error + exit 1

**MIME type when compression fails:**
- [x] `_call_gemini_transcribe()` hardcodes `mime_type = "audio/ogg"` — if ffmpeg fails and original file is MP3/WAV, wrong MIME type is sent. Detect actual format from file extension.

**Empty API response handling:**
- [x] `_call_gemini_transcribe()` — if `choices` is empty list, raise RuntimeError instead of returning empty string (distinguishes API failure from silence)

**Reduce max_tokens:**
- [x] Change `max_tokens=4096` to `max_tokens=512` — voice messages rarely exceed 100 words

**Tests to add:**
- [x] Path traversal lateral escape
- [x] Malformed JSON stdin
- [x] Empty choices array from API → RuntimeError
- [x] Malformed API response (missing message/content keys) → "No speech detected"
- [x] Whitespace-only transcription → "No speech detected"
- [x] `uv run pytest tests/ -q` passes — 44 tests

## Known Issues

- Gemini audio input: no speaker diarization (who said what) — returns flat text
- Compression to 32kbps is speech-optimized — music/complex audio may lose quality (acceptable for voice messages)
- Very noisy audio may produce lower quality transcription than Whisper (trade-off for cost + simplicity)

---

### M7 — Declare `consumes` in kiso.toml (core M826)

**Context:** Core M826 adds a `consumes` field to `[kiso.tool]` in kiso.toml. The planner uses
this to auto-route session workspace files to the right tool. Vocabulary: `image`, `document`,
`audio`, `video`, `code`, `web_page`.

**Changes:**
- [x] Add `consumes = ["audio"]` to `[kiso.tool]` in kiso.toml
- [ ] Enrich `usage_guide` with concrete arg examples and supported formats list
