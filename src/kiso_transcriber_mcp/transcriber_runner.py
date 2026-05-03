"""Audio transcription core — pluggable backend (Whisper.cpp local default, Gemini opt-in).

Two backends, selected by ``KISO_TRANSCRIBER_BACKEND``:

- ``whisper-cpp`` (default) — local OCR-style transcription via the
  whisper.cpp ``whisper-cli`` binary. No API key, no data egress, runs
  in the appliance. Default model file path is configured via
  ``KISO_TRANSCRIBER_WHISPER_MODEL_PATH`` (point at e.g.
  ``ggml-small.bin``). The binary defaults to ``whisper-cli`` and can
  be overridden via ``KISO_TRANSCRIBER_WHISPER_BIN``.
- ``gemini`` — Gemini 2.5 Flash Lite via OpenRouter. Higher quality on
  noisy multi-speaker scenes, language identification on unknown-language
  input. Audio is compressed to OGG Opus 32kbps mono 16kHz before
  upload. Requires ``OPENROUTER_API_KEY``.

Both backends share the ffmpeg compression and ffprobe duration
detection — these stay system-level dependencies.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


_MAX_OUTPUT_CHARS = 50_000
_MAX_DURATION_GEMINI = 300       # 5 min — Gemini-side cap
_MAX_DURATION_WHISPER = 60 * 60  # 60 min — internal cap to avoid runaway
_COMPRESS_THRESHOLD = 500 * 1024
_EMPTY_RETRIES = 2
_RETRY_BACKOFF = (1, 2)
_WHISPER_TIMEOUT_SECS = 600
_DEFAULT_WHISPER_BIN = "whisper-cli"
_SUPPORTED_BACKENDS = {"whisper-cpp", "gemini"}

_AUDIO_EXTENSIONS = frozenset({
    ".ogg", ".mp3", ".m4a", ".wav", ".webm", ".flac",
    ".opus", ".aac", ".wma", ".mp4",
})

_SYSTEM_PROMPT = (
    "Transcribe the audio exactly as spoken. "
    "Return only the transcription text, no commentary, timestamps, or formatting."
)

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_GEMINI_MODEL = "google/gemini-2.5-flash-lite"


def transcribe_audio(
    *, file_path: str, language: str | None = None,
) -> dict:
    path = Path(file_path).expanduser()
    if not path.is_file():
        return _fail(f"file not found: {file_path}")
    if path.suffix.lower() not in _AUDIO_EXTENSIONS:
        return _fail(f"unsupported audio format: {path.suffix}")

    backend = _backend()
    if backend not in _SUPPORTED_BACKENDS:
        return _fail(
            f"KISO_TRANSCRIBER_BACKEND={backend!r} is not supported "
            f"(use one of: {sorted(_SUPPORTED_BACKENDS)})"
        )

    if backend == "gemini" and not os.environ.get("OPENROUTER_API_KEY"):
        return _fail("OPENROUTER_API_KEY is not set")

    duration = _get_duration(path)
    cap = _MAX_DURATION_GEMINI if backend == "gemini" else _MAX_DURATION_WHISPER
    if duration is not None and duration > cap:
        return _fail(
            f"audio too long ({_format_duration(duration)}); max for backend "
            f"{backend!r} is {_format_duration(cap)}; split the file first"
        )

    compressed = _compress_audio(path)
    try:
        if backend == "whisper-cpp":
            text = _transcribe_whisper(compressed, language=language)
        else:
            api_key = os.environ["OPENROUTER_API_KEY"]
            text = _call_gemini(compressed, api_key, language)
    except RuntimeError as exc:
        return _fail(str(exc), backend=backend)
    finally:
        if compressed != path and compressed.exists():
            compressed.unlink()

    truncated = False
    if len(text) > _MAX_OUTPUT_CHARS:
        shown = text[:_MAX_OUTPUT_CHARS]
        last_nl = shown.rfind("\n")
        if last_nl > 0:
            shown = shown[:last_nl]
        text = shown
        truncated = True

    return {
        "success": True,
        "text": text,
        "duration_sec": duration,
        "format": path.suffix.lower().lstrip("."),
        "truncated": truncated,
        "backend": backend,
        "stderr": "",
    }


def audio_info(*, file_path: str) -> dict:
    path = Path(file_path).expanduser()
    if not path.is_file():
        return {
            "success": False,
            "file_name": None,
            "size_bytes": None,
            "format": None,
            "duration_sec": None,
            "estimated_chars": None,
            "stderr": f"file not found: {file_path}",
        }
    size = path.stat().st_size
    duration = _get_duration(path)
    estimated = int(duration / 60 * 750) if duration else None
    return {
        "success": True,
        "file_name": path.name,
        "size_bytes": size,
        "format": path.suffix.lower().lstrip("."),
        "duration_sec": duration,
        "estimated_chars": estimated,
        "stderr": "",
    }


def check_health() -> dict:
    issues: list[str] = []
    backend = _backend()
    result: dict = {
        "healthy": False,
        "issues": issues,
        "backend": backend,
    }
    if backend not in _SUPPORTED_BACKENDS:
        issues.append(
            f"KISO_TRANSCRIBER_BACKEND={backend!r} is not supported "
            f"(use one of: {sorted(_SUPPORTED_BACKENDS)})"
        )
        return result

    if not _binary_exists("ffmpeg"):
        issues.append("ffmpeg not found on PATH (required for audio compression)")
    if not _binary_exists("ffprobe"):
        issues.append("ffprobe not found on PATH (required for duration detection)")

    if backend == "gemini":
        if not os.environ.get("OPENROUTER_API_KEY"):
            issues.append("OPENROUTER_API_KEY is not set")
    elif backend == "whisper-cpp":
        whisper_bin = os.environ.get("KISO_TRANSCRIBER_WHISPER_BIN", _DEFAULT_WHISPER_BIN)
        if not _binary_exists(whisper_bin):
            issues.append(
                f"whisper-cli binary not found on PATH (looked for {whisper_bin!r}). "
                "Install whisper.cpp or set KISO_TRANSCRIBER_WHISPER_BIN to its path."
            )
        model_path = os.environ.get("KISO_TRANSCRIBER_WHISPER_MODEL_PATH")
        if not model_path:
            issues.append(
                "KISO_TRANSCRIBER_WHISPER_MODEL_PATH is not set "
                "(point it at a whisper.cpp ggml model file, e.g. ggml-small.bin)"
            )
        else:
            result["whisper_model_path"] = model_path
            if not Path(model_path).is_file():
                issues.append(
                    f"whisper model file not found at {model_path!r}"
                )

    result["healthy"] = not issues
    return result


def _backend() -> str:
    return os.environ.get("KISO_TRANSCRIBER_BACKEND", "whisper-cpp").lower()


def _compress_audio(path: Path) -> Path:
    if path.stat().st_size <= _COMPRESS_THRESHOLD and path.suffix.lower() in (".ogg", ".opus"):
        return path
    tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
    tmp.close()
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", str(path),
                "-ac", "1", "-ar", "16000", "-b:a", "32k", "-f", "ogg",
                "-y", tmp.name,
            ],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            Path(tmp.name).unlink(missing_ok=True)
            return path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        Path(tmp.name).unlink(missing_ok=True)
        return path
    return Path(tmp.name)


def _transcribe_whisper(file_path: Path, *, language: str | None) -> str:
    """Invoke the local whisper.cpp binary on the audio file. Returns
    the extracted text (raw stdout). Raises ``RuntimeError`` on failure."""
    model_path = os.environ.get("KISO_TRANSCRIBER_WHISPER_MODEL_PATH")
    if not model_path:
        raise RuntimeError(
            "KISO_TRANSCRIBER_WHISPER_MODEL_PATH is not set "
            "(point it at a whisper.cpp ggml model file, e.g. ggml-small.bin)"
        )
    binary = os.environ.get("KISO_TRANSCRIBER_WHISPER_BIN", _DEFAULT_WHISPER_BIN)
    cmd = [
        binary,
        "-m", model_path,
        "-f", str(file_path),
        "--output-txt",
        "--no-timestamps",
    ]
    if language:
        cmd.extend(["-l", language])
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_WHISPER_TIMEOUT_SECS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"whisper-cli binary not found in PATH: {exc}. "
            "Install whisper.cpp (build from https://github.com/ggerganov/whisper.cpp) "
            "or set KISO_TRANSCRIBER_WHISPER_BIN, or switch to KISO_TRANSCRIBER_BACKEND=gemini."
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"whisper-cli error ({completed.returncode}): "
            f"{completed.stderr.strip()[:500]}"
        )
    return _strip_whisper_artifacts(completed.stdout)


def _strip_whisper_artifacts(text: str) -> str:
    """whisper-cli prefixes output with the language tag (e.g. ``[it]``)
    and may include leading/trailing blank lines. Clean both."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # Language tag at start: [xx] some text → some text
        if stripped.startswith("[") and "]" in stripped:
            close = stripped.index("]")
            tag = stripped[1:close]
            if len(tag) <= 4 and tag.isalpha():
                stripped = stripped[close + 1:].strip()
        lines.append(stripped)
    return "\n".join(l for l in lines if l)


def _call_gemini(file_path: Path, api_key: str, language: str | None) -> str:
    import httpx

    audio_data = base64.b64encode(file_path.read_bytes()).decode()
    format_map = {".ogg": "ogg", ".mp3": "mp3", ".wav": "wav", ".m4a": "m4a"}
    audio_format = format_map.get(file_path.suffix.lower(), "ogg")

    prompt = _SYSTEM_PROMPT
    if language:
        prompt += f" The audio is in {language}."

    payload = {
        "model": _GEMINI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio_data, "format": audio_format}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "max_tokens": 512,
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(_EMPTY_RETRIES + 1):
        response = httpx.post(_OPENROUTER_URL, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(
                f"Gemini API error ({response.status_code}): {response.text[:500]}"
            )
        result = response.json()
        choices = result.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        if content.strip():
            return content
        if attempt < _EMPTY_RETRIES:
            time.sleep(_RETRY_BACKOFF[attempt])
    return ""


def _get_duration(path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


def _binary_exists(name: str) -> bool:
    if "/" in name and Path(name).is_file():
        return True
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if p and (Path(p) / name).exists():
            return True
    return False


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _fail(message: str, *, backend: str | None = None) -> dict:
    result = {
        "success": False,
        "text": "",
        "duration_sec": None,
        "format": None,
        "truncated": False,
        "stderr": message,
    }
    if backend is not None:
        result["backend"] = backend
    return result
