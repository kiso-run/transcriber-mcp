"""Audio transcription core — cloud-only MCP wrapper.

Two backends are supported, selected by ``KISO_TRANSCRIBER_BACKEND``:

- ``openrouter`` (default) — direct OpenRouter calls. Requires
  ``OPENROUTER_API_KEY``.
- ``litellm`` — calls go through the consumer's local LiteLLM gateway
  at ``LITELLM_BASE_URL``. Optional ``LITELLM_API_KEY`` (when the
  gateway requires auth). Use this when audio traffic must flow through
  the same governance layer as other LLM calls (cost tracking, PII
  filter, residency tag, quota, fallback).

ffmpeg + ffprobe remain local system-level dependencies — they compress
audio to OGG Opus 32kbps mono 16kHz before upload (dramatically smaller
file, speech-optimised) and detect duration for the input cap.
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
_MAX_DURATION = 300       # 5 min — typical cloud audio-LLM cap
_COMPRESS_THRESHOLD = 500 * 1024
_EMPTY_RETRIES = 2
_RETRY_BACKOFF = (1, 2)
_SUPPORTED_BACKENDS = {"openrouter", "litellm"}

_AUDIO_EXTENSIONS = frozenset({
    ".ogg", ".mp3", ".m4a", ".wav", ".webm", ".flac",
    ".opus", ".aac", ".wma", ".mp4",
})

_SYSTEM_PROMPT = (
    "Transcribe the audio exactly as spoken. "
    "Return only the transcription text, no commentary, timestamps, or formatting."
)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "google/gemini-2.5-flash-lite"


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

    duration = _get_duration(path)
    if duration is not None and duration > _MAX_DURATION:
        return _fail(
            f"audio too long ({_format_duration(duration)}); max is "
            f"{_format_duration(_MAX_DURATION)}; split the file first"
        )

    try:
        base_url, api_key, model = _resolve_endpoint(backend)
    except RuntimeError as exc:
        return _fail(str(exc), backend=backend)

    compressed = _compress_audio(path)
    try:
        text = _call_audio_llm(
            base_url=base_url,
            api_key=api_key,
            model=model,
            file_path=compressed,
            language=language,
        )
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

    try:
        _resolve_endpoint(backend)
    except RuntimeError as exc:
        issues.append(str(exc))

    result["healthy"] = not issues
    return result


def _backend() -> str:
    return os.environ.get("KISO_TRANSCRIBER_BACKEND", "openrouter").lower()


def _resolve_endpoint(backend: str) -> tuple[str, str, str]:
    """Return ``(base_url, api_key, model)`` for the selected backend.

    Raises ``RuntimeError`` if required env vars are missing.
    """
    model = os.environ.get("KISO_TRANSCRIBER_MODEL", _DEFAULT_MODEL)
    if backend == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        return _OPENROUTER_BASE_URL, api_key, model
    if backend == "litellm":
        base_url = os.environ.get("LITELLM_BASE_URL", "").rstrip("/")
        if not base_url:
            raise RuntimeError("LITELLM_BASE_URL is not set")
        return base_url, os.environ.get("LITELLM_API_KEY", ""), model
    raise RuntimeError(f"unknown backend: {backend}")


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


def _call_audio_llm(
    *,
    base_url: str,
    api_key: str,
    model: str,
    file_path: Path,
    language: str | None,
) -> str:
    """POST a multimodal `/chat/completions` request to an OpenAI-compatible
    endpoint with the audio inline-encoded. Returns the transcript text.
    """
    import httpx

    audio_data = base64.b64encode(file_path.read_bytes()).decode()
    format_map = {".ogg": "ogg", ".mp3": "mp3", ".wav": "wav", ".m4a": "m4a"}
    audio_format = format_map.get(file_path.suffix.lower(), "ogg")

    prompt = _SYSTEM_PROMPT
    if language:
        prompt += f" The audio is in {language}."

    payload = {
        "model": model,
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
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = f"{base_url.rstrip('/')}/chat/completions"

    for attempt in range(_EMPTY_RETRIES + 1):
        response = httpx.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(
                f"audio API error ({response.status_code}): {response.text[:500]}"
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
