"""tool-transcriber — transcribe audio files to text via Gemini multimodal.

Subprocess contract (same as all kiso tools):
  stdin:  JSON {args, session, workspace, session_secrets, plan_outputs}
  stdout: result text on success
  stderr: error description on failure
  exit 0: success, exit 1: failure

Uses Gemini 2.5 Flash Lite via OpenRouter /chat/completions.
Audio is compressed to OGG Opus 32kbps mono 16kHz, then sent as base64.
Same API key as all kiso LLM calls — zero extra config.

Cost: 32 tokens/sec of audio. 5 min = 9600 tokens = $0.0014.
"""
from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

_MAX_OUTPUT_CHARS = 50_000
_MAX_DURATION_SECS = 300  # 5 min — hard cap
_COMPRESS_THRESHOLD = 500 * 1024  # 500 KB — skip compression if already small
_EMPTY_RETRIES = 2  # retry up to 2 times on empty API response
_RETRY_BACKOFF = (1, 2)  # seconds between retries

_AUDIO_EXTENSIONS = frozenset({
    ".ogg", ".mp3", ".m4a", ".wav", ".webm", ".flac",
    ".opus", ".aac", ".wma", ".mp4",
})

_SYSTEM_PROMPT = (
    "Transcribe the audio exactly as spoken. "
    "Return only the transcription text, no commentary, timestamps, or formatting."
)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    args = data.get("args", {})
    workspace = data.get("workspace", ".")

    action = args.get("action", "transcribe")

    try:
        if action == "list":
            result = do_list(workspace)
        elif action == "info":
            result = do_info(workspace, args)
        elif action == "transcribe":
            result = do_transcribe(workspace, args)
        else:
            print(f"Unknown action: {action}", file=sys.stderr)
            sys.exit(1)
    except FileNotFoundError as e:
        print(f"File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(result)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def do_list(workspace: str) -> str:
    """List audio files in the uploads/ directory."""
    uploads = Path(workspace) / "uploads"
    if not uploads.is_dir():
        return "No uploads/ directory found."
    files = sorted(
        f for f in uploads.rglob("*")
        if f.is_file() and f.suffix.lower() in _AUDIO_EXTENSIONS
    )
    if not files:
        return "No audio files found in uploads/."
    lines = [f"Audio files in uploads/ ({len(files)}):"]
    for f in files:
        rel = f.relative_to(uploads)
        size = f.stat().st_size
        duration = _get_duration(f)
        dur_str = f", {_format_duration(duration)}" if duration else ""
        lines.append(f"  {rel} ({_format_size(size)}{dur_str})")
    return "\n".join(lines)


def do_info(workspace: str, args: dict) -> str:
    """Get audio file metadata without transcribing."""
    file_path = _resolve_path(workspace, args)
    size = file_path.stat().st_size
    duration = _get_duration(file_path)
    lines = [
        f"File: {file_path.name}",
        f"Size: {_format_size(size)}",
        f"Format: {file_path.suffix.lower()}",
    ]
    if duration is not None:
        lines.append(f"Duration: {_format_duration(duration)}")
        est_chars = int(duration / 60 * 750)
        lines.append(f"Estimated transcript: ~{est_chars} chars")
    return "\n".join(lines)


def do_transcribe(workspace: str, args: dict) -> str:
    """Transcribe an audio file to text via Gemini multimodal."""
    file_path = _resolve_path(workspace, args)
    language = args.get("language")
    duration = _get_duration(file_path)

    # Duration guard
    if duration is not None and duration > _MAX_DURATION_SECS:
        return (
            f"Audio too long ({_format_duration(duration)}). "
            f"Max duration is {_format_duration(_MAX_DURATION_SECS)}. "
            f"Split the file first (e.g. ffmpeg -t 300 -i input.ogg output.ogg)."
        )

    header = f"Transcription: {file_path.name}"
    if duration is not None:
        header += f" ({_format_duration(duration)})"

    # Compress if needed, then transcribe
    compressed = _compress_audio(file_path)
    try:
        api_key = _get_api_key()
        text = _call_gemini_transcribe(compressed, api_key, language)
    finally:
        # Clean up temp file if we created one
        if compressed != file_path and compressed.exists():
            compressed.unlink()

    if not text.strip():
        return f"{header}\nNo speech detected in audio."

    if len(text) > _MAX_OUTPUT_CHARS:
        shown = text[:_MAX_OUTPUT_CHARS]
        last_nl = shown.rfind("\n")
        if last_nl > 0:
            shown = shown[:last_nl]
        text = (
            f"{shown}\n\n"
            f"Showing first {len(shown)} of {len(text)} chars."
        )

    return f"{header}\n\n{text}"


# ---------------------------------------------------------------------------
# Audio compression
# ---------------------------------------------------------------------------


def _compress_audio(path: Path) -> Path:
    """Compress audio to OGG Opus mono 16kHz 32kbps for efficient API transfer.

    Skips compression if file is already small and in OGG format.
    Returns the path to use (original or temp file).
    """
    if path.stat().st_size <= _COMPRESS_THRESHOLD and path.suffix.lower() in (".ogg", ".opus"):
        return path

    tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
    tmp.close()
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", str(path),
                "-ac", "1",        # mono
                "-ar", "16000",    # 16kHz
                "-b:a", "32k",     # 32kbps
                "-f", "ogg",
                "-y", tmp.name,
            ],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            # Compression failed — use original
            Path(tmp.name).unlink(missing_ok=True)
            return path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        Path(tmp.name).unlink(missing_ok=True)
        return path

    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------


def _get_api_key() -> str:
    """Get API key — same as kiso LLM key."""
    key = os.environ.get("KISO_LLM_API_KEY")
    if key:
        return key
    raise RuntimeError(
        "No API key found. Set KISO_LLM_API_KEY (same key used by kiso for all LLM calls)."
    )


def _call_gemini_transcribe(
    file_path: Path, api_key: str, language: str | None = None,
) -> str:
    """Send audio to Gemini via OpenRouter chat completion."""
    import httpx

    base_url = os.environ.get(
        "KISO_TOOL_TRANSCRIBER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )
    url = f"{base_url}/chat/completions"

    audio_data = base64.b64encode(file_path.read_bytes()).decode()

    # Detect actual audio format from extension (compression may have failed)
    _FORMAT_MAP = {".ogg": "ogg", ".mp3": "mp3", ".wav": "wav", ".m4a": "m4a"}
    audio_format = _FORMAT_MAP.get(file_path.suffix.lower(), "ogg")

    prompt = _SYSTEM_PROMPT
    if language:
        prompt += f" The audio is in {language}."

    payload = {
        "model": "google/gemini-2.5-flash-lite",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_data,
                            "format": audio_format,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
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
        response = httpx.post(url, headers=headers, json=payload, timeout=120)

        if response.status_code != 200:
            raise RuntimeError(
                f"Gemini API error ({response.status_code}): {response.text[:500]}"
            )

        result = response.json()
        choices = result.get("choices", [])
        content = choices[0].get("message", {}).get("content", "") if choices else ""

        if content.strip():
            return content

        # Empty response — retry with backoff if attempts remain
        if attempt < _EMPTY_RETRIES:
            time.sleep(_RETRY_BACKOFF[attempt])

    return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_path(workspace: str, args: dict) -> Path:
    """Resolve file_path arg to an absolute Path."""
    file_path = args.get("file_path")
    if not file_path:
        raise ValueError("file_path argument is required for transcribe/info actions")
    resolved = (Path(workspace) / file_path).resolve()
    ws_resolved = Path(workspace).resolve()
    try:
        resolved.relative_to(ws_resolved)
    except ValueError:
        raise ValueError(f"Path traversal denied: {file_path}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved.name)
    return resolved


def _get_duration(path: Path) -> float | None:
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "json",
                str(path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
    except Exception:
        pass
    return None


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _format_size(size: int) -> str:
    """Format byte size as human-readable string."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


if __name__ == "__main__":
    main()
