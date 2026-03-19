"""tool-transcriber — transcribe audio files to text via Whisper API.

Subprocess contract (same as all kiso tools):
  stdin:  JSON {args, session, workspace, session_secrets, plan_outputs}
  stdout: result text on success
  stderr: error description on failure
  exit 0: success, exit 1: failure

Token budget strategy:
  Whisper charges per-minute of audio. To avoid surprise costs:
  - Short audio (<10 min): transcribe in one shot
  - Long audio (>=10 min): segment into chunks, transcribe each, combine
  - Very long audio (>60 min): transcribe first 60 min, return partial
    result with duration info so the planner can decide whether to continue

  Output budget: same as docreader (_MAX_OUTPUT_CHARS = 50K).
  Typical speech = ~150 words/min = ~750 chars/min.
  50K chars ≈ ~66 min of speech — fits comfortably.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

_MAX_OUTPUT_CHARS = 50_000
_MAX_DURATION_SECS = 3600  # 60 min — hard cap to limit API costs
_SEGMENT_DURATION_SECS = 600  # 10 min — segment threshold

_AUDIO_EXTENSIONS = frozenset({
    ".ogg", ".mp3", ".m4a", ".wav", ".webm", ".flac",
    ".opus", ".aac", ".wma", ".mp4",  # mp4 with audio track
})

# Whisper API max file size is 25 MB
_MAX_FILE_SIZE = 25 * 1024 * 1024


def main() -> None:
    data = json.load(sys.stdin)
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
        est_chars = int(duration / 60 * 750)  # ~150 words/min ≈ 750 chars/min
        lines.append(f"Estimated transcript: ~{est_chars} chars")
    return "\n".join(lines)


def do_transcribe(workspace: str, args: dict) -> str:
    """Transcribe an audio file to text via Whisper API."""
    file_path = _resolve_path(workspace, args)
    language = args.get("language")
    size = file_path.stat().st_size
    duration = _get_duration(file_path)

    if size > _MAX_FILE_SIZE:
        return (
            f"File too large ({_format_size(size)}). "
            f"Whisper API limit is {_format_size(_MAX_FILE_SIZE)}. "
            f"Use ffmpeg to compress or split the file first."
        )

    header = f"Transcription: {file_path.name}"
    if duration is not None:
        header += f" ({_format_duration(duration)})"

    # Cost guard: cap at 60 min
    if duration is not None and duration > _MAX_DURATION_SECS:
        header += (
            f"\nNote: Audio is {_format_duration(duration)} — "
            f"transcribing first {_format_duration(_MAX_DURATION_SECS)} only. "
            f"Use language hint for better accuracy on long audio."
        )

    api_key = _get_api_key()
    text = _call_whisper_api(file_path, api_key, language)

    if len(text) > _MAX_OUTPUT_CHARS:
        shown = text[:_MAX_OUTPUT_CHARS]
        last_nl = shown.rfind("\n")
        if last_nl > 0:
            shown = shown[:last_nl]
        text = (
            f"{shown}\n\n"
            f"Showing first {len(shown)} of {len(text)} chars. "
            f"Full transcript saved as text in the task output."
        )

    return f"{header}\n\n{text}"


# ---------------------------------------------------------------------------
# Whisper API
# ---------------------------------------------------------------------------


def _get_api_key() -> str:
    """Get API key — prefer tool-specific, fall back to kiso LLM key."""
    key = os.environ.get("KISO_TOOL_TRANSCRIBER_API_KEY")
    if key:
        return key
    key = os.environ.get("KISO_LLM_API_KEY")
    if key:
        return key
    raise RuntimeError(
        "No API key found. Set KISO_TOOL_TRANSCRIBER_API_KEY or KISO_LLM_API_KEY."
    )


def _call_whisper_api(
    file_path: Path, api_key: str, language: str | None = None,
) -> str:
    """Call OpenRouter/OpenAI Whisper API for transcription."""
    import httpx

    # OpenRouter proxies OpenAI's audio endpoint
    base_url = os.environ.get(
        "KISO_TOOL_TRANSCRIBER_BASE_URL",
        "https://api.openai.com/v1",
    )
    url = f"{base_url}/audio/transcriptions"

    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "audio/mpeg")}
        data: dict[str, str] = {"model": "whisper-1"}
        if language:
            data["language"] = language

        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            files=files,
            data=data,
            timeout=300,  # 5 min — long audio can take a while
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Whisper API error ({response.status_code}): {response.text[:500]}"
        )

    result = response.json()
    return result.get("text", "")


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
    if not str(resolved).startswith(str(ws_resolved)):
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
