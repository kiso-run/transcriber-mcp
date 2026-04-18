"""MCP server exposing audio transcription as a tool."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import transcriber_runner


mcp = FastMCP("kiso-transcriber")


@mcp.tool()
def transcribe_audio(file_path: str, language: str | None = None) -> dict:
    """Transcribe an audio file to text via Gemini multimodal (OpenRouter).

    Args:
        file_path: Path to the audio file (absolute, or relative to the
            server's working directory). Supported formats: ogg, mp3, m4a,
            wav, webm, flac, opus, aac, wma, mp4.
        language: Optional ISO 639-1 language hint (e.g. ``"en"``, ``"it"``)
            to improve accuracy.

    Returns:
        ``{"success": bool, "text": str, "duration_sec": float | None,
           "format": str | None, "truncated": bool, "stderr": str}``.

    Max audio length is 5 minutes; longer files should be split first. The
    file is auto-compressed to OGG Opus 32kbps mono 16kHz before transfer.
    """
    return transcriber_runner.transcribe_audio(
        file_path=file_path, language=language,
    )


@mcp.tool()
def audio_info(file_path: str) -> dict:
    """Return audio file metadata (duration, size, format) without transcribing.

    Args:
        file_path: Path to the audio file.

    Returns:
        ``{"success": bool, "file_name": str | None, "size_bytes": int | None,
           "format": str | None, "duration_sec": float | None,
           "estimated_chars": int | None, "stderr": str}``.
    """
    return transcriber_runner.audio_info(file_path=file_path)


@mcp.tool()
def doctor() -> dict:
    """Check ffmpeg/ffprobe binaries and OpenRouter credentials.

    Returns ``{"healthy": bool, "issues": [str]}``.
    """
    return transcriber_runner.check_health()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
