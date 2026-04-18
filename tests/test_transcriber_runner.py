"""Unit tests for kiso_transcriber_mcp.transcriber_runner."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kiso_transcriber_mcp.transcriber_runner import (
    audio_info,
    check_health,
    transcribe_audio,
)


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    f = tmp_path / "sample.ogg"
    f.write_bytes(b"fake audio bytes")
    return f


class TestTranscribeAudio:
    def test_missing_api_key_fails(self, monkeypatch, audio_file):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        result = transcribe_audio(file_path=str(audio_file))
        assert result["success"] is False
        assert "OPENROUTER_API_KEY" in result["stderr"]

    def test_file_not_found(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        result = transcribe_audio(file_path=str(tmp_path / "missing.ogg"))
        assert result["success"] is False
        assert "not found" in result["stderr"].lower()

    def test_unsupported_extension_rejected(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        bad = tmp_path / "x.txt"
        bad.write_text("nope")
        result = transcribe_audio(file_path=str(bad))
        assert result["success"] is False
        assert "unsupported" in result["stderr"].lower()

    def test_duration_too_long_rejected(self, monkeypatch, audio_file):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._get_duration",
            return_value=900.0,  # 15 min
        ):
            result = transcribe_audio(file_path=str(audio_file))
        assert result["success"] is False
        assert "too long" in result["stderr"].lower()

    def test_success_returns_transcript(self, monkeypatch, audio_file):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._get_duration",
            return_value=12.0,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._compress_audio",
            side_effect=lambda p: p,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._call_gemini",
            return_value="hello world",
        ):
            result = transcribe_audio(file_path=str(audio_file))
        assert result["success"] is True
        assert result["text"] == "hello world"
        assert result["duration_sec"] == 12.0
        assert result["format"] == "ogg"
        assert result["truncated"] is False

    def test_large_transcript_truncates(self, monkeypatch, audio_file):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        huge = "a" * 100_000
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._get_duration",
            return_value=12.0,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._compress_audio",
            side_effect=lambda p: p,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._call_gemini",
            return_value=huge,
        ):
            result = transcribe_audio(file_path=str(audio_file))
        assert result["success"] is True
        assert result["truncated"] is True
        assert len(result["text"]) <= 50_000

    def test_api_error_surfaces_as_failure(self, monkeypatch, audio_file):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._get_duration",
            return_value=12.0,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._compress_audio",
            side_effect=lambda p: p,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._call_gemini",
            side_effect=RuntimeError("API error 500"),
        ):
            result = transcribe_audio(file_path=str(audio_file))
        assert result["success"] is False
        assert "500" in result["stderr"]

    def test_language_hint_passed_to_api(self, monkeypatch, audio_file):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._get_duration",
            return_value=12.0,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._compress_audio",
            side_effect=lambda p: p,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._call_gemini",
            return_value="ciao",
        ) as call:
            transcribe_audio(file_path=str(audio_file), language="it")
        assert call.call_args.args[2] == "it"


class TestCallGeminiRetry:
    """Empty-response retry behavior for _call_gemini."""

    def test_empty_response_retries_then_returns_empty(self, monkeypatch, audio_file):
        from kiso_transcriber_mcp import transcriber_runner

        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "choices": [{"message": {"content": ""}}]
        }
        with patch(
            "httpx.post", return_value=mock_response,
        ) as post, patch(
            "kiso_transcriber_mcp.transcriber_runner.time.sleep",
        ):
            result = transcriber_runner._call_gemini(audio_file, "k", None)
        assert result == ""
        # 1 initial + 2 retries = 3 calls
        assert post.call_count == 3

    def test_non_empty_response_returns_immediately(self, monkeypatch, audio_file):
        from kiso_transcriber_mcp import transcriber_runner

        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "hello"}}]
        }
        with patch(
            "httpx.post", return_value=mock_response,
        ) as post:
            result = transcriber_runner._call_gemini(audio_file, "k", None)
        assert result == "hello"
        assert post.call_count == 1

    def test_http_error_raises(self, monkeypatch, audio_file):
        from kiso_transcriber_mcp import transcriber_runner

        mock_response = MagicMock(status_code=500, text="server boom")
        with patch("httpx.post", return_value=mock_response), \
             pytest.raises(RuntimeError, match="500"):
            transcriber_runner._call_gemini(audio_file, "k", None)


class TestAudioInfo:
    def test_success(self, audio_file):
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._get_duration",
            return_value=60.0,
        ):
            info = audio_info(file_path=str(audio_file))
        assert info["success"] is True
        assert info["file_name"] == "sample.ogg"
        assert info["format"] == "ogg"
        assert info["duration_sec"] == 60.0
        assert info["estimated_chars"] == 750

    def test_file_not_found(self, tmp_path):
        info = audio_info(file_path=str(tmp_path / "missing.ogg"))
        assert info["success"] is False
        assert "not found" in info["stderr"].lower()

    def test_no_duration_no_estimate(self, audio_file):
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._get_duration",
            return_value=None,
        ):
            info = audio_info(file_path=str(audio_file))
        assert info["success"] is True
        assert info["estimated_chars"] is None


class TestCheckHealth:
    def test_all_good(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._binary_exists",
            return_value=True,
        ):
            h = check_health()
        assert h["healthy"] is True
        assert h["issues"] == []

    def test_missing_ffmpeg(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")

        def _missing(name):
            return name == "ffprobe"

        with patch(
            "kiso_transcriber_mcp.transcriber_runner._binary_exists",
            side_effect=_missing,
        ):
            h = check_health()
        assert h["healthy"] is False
        assert any("ffmpeg" in i for i in h["issues"])

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._binary_exists",
            return_value=True,
        ):
            h = check_health()
        assert h["healthy"] is False
        assert any("OPENROUTER_API_KEY" in i for i in h["issues"])
