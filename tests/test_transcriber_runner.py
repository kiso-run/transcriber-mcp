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
    """Transcription via the Gemini backend (opt-in). Set
    KISO_TRANSCRIBER_BACKEND=gemini to force this path; the new default
    is whisper-cpp (see TestTranscribeAudioWhisper)."""

    @pytest.fixture(autouse=True)
    def _force_gemini_backend(self, monkeypatch):
        monkeypatch.setenv("KISO_TRANSCRIBER_BACKEND", "gemini")

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
    @pytest.fixture(autouse=True)
    def _force_gemini_backend(self, monkeypatch):
        monkeypatch.setenv("KISO_TRANSCRIBER_BACKEND", "gemini")

    def test_all_good(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._binary_exists",
            return_value=True,
        ):
            h = check_health()
        assert h["healthy"] is True
        assert h["issues"] == []
        assert h["backend"] == "gemini"

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


class TestCheckHealthWhisper:
    """Health check for the new default backend (whisper-cpp local)."""

    def test_default_backend_is_whisper_cpp(self, monkeypatch):
        monkeypatch.delenv("KISO_TRANSCRIBER_BACKEND", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        # Need ffmpeg/ffprobe still (used for compression and duration);
        # whisper-cli also needed.
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._binary_exists",
            return_value=True,
        ):
            h = check_health()
        assert h["backend"] == "whisper-cpp"

    def test_reports_model_size(self, monkeypatch):
        monkeypatch.setenv("KISO_TRANSCRIBER_BACKEND", "whisper-cpp")
        monkeypatch.setenv("KISO_TRANSCRIBER_WHISPER_MODEL_PATH", "/var/cache/ggml-small.bin")
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._binary_exists",
            return_value=True,
        ), patch("pathlib.Path.is_file", return_value=True):
            h = check_health()
        assert h.get("whisper_model_path") == "/var/cache/ggml-small.bin"

    def test_missing_whisper_binary_unhealthy(self, monkeypatch):
        monkeypatch.setenv("KISO_TRANSCRIBER_BACKEND", "whisper-cpp")
        monkeypatch.setenv("KISO_TRANSCRIBER_WHISPER_MODEL_PATH", "/var/cache/ggml-small.bin")

        def _exists(name):
            return name != "whisper-cli"  # everything except whisper-cli present

        with patch(
            "kiso_transcriber_mcp.transcriber_runner._binary_exists",
            side_effect=_exists,
        ), patch("pathlib.Path.is_file", return_value=True):
            h = check_health()
        assert h["healthy"] is False
        assert any("whisper" in i.lower() for i in h["issues"])

    def test_missing_model_path_unhealthy(self, monkeypatch):
        monkeypatch.setenv("KISO_TRANSCRIBER_BACKEND", "whisper-cpp")
        monkeypatch.delenv("KISO_TRANSCRIBER_WHISPER_MODEL_PATH", raising=False)
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._binary_exists",
            return_value=True,
        ):
            h = check_health()
        assert h["healthy"] is False
        assert any("WHISPER_MODEL_PATH" in i for i in h["issues"])

    def test_unknown_backend_unhealthy(self, monkeypatch):
        monkeypatch.setenv("KISO_TRANSCRIBER_BACKEND", "bogus")
        h = check_health()
        assert h["healthy"] is False
        assert any("bogus" in i for i in h["issues"])


class TestTranscribeAudioWhisper:
    """Transcription via the local Whisper.cpp backend (new default in v0.2)."""

    @pytest.fixture(autouse=True)
    def _force_whisper_backend(self, monkeypatch):
        monkeypatch.setenv("KISO_TRANSCRIBER_BACKEND", "whisper-cpp")
        monkeypatch.setenv(
            "KISO_TRANSCRIBER_WHISPER_MODEL_PATH",
            "/var/cache/ggml-small.bin",
        )

    def test_no_api_key_required(self, monkeypatch, audio_file):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._get_duration",
            return_value=12.0,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._compress_audio",
            side_effect=lambda p: p,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._transcribe_whisper",
            return_value="hello local world",
        ):
            result = transcribe_audio(file_path=str(audio_file))
        assert result["success"] is True
        assert result["text"] == "hello local world"
        assert result["backend"] == "whisper-cpp"

    def test_no_duration_cap_for_whisper(self, monkeypatch, audio_file):
        """Gemini's 5-min cap doesn't apply to local Whisper; raise to 60 min."""
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._get_duration",
            return_value=20 * 60.0,  # 20 minutes — would fail under Gemini cap
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._compress_audio",
            side_effect=lambda p: p,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._transcribe_whisper",
            return_value="long meeting transcript",
        ):
            result = transcribe_audio(file_path=str(audio_file))
        assert result["success"] is True

    def test_extreme_duration_still_capped(self, monkeypatch, audio_file):
        """Even Whisper has an internal cap to avoid runaway (60 min)."""
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._get_duration",
            return_value=120 * 60.0,  # 2 hours
        ):
            result = transcribe_audio(file_path=str(audio_file))
        assert result["success"] is False
        assert "too long" in result["stderr"].lower()

    def test_language_hint_passed_to_whisper(self, monkeypatch, audio_file):
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._get_duration",
            return_value=12.0,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._compress_audio",
            side_effect=lambda p: p,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._transcribe_whisper",
            return_value="ciao",
        ) as run:
            transcribe_audio(file_path=str(audio_file), language="it")
        assert run.call_args.kwargs.get("language") == "it"

    def test_subprocess_error_surfaces(self, monkeypatch, audio_file):
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._get_duration",
            return_value=12.0,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._compress_audio",
            side_effect=lambda p: p,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._transcribe_whisper",
            side_effect=RuntimeError("whisper-cli: model not found"),
        ):
            result = transcribe_audio(file_path=str(audio_file))
        assert result["success"] is False
        assert "whisper" in result["stderr"].lower()

    def test_truncates_long_output(self, monkeypatch, audio_file):
        long_text = "x" * 100_000
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._get_duration",
            return_value=12.0,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._compress_audio",
            side_effect=lambda p: p,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._transcribe_whisper",
            return_value=long_text,
        ):
            result = transcribe_audio(file_path=str(audio_file))
        assert result["truncated"] is True
        assert len(result["text"]) <= 50_000

    def test_response_includes_backend_field(self, monkeypatch, audio_file):
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._get_duration",
            return_value=12.0,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._compress_audio",
            side_effect=lambda p: p,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._transcribe_whisper",
            return_value="text",
        ):
            result = transcribe_audio(file_path=str(audio_file))
        assert result.get("backend") == "whisper-cpp"


class TestWhisperRunner:
    """Direct unit tests for the _transcribe_whisper subprocess wrapper."""

    def test_invokes_whisper_cli(self, audio_file, monkeypatch):
        from kiso_transcriber_mcp import transcriber_runner

        monkeypatch.setenv(
            "KISO_TRANSCRIBER_WHISPER_MODEL_PATH", "/cache/ggml-small.bin",
        )
        completed = MagicMock(returncode=0, stdout="transcript text\n", stderr="")
        with patch(
            "kiso_transcriber_mcp.transcriber_runner.subprocess.run",
            return_value=completed,
        ) as run:
            text = transcriber_runner._transcribe_whisper(audio_file, language="it")
        # Trailing newline stripped by _strip_whisper_artifacts.
        assert text == "transcript text"
        cmd = run.call_args.args[0]
        assert cmd[0] == "whisper-cli" or cmd[0].endswith("whisper-cli")
        assert "/cache/ggml-small.bin" in cmd  # model path passed via -m
        assert str(audio_file) in cmd
        assert "it" in cmd  # language hint

    def test_binary_override_via_env(self, audio_file, monkeypatch):
        from kiso_transcriber_mcp import transcriber_runner

        monkeypatch.setenv(
            "KISO_TRANSCRIBER_WHISPER_MODEL_PATH", "/cache/ggml-small.bin",
        )
        monkeypatch.setenv("KISO_TRANSCRIBER_WHISPER_BIN", "/opt/whisper.cpp/main")
        completed = MagicMock(returncode=0, stdout="t", stderr="")
        with patch(
            "kiso_transcriber_mcp.transcriber_runner.subprocess.run",
            return_value=completed,
        ) as run:
            transcriber_runner._transcribe_whisper(audio_file, language=None)
        assert run.call_args.args[0][0] == "/opt/whisper.cpp/main"

    def test_missing_model_path_raises(self, audio_file, monkeypatch):
        from kiso_transcriber_mcp import transcriber_runner

        monkeypatch.delenv("KISO_TRANSCRIBER_WHISPER_MODEL_PATH", raising=False)
        with pytest.raises(RuntimeError, match="WHISPER_MODEL_PATH"):
            transcriber_runner._transcribe_whisper(audio_file, language=None)

    def test_nonzero_exit_raises(self, audio_file, monkeypatch):
        from kiso_transcriber_mcp import transcriber_runner

        monkeypatch.setenv(
            "KISO_TRANSCRIBER_WHISPER_MODEL_PATH", "/cache/ggml-small.bin",
        )
        completed = MagicMock(returncode=2, stdout="", stderr="model load failed")
        with patch(
            "kiso_transcriber_mcp.transcriber_runner.subprocess.run",
            return_value=completed,
        ), pytest.raises(RuntimeError, match="whisper"):
            transcriber_runner._transcribe_whisper(audio_file, language=None)

    def test_binary_missing_raises(self, audio_file, monkeypatch):
        from kiso_transcriber_mcp import transcriber_runner

        monkeypatch.setenv(
            "KISO_TRANSCRIBER_WHISPER_MODEL_PATH", "/cache/ggml-small.bin",
        )
        with patch(
            "kiso_transcriber_mcp.transcriber_runner.subprocess.run",
            side_effect=FileNotFoundError("whisper-cli not in PATH"),
        ), pytest.raises(RuntimeError, match="whisper"):
            transcriber_runner._transcribe_whisper(audio_file, language=None)
