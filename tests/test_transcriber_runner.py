"""Unit tests for kiso_transcriber_mcp.transcriber_runner — v0.3 cloud-only contract."""
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


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "KISO_TRANSCRIBER_BACKEND",
        "OPENROUTER_API_KEY",
        "LITELLM_BASE_URL",
        "LITELLM_API_KEY",
        "KISO_TRANSCRIBER_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


class TestBackendSelection:
    def test_default_backend_is_openrouter(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._binary_exists",
            return_value=True,
        ):
            h = check_health()
        assert h["backend"] == "openrouter"

    def test_litellm_backend_selectable(self, monkeypatch):
        monkeypatch.setenv("KISO_TRANSCRIBER_BACKEND", "litellm")
        monkeypatch.setenv("LITELLM_BASE_URL", "http://lite:4000")
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._binary_exists",
            return_value=True,
        ):
            h = check_health()
        assert h["backend"] == "litellm"

    def test_unknown_backend_unhealthy(self, monkeypatch):
        monkeypatch.setenv("KISO_TRANSCRIBER_BACKEND", "bogus")
        h = check_health()
        assert h["healthy"] is False
        assert any("bogus" in i for i in h["issues"])

    def test_whisper_cpp_backend_no_longer_supported(self, monkeypatch):
        """v0.3 removes the local whisper.cpp backend entirely."""
        monkeypatch.setenv("KISO_TRANSCRIBER_BACKEND", "whisper-cpp")
        h = check_health()
        assert h["healthy"] is False
        assert any("whisper-cpp" in i.lower() for i in h["issues"])

    def test_gemini_alias_no_longer_supported(self, monkeypatch):
        """v0.3 renames 'gemini' (old opt-in name) to 'openrouter'."""
        monkeypatch.setenv("KISO_TRANSCRIBER_BACKEND", "gemini")
        h = check_health()
        assert h["healthy"] is False
        assert any("gemini" in i.lower() for i in h["issues"])


class TestTranscribeAudioOpenrouter:
    """Transcription via the default openrouter backend."""

    def test_missing_api_key_fails(self, monkeypatch, audio_file):
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
            return_value=900.0,  # 15 min — exceeds the 5-min cap
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
            "kiso_transcriber_mcp.transcriber_runner._call_audio_llm",
            return_value="hello world",
        ):
            result = transcribe_audio(file_path=str(audio_file))
        assert result["success"] is True
        assert result["text"] == "hello world"
        assert result["duration_sec"] == 12.0
        assert result["format"] == "ogg"
        assert result["truncated"] is False
        assert result["backend"] == "openrouter"

    def test_large_transcript_truncates(self, monkeypatch, audio_file):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._get_duration",
            return_value=12.0,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._compress_audio",
            side_effect=lambda p: p,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._call_audio_llm",
            return_value="a" * 100_000,
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
            "kiso_transcriber_mcp.transcriber_runner._call_audio_llm",
            side_effect=RuntimeError("audio API error (500): boom"),
        ):
            result = transcribe_audio(file_path=str(audio_file))
        assert result["success"] is False
        assert "500" in result["stderr"]

    def test_language_hint_passed_to_call(self, monkeypatch, audio_file):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._get_duration",
            return_value=12.0,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._compress_audio",
            side_effect=lambda p: p,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._call_audio_llm",
            return_value="ciao",
        ) as call:
            transcribe_audio(file_path=str(audio_file), language="it")
        assert call.call_args.kwargs.get("language") == "it"


class TestTranscribeAudioLitellm:
    """Transcription via the litellm backend — consumer's local LiteLLM gateway."""

    @pytest.fixture(autouse=True)
    def _force_litellm(self, monkeypatch):
        monkeypatch.setenv("KISO_TRANSCRIBER_BACKEND", "litellm")
        monkeypatch.setenv("LITELLM_BASE_URL", "http://lite:4000")

    def test_missing_base_url_fails(self, monkeypatch, audio_file):
        monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
        result = transcribe_audio(file_path=str(audio_file))
        assert result["success"] is False
        assert "LITELLM_BASE_URL" in result["stderr"]

    def test_success_uses_litellm_base_url(self, monkeypatch, audio_file):
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._get_duration",
            return_value=12.0,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._compress_audio",
            side_effect=lambda p: p,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._call_audio_llm",
            return_value="text from litellm",
        ) as call:
            result = transcribe_audio(file_path=str(audio_file))
        assert result["success"] is True
        assert result["text"] == "text from litellm"
        assert result["backend"] == "litellm"
        assert call.call_args.kwargs["base_url"].startswith("http://lite:4000")

    def test_litellm_api_key_optional(self, monkeypatch, audio_file):
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._get_duration",
            return_value=12.0,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._compress_audio",
            side_effect=lambda p: p,
        ), patch(
            "kiso_transcriber_mcp.transcriber_runner._call_audio_llm",
            return_value="text",
        ):
            result = transcribe_audio(file_path=str(audio_file))
        assert result["success"] is True


class TestCallAudioLlm:
    """Direct unit tests for the single _call_audio_llm function."""

    def test_posts_to_base_url(self, audio_file):
        from kiso_transcriber_mcp import transcriber_runner

        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "hello"}}]
        }
        with patch("httpx.post", return_value=mock_response) as post:
            text = transcriber_runner._call_audio_llm(
                base_url="http://example/v1",
                api_key="k",
                model="vendor/model",
                file_path=audio_file,
                language=None,
            )
        assert text == "hello"
        url = post.call_args.args[0]
        assert url == "http://example/v1/chat/completions"

    def test_authorization_header_when_key_set(self, audio_file):
        from kiso_transcriber_mcp import transcriber_runner

        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        with patch("httpx.post", return_value=mock_response) as post:
            transcriber_runner._call_audio_llm(
                base_url="http://x/v1",
                api_key="secret",
                model="m",
                file_path=audio_file,
                language=None,
            )
        headers = post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer secret"

    def test_no_auth_header_when_key_empty(self, audio_file):
        from kiso_transcriber_mcp import transcriber_runner

        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        with patch("httpx.post", return_value=mock_response) as post:
            transcriber_runner._call_audio_llm(
                base_url="http://x/v1",
                api_key="",
                model="m",
                file_path=audio_file,
                language=None,
            )
        headers = post.call_args.kwargs["headers"]
        assert "Authorization" not in headers

    def test_language_hint_in_prompt(self, audio_file):
        from kiso_transcriber_mcp import transcriber_runner

        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ciao"}}]
        }
        with patch("httpx.post", return_value=mock_response) as post:
            transcriber_runner._call_audio_llm(
                base_url="http://x/v1",
                api_key="k",
                model="m",
                file_path=audio_file,
                language="it",
            )
        body = post.call_args.kwargs["json"]
        # Last message content has both the audio block and a text block;
        # the text block carries the language hint.
        text_parts = [
            c["text"] for c in body["messages"][0]["content"] if c.get("type") == "text"
        ]
        assert any("it" in t for t in text_parts)

    def test_empty_response_retries_then_returns_empty(self, audio_file):
        from kiso_transcriber_mcp import transcriber_runner

        mock_response = MagicMock(status_code=200)
        mock_response.json.return_value = {
            "choices": [{"message": {"content": ""}}]
        }
        with patch("httpx.post", return_value=mock_response) as post, \
             patch("kiso_transcriber_mcp.transcriber_runner.time.sleep"):
            result = transcriber_runner._call_audio_llm(
                base_url="http://x/v1",
                api_key="k",
                model="m",
                file_path=audio_file,
                language=None,
            )
        assert result == ""
        assert post.call_count == 3

    def test_http_error_raises(self, audio_file):
        from kiso_transcriber_mcp import transcriber_runner

        mock_response = MagicMock(status_code=500, text="server boom")
        with patch("httpx.post", return_value=mock_response), \
             pytest.raises(RuntimeError, match="500"):
            transcriber_runner._call_audio_llm(
                base_url="http://x/v1",
                api_key="k",
                model="m",
                file_path=audio_file,
                language=None,
            )


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
    def test_openrouter_healthy(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._binary_exists",
            return_value=True,
        ):
            h = check_health()
        assert h["healthy"] is True
        assert h["issues"] == []
        assert h["backend"] == "openrouter"

    def test_openrouter_missing_key_unhealthy(self, monkeypatch):
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._binary_exists",
            return_value=True,
        ):
            h = check_health()
        assert h["healthy"] is False
        assert any("OPENROUTER_API_KEY" in i for i in h["issues"])

    def test_litellm_healthy(self, monkeypatch):
        monkeypatch.setenv("KISO_TRANSCRIBER_BACKEND", "litellm")
        monkeypatch.setenv("LITELLM_BASE_URL", "http://lite:4000")
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._binary_exists",
            return_value=True,
        ):
            h = check_health()
        assert h["healthy"] is True
        assert h["backend"] == "litellm"

    def test_litellm_missing_base_url_unhealthy(self, monkeypatch):
        monkeypatch.setenv("KISO_TRANSCRIBER_BACKEND", "litellm")
        with patch(
            "kiso_transcriber_mcp.transcriber_runner._binary_exists",
            return_value=True,
        ):
            h = check_health()
        assert h["healthy"] is False
        assert any("LITELLM_BASE_URL" in i for i in h["issues"])

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

    def test_no_whisper_probe(self, monkeypatch):
        """v0.3 removes whisper.cpp — doctor must not probe for whisper-cli."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "k")
        calls = []

        def _track(name):
            calls.append(name)
            return True

        with patch(
            "kiso_transcriber_mcp.transcriber_runner._binary_exists",
            side_effect=_track,
        ):
            check_health()
        assert not any("whisper" in c.lower() for c in calls)
