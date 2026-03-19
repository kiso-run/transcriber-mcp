"""Unit tests for tool-transcriber (Gemini multimodal version)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from run import (
    do_list, do_info, do_transcribe,
    _resolve_path, _get_duration, _format_duration, _format_size,
    _get_api_key, _compress_audio, _MAX_OUTPUT_CHARS, _MAX_DURATION_SECS,
    _COMPRESS_THRESHOLD,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    return tmp_path


@pytest.fixture
def ogg_file(workspace):
    f = workspace / "uploads" / "voice-message.ogg"
    f.write_bytes(b"\x00" * 1024)
    return f


@pytest.fixture
def mp3_file(workspace):
    f = workspace / "uploads" / "recording.mp3"
    f.write_bytes(b"\x00" * 2048)
    return f


@pytest.fixture
def mixed_files(workspace):
    (workspace / "uploads" / "voice.ogg").write_bytes(b"\x00" * 100)
    (workspace / "uploads" / "song.mp3").write_bytes(b"\x00" * 200)
    (workspace / "uploads" / "report.pdf").write_bytes(b"\x00" * 300)
    (workspace / "uploads" / "notes.txt").write_text("hello")
    return workspace


# ---------------------------------------------------------------------------
# do_list
# ---------------------------------------------------------------------------


class TestDoList:
    def test_list_audio_files(self, workspace, ogg_file, mp3_file):
        with patch("run._get_duration", return_value=15.5):
            result = do_list(str(workspace))
        assert "Audio files in uploads/ (2):" in result
        assert "voice-message.ogg" in result
        assert "recording.mp3" in result

    def test_list_filters_non_audio(self, mixed_files):
        with patch("run._get_duration", return_value=None):
            result = do_list(str(mixed_files))
        assert "voice.ogg" in result
        assert "song.mp3" in result
        assert "report.pdf" not in result

    def test_list_empty(self, workspace):
        result = do_list(str(workspace))
        assert "No audio files" in result

    def test_list_no_uploads(self, tmp_path):
        result = do_list(str(tmp_path))
        assert "No uploads/" in result

    def test_list_shows_duration(self, workspace, ogg_file):
        with patch("run._get_duration", return_value=32.0):
            result = do_list(str(workspace))
        assert "32s" in result


# ---------------------------------------------------------------------------
# do_info
# ---------------------------------------------------------------------------


class TestDoInfo:
    def test_info_basic(self, workspace, ogg_file):
        with patch("run._get_duration", return_value=12.5):
            result = do_info(str(workspace), {"file_path": "uploads/voice-message.ogg"})
        assert "voice-message.ogg" in result
        assert ".ogg" in result
        assert "12s" in result
        assert "Estimated transcript" in result

    def test_info_no_duration(self, workspace, ogg_file):
        with patch("run._get_duration", return_value=None):
            result = do_info(str(workspace), {"file_path": "uploads/voice-message.ogg"})
        assert "Duration" not in result

    def test_info_missing(self, workspace):
        with pytest.raises(FileNotFoundError):
            do_info(str(workspace), {"file_path": "uploads/nope.ogg"})


# ---------------------------------------------------------------------------
# do_transcribe
# ---------------------------------------------------------------------------


def _mock_gemini_response(text: str) -> MagicMock:
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "choices": [{"message": {"content": text}}],
    }
    return mock


class TestDoTranscribe:
    def test_transcribe_success(self, workspace, ogg_file):
        with (
            patch("run._get_duration", return_value=5.0),
            patch("run._get_api_key", return_value="sk-test"),
            patch("run._compress_audio", return_value=ogg_file),
            patch("httpx.post", return_value=_mock_gemini_response("Hello, this is a test.")),
        ):
            result = do_transcribe(str(workspace), {"file_path": "uploads/voice-message.ogg"})
        assert "Transcription: voice-message.ogg (5s)" in result
        assert "Hello, this is a test." in result

    def test_transcribe_with_language(self, workspace, ogg_file):
        with (
            patch("run._get_duration", return_value=3.0),
            patch("run._get_api_key", return_value="sk-test"),
            patch("run._compress_audio", return_value=ogg_file),
            patch("httpx.post", return_value=_mock_gemini_response("Ciao, questa è una prova.")) as mock_post,
        ):
            result = do_transcribe(str(workspace), {
                "file_path": "uploads/voice-message.ogg",
                "language": "it",
            })
        # Language hint in the prompt
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs["json"]
        user_content = payload["messages"][0]["content"]
        text_part = next(p for p in user_content if p.get("type") == "text")
        assert "it" in text_part["text"]
        assert "Ciao" in result

    def test_transcribe_too_long(self, workspace, ogg_file):
        with patch("run._get_duration", return_value=600.0):  # 10 min > 5 min cap
            result = do_transcribe(str(workspace), {"file_path": "uploads/voice-message.ogg"})
        assert "too long" in result.lower()
        assert "5m" in result

    def test_transcribe_no_speech(self, workspace, ogg_file):
        with (
            patch("run._get_duration", return_value=2.0),
            patch("run._get_api_key", return_value="sk-test"),
            patch("run._compress_audio", return_value=ogg_file),
            patch("httpx.post", return_value=_mock_gemini_response("")),
        ):
            result = do_transcribe(str(workspace), {"file_path": "uploads/voice-message.ogg"})
        assert "No speech detected" in result

    def test_transcribe_api_error(self, workspace, ogg_file):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal error"
        with (
            patch("run._get_duration", return_value=5.0),
            patch("run._get_api_key", return_value="sk-test"),
            patch("run._compress_audio", return_value=ogg_file),
            patch("httpx.post", return_value=mock_response),
        ):
            with pytest.raises(RuntimeError, match="500"):
                do_transcribe(str(workspace), {"file_path": "uploads/voice-message.ogg"})

    def test_transcribe_output_truncation(self, workspace, ogg_file):
        long_text = "word " * (_MAX_OUTPUT_CHARS // 3)
        with (
            patch("run._get_duration", return_value=120.0),
            patch("run._get_api_key", return_value="sk-test"),
            patch("run._compress_audio", return_value=ogg_file),
            patch("httpx.post", return_value=_mock_gemini_response(long_text)),
        ):
            result = do_transcribe(str(workspace), {"file_path": "uploads/voice-message.ogg"})
        assert len(result) < _MAX_OUTPUT_CHARS + 500
        assert "Showing first" in result

    def test_transcribe_missing_file_path(self, workspace):
        with pytest.raises(ValueError, match="file_path"):
            do_transcribe(str(workspace), {})

    def test_transcribe_empty_choices(self, workspace, ogg_file):
        """Empty choices array from API → RuntimeError, not 'no speech'."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}
        with (
            patch("run._get_duration", return_value=5.0),
            patch("run._get_api_key", return_value="sk-test"),
            patch("run._compress_audio", return_value=ogg_file),
            patch("httpx.post", return_value=mock_response),
        ):
            with pytest.raises(RuntimeError, match="no output"):
                do_transcribe(str(workspace), {"file_path": "uploads/voice-message.ogg"})

    def test_transcribe_malformed_response(self, workspace, ogg_file):
        """API returns choices with missing message/content keys → empty text → 'No speech'."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{}]}
        with (
            patch("run._get_duration", return_value=5.0),
            patch("run._get_api_key", return_value="sk-test"),
            patch("run._compress_audio", return_value=ogg_file),
            patch("httpx.post", return_value=mock_response),
        ):
            result = do_transcribe(str(workspace), {"file_path": "uploads/voice-message.ogg"})
        assert "No speech detected" in result

    def test_transcribe_whitespace_only(self, workspace, ogg_file):
        """Whitespace-only transcription treated as no speech."""
        with (
            patch("run._get_duration", return_value=5.0),
            patch("run._get_api_key", return_value="sk-test"),
            patch("run._compress_audio", return_value=ogg_file),
            patch("httpx.post", return_value=_mock_gemini_response("   \n\n   ")),
        ):
            result = do_transcribe(str(workspace), {"file_path": "uploads/voice-message.ogg"})
        assert "No speech detected" in result


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


class TestCompressAudio:
    def test_skip_small_ogg(self, workspace, ogg_file):
        """Small OGG files are not recompressed."""
        result = _compress_audio(ogg_file)
        assert result == ogg_file  # same path, no temp file

    def test_compress_large_file(self, workspace):
        """Large files get compressed via ffmpeg."""
        big = workspace / "uploads" / "big.mp3"
        big.write_bytes(b"\x00" * (_COMPRESS_THRESHOLD + 1))
        mock_run = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_run) as mock:
            result = _compress_audio(big)
            # ffmpeg was called
            assert mock.called
            call_args = mock.call_args[0][0]
            assert "ffmpeg" in call_args
            assert "-ac" in call_args
            assert "1" in call_args  # mono

    def test_compress_non_ogg_small(self, workspace):
        """Small MP3 gets compressed (not OGG format)."""
        mp3 = workspace / "uploads" / "small.mp3"
        mp3.write_bytes(b"\x00" * 100)  # small but not OGG
        mock_run = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_run):
            _compress_audio(mp3)
            assert mock_run is not None  # ffmpeg called because format is not OGG

    def test_compress_ffmpeg_failure_returns_original(self, workspace):
        """If ffmpeg fails, return the original file."""
        mp3 = workspace / "uploads" / "bad.mp3"
        mp3.write_bytes(b"\x00" * 100)
        mock_run = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=mock_run):
            result = _compress_audio(mp3)
        assert result == mp3


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------


class TestGetApiKey:
    def test_key_found(self):
        with patch.dict("os.environ", {"KISO_LLM_API_KEY": "sk-test"}, clear=True):
            assert _get_api_key() == "sk-test"

    def test_no_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="No API key"):
                _get_api_key()


# ---------------------------------------------------------------------------
# Path traversal
# ---------------------------------------------------------------------------


class TestPathTraversal:
    def test_traversal_rejected(self, workspace):
        with pytest.raises(ValueError, match="traversal"):
            _resolve_path(str(workspace), {"file_path": "../../etc/passwd"})

    def test_valid_path(self, workspace, ogg_file):
        result = _resolve_path(str(workspace), {"file_path": "uploads/voice-message.ogg"})
        assert result.name == "voice-message.ogg"

    def test_traversal_lateral_escape(self, tmp_path):
        """Sibling directory escape via prefix attack."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        sibling = tmp_path / "workspace-data"
        sibling.mkdir()
        secret = sibling / "file.ogg"
        secret.write_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="traversal"):
            _resolve_path(str(workspace), {"file_path": "../workspace-data/file.ogg"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestFormatDuration:
    def test_seconds(self):
        assert _format_duration(5) == "5s"

    def test_minutes(self):
        assert _format_duration(125) == "2m 5s"

    def test_hours(self):
        assert _format_duration(3725) == "1h 2m 5s"

    def test_zero(self):
        assert _format_duration(0) == "0s"


class TestFormatSize:
    def test_bytes(self):
        assert _format_size(500) == "500 B"

    def test_kb(self):
        assert "KB" in _format_size(2048)

    def test_mb(self):
        assert "MB" in _format_size(5 * 1024 * 1024)


class TestGetDuration:
    def test_ffprobe_success(self, workspace, ogg_file):
        output = json.dumps({"format": {"duration": "12.345"}})
        with patch("subprocess.run") as mock:
            mock.return_value = MagicMock(returncode=0, stdout=output)
            result = _get_duration(ogg_file)
        assert result == pytest.approx(12.345)

    def test_ffprobe_failure(self, workspace, ogg_file):
        with patch("subprocess.run") as mock:
            mock.return_value = MagicMock(returncode=1, stdout="")
            assert _get_duration(ogg_file) is None

    def test_ffprobe_not_installed(self, workspace, ogg_file):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _get_duration(ogg_file) is None

    def test_real_ffprobe_on_fixture(self):
        fixture = FIXTURES / "sample.ogg"
        if not fixture.exists():
            pytest.skip("fixture not found")
        import shutil
        if not shutil.which("ffprobe"):
            pytest.skip("ffprobe not installed")
        duration = _get_duration(fixture)
        assert duration is not None
        assert 1.5 < duration < 3.0


class TestFixtureIntegration:
    @pytest.fixture
    def fixture_workspace(self, tmp_path):
        import shutil
        fixture = FIXTURES / "sample.ogg"
        if not fixture.exists():
            pytest.skip("fixture not found")
        if not shutil.which("ffprobe"):
            pytest.skip("ffprobe not installed")
        uploads = tmp_path / "uploads"
        uploads.mkdir()
        shutil.copy2(fixture, uploads / "sample.ogg")
        return tmp_path

    def test_info_real_ffprobe(self, fixture_workspace):
        result = do_info(str(fixture_workspace), {"file_path": "uploads/sample.ogg"})
        assert "sample.ogg" in result
        assert "Duration:" in result
        assert "2s" in result

    def test_list_real_ffprobe(self, fixture_workspace):
        result = do_list(str(fixture_workspace))
        assert "sample.ogg" in result
        assert "2s" in result


# ---------------------------------------------------------------------------
# Functional: stdin/stdout
# ---------------------------------------------------------------------------


class TestFunctional:
    def test_list_via_stdin(self, workspace, ogg_file):
        input_data = json.dumps({
            "args": {"action": "list"},
            "workspace": str(workspace),
        })
        with patch("run._get_duration", return_value=None):
            result = subprocess.run(
                [sys.executable, "run.py"],
                input=input_data, capture_output=True, text=True,
                cwd=str(Path(__file__).parent.parent),
            )
        assert result.returncode == 0
        assert "voice-message.ogg" in result.stdout

    def test_missing_file_exits_1(self, workspace):
        input_data = json.dumps({
            "args": {"action": "transcribe", "file_path": "uploads/nope.ogg"},
            "workspace": str(workspace),
        })
        result = subprocess.run(
            [sys.executable, "run.py"],
            input=input_data, capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 1
        assert "not found" in result.stderr.lower()

    def test_unknown_action_exits_1(self, workspace):
        input_data = json.dumps({
            "args": {"action": "explode"},
            "workspace": str(workspace),
        })
        result = subprocess.run(
            [sys.executable, "run.py"],
            input=input_data, capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 1
        assert "unknown" in result.stderr.lower()

    def test_malformed_json_stdin(self, workspace):
        """Malformed JSON input → exit 1, stderr contains error."""
        result = subprocess.run(
            [sys.executable, "run.py"],
            input="not json", capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 1
        assert "invalid json" in result.stderr.lower() or "json" in result.stderr.lower()
