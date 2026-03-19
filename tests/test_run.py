"""Unit tests for tool-transcriber."""
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
    _get_api_key, _MAX_OUTPUT_CHARS, _MAX_FILE_SIZE,
)


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
    """Create a fake .ogg file (not real audio, just for path/size tests)."""
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
    """Create audio + non-audio files."""
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
        assert "2)" in result  # only 2 audio files
        assert "voice.ogg" in result
        assert "song.mp3" in result
        assert "report.pdf" not in result
        assert "notes.txt" not in result

    def test_list_empty_directory(self, workspace):
        result = do_list(str(workspace))
        assert "No audio files" in result

    def test_list_no_uploads_dir(self, tmp_path):
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
        assert "1.0 KB" in result
        assert "12s" in result
        assert "Estimated transcript" in result

    def test_info_no_duration(self, workspace, ogg_file):
        with patch("run._get_duration", return_value=None):
            result = do_info(str(workspace), {"file_path": "uploads/voice-message.ogg"})
        assert "voice-message.ogg" in result
        assert "Duration" not in result

    def test_info_missing_file(self, workspace):
        with pytest.raises(FileNotFoundError):
            do_info(str(workspace), {"file_path": "uploads/nope.ogg"})


# ---------------------------------------------------------------------------
# do_transcribe
# ---------------------------------------------------------------------------


class TestDoTranscribe:
    def test_transcribe_success(self, workspace, ogg_file):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Hello, this is a test message."}

        with (
            patch("run._get_duration", return_value=5.0),
            patch("run._get_api_key", return_value="sk-test"),
            patch("httpx.post", return_value=mock_response),
        ):
            result = do_transcribe(str(workspace), {"file_path": "uploads/voice-message.ogg"})
        assert "Transcription: voice-message.ogg (5s)" in result
        assert "Hello, this is a test message." in result

    def test_transcribe_with_language(self, workspace, ogg_file):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Ciao, questo è un test."}

        with (
            patch("run._get_duration", return_value=3.0),
            patch("run._get_api_key", return_value="sk-test"),
            patch("httpx.post", return_value=mock_response) as mock_post,
        ):
            result = do_transcribe(str(workspace), {
                "file_path": "uploads/voice-message.ogg",
                "language": "it",
            })
        # Verify language was passed to API
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["data"]["language"] == "it"
        assert "Ciao" in result

    def test_transcribe_file_too_large(self, workspace):
        big_file = workspace / "uploads" / "huge.ogg"
        big_file.write_bytes(b"\x00" * (_MAX_FILE_SIZE + 1))

        with patch("run._get_duration", return_value=100.0):
            result = do_transcribe(str(workspace), {"file_path": "uploads/huge.ogg"})
        assert "too large" in result.lower()
        assert "25" in result  # mentions the limit

    def test_transcribe_long_audio_cap(self, workspace, ogg_file):
        """Audio >60 min gets a warning note."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": "Long audio content."}

        with (
            patch("run._get_duration", return_value=4500.0),  # 75 min
            patch("run._get_api_key", return_value="sk-test"),
            patch("httpx.post", return_value=mock_response),
        ):
            result = do_transcribe(str(workspace), {"file_path": "uploads/voice-message.ogg"})
        assert "1h 15m" in result  # shows actual duration
        assert "first" in result.lower()  # mentions partial transcription
        assert "Long audio content." in result

    def test_transcribe_api_error(self, workspace, ogg_file):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with (
            patch("run._get_duration", return_value=5.0),
            patch("run._get_api_key", return_value="sk-bad"),
            patch("httpx.post", return_value=mock_response),
        ):
            with pytest.raises(RuntimeError, match="401"):
                do_transcribe(str(workspace), {"file_path": "uploads/voice-message.ogg"})

    def test_transcribe_output_truncation(self, workspace, ogg_file):
        """Very long transcript gets truncated."""
        long_text = "word " * (_MAX_OUTPUT_CHARS // 3)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": long_text}

        with (
            patch("run._get_duration", return_value=120.0),
            patch("run._get_api_key", return_value="sk-test"),
            patch("httpx.post", return_value=mock_response),
        ):
            result = do_transcribe(str(workspace), {"file_path": "uploads/voice-message.ogg"})
        assert len(result) < _MAX_OUTPUT_CHARS + 500  # header + hint overhead
        assert "Showing first" in result

    def test_transcribe_missing_file_path(self, workspace):
        with pytest.raises(ValueError, match="file_path"):
            do_transcribe(str(workspace), {})


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------


class TestGetApiKey:
    def test_tool_specific_key(self):
        with patch.dict("os.environ", {"KISO_TOOL_TRANSCRIBER_API_KEY": "sk-tool"}, clear=True):
            assert _get_api_key() == "sk-tool"

    def test_fallback_to_llm_key(self):
        with patch.dict("os.environ", {"KISO_LLM_API_KEY": "sk-llm"}, clear=True):
            assert _get_api_key() == "sk-llm"

    def test_tool_key_takes_priority(self):
        with patch.dict("os.environ", {
            "KISO_TOOL_TRANSCRIBER_API_KEY": "sk-tool",
            "KISO_LLM_API_KEY": "sk-llm",
        }, clear=True):
            assert _get_api_key() == "sk-tool"

    def test_no_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="No API key"):
                _get_api_key()


# ---------------------------------------------------------------------------
# Path traversal guard
# ---------------------------------------------------------------------------


class TestPathTraversal:
    def test_traversal_rejected(self, workspace):
        with pytest.raises(ValueError, match="traversal"):
            _resolve_path(str(workspace), {"file_path": "../../etc/passwd"})

    def test_valid_path_accepted(self, workspace, ogg_file):
        result = _resolve_path(str(workspace), {"file_path": "uploads/voice-message.ogg"})
        assert result.name == "voice-message.ogg"


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

    def test_kilobytes(self):
        assert "KB" in _format_size(2048)

    def test_megabytes(self):
        assert "MB" in _format_size(5 * 1024 * 1024)


FIXTURES = Path(__file__).parent / "fixtures"


class TestGetDuration:
    def test_ffprobe_success(self, workspace, ogg_file):
        ffprobe_output = json.dumps({"format": {"duration": "12.345"}})
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=ffprobe_output)
            result = _get_duration(ogg_file)
        assert result == pytest.approx(12.345)

    def test_ffprobe_failure(self, workspace, ogg_file):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            result = _get_duration(ogg_file)
        assert result is None

    def test_ffprobe_not_installed(self, workspace, ogg_file):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _get_duration(ogg_file)
        assert result is None

    def test_real_ffprobe_on_fixture(self):
        """Real ffprobe on the static OGG fixture — no mocks."""
        fixture = FIXTURES / "sample.ogg"
        if not fixture.exists():
            pytest.skip("fixture not found — run tests/create_fixtures.sh")
        duration = _get_duration(fixture)
        assert duration is not None
        assert 1.5 < duration < 3.0  # 2-second tone


class TestFixtureIntegration:
    """Tests using the real audio fixture with real ffprobe."""

    @pytest.fixture
    def fixture_workspace(self, tmp_path):
        """Workspace with the static OGG fixture copied to uploads/."""
        import shutil
        fixture = FIXTURES / "sample.ogg"
        if not fixture.exists():
            pytest.skip("fixture not found — run tests/create_fixtures.sh")
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
# Functional: stdin/stdout contract
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
