"""Live integration test — transcribes a tiny fixture audio file via OpenRouter."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from kiso_transcriber_mcp.transcriber_runner import transcribe_audio


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY required for live transcriber test",
    ),
]


def test_transcribe_sample_audio():
    fixture = Path(__file__).parent / "fixtures" / "sample.ogg"
    assert fixture.exists(), "sample.ogg fixture missing"

    result = transcribe_audio(file_path=str(fixture))

    # The fixture is a real (brief) recording; any non-empty transcript
    # proves the end-to-end path works.
    assert result["success"], (
        f"transcription failed: stderr={result['stderr']!r}"
    )
    assert result["text"].strip(), "expected non-empty transcript"
