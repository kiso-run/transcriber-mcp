"""Tests for the MCP tool surface exposed by kiso_transcriber_mcp.server."""
from __future__ import annotations

import json
from unittest.mock import patch


def _decode(result) -> dict:
    blocks = result if isinstance(result, list) else list(result)
    return json.loads(blocks[0].text)


def test_mcp_instance_named():
    from kiso_transcriber_mcp import server
    assert server.mcp.name == "kiso-transcriber"


async def test_all_tools_registered():
    from kiso_transcriber_mcp import server
    tools = await server.mcp.list_tools()
    names = {t.name for t in tools}
    assert {"transcribe_audio", "audio_info", "doctor"} <= names


async def test_transcribe_schema_requires_file_path():
    from kiso_transcriber_mcp import server
    tools = await server.mcp.list_tools()
    t = next(x for x in tools if x.name == "transcribe_audio")
    assert "file_path" in t.inputSchema.get("required", [])


async def test_transcribe_audio_delegates():
    from kiso_transcriber_mcp import server
    stub = {
        "success": True, "text": "hello", "duration_sec": 10.0,
        "format": "ogg", "truncated": False, "stderr": "",
    }
    with patch(
        "kiso_transcriber_mcp.server.transcriber_runner.transcribe_audio",
        return_value=stub,
    ) as run:
        result = await server.mcp.call_tool(
            "transcribe_audio",
            {"file_path": "/tmp/x.ogg", "language": "it"},
        )
    run.assert_called_once_with(file_path="/tmp/x.ogg", language="it")
    assert _decode(result) == stub


async def test_audio_info_delegates():
    from kiso_transcriber_mcp import server
    stub = {
        "success": True, "file_name": "x.ogg", "size_bytes": 1234,
        "format": "ogg", "duration_sec": 12.0, "estimated_chars": 150,
        "stderr": "",
    }
    with patch(
        "kiso_transcriber_mcp.server.transcriber_runner.audio_info",
        return_value=stub,
    ) as run:
        result = await server.mcp.call_tool(
            "audio_info", {"file_path": "/tmp/x.ogg"},
        )
    run.assert_called_once_with(file_path="/tmp/x.ogg")
    assert _decode(result) == stub


async def test_doctor_delegates():
    from kiso_transcriber_mcp import server
    stub = {"healthy": True, "issues": []}
    with patch(
        "kiso_transcriber_mcp.server.transcriber_runner.check_health",
        return_value=stub,
    ) as run:
        result = await server.mcp.call_tool("doctor", {})
    run.assert_called_once_with()
    assert _decode(result) == stub


def test_main_calls_run():
    from kiso_transcriber_mcp import server
    with patch.object(server.mcp, "run") as run:
        server.main()
    run.assert_called_once()
