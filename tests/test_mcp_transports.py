from __future__ import annotations

import asyncio
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path

from omni_iot.mcp_client import McpManager


FIXTURE = Path(__file__).parent / "fixtures" / "mcp_test_server.py"


def _write_config(path: Path, server: dict[str, object]) -> None:
    path.write_text(
        json.dumps({"version": 1, "mcpServers": {"fixture": server}}),
        encoding="utf-8",
    )


class McpTransportIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_official_sdk_stdio_transport(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".mcp.json"
            _write_config(
                config_path,
                {
                    "command": sys.executable,
                    "args": [str(FIXTURE)],
                    "allowedTools": ["add", "echo"],
                    "timeoutSeconds": 10,
                },
            )
            manager = McpManager(config_path)
            await manager.start()
            try:
                result = await manager.call_tool(
                    "fixture__add",
                    {"left": 2, "right": 3},
                )
                health = manager.health()
            finally:
                await manager.stop()

        self.assertTrue(health["ok"])
        self.assertEqual(json.loads(result.content)["structuredContent"], {"sum": 5})

    async def test_official_sdk_streamable_http_transport(self) -> None:
        port = _unused_local_port()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(FIXTURE),
            "--http",
            "--port",
            str(port),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await _wait_for_port(port, process)
            with tempfile.TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / ".mcp.json"
                _write_config(
                    config_path,
                    {
                        "url": f"http://127.0.0.1:{port}/mcp",
                        "allowedTools": ["echo"],
                        "timeoutSeconds": 10,
                    },
                )
                manager = McpManager(config_path)
                await manager.start()
                try:
                    result = await manager.call_tool(
                        "fixture__echo",
                        {"text": "hello"},
                    )
                    health = manager.health()
                finally:
                    await manager.stop()
        finally:
            if process.returncode is None:
                process.terminate()
            await process.wait()

        self.assertTrue(health["ok"])
        self.assertEqual(json.loads(result.content)["content"], "hello")


def _unused_local_port() -> int:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


async def _wait_for_port(port: int, process: asyncio.subprocess.Process) -> None:
    for _ in range(50):
        if process.returncode is not None:
            raise RuntimeError("MCP HTTP fixture exited during startup.")
        try:
            _reader, writer = await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            await asyncio.sleep(0.1)
            continue
        writer.close()
        await writer.wait_closed()
        return
    raise TimeoutError("MCP HTTP fixture did not start.")


if __name__ == "__main__":
    unittest.main()
