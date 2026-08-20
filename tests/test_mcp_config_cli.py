from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from omni_iot.mcp_cli import _parser, _run
from omni_iot.mcp_config import McpConfigError, expand_env, load_mcp_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class McpConfigTest(unittest.TestCase):
    def test_example_brave_server_is_disabled_and_exactly_allowlisted(self) -> None:
        with patch.dict(os.environ, {"BRAVE_API_KEY": "test-only"}):
            server = load_mcp_config(PROJECT_ROOT / ".mcp.example.json").servers[
                "brave-search"
            ]
            self.assertEqual(server.resolved_env(), {"BRAVE_API_KEY": "test-only"})

        self.assertFalse(server.enabled)
        self.assertEqual(
            server.allowed_tools,
            ("brave_web_search", "brave_news_search"),
        )
        self.assertEqual(server.env, {"BRAVE_API_KEY": "${BRAVE_API_KEY}"})

    def test_loads_stdio_http_allowlists_and_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".mcp.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "mcpServers": {
                            "local": {
                                "command": sys.executable,
                                "args": ["server.py"],
                                "enabled": True,
                                "allowedTools": ["status"],
                            },
                            "remote": {
                                "url": "http://127.0.0.1:8765/mcp",
                                "headers": {"Authorization": "Bearer ${TEST_TOKEN}"},
                                "enabled": False,
                                "allowedTools": ["*"],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"TEST_TOKEN": "secret-value"}):
                config = load_mcp_config(path)
                headers = config.servers["remote"].resolved_headers()

            self.assertTrue(config.servers["local"].allows("status"))
            self.assertFalse(config.servers["local"].allows("other"))
            self.assertTrue(config.servers["remote"].allows("anything"))
            self.assertEqual(headers, {"Authorization": "Bearer secret-value"})

    def test_rejects_duplicate_server_ids_and_mixed_transports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".mcp.json"
            path.write_text(
                '{"version":1,"mcpServers":{"same":{"command":"x",'
                '"allowedTools":[]},"same":{"url":"http://localhost/mcp",'
                '"allowedTools":[]}}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(McpConfigError, "Duplicate JSON key"):
                load_mcp_config(path)

            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "mcpServers": {
                            "mixed": {
                                "command": "x",
                                "url": "http://localhost/mcp",
                                "allowedTools": [],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(McpConfigError, "exactly one"):
                load_mcp_config(path)

    def test_missing_environment_error_never_contains_other_secret_values(self) -> None:
        with patch.dict(os.environ, {"REAL_SECRET": "do-not-print"}, clear=True):
            with self.assertRaises(McpConfigError) as raised:
                expand_env("Bearer ${MISSING_TOKEN}", "authorization header")
        self.assertIn("MISSING_TOKEN", str(raised.exception))
        self.assertNotIn("do-not-print", str(raised.exception))


class McpCliTest(unittest.TestCase):
    def test_doctor_uses_runtime_catalog_and_result_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".mcp.json"
            manager = SimpleNamespace(
                start=AsyncMock(),
                stop=AsyncMock(),
                health=lambda: {"ok": True, "error": None, "servers": []},
            )
            settings = SimpleNamespace(
                mcp_tool_catalog_max_chars=16_000,
                mcp_tool_result_max_chars=6_000,
                mcp_brave_search_max_results=5,
            )
            parser = _parser()
            args = parser.parse_args(
                ["--config", str(path), "doctor", "--json"]
            )
            with (
                patch("omni_iot.mcp_cli.Settings", return_value=settings),
                patch("omni_iot.mcp_cli.McpManager", return_value=manager) as factory,
            ):
                self.assertEqual(_run(args, path), 0)

        factory.assert_called_once_with(
            path,
            catalog_max_chars=16_000,
            result_max_chars=6_000,
            brave_search_max_results=5,
        )

    def test_registration_and_allowlist_updates_are_atomic_and_private(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".mcp.json"
            parser = _parser()
            args = parser.parse_args(
                [
                    "--config",
                    str(path),
                    "add-stdio",
                    "switchbot",
                    "--command",
                    sys.executable,
                    "--arg",
                    "server.py",
                    "--allow-tool",
                    "list_devices",
                ]
            )
            self.assertEqual(_run(args, path), 0)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

            allow = parser.parse_args(
                ["--config", str(path), "allow", "switchbot", "send_command"]
            )
            self.assertEqual(_run(allow, path), 0)
            server = load_mcp_config(path).servers["switchbot"]
            self.assertEqual(
                server.allowed_tools,
                ("list_devices", "send_command"),
            )
            self.assertTrue(Path(server.command or "").is_absolute())
            self.assertFalse(any(path.parent.glob(f".{path.name}.*")))


if __name__ == "__main__":
    unittest.main()
