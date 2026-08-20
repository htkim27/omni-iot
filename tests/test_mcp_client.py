from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omni_iot.mcp_client import McpManager, _tool_result_text, _truncate_result


class _FakeClient:
    def __init__(
        self,
        *_args: object,
        tool_name: str = "allowed",
        **_kwargs: object,
    ) -> None:
        self.tool_name = tool_name
        self.list_cursors: list[str | None] = []
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def list_tools(self, *, cursor: str | None = None) -> object:
        self.list_cursors.append(cursor)
        if cursor is None:
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name=self.tool_name,
                        description="Allowed tool",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "count": {"type": "integer", "maximum": 20},
                                "extra_snippets": {"type": "boolean"},
                                "summary": {"type": "boolean"},
                            },
                        },
                    )
                ],
                next_cursor="page-2",
            )
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="hidden",
                    description="Hidden tool",
                    input_schema={"type": "object"},
                )
            ],
            next_cursor=None,
        )

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, object],
        **_kwargs: object,
    ) -> object:
        self.calls.append((name, arguments))
        return SimpleNamespace(
            structured_content={"value": 7},
            content=[],
            is_error=False,
        )


class _FakeHttpClient:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    async def __aenter__(self) -> _FakeHttpClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None


class McpManagerTest(unittest.IsolatedAsyncioTestCase):
    async def test_brave_search_is_small_by_default_and_hard_capped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".mcp.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "mcpServers": {
                            "brave-search": {
                                "command": sys.executable,
                                "allowedTools": ["brave_web_search"],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            fake = _FakeClient(tool_name="brave_web_search")
            with patch("omni_iot.mcp_client.Client", return_value=fake):
                manager = McpManager(path, brave_search_max_results=5)
                await manager.start()
                try:
                    tool = manager.openai_tools()[0]["function"]
                    await manager.call_tool(
                        "brave-search__brave_web_search",
                        {
                            "query": "first query",
                            "count": 20,
                            "extra_snippets": True,
                            "summary": True,
                        },
                    )
                    await manager.call_tool(
                        "brave-search__brave_web_search",
                        {"query": "different query"},
                    )
                finally:
                    await manager.stop()

        self.assertIn("meaningfully different query", tool["description"])
        count_schema = tool["parameters"]["properties"]["count"]
        self.assertEqual(count_schema["maximum"], 5)
        self.assertEqual(
            fake.calls,
            [
                (
                    "brave_web_search",
                    {
                        "query": "first query",
                        "count": 5,
                        "extra_snippets": False,
                        "summary": False,
                    },
                ),
                (
                    "brave_web_search",
                    {
                        "query": "different query",
                        "count": 5,
                        "extra_snippets": False,
                        "summary": False,
                    },
                ),
            ],
        )

    async def test_paginates_filters_namespaces_and_calls_allowed_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".mcp.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "mcpServers": {
                            "fixture": {
                                "command": sys.executable,
                                "enabled": True,
                                "allowedTools": ["allowed"],
                                "timeoutSeconds": 2,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            fake = _FakeClient()
            with patch("omni_iot.mcp_client.Client", return_value=fake):
                manager = McpManager(path)
                await manager.start()
                try:
                    tools = manager.openai_tools()
                    result = await manager.call_tool("fixture__allowed", {"x": 1})
                    denied = await manager.call_tool("fixture__hidden", {})
                finally:
                    await manager.stop()

            self.assertEqual(fake.list_cursors, [None, "page-2"])
            self.assertEqual([item["function"]["name"] for item in tools], ["fixture__allowed"])
            self.assertEqual(
                manager.model_name_for("fixture", "allowed"),
                "fixture__allowed",
            )
            self.assertEqual(fake.calls, [("allowed", {"x": 1})])
            self.assertEqual(json.loads(result.content)["structuredContent"], {"value": 7})
            self.assertTrue(denied.is_error)

    async def test_catalog_overflow_is_fail_closed_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".mcp.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "mcpServers": {
                            "fixture": {
                                "command": sys.executable,
                                "allowedTools": ["*"],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch("omni_iot.mcp_client.Client", return_value=_FakeClient()):
                manager = McpManager(path, catalog_max_chars=20)
                await manager.start()
                try:
                    self.assertEqual(manager.openai_tools(), [])
                    self.assertIn("Reduce allowedTools", manager.health()["error"])
                finally:
                    await manager.stop()

    async def test_streamable_http_uses_resolved_headers_without_exposing_them(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".mcp.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "mcpServers": {
                            "remote": {
                                "url": "http://127.0.0.1:8765/mcp",
                                "headers": {"Authorization": "Bearer ${REMOTE_TOKEN}"},
                                "allowedTools": ["allowed"],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            http = _FakeHttpClient()
            with (
                patch.dict("os.environ", {"REMOTE_TOKEN": "top-secret"}),
                patch("omni_iot.mcp_client.httpx2.AsyncClient", return_value=http) as factory,
                patch("omni_iot.mcp_client.streamable_http_client", return_value=object()),
                patch("omni_iot.mcp_client.Client", return_value=_FakeClient()),
            ):
                manager = McpManager(path)
                await manager.start()
                try:
                    health = manager.health()
                finally:
                    await manager.stop()

            self.assertEqual(
                factory.call_args.kwargs["headers"],
                {"Authorization": "Bearer top-secret"},
            )
            self.assertNotIn("top-secret", json.dumps(health))
            self.assertEqual(health["servers"][0]["transport"], "streamable-http")

    def test_truncation_is_explicit_and_bounded(self) -> None:
        truncated = _truncate_result("x" * 500, 120)
        payload = json.loads(truncated)
        self.assertLessEqual(len(truncated), 120)
        self.assertTrue(payload["truncated"])
        self.assertEqual(payload["originalCharacters"], 500)

    def test_text_and_error_result_is_preserved(self) -> None:
        result = SimpleNamespace(
            structured_content=None,
            content=[
                SimpleNamespace(type="text", text="first"),
                SimpleNamespace(type="text", text="second"),
            ],
            is_error=True,
        )

        payload = json.loads(_tool_result_text(result))

        self.assertTrue(payload["isError"])
        self.assertEqual(payload["content"], "first\nsecond")


if __name__ == "__main__":
    unittest.main()
