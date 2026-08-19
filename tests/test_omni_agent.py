from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

from omni_iot.config import Settings
from omni_iot.mcp_client import McpCallResult
from omni_iot.omni_agent import OmniAgent
from omni_iot.omni_llama import ChatCompletion, ToolCall


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "switchbot__send_command",
            "description": "Control a device",
            "parameters": {"type": "object"},
        },
    }
]


async def _direct_to_thread(function: Callable[..., Any], *args: object, **kwargs: object) -> Any:
    return function(*args, **kwargs)


class _FakeMcp:
    def __init__(self, tools: list[dict[str, object]] | None = None) -> None:
        self.tools = TOOLS if tools is None else tools
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def prepare_turn(self) -> list[dict[str, object]]:
        return self.tools

    async def call_tool(self, name: str, arguments: dict[str, object]) -> McpCallResult:
        self.calls.append((name, arguments))
        return McpCallResult(
            content=json.dumps({"isError": False, "content": "ok"}),
            is_error=False,
            server_name="switchbot",
            tool_name="send_command",
            seconds=0.01,
        )


class _FakeLlama:
    def __init__(self, completions: list[ChatCompletion]) -> None:
        self.completions = iter(completions)
        self.chat_requests: list[tuple[list[dict[str, object]], object]] = []
        self.generate_calls = 0

    def chat(
        self,
        messages: list[dict[str, object]],
        _max_tokens: int | None,
        tools: list[dict[str, object]] | None,
    ) -> ChatCompletion:
        self.chat_requests.append((list(messages), tools))
        return next(self.completions)

    def generate(self, *_args: object) -> str:
        self.generate_calls += 1
        return '{"user_text":"안녕","text":"반가워요."}'


class OmniAgentTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.to_thread = patch(
            "omni_iot.omni_agent.asyncio.to_thread",
            side_effect=_direct_to_thread,
        )
        self.to_thread.start()

    def tearDown(self) -> None:
        self.to_thread.stop()

    async def test_executes_multiple_calls_in_order_and_returns_final_json(self) -> None:
        calls = (
            ToolCall("one", "switchbot__send_command", '{"command":"turnOn"}'),
            ToolCall("two", "switchbot__send_command", '{"command":"turnOff"}'),
        )
        llama = _FakeLlama(
            [
                ChatCompletion(content=None, tool_calls=calls),
                ChatCompletion(
                    content='{"transcript":"불을 껐다 켜줘","response":"완료했습니다."}'
                ),
            ]
        )
        mcp = _FakeMcp()
        with tempfile.TemporaryDirectory() as temp_dir:
            audio = Path(temp_dir) / "input.wav"
            audio.write_bytes(b"wav")
            agent = OmniAgent(  # type: ignore[arg-type]
                llama,
                mcp,
                Settings(runtime_dir=Path(temp_dir)),
            )
            generation = await agent.generate(audio, [])

        self.assertEqual(
            [arguments["command"] for _, arguments in mcp.calls],
            ["turnOn", "turnOff"],
        )
        self.assertEqual(json.loads(generation.output)["text"], "완료했습니다.")
        self.assertEqual(len(generation.tool_trace), 2)
        second_messages = llama.chat_requests[1][0]
        self.assertEqual(second_messages[-2]["tool_call_id"], "one")
        self.assertEqual(second_messages[-1]["tool_call_id"], "two")

    async def test_invalid_arguments_are_model_visible_without_calling_mcp(self) -> None:
        llama = _FakeLlama(
            [
                ChatCompletion(
                    content=None,
                    tool_calls=(ToolCall("bad", "switchbot__send_command", "{"),),
                ),
                ChatCompletion(
                    content=(
                        '{"transcript":null,'
                        '"response":"명령 형식이 잘못됐습니다."}'
                    )
                ),
            ]
        )
        mcp = _FakeMcp()
        with tempfile.TemporaryDirectory() as temp_dir:
            audio = Path(temp_dir) / "input.wav"
            audio.write_bytes(b"wav")
            generation = await OmniAgent(
                llama, mcp, Settings(runtime_dir=Path(temp_dir))  # type: ignore[arg-type]
            ).generate(audio, [])

        self.assertEqual(mcp.calls, [])
        self.assertIn("잘못", json.loads(generation.output)["text"])
        tool_message = llama.chat_requests[1][0][-1]
        self.assertTrue(json.loads(tool_message["content"])["isError"])

    async def test_no_catalog_preserves_direct_generation_path(self) -> None:
        llama = _FakeLlama([])
        mcp = _FakeMcp(tools=[])
        with tempfile.TemporaryDirectory() as temp_dir:
            audio = Path(temp_dir) / "input.wav"
            audio.write_bytes(b"wav")
            result = await OmniAgent(
                llama, mcp, Settings(runtime_dir=Path(temp_dir))  # type: ignore[arg-type]
            ).generate(audio, [])

        self.assertEqual(llama.generate_calls, 1)
        self.assertEqual(json.loads(result.output)["text"], "반가워요.")

    async def test_round_limit_forces_a_tool_free_final_request(self) -> None:
        llama = _FakeLlama(
            [
                ChatCompletion(
                    content=None,
                    tool_calls=(
                        ToolCall("one", "switchbot__send_command", "{}"),
                    ),
                ),
                ChatCompletion(
                    content='{"transcript":"불 켜줘","response":"현재 결과를 요약했습니다."}'
                ),
            ]
        )
        mcp = _FakeMcp()
        with tempfile.TemporaryDirectory() as temp_dir:
            audio = Path(temp_dir) / "input.wav"
            audio.write_bytes(b"wav")
            settings = Settings(
                runtime_dir=Path(temp_dir),
                mcp_max_tool_rounds=1,
            )
            result = await OmniAgent(llama, mcp, settings).generate(  # type: ignore[arg-type]
                audio,
                [],
            )

        self.assertIsNone(llama.chat_requests[-1][1])
        self.assertIn("요약", json.loads(result.output)["text"])


if __name__ == "__main__":
    unittest.main()
