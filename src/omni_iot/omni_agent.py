from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .mcp_client import McpManager
from .omni_llama import (
    DEFAULT_PROMPT,
    DEFAULT_SYSTEM,
    ChatCompletion,
    LlamaServer,
    ToolCall,
    _build_chat_messages,
    _parse_model_turn,
)


@dataclass(frozen=True)
class AgentGeneration:
    output: str
    tool_trace: tuple[dict[str, object], ...] = ()


class OmniAgent:
    """Run Qwen3-Omni tool calls through the MCP manager."""

    def __init__(
        self,
        llama: LlamaServer,
        mcp: McpManager,
        settings: Settings,
    ) -> None:
        self.llama = llama
        self.mcp = mcp
        self.settings = settings

    async def generate(
        self,
        audio_path: Path,
        history: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> AgentGeneration:
        tools = await self.mcp.prepare_turn()
        if not tools:
            output = await asyncio.to_thread(
                self.llama.generate,
                audio_path,
                history,
                max_tokens,
            )
            return AgentGeneration(output=output)

        audio_data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        messages = _build_chat_messages(
            audio_data=audio_data,
            history=history,
            system_prompt=os.getenv("OMNI_SYSTEM_PROMPT", DEFAULT_SYSTEM),
            prompt=os.getenv("OMNI_PROMPT", DEFAULT_PROMPT),
        )
        trace: list[dict[str, object]] = []
        total_calls = 0

        for _round in range(self.settings.mcp_max_tool_rounds):
            completion = await asyncio.to_thread(
                self.llama.chat,
                messages,
                max_tokens,
                tools,
            )
            if not completion.tool_calls:
                return AgentGeneration(
                    output=_completion_output(completion),
                    tool_trace=tuple(trace),
                )

            messages.append(completion.as_assistant_message())
            remaining = self.settings.mcp_max_tool_calls - total_calls
            for index, call in enumerate(completion.tool_calls):
                if index >= remaining:
                    result_content = json.dumps(
                        {
                            "isError": True,
                            "error": "Tool call limit reached for this turn.",
                        },
                        ensure_ascii=False,
                    )
                else:
                    result_content, result_trace = await self._execute(call)
                    trace.append(result_trace)
                    total_calls += 1
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": result_content,
                    }
                )
            if total_calls >= self.settings.mcp_max_tool_calls:
                break

        return await self._forced_final(messages, max_tokens, trace)

    async def _execute(self, call: ToolCall) -> tuple[str, dict[str, object]]:
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError:
            return (
                json.dumps(
                    {"isError": True, "error": "Tool arguments are not valid JSON."},
                    ensure_ascii=False,
                ),
                {"server": "unknown", "tool": call.name, "seconds": 0.0, "ok": False},
            )
        if not isinstance(arguments, dict):
            return (
                json.dumps(
                    {"isError": True, "error": "Tool arguments must be a JSON object."},
                    ensure_ascii=False,
                ),
                {"server": "unknown", "tool": call.name, "seconds": 0.0, "ok": False},
            )
        result = await self.mcp.call_tool(call.name, arguments)
        return result.content, result.trace()

    async def _forced_final(
        self,
        messages: list[dict[str, object]],
        max_tokens: int | None,
        trace: list[dict[str, object]],
    ) -> AgentGeneration:
        messages.append(
            {
                "role": "system",
                "content": (
                    "도구 호출 한도에 도달했다. 추가 도구를 호출하지 말고 지금까지의 "
                    "결과로 사용자에게 최종 JSON 답변을 작성하라."
                ),
            }
        )
        try:
            completion = await asyncio.to_thread(
                self.llama.chat,
                messages,
                max_tokens,
                None,
            )
            if completion.tool_calls:
                raise RuntimeError("Model emitted a tool call during forced finalization.")
            output = _completion_output(completion)
        except Exception:  # noqa: BLE001 - deterministic turn fallback after exhausted loop
            output = json.dumps(
                {
                    "user_text": None,
                    "text": "도구 호출 횟수 제한에 도달해 요청을 완료하지 못했습니다.",
                },
                ensure_ascii=False,
            )
        return AgentGeneration(output=output, tool_trace=tuple(trace))


def _completion_output(completion: ChatCompletion) -> str:
    if not completion.content:
        raise RuntimeError("Qwen3-Omni returned no final answer after tool execution.")
    transcript, answer = _parse_model_turn(completion.content)
    return json.dumps(
        {"user_text": transcript, "text": answer},
        ensure_ascii=False,
    )
