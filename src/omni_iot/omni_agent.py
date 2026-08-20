from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .mcp_client import McpManager
from .observability import AiObservability, NoopObservability
from .omni_llama import (
    DEFAULT_PROMPT,
    DEFAULT_SYSTEM,
    ChatCompletion,
    LlamaServer,
    ParsedModelTurn,
    ToolCall,
    _build_chat_messages,
    _parse_model_turn,
)


@dataclass(frozen=True)
class AgentGeneration:
    output: str
    tool_trace: tuple[dict[str, object], ...] = ()
    parsed: ParsedModelTurn | None = None
    tool_attempted: bool = False
    tool_succeeded: bool = True
    tool_loop_limit_hit: bool = False


class OmniAgent:
    """Run Qwen3-Omni tool calls through the MCP manager."""

    def __init__(
        self,
        llama: LlamaServer,
        mcp: McpManager,
        settings: Settings,
        observability: AiObservability | None = None,
    ) -> None:
        self.llama = llama
        self.mcp = mcp
        self.settings = settings
        self.observability = observability or NoopObservability()

    async def generate(
        self,
        audio_path: Path,
        history: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> AgentGeneration:
        tools = await self.mcp.prepare_turn()
        audio_data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        messages = _build_chat_messages(
            audio_data=audio_data,
            history=history,
            system_prompt=os.getenv("OMNI_SYSTEM_PROMPT", DEFAULT_SYSTEM),
            prompt=os.getenv("OMNI_PROMPT", DEFAULT_PROMPT),
        )
        trace: list[dict[str, object]] = []
        total_calls = 0
        tool_attempted = False
        tool_succeeded = True

        if not tools:
            completion = await self._chat(messages, max_tokens, None, 1)
            parsed = self._parse_completion(completion)
            return AgentGeneration(
                output=parsed.output_json(),
                parsed=parsed,
            )

        for round_index in range(1, self.settings.mcp_max_tool_rounds + 1):
            completion = await self._chat(messages, max_tokens, tools, round_index)
            if not completion.tool_calls:
                parsed = self._parse_completion(completion)
                return AgentGeneration(
                    output=parsed.output_json(),
                    tool_trace=tuple(trace),
                    parsed=parsed,
                    tool_attempted=tool_attempted,
                    tool_succeeded=tool_succeeded,
                )

            tool_attempted = True
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
                    result_trace = _invalid_tool_trace(call.name)
                    trace.append(result_trace)
                    with self.observability.start_observation(
                        name="mcp-tool",
                        as_type="tool",
                        input=_tool_trace_input(call),
                        metadata={"tool_call_id": call.id},
                    ) as observation:
                        observation.update(
                            output={"content": result_content, "is_error": True},
                            level="WARNING",
                            status_message="Tool call limit reached for this turn.",
                        )
                    tool_succeeded = False
                else:
                    result_content, result_trace, call_succeeded = await self._execute(
                        call
                    )
                    trace.append(result_trace)
                    total_calls += 1
                    tool_succeeded = tool_succeeded and call_succeeded
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

        return await self._forced_final(
            messages,
            max_tokens,
            trace,
            tool_attempted=tool_attempted,
            tool_succeeded=tool_succeeded,
        )

    async def _execute(
        self,
        call: ToolCall,
    ) -> tuple[str, dict[str, object], bool]:
        with self.observability.start_observation(
            name="mcp-tool",
            as_type="tool",
            input=_tool_trace_input(call),
            metadata={"tool_call_id": call.id},
        ) as observation:
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError:
                content = json.dumps(
                    {"isError": True, "error": "Tool arguments are not valid JSON."},
                    ensure_ascii=False,
                )
                trace = _invalid_tool_trace(call.name)
                observation.update(
                    output={"content": content, "is_error": True},
                    level="WARNING",
                    status_message="Tool arguments were not valid JSON.",
                )
                return content, trace, False
            if not isinstance(arguments, dict):
                content = json.dumps(
                    {"isError": True, "error": "Tool arguments must be a JSON object."},
                    ensure_ascii=False,
                )
                trace = _invalid_tool_trace(call.name)
                observation.update(
                    output={"content": content, "is_error": True},
                    level="WARNING",
                    status_message="Tool arguments were not an object.",
                )
                return content, trace, False
            try:
                result = await self.mcp.call_tool(call.name, arguments)
            except Exception as exc:
                observation.fail(exc)
                raise
            observation.update(
                output={
                    "server": result.server_name,
                    "tool": result.tool_name,
                    "content": result.content,
                    "is_error": result.is_error,
                    "truncated": result.truncated,
                    "seconds": round(result.seconds, 3),
                },
                level="WARNING" if result.is_error else "DEFAULT",
            )
            return result.content, result.trace(), not result.is_error

    async def _chat(
        self,
        messages: list[dict[str, object]],
        max_tokens: int | None,
        tools: list[dict[str, object]] | None,
        round_index: int,
    ) -> ChatCompletion:
        params = {
            "max_tokens": (
                max_tokens if max_tokens is not None else self.settings.llama_n_predict
            ),
            "temperature": self.settings.llama_temperature,
            "cache_prompt": self.settings.llama_cache_prompt,
        }
        with self.observability.start_observation(
            name=f"omni-generation-{round_index}",
            as_type="generation",
            input=self.observability.trace_value(
                {"messages": messages, "tools": tools}
            ),
            metadata={"round": round_index, "tools_enabled": bool(tools)},
            model=self.settings.omni_model.name,
            model_parameters=params,
        ) as observation:
            try:
                completion = await asyncio.to_thread(
                    self.llama.chat,
                    messages,
                    max_tokens,
                    tools,
                )
            except Exception as exc:
                observation.fail(exc)
                raise
            observation.update(
                output={
                    "content": completion.content,
                    "tool_calls": [
                        call.as_message_item() for call in completion.tool_calls
                    ],
                },
                usage_details=_usage_details(completion.usage),
            )
            return completion

    def _parse_completion(self, completion: ChatCompletion) -> ParsedModelTurn:
        if not completion.content:
            raise RuntimeError(
                "Qwen3-Omni returned no final answer after tool execution."
            )
        with self.observability.start_observation(
            name="parse-model-turn",
            input={"raw_content": completion.content},
        ) as observation:
            parsed = _parse_model_turn(completion.content)
            observation.update(
                output=parsed.trace(),
                level="DEFAULT" if parsed.structured_output_valid else "WARNING",
                status_message=parsed.fallback_reason,
            )
            return parsed

    async def _forced_final(
        self,
        messages: list[dict[str, object]],
        max_tokens: int | None,
        trace: list[dict[str, object]],
        *,
        tool_attempted: bool,
        tool_succeeded: bool,
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
            completion = await self._chat(
                messages,
                max_tokens,
                None,
                self.settings.mcp_max_tool_rounds + 1,
            )
            if completion.tool_calls:
                raise RuntimeError(
                    "Model emitted a tool call during forced finalization."
                )
            parsed = self._parse_completion(completion)
        except Exception:  # noqa: BLE001 - deterministic turn fallback after exhausted loop
            parsed = ParsedModelTurn(
                transcript=None,
                response="도구 호출 횟수 제한에 도달해 요청을 완료하지 못했습니다.",
                fallback_reason="Forced finalization failed after the tool loop limit.",
            )
        return AgentGeneration(
            output=parsed.output_json(),
            tool_trace=tuple(trace),
            parsed=parsed,
            tool_attempted=tool_attempted,
            tool_succeeded=tool_succeeded,
            tool_loop_limit_hit=True,
        )


def _completion_output(completion: ChatCompletion) -> str:
    if not completion.content:
        raise RuntimeError("Qwen3-Omni returned no final answer after tool execution.")
    return _parse_model_turn(completion.content).output_json()


def _invalid_tool_trace(name: str) -> dict[str, object]:
    return {
        "server": "unknown",
        "tool": name,
        "seconds": 0.0,
        "ok": False,
        "truncated": False,
    }


def _usage_details(usage: dict[str, int] | None) -> dict[str, int] | None:
    if not usage:
        return None
    aliases = {
        "prompt_tokens": "input",
        "completion_tokens": "output",
        "total_tokens": "total",
    }
    return {aliases.get(key, key): value for key, value in usage.items()}


def _tool_trace_input(call: ToolCall) -> dict[str, object]:
    try:
        arguments: object = json.loads(call.arguments)
    except json.JSONDecodeError:
        arguments = {
            "invalid_json": True,
            "characters": len(call.arguments),
        }
    return {"model_tool": call.name, "arguments": arguments}
