from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
    timings: dict[str, float] | None = None


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
        started_at = time.perf_counter()
        prepare_started_at = time.perf_counter()
        tools = await self.mcp.prepare_turn()
        prepare_seconds = time.perf_counter() - prepare_started_at
        if not tools:
            inference_started_at = time.perf_counter()
            output = await asyncio.to_thread(
                self.llama.generate,
                audio_path,
                history,
                max_tokens,
            )
            inference_seconds = time.perf_counter() - inference_started_at
            return AgentGeneration(
                output=output,
                timings={
                    "mcp_prepare_seconds": round(prepare_seconds, 3),
                    "llm_inference_seconds": round(inference_seconds, 3),
                    "llm_rounds": 1.0,
                    "tool_execution_seconds": 0.0,
                    "agent_total_seconds": round(time.perf_counter() - started_at, 3),
                },
            )

        audio_data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        messages = _build_chat_messages(
            audio_data=audio_data,
            history=history,
            system_prompt=os.getenv("OMNI_SYSTEM_PROMPT", DEFAULT_SYSTEM),
            prompt=os.getenv("OMNI_PROMPT", DEFAULT_PROMPT),
        )
        trace: list[dict[str, object]] = []
        total_calls = 0
        inference_seconds = 0.0
        inference_rounds = 0
        prompt_tokens = 0
        completion_tokens = 0

        for _round in range(self.settings.mcp_max_tool_rounds):
            inference_started_at = time.perf_counter()
            completion = await asyncio.to_thread(
                self.llama.chat,
                messages,
                max_tokens,
                tools,
            )
            inference_seconds += time.perf_counter() - inference_started_at
            inference_rounds += 1
            prompt_tokens += completion.prompt_tokens
            completion_tokens += completion.completion_tokens
            if not completion.tool_calls:
                return AgentGeneration(
                    output=_completion_output(completion),
                    tool_trace=tuple(trace),
                    timings=_agent_timings(
                        started_at,
                        prepare_seconds,
                        inference_seconds,
                        inference_rounds,
                        trace,
                        prompt_tokens,
                        completion_tokens,
                    ),
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

        forced_started_at = time.perf_counter()
        generation = await self._forced_final(messages, max_tokens, trace)
        inference_seconds += time.perf_counter() - forced_started_at
        inference_rounds += 1
        return AgentGeneration(
            output=generation.output,
            tool_trace=generation.tool_trace,
            timings=_agent_timings(
                started_at,
                prepare_seconds,
                inference_seconds,
                inference_rounds,
                trace,
                prompt_tokens,
                completion_tokens,
            ),
        )

    async def generate_stream(
        self,
        audio_path: Path,
        history: list[dict[str, str]],
        on_response_delta: Callable[[str], None],
        max_tokens: int | None = None,
    ) -> AgentGeneration:
        """Stream the final response text while retaining normal tool execution."""
        started_at = time.perf_counter()
        prepare_started_at = time.perf_counter()
        tools = await self.mcp.prepare_turn()
        prepare_seconds = time.perf_counter() - prepare_started_at
        audio_data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        messages = _build_chat_messages(
            audio_data=audio_data,
            history=history,
            system_prompt=os.getenv("OMNI_SYSTEM_PROMPT", DEFAULT_SYSTEM),
            prompt=os.getenv("OMNI_PROMPT", DEFAULT_PROMPT),
            response_first=True,
        )
        trace: list[dict[str, object]] = []
        total_calls = 0
        inference_seconds = 0.0
        inference_rounds = 0
        prompt_tokens = 0
        completion_tokens = 0

        for _round in range(self.settings.mcp_max_tool_rounds):
            extractor = _ResponseDeltaExtractor(on_response_delta)
            inference_started_at = time.perf_counter()
            completion = await asyncio.to_thread(
                self.llama.chat_stream,
                messages,
                max_tokens,
                tools or None,
                extractor.feed,
            )
            inference_seconds += time.perf_counter() - inference_started_at
            inference_rounds += 1
            prompt_tokens += completion.prompt_tokens
            completion_tokens += completion.completion_tokens
            if not completion.tool_calls:
                output = _completion_output(completion)
                extractor.finish(json.loads(output)["text"])
                return AgentGeneration(
                    output=output,
                    tool_trace=tuple(trace),
                    timings=_agent_timings(
                        started_at,
                        prepare_seconds,
                        inference_seconds,
                        inference_rounds,
                        trace,
                        prompt_tokens,
                        completion_tokens,
                    ),
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

        # The forced-final path is rare and intentionally remains non-streaming.
        generation = await self._forced_final(messages, max_tokens, trace)
        answer = json.loads(generation.output)["text"]
        on_response_delta(answer)
        return AgentGeneration(
            output=generation.output,
            tool_trace=generation.tool_trace,
            timings=_agent_timings(
                started_at,
                prepare_seconds,
                inference_seconds,
                inference_rounds,
                trace,
                prompt_tokens,
                completion_tokens,
            ),
        )

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


class _ResponseDeltaExtractor:
    """Expose the partially generated JSON response string as plain text."""

    def __init__(self, callback: Callable[[str], None]) -> None:
        self.callback = callback
        self.raw = ""
        self.emitted = ""

    def feed(self, delta: str) -> None:
        self.raw += delta
        partial = _partial_json_string(self.raw, ("response", "text"))
        if partial is None or not partial.startswith(self.emitted):
            return
        new_text = partial[len(self.emitted) :]
        if new_text:
            self.emitted = partial
            self.callback(new_text)

    def finish(self, answer: str) -> None:
        if answer.startswith(self.emitted):
            remaining = answer[len(self.emitted) :]
            if remaining:
                self.emitted = answer
                self.callback(remaining)
        elif not self.emitted:
            self.emitted = answer
            self.callback(answer)


def _partial_json_string(text: str, keys: tuple[str, ...]) -> str | None:
    key_pattern = "|".join(re.escape(key) for key in keys)
    match = re.search(rf'"(?:{key_pattern})"\s*:\s*"', text)
    if match is None:
        return None
    source = text[match.end() :]
    decoded: list[str] = []
    index = 0
    escapes = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    while index < len(source):
        character = source[index]
        if character == '"':
            break
        if character != "\\":
            decoded.append(character)
            index += 1
            continue
        if index + 1 >= len(source):
            break
        escaped = source[index + 1]
        if escaped == "u":
            digits = source[index + 2 : index + 6]
            if len(digits) < 4 or not all(item in "0123456789abcdefABCDEF" for item in digits):
                break
            decoded.append(chr(int(digits, 16)))
            index += 6
            continue
        replacement = escapes.get(escaped)
        if replacement is None:
            break
        decoded.append(replacement)
        index += 2
    return "".join(decoded)


def _agent_timings(
    started_at: float,
    prepare_seconds: float,
    inference_seconds: float,
    inference_rounds: int,
    trace: list[dict[str, object]],
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, float]:
    tool_seconds = sum(
        float(call.get("seconds", 0.0))
        for call in trace
        if isinstance(call.get("seconds", 0.0), (int, float))
    )
    return {
        "mcp_prepare_seconds": round(prepare_seconds, 3),
        "llm_inference_seconds": round(inference_seconds, 3),
        "llm_rounds": float(inference_rounds),
        "tool_execution_seconds": round(tool_seconds, 3),
        "agent_total_seconds": round(time.perf_counter() - started_at, 3),
        "prompt_tokens": float(prompt_tokens),
        "completion_tokens": float(completion_tokens),
    }
