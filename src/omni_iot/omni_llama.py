from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

from .conversation import normalize_user_transcript

if TYPE_CHECKING:
    from .config import Settings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLAMA_CLI = (
    PROJECT_ROOT
    / "vendor"
    / "llama.cpp"
    / "build-cuda131-sm120-gcc13"
    / "bin"
    / "llama-cli"
)
DEFAULT_MODEL = PROJECT_ROOT / "models" / "Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf"
DEFAULT_MMPROJ = PROJECT_ROOT / "models" / "mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf"
DEFAULT_SYSTEM = (
    "You are a private local Korean voice assistant for a Jarvis-style IoT project. "
    "Answer naturally and concisely in Korean."
)
DEFAULT_PROMPT = "사용자의 음성 입력을 듣고 한국어로 자연스럽게 대답해줘."


def _structured_output_instruction(response_first: bool = False) -> str:
    key_instruction = (
        "키는 response와 transcript 두 개이며 response를 반드시 먼저 출력한다. "
        if response_first
        else "키는 transcript와 response 두 개다. "
    )
    return (
        "사용 가능한 도구가 있고 답변에 필요하면 먼저 도구를 호출하라. "
        "도구 호출 자체를 최종 JSON 안에 넣지 말고, 도구 결과를 받은 뒤 최종 답변을 작성하라. "
        "반드시 다른 설명이나 Markdown 없이 JSON 객체 하나만 출력하라. "
        f"{key_instruction}"
        "transcript에는 오디오에서 실제로 들은 사용자 발화를 그대로 적고, "
        "지시문이나 형식 설명을 복사하지 마라. 발화를 판별할 수 없으면 null을 사용하라. "
        "response에는 사용자에게 말할 자연스러운 한국어 답변을 적어라."
    )


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def as_message_item(self) -> dict[str, object]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass(frozen=True)
class ChatCompletion:
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def as_assistant_message(self) -> dict[str, object]:
        message: dict[str, object] = {
            "role": "assistant",
            "content": self.content,
        }
        if self.tool_calls:
            message["tool_calls"] = [call.as_message_item() for call in self.tool_calls]
        return message


class LlamaServer:
    """Own a llama-server process and send multimodal chat requests to it."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = (
            f"http://{settings.llama_server_host}:{settings.llama_server_port}"
        )
        self.process: subprocess.Popen[str] | None = None
        self._log_file = None
        self.owns_process = False

    @property
    def ready(self) -> bool:
        return self._is_healthy()

    def start(self) -> None:
        if self._is_healthy():
            return

        required_paths = (
            self.settings.llama_server,
            self.settings.omni_model,
            self.settings.omni_mmproj,
        )
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "llama-server assets not found: " + ", ".join(missing)
            )

        log_path = self.settings.runtime_dir / "llama-server.log"
        self._log_file = log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            self._command(),
            cwd=self.settings.project_root,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self.owns_process = True

        deadline = time.monotonic() + self.settings.llama_server_startup_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                error = self._log_tail(log_path)
                self.stop()
                raise RuntimeError(
                    "llama-server exited during startup.\n" + error
                )
            if self._is_healthy():
                return
            time.sleep(0.25)

        self.stop()
        raise TimeoutError(
            "llama-server did not become ready within "
            f"{self.settings.llama_server_startup_seconds} seconds. "
            f"See {log_path}."
        )

    def stop(self) -> None:
        if self.owns_process and self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        self.owns_process = False
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def generate(
        self,
        audio_path: Path,
        history: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> str:
        audio_data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        completion = self.chat(
            messages=_build_chat_messages(
                audio_data=audio_data,
                history=history,
                system_prompt=os.getenv("OMNI_SYSTEM_PROMPT", DEFAULT_SYSTEM),
                prompt=os.getenv("OMNI_PROMPT", DEFAULT_PROMPT),
            ),
            max_tokens=max_tokens,
        )
        if completion.tool_calls:
            raise RuntimeError("llama-server requested a tool but no tool runner is active.")
        if not completion.content:
            raise RuntimeError("llama-server returned an empty assistant response.")

        transcript, answer = _parse_model_turn(completion.content)
        return json.dumps(
            {"user_text": transcript, "text": answer},
            ensure_ascii=False,
        )

    def chat(
        self,
        messages: list[dict[str, object]],
        max_tokens: int | None = None,
        tools: list[dict[str, object]] | None = None,
    ) -> ChatCompletion:
        payload: dict[str, object] = {
            "messages": messages,
            "max_tokens": (
                max_tokens if max_tokens is not None else self.settings.llama_n_predict
            ),
            "temperature": self.settings.llama_temperature,
            "seed": self.settings.llama_seed,
            "cache_prompt": self.settings.llama_cache_prompt,
        }
        if tools:
            payload.update(
                {
                    "tools": tools,
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                }
            )
        response = self._request_json(
            "/v1/chat/completions",
            payload,
            timeout=self.settings.omni_timeout_seconds,
        )
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "llama-server returned an unexpected response: "
                f"{json.dumps(response, ensure_ascii=False)}"
            ) from exc

        if not isinstance(message, dict):
            raise RuntimeError("llama-server assistant message must be an object.")
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        if content is not None and not isinstance(content, str):
            raise RuntimeError("llama-server assistant content must be text or null.")
        normalized_content = content.strip() if isinstance(content, str) else None
        calls = _parse_tool_calls(message.get("tool_calls"))
        if not normalized_content and not calls:
            raise RuntimeError("llama-server returned an empty assistant response.")
        usage = response.get("usage")
        prompt_tokens = 0
        completion_tokens = 0
        if isinstance(usage, dict):
            if isinstance(usage.get("prompt_tokens"), int):
                prompt_tokens = usage["prompt_tokens"]
            if isinstance(usage.get("completion_tokens"), int):
                completion_tokens = usage["completion_tokens"]
        return ChatCompletion(
            content=normalized_content,
            tool_calls=tuple(calls),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def chat_stream(
        self,
        messages: list[dict[str, object]],
        max_tokens: int | None = None,
        tools: list[dict[str, object]] | None = None,
        on_content_delta: Callable[[str], None] | None = None,
    ) -> ChatCompletion:
        """Consume llama.cpp's SSE stream while exposing text deltas immediately."""
        payload: dict[str, object] = {
            "messages": messages,
            "max_tokens": (
                max_tokens if max_tokens is not None else self.settings.llama_n_predict
            ),
            "temperature": self.settings.llama_temperature,
            "seed": self.settings.llama_seed,
            "cache_prompt": self.settings.llama_cache_prompt,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload.update(
                {
                    "tools": tools,
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                }
            )

        content_parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        prompt_tokens = 0
        completion_tokens = 0
        for event in self._request_sse(
            "/v1/chat/completions",
            payload,
            timeout=self.settings.omni_timeout_seconds,
        ):
            usage = event.get("usage")
            if isinstance(usage, dict):
                if isinstance(usage.get("prompt_tokens"), int):
                    prompt_tokens = usage["prompt_tokens"]
                if isinstance(usage.get("completion_tokens"), int):
                    completion_tokens = usage["completion_tokens"]

            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str) and content:
                content_parts.append(content)
                if on_content_delta is not None:
                    on_content_delta(content)
            _merge_stream_tool_calls(calls, delta.get("tool_calls"))

        content = "".join(content_parts).strip()
        tool_calls = tuple(
            ToolCall(
                id=call.get("id") or f"call_{index}",
                name=call.get("name", ""),
                arguments=call.get("arguments") or "{}",
            )
            for index, call in sorted(calls.items())
        )
        if any(not call.name for call in tool_calls):
            raise RuntimeError("llama-server streamed a tool call without a name.")
        if not content and not tool_calls:
            raise RuntimeError("llama-server returned an empty assistant stream.")
        return ChatCompletion(
            content=content or None,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _command(self) -> list[str]:
        command = [
            str(self.settings.llama_server),
            "-m",
            str(self.settings.omni_model),
            "--mmproj",
            str(self.settings.omni_mmproj),
            "--host",
            self.settings.llama_server_host,
            "--port",
            str(self.settings.llama_server_port),
            "--gpu-layers",
            self.settings.llama_gpu_layers,
            "--ctx-size",
            str(self.settings.llama_ctx_size),
            "--flash-attn",
            self.settings.llama_flash_attn,
            "--parallel",
            str(self.settings.llama_parallel),
            "--threads",
            str(self.settings.llama_threads),
            "--seed",
            str(self.settings.llama_seed),
            "--jinja",
        ]
        command.append(
            "--cache-prompt"
            if self.settings.llama_cache_prompt
            else "--no-cache-prompt"
        )
        if self.settings.llama_device:
            command.extend(["--device", self.settings.llama_device])
        if self.settings.llama_slot_save_path:
            self.settings.llama_slot_save_path.mkdir(parents=True, exist_ok=True)
            command.extend(
                ["--slot-save-path", str(self.settings.llama_slot_save_path)]
            )
        if not self.settings.llama_op_offload:
            command.append("--no-op-offload")
        if not self.settings.llama_mmproj_offload:
            command.append("--no-mmproj-offload")
        if not self.settings.llama_warmup:
            command.append("--no-warmup")
        return command

    def erase_slot(self, slot_id: int = 0) -> None:
        self._request_json(
            f"/slots/{slot_id}?action=erase",
            {},
            timeout=self.settings.omni_timeout_seconds,
        )

    def _is_healthy(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/health",
                timeout=1,
            ) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    def _request_json(
        self,
        path: str,
        payload: dict[str, object],
        timeout: int,
    ) -> dict[str, object]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"llama-server request failed ({exc.code}): {detail}"
            ) from exc
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"llama-server request failed: {exc}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("llama-server response must be a JSON object.")
        return parsed

    def _request_sse(
        self,
        path: str,
        payload: dict[str, object],
        timeout: int,
    ) -> Iterator[dict[str, object]]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    event = json.loads(data)
                    if isinstance(event, dict):
                        yield event
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"llama-server stream failed ({exc.code}): {detail}"
            ) from exc
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"llama-server stream failed: {exc}") from exc

    @staticmethod
    def _log_tail(path: Path, lines: int = 30) -> str:
        if not path.exists():
            return "No llama-server log was created."
        content = path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(content.splitlines()[-lines:])


def _parse_tool_calls(value: object) -> list[ToolCall]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RuntimeError("llama-server tool_calls must be an array.")
    calls: list[ToolCall] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise RuntimeError("llama-server tool call must be an object.")
        function = raw.get("function")
        source = function if isinstance(function, dict) else raw
        name = source.get("name")
        arguments = source.get("arguments", {})
        if not isinstance(name, str) or not name:
            raise RuntimeError("llama-server tool call has no function name.")
        if isinstance(arguments, dict):
            arguments_text = json.dumps(arguments, ensure_ascii=False)
        elif isinstance(arguments, str):
            arguments_text = arguments
        else:
            raise RuntimeError("llama-server tool call arguments must be JSON.")
        call_id = raw.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{index}"
        calls.append(ToolCall(id=call_id, name=name, arguments=arguments_text))
    return calls


def _merge_stream_tool_calls(
    accumulated: dict[int, dict[str, str]],
    value: object,
) -> None:
    if not isinstance(value, list):
        return
    for fallback_index, raw in enumerate(value):
        if not isinstance(raw, dict):
            continue
        index = raw.get("index", fallback_index)
        if not isinstance(index, int):
            index = fallback_index
        target = accumulated.setdefault(
            index,
            {"id": "", "name": "", "arguments": ""},
        )
        call_id = raw.get("id")
        if isinstance(call_id, str):
            target["id"] += call_id
        function = raw.get("function")
        source = function if isinstance(function, dict) else raw
        name = source.get("name")
        if isinstance(name, str):
            target["name"] += name
        arguments = source.get("arguments")
        if isinstance(arguments, str):
            target["arguments"] += arguments


def _build_chat_messages(
    audio_data: str,
    history: list[dict[str, str]],
    system_prompt: str = DEFAULT_SYSTEM,
    prompt: str = DEFAULT_PROMPT,
    response_first: bool = False,
) -> list[dict[str, object]]:
    system_instruction = (
        f"{system_prompt}\n\n"
        f"{prompt}\n\n"
        f"{_structured_output_instruction(response_first=response_first)}"
    )
    messages: list[dict[str, object]] = [
        {"role": "system", "content": system_instruction},
    ]
    messages.extend(
        {"role": item["role"], "content": item["content"]}
        for item in history
        if item.get("role") in {"user", "assistant"} and item.get("content")
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_data, "format": "wav"},
                },
            ],
        }
    )
    return messages


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")


def run_llama_omni(
    audio_path: Path,
    llama_cli: Path = DEFAULT_LLAMA_CLI,
    model_path: Path = DEFAULT_MODEL,
    mmproj_path: Path = DEFAULT_MMPROJ,
    gpu_layers: str = "32",
    device: str | None = None,
    op_offload: bool = True,
    mmproj_offload: bool = True,
    prompt: str = DEFAULT_PROMPT,
    history: list[dict[str, str]] | None = None,
    system_prompt: str = DEFAULT_SYSTEM,
    ctx_size: int = 4096,
    n_predict: int = 192,
    flash_attn: str = "off",
    warmup: bool = False,
    timeout: int = 180,
) -> str:
    with tempfile.NamedTemporaryFile("r", encoding="utf-8", suffix=".txt", delete=False) as output:
        output_path = Path(output.name)

    turn_prompt = _build_turn_prompt(prompt, history or [])
    command = [
        str(llama_cli),
        "-m",
        str(model_path),
        "--mmproj",
        str(mmproj_path),
        "--audio",
        str(audio_path),
        "--gpu-layers",
        str(gpu_layers),
        "--ctx-size",
        str(ctx_size),
        "--predict",
        str(n_predict),
        "--flash-attn",
        flash_attn,
        "--system-prompt",
        system_prompt,
        "--prompt",
        turn_prompt,
        "--single-turn",
        "--no-display-prompt",
        "--no-show-timings",
        "--output",
        str(output_path),
    ]
    if device:
        command.extend(["--device", device])
    if not op_offload:
        command.append("--no-op-offload")
    if not mmproj_offload:
        command.append("--no-mmproj-offload")
    if not warmup:
        command.append("--no-warmup")

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "llama.cpp OMNI command failed\n"
            f"exit={completed.returncode}\n"
            f"stderr={completed.stderr.strip()}\n"
            f"stdout={completed.stdout.strip()}"
        )

    text = output_path.read_text(encoding="utf-8").strip()
    if not text:
        text = completed.stdout.strip()
    text = _extract_assistant_text(text)
    if not text:
        raise RuntimeError("llama.cpp OMNI command completed but produced no text output.")
    return text


def _extract_assistant_text(text: str) -> str:
    marker = "Assistant:"
    if marker in text:
        return text.rsplit(marker, 1)[-1].strip()
    return text.strip()


def _build_turn_prompt(prompt: str, history: list[dict[str, str]]) -> str:
    history_section = ""
    if history:
        history_section = (
            "다음은 같은 대화 세션의 이전 대화 기록이다. 문맥을 이어서 답하라.\n"
            f"<conversation_history>\n{json.dumps(history, ensure_ascii=False)}\n"
            "</conversation_history>\n\n"
        )
    return (
        f"{history_section}"
        "첨부된 오디오는 사용자의 다음 발화다. 발화를 정확히 이해하고 이전 문맥을 이어라.\n"
        f"{prompt}\n\n"
        f"{_structured_output_instruction()}"
    )


def _parse_model_turn(text: str) -> tuple[str | None, str]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            response = payload.get("response") or payload.get("text")
            transcript = payload.get("transcript") or payload.get("user_text")
            if isinstance(response, str) and response.strip():
                normalized_transcript = normalize_user_transcript(transcript)
                return normalized_transcript, response.strip()

    return None, stripped


def _load_history(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Conversation history must be a JSON array.")

    history: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            history.append({"role": role, "content": content})
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Qwen3-Omni through llama.cpp.")
    parser.add_argument("--audio", "--input", dest="audio", type=Path, required=True)
    parser.add_argument("--llama-cli", type=Path, default=Path(os.getenv("LLAMA_CLI", DEFAULT_LLAMA_CLI)))
    parser.add_argument("--model", type=Path, default=Path(os.getenv("OMNI_MODEL", DEFAULT_MODEL)))
    parser.add_argument("--mmproj", type=Path, default=Path(os.getenv("OMNI_MMPROJ", DEFAULT_MMPROJ)))
    parser.add_argument("--gpu-layers", default=os.getenv("LLAMA_N_GPU_LAYERS", "32"))
    parser.add_argument("--device", default=os.getenv("LLAMA_DEVICE") or None)
    parser.add_argument("--no-op-offload", action="store_true", default=os.getenv("LLAMA_OP_OFFLOAD", "true").lower() in {"0", "false", "no", "off"})
    parser.add_argument("--no-mmproj-offload", action="store_true", default=os.getenv("LLAMA_MMPROJ_OFFLOAD", "true").lower() in {"0", "false", "no", "off"})
    parser.add_argument("--prompt", default=os.getenv("OMNI_PROMPT", DEFAULT_PROMPT))
    parser.add_argument("--history-file", type=Path)
    parser.add_argument("--json-output", action="store_true")
    parser.add_argument("--system-prompt", default=os.getenv("OMNI_SYSTEM_PROMPT", DEFAULT_SYSTEM))
    parser.add_argument("--ctx-size", type=int, default=int(os.getenv("LLAMA_CTX_SIZE", "4096")))
    parser.add_argument("--n-predict", type=int, default=int(os.getenv("LLAMA_N_PREDICT", "192")))
    parser.add_argument("--flash-attn", default=os.getenv("LLAMA_FLASH_ATTN", "off"))
    parser.add_argument("--warmup", action="store_true", default=os.getenv("LLAMA_WARMUP", "false").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--timeout", type=int, default=int(os.getenv("OMNI_TIMEOUT_SECONDS", "180")))
    args = parser.parse_args()

    generated = run_llama_omni(
        audio_path=args.audio,
        llama_cli=args.llama_cli,
        model_path=args.model,
        mmproj_path=args.mmproj,
        gpu_layers=args.gpu_layers,
        device=args.device,
        op_offload=not args.no_op_offload,
        mmproj_offload=not args.no_mmproj_offload,
        prompt=args.prompt,
        history=_load_history(args.history_file),
        system_prompt=args.system_prompt,
        ctx_size=args.ctx_size,
        n_predict=args.n_predict,
        flash_attn=args.flash_attn,
        warmup=args.warmup,
        timeout=args.timeout,
    )
    transcript, response = _parse_model_turn(generated)
    if args.json_output:
        print(
            json.dumps(
                {"user_text": transcript, "text": response},
                ensure_ascii=False,
            )
        )
    else:
        print(response)


if __name__ == "__main__":
    main()
