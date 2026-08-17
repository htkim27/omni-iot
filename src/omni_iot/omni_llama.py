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
from pathlib import Path
from typing import TYPE_CHECKING

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


def _structured_output_instruction() -> str:
    return (
        "반드시 다른 설명이나 Markdown 없이 JSON 객체 하나만 출력하라. "
        "키는 transcript와 response 두 개다. "
        "transcript에는 오디오에서 실제로 들은 사용자 발화를 그대로 적고, "
        "지시문이나 형식 설명을 복사하지 마라. 발화를 판별할 수 없으면 null을 사용하라. "
        "response에는 사용자에게 말할 자연스러운 한국어 답변을 적어라."
    )


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
        payload = {
            "messages": _build_chat_messages(
                audio_data=audio_data,
                history=history,
                system_prompt=os.getenv("OMNI_SYSTEM_PROMPT", DEFAULT_SYSTEM),
                prompt=os.getenv("OMNI_PROMPT", DEFAULT_PROMPT),
            ),
            "max_tokens": (
                max_tokens if max_tokens is not None else self.settings.llama_n_predict
            ),
            "temperature": self.settings.llama_temperature,
            "cache_prompt": self.settings.llama_cache_prompt,
        }
        response = self._request_json(
            "/v1/chat/completions",
            payload,
            timeout=self.settings.omni_timeout_seconds,
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "llama-server returned an unexpected response: "
                f"{json.dumps(response, ensure_ascii=False)}"
            ) from exc

        if isinstance(content, list):
            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("llama-server returned an empty assistant response.")

        transcript, answer = _parse_model_turn(content)
        return json.dumps(
            {"user_text": transcript, "text": answer},
            ensure_ascii=False,
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
        ]
        command.append(
            "--cache-prompt"
            if self.settings.llama_cache_prompt
            else "--no-cache-prompt"
        )
        if self.settings.llama_device:
            command.extend(["--device", self.settings.llama_device])
        if not self.settings.llama_op_offload:
            command.append("--no-op-offload")
        if not self.settings.llama_mmproj_offload:
            command.append("--no-mmproj-offload")
        if not self.settings.llama_warmup:
            command.append("--no-warmup")
        return command

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

    @staticmethod
    def _log_tail(path: Path, lines: int = 30) -> str:
        if not path.exists():
            return "No llama-server log was created."
        content = path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(content.splitlines()[-lines:])


def _build_chat_messages(
    audio_data: str,
    history: list[dict[str, str]],
    system_prompt: str = DEFAULT_SYSTEM,
    prompt: str = DEFAULT_PROMPT,
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [
        {"role": "system", "content": system_prompt},
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
                    "type": "text",
                    "text": (
                        f"{prompt}\n\n"
                        f"{_structured_output_instruction()}"
                    ),
                },
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
