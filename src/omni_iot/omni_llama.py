from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLAMA_CLI = PROJECT_ROOT / "vendor" / "llama.cpp" / "build-cuda124-sm89" / "bin" / "llama-cli"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf"
DEFAULT_MMPROJ = PROJECT_ROOT / "models" / "mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf"
DEFAULT_SYSTEM = (
    "You are a private local Korean voice assistant for a Jarvis-style IoT project. "
    "Answer naturally and concisely in Korean."
)
DEFAULT_PROMPT = "사용자의 음성 입력을 듣고 한국어로 자연스럽게 대답해줘."


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
    system_prompt: str = DEFAULT_SYSTEM,
    ctx_size: int = 4096,
    n_predict: int = 192,
    flash_attn: str = "off",
    warmup: bool = False,
    timeout: int = 180,
) -> str:
    with tempfile.NamedTemporaryFile("r", encoding="utf-8", suffix=".txt", delete=False) as output:
        output_path = Path(output.name)

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
        prompt,
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
    parser.add_argument("--system-prompt", default=os.getenv("OMNI_SYSTEM_PROMPT", DEFAULT_SYSTEM))
    parser.add_argument("--ctx-size", type=int, default=int(os.getenv("LLAMA_CTX_SIZE", "4096")))
    parser.add_argument("--n-predict", type=int, default=int(os.getenv("LLAMA_N_PREDICT", "192")))
    parser.add_argument("--flash-attn", default=os.getenv("LLAMA_FLASH_ATTN", "off"))
    parser.add_argument("--warmup", action="store_true", default=os.getenv("LLAMA_WARMUP", "false").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--timeout", type=int, default=int(os.getenv("OMNI_TIMEOUT_SECONDS", "180")))
    args = parser.parse_args()

    print(
        run_llama_omni(
            audio_path=args.audio,
            llama_cli=args.llama_cli,
            model_path=args.model,
            mmproj_path=args.mmproj,
            gpu_layers=args.gpu_layers,
            device=args.device,
            op_offload=not args.no_op_offload,
            mmproj_offload=not args.no_mmproj_offload,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            ctx_size=args.ctx_size,
            n_predict=args.n_predict,
            flash_attn=args.flash_attn,
            warmup=args.warmup,
            timeout=args.timeout,
        )
    )


if __name__ == "__main__":
    main()
