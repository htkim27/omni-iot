from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .runtime import prune_runtime_turns


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(PROJECT_ROOT / ".env")


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    return int(value)


def _path_env(name: str, default: Path) -> Path:
    path = Path(os.getenv(name, default))
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    static_dir: Path = PROJECT_ROOT / "src" / "omni_iot" / "static"
    runtime_dir: Path = PROJECT_ROOT / ".runtime"
    runtime_turn_limit: int = _int_env("RUNTIME_TURN_LIMIT", 20)
    conversation_history_messages: int = _int_env(
        "CONVERSATION_HISTORY_MESSAGES",
        12,
    )
    omni_backend: str = os.getenv("OMNI_BACKEND", "command").lower()
    omni_command: str | None = os.getenv("OMNI_COMMAND") or None
    llama_server: Path = _path_env(
        "LLAMA_SERVER",
        PROJECT_ROOT
        / "vendor"
        / "llama.cpp"
        / "build-cuda131-sm120-gcc13"
        / "bin"
        / "llama-server",
    )
    omni_model: Path = _path_env(
        "OMNI_MODEL",
        PROJECT_ROOT / "models" / "Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf",
    )
    omni_mmproj: Path = _path_env(
        "OMNI_MMPROJ",
        PROJECT_ROOT / "models" / "mmproj-Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf",
    )
    llama_server_host: str = os.getenv("LLAMA_SERVER_HOST", "127.0.0.1")
    llama_server_port: int = _int_env("LLAMA_SERVER_PORT", 8081)
    llama_server_startup_seconds: int = _int_env("LLAMA_SERVER_STARTUP_SECONDS", 90)
    llama_gpu_layers: str = os.getenv("LLAMA_N_GPU_LAYERS", "20")
    llama_device: str | None = os.getenv("LLAMA_DEVICE") or None
    llama_op_offload: bool = os.getenv("LLAMA_OP_OFFLOAD", "true").lower() in {
        "1", "true", "yes", "on",
    }
    llama_mmproj_offload: bool = os.getenv(
        "LLAMA_MMPROJ_OFFLOAD", "true"
    ).lower() in {"1", "true", "yes", "on"}
    llama_ctx_size: int = _int_env("LLAMA_CTX_SIZE", 4096)
    llama_n_predict: int = _int_env("LLAMA_N_PREDICT", 192)
    llama_flash_attn: str = os.getenv("LLAMA_FLASH_ATTN", "off")
    llama_warmup: bool = os.getenv("LLAMA_WARMUP", "false").lower() in {
        "1", "true", "yes", "on",
    }
    tts_backend: str = os.getenv("TTS_BACKEND", "command").lower()
    tts_command: str | None = os.getenv("TTS_COMMAND") or None
    omni_timeout_seconds: int = _int_env("OMNI_TIMEOUT_SECONDS", 180)
    tts_timeout_seconds: int = _int_env("TTS_TIMEOUT_SECONDS", 180)
    omnivoice_model_id: str = os.getenv("OMNIVOICE_MODEL_ID", "k2-fsa/OmniVoice")
    omnivoice_language: str | None = os.getenv("OMNIVOICE_LANGUAGE", "ko") or None
    omnivoice_instruct: str | None = os.getenv("OMNIVOICE_INSTRUCT") or None
    omnivoice_speed: float | None = (
        float(os.environ["OMNIVOICE_SPEED"])
        if os.getenv("OMNIVOICE_SPEED")
        else None
    )
    omnivoice_warmup: bool = os.getenv("OMNIVOICE_WARMUP", "true").lower() in {
        "1", "true", "yes", "on",
    }


def get_settings() -> Settings:
    settings = Settings()
    if settings.runtime_turn_limit < 1:
        raise ValueError("RUNTIME_TURN_LIMIT must be at least 1.")
    if settings.conversation_history_messages < 0:
        raise ValueError("CONVERSATION_HISTORY_MESSAGES must not be negative.")
    if settings.omni_backend not in {"command", "server"}:
        raise ValueError("OMNI_BACKEND must be either 'command' or 'server'.")
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    prune_runtime_turns(settings.runtime_dir, keep=settings.runtime_turn_limit)
    return settings
