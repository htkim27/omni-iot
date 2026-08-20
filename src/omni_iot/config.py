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


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    return float(value)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


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
    mcp_config: Path = _path_env("MCP_CONFIG", PROJECT_ROOT / ".mcp.json")
    mcp_tool_catalog_max_chars: int = _int_env(
        "MCP_TOOL_CATALOG_MAX_CHARS",
        12_000,
    )
    mcp_tool_result_max_chars: int = _int_env(
        "MCP_TOOL_RESULT_MAX_CHARS",
        12_000,
    )
    mcp_brave_search_max_results: int = _int_env(
        "MCP_BRAVE_SEARCH_MAX_RESULTS",
        5,
    )
    mcp_max_tool_rounds: int = _int_env("MCP_MAX_TOOL_ROUNDS", 4)
    mcp_max_tool_calls: int = _int_env("MCP_MAX_TOOL_CALLS", 8)
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
    llama_temperature: float = _float_env("LLAMA_TEMPERATURE", 0.2)
    llama_parallel: int = _int_env("LLAMA_PARALLEL", 1)
    llama_threads: int = _int_env("LLAMA_THREADS", 8)
    llama_cache_prompt: bool = _bool_env("LLAMA_CACHE_PROMPT", True)
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
    omnivoice_num_steps: int = _int_env("OMNIVOICE_NUM_STEPS", 32)
    vad_threshold: float = _float_env("VAD_THRESHOLD", 0.04)
    vad_silence_end_ms: int = _int_env("VAD_SILENCE_END_MS", 600)
    vad_pre_roll_ms: int = _int_env("VAD_PRE_ROLL_MS", 450)
    vad_max_turn_ms: int = _int_env("VAD_MAX_TURN_MS", 14_000)
    vad_barge_in_multiplier: float = _float_env("VAD_BARGE_IN_MULTIPLIER", 1.4)
    vad_continue_multiplier: float = _float_env("VAD_CONTINUE_MULTIPLIER", 0.72)
    wakeword_model: Path = _path_env(
        "WAKEWORD_MODEL",
        PROJECT_ROOT / "models" / "openwakeword" / "오둥아.onnx",
    )
    wakeword_label: str = os.getenv("WAKEWORD_LABEL", "오둥아")
    wakeword_threshold: float = _float_env("WAKEWORD_THRESHOLD", 0.5)
    wakeword_inference_framework: str = os.getenv(
        "WAKEWORD_INFERENCE_FRAMEWORK", "onnx"
    )
    follow_up_timeout_ms: int = _int_env("FOLLOW_UP_TIMEOUT_MS", 8_000)


def get_settings() -> Settings:
    settings = Settings()
    if settings.runtime_turn_limit < 1:
        raise ValueError("RUNTIME_TURN_LIMIT must be at least 1.")
    if settings.conversation_history_messages < 0:
        raise ValueError("CONVERSATION_HISTORY_MESSAGES must not be negative.")
    if settings.mcp_tool_catalog_max_chars < 1:
        raise ValueError("MCP_TOOL_CATALOG_MAX_CHARS must be positive.")
    if settings.mcp_tool_result_max_chars < 1:
        raise ValueError("MCP_TOOL_RESULT_MAX_CHARS must be positive.")
    if not 1 <= settings.mcp_brave_search_max_results <= 20:
        raise ValueError("MCP_BRAVE_SEARCH_MAX_RESULTS must be between 1 and 20.")
    if settings.mcp_max_tool_rounds < 1 or settings.mcp_max_tool_calls < 1:
        raise ValueError("MCP tool round and call limits must be positive.")
    if settings.omni_backend not in {"command", "server"}:
        raise ValueError("OMNI_BACKEND must be either 'command' or 'server'.")
    if not 16 <= settings.llama_n_predict <= 512:
        raise ValueError("LLAMA_N_PREDICT must be between 16 and 512.")
    if not 0 <= settings.llama_temperature <= 2:
        raise ValueError("LLAMA_TEMPERATURE must be between 0 and 2.")
    if settings.llama_parallel < 1 or settings.llama_threads < 1:
        raise ValueError("LLAMA_PARALLEL and LLAMA_THREADS must be at least 1.")
    if not 4 <= settings.omnivoice_num_steps <= 64:
        raise ValueError("OMNIVOICE_NUM_STEPS must be between 4 and 64.")
    if not 0.01 <= settings.vad_threshold <= 0.2:
        raise ValueError("VAD_THRESHOLD must be between 0.01 and 0.2.")
    if not 250 <= settings.vad_silence_end_ms <= 2_000:
        raise ValueError("VAD_SILENCE_END_MS must be between 250 and 2000.")
    if settings.vad_pre_roll_ms < 0 or settings.vad_max_turn_ms < 1_000:
        raise ValueError("VAD pre-roll/max-turn settings are invalid.")
    if settings.vad_barge_in_multiplier <= 0 or settings.vad_continue_multiplier <= 0:
        raise ValueError("VAD threshold multipliers must be positive.")
    if not settings.wakeword_label.strip():
        raise ValueError("WAKEWORD_LABEL must not be empty.")
    if not 0 < settings.wakeword_threshold <= 1:
        raise ValueError("WAKEWORD_THRESHOLD must be greater than 0 and at most 1.")
    if settings.wakeword_inference_framework not in {"onnx", "tflite"}:
        raise ValueError("WAKEWORD_INFERENCE_FRAMEWORK must be 'onnx' or 'tflite'.")
    if settings.follow_up_timeout_ms < 1_000:
        raise ValueError("FOLLOW_UP_TIMEOUT_MS must be at least 1000.")
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    prune_runtime_turns(settings.runtime_dir, keep=settings.runtime_turn_limit)
    return settings
