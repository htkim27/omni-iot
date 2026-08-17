from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    static_dir: Path = PROJECT_ROOT / "src" / "omni_iot" / "static"
    runtime_dir: Path = PROJECT_ROOT / ".runtime"
    omni_command: str | None = os.getenv("OMNI_COMMAND") or None
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


def get_settings() -> Settings:
    settings = Settings()
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    return settings
