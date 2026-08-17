from __future__ import annotations

import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .conversation import ConversationSession
from .config import Settings


@dataclass(frozen=True)
class TurnResult:
    turn_id: str
    text: str
    audio_path: Path | None
    used_mock_omni: bool
    used_tts: bool
    timings: dict[str, float] = field(default_factory=dict)


def run_turn_pipeline(
    input_audio: bytes,
    settings: Settings,
    session: ConversationSession | None = None,
) -> TurnResult:
    turn_id = uuid.uuid4().hex
    turn_dir = settings.runtime_dir / turn_id
    turn_dir.mkdir(parents=True, exist_ok=True)

    input_path = turn_dir / "input.wav"
    input_path.write_bytes(input_audio)

    started_at = time.perf_counter()
    if session:
        session.add_user_audio_turn()

    omni_started_at = time.perf_counter()
    text, used_mock = run_omni(input_path, settings)
    omni_elapsed = time.perf_counter() - omni_started_at

    if session:
        session.add_assistant_message(text)

    tts_started_at = time.perf_counter()
    audio_path = run_tts(text, turn_dir, settings)
    tts_elapsed = time.perf_counter() - tts_started_at

    return TurnResult(
        turn_id=turn_id,
        text=text,
        audio_path=audio_path,
        used_mock_omni=used_mock,
        used_tts=audio_path is not None,
        timings={
            "omni_seconds": round(omni_elapsed, 3),
            "tts_seconds": round(tts_elapsed, 3),
            "total_seconds": round(time.perf_counter() - started_at, 3),
        },
    )


def run_demo_pipeline(input_audio: bytes, settings: Settings) -> TurnResult:
    return run_turn_pipeline(input_audio, settings)


def run_omni(input_path: Path, settings: Settings) -> tuple[str, bool]:
    if not settings.omni_command:
        return (
            "옴니 명령은 아직 연결되지 않았지만, 실시간 음성 하네스가 사용자의 음성 턴을 정상적으로 받았습니다.",
            True,
        )

    command = _format_command(
        settings.omni_command,
        audio=input_path,
        input=input_path,
    )
    completed = subprocess.run(
        command,
        cwd=settings.project_root,
        capture_output=True,
        text=True,
        timeout=settings.omni_timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "OMNI command failed\n"
            f"exit={completed.returncode}\n"
            f"stderr={completed.stderr.strip()}\n"
            f"stdout={completed.stdout.strip()}"
        )

    response = completed.stdout.strip()
    if not response:
        response = completed.stderr.strip()
    if not response:
        raise RuntimeError("OMNI command completed but produced no text output.")
    return response, False


def run_tts(text: str, turn_dir: Path, settings: Settings) -> Path | None:
    if settings.tts_backend == "omnivoice":
        output_path = turn_dir / "reply.wav"
        command = [
            sys.executable,
            "-m",
            "omni_iot.tts_omnivoice",
            "--text",
            text,
            "--output",
            str(output_path),
            "--model-id",
            settings.omnivoice_model_id,
        ]
        if settings.omnivoice_language:
            command.extend(["--language", settings.omnivoice_language])
        if settings.omnivoice_instruct:
            command.extend(["--instruct", settings.omnivoice_instruct])
        if settings.omnivoice_speed is not None:
            command.extend(["--speed", str(settings.omnivoice_speed)])

        completed = subprocess.run(
            command,
            cwd=settings.project_root,
            capture_output=True,
            text=True,
            timeout=settings.tts_timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "OmniVoice TTS command failed\n"
                f"exit={completed.returncode}\n"
                f"stderr={completed.stderr.strip()}\n"
                f"stdout={completed.stdout.strip()}"
            )
        if not output_path.exists():
            raise RuntimeError("OmniVoice TTS completed but produced no WAV output.")
        return output_path

    if not settings.tts_command:
        return None

    output_path = turn_dir / "reply.wav"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".txt",
        dir=turn_dir,
        delete=False,
    ) as text_file:
        text_file.write(text)
        text_file_path = Path(text_file.name)

    command = _format_command(
        settings.tts_command,
        text=text,
        text_file=text_file_path,
        output=output_path,
    )
    completed = subprocess.run(
        command,
        cwd=settings.project_root,
        capture_output=True,
        text=True,
        timeout=settings.tts_timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "TTS command failed\n"
            f"exit={completed.returncode}\n"
            f"stderr={completed.stderr.strip()}\n"
            f"stdout={completed.stdout.strip()}"
        )
    if not output_path.exists():
        raise RuntimeError(f"TTS command did not create {output_path}")
    return output_path


def _format_command(template: str, **values: object) -> list[str]:
    quoted = {
        key: shlex.quote(str(value))
        for key, value in values.items()
    }
    return shlex.split(template.format(**quoted))
