from __future__ import annotations

import json
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
from .runtime import prune_runtime_turns


@dataclass(frozen=True)
class TurnResult:
    turn_id: str
    text: str
    user_text: str | None
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
    prune_runtime_turns(
        settings.runtime_dir,
        keep=settings.runtime_turn_limit,
        protected={turn_dir},
    )

    input_path = turn_dir / "input.wav"
    input_path.write_bytes(input_audio)

    started_at = time.perf_counter()
    history = (
        session.prompt_history(settings.conversation_history_messages)
        if session
        else []
    )

    omni_started_at = time.perf_counter()
    omni_result = run_omni(input_path, settings, history=history)
    omni_elapsed = time.perf_counter() - omni_started_at

    if session:
        session.add_user_audio_turn(omni_result.user_text)
        session.add_assistant_message(omni_result.text)

    tts_started_at = time.perf_counter()
    audio_path = run_tts(omni_result.text, turn_dir, settings)
    tts_elapsed = time.perf_counter() - tts_started_at

    return TurnResult(
        turn_id=turn_id,
        text=omni_result.text,
        user_text=omni_result.user_text,
        audio_path=audio_path,
        used_mock_omni=omni_result.used_mock,
        used_tts=audio_path is not None,
        timings={
            "omni_seconds": round(omni_elapsed, 3),
            "tts_seconds": round(tts_elapsed, 3),
            "total_seconds": round(time.perf_counter() - started_at, 3),
        },
    )


def run_demo_pipeline(input_audio: bytes, settings: Settings) -> TurnResult:
    return run_turn_pipeline(input_audio, settings)


@dataclass(frozen=True)
class OmniResult:
    text: str
    user_text: str | None
    used_mock: bool


def run_omni(
    input_path: Path,
    settings: Settings,
    history: list[dict[str, str]] | None = None,
) -> OmniResult:
    if not settings.omni_command:
        return OmniResult(
            text="옴니 명령은 아직 연결되지 않았지만, 실시간 음성 하네스가 사용자의 음성 턴을 정상적으로 받았습니다.",
            user_text=None,
            used_mock=True,
        )

    history_payload = history or []
    history_path = input_path.parent / "history.json"
    history_path.write_text(
        json.dumps(history_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    command = _format_command(
        settings.omni_command,
        audio=input_path,
        input=input_path,
        history=json.dumps(history_payload, ensure_ascii=False),
        history_file=history_path,
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

    output = completed.stdout.strip()
    if not output:
        output = completed.stderr.strip()
    if not output:
        raise RuntimeError("OMNI command completed but produced no text output.")
    return _parse_omni_output(output)


def _parse_omni_output(output: str) -> OmniResult:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return OmniResult(text=output, user_text=None, used_mock=False)

    if not isinstance(payload, dict):
        return OmniResult(text=output, user_text=None, used_mock=False)

    text = payload.get("text") or payload.get("response")
    if not isinstance(text, str) or not text.strip():
        return OmniResult(text=output, user_text=None, used_mock=False)

    user_text = payload.get("user_text") or payload.get("transcript")
    if not isinstance(user_text, str) or not user_text.strip():
        user_text = None
    return OmniResult(
        text=text.strip(),
        user_text=user_text.strip() if user_text else None,
        used_mock=False,
    )


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
