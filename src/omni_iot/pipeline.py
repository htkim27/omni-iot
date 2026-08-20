from __future__ import annotations

import asyncio
import json
import shlex
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .config import Settings
from .conversation import ConversationSession
from .observability import AiObservability, NoopObservability
from .omni_agent import AgentGeneration
from .omni_llama import ParsedModelTurn, _parse_model_turn
from .runtime import prune_runtime_turns

MOCK_OMNI_RESPONSE = (
    "옴니 명령은 아직 연결되지 않았지만, 실시간 음성 하네스가 "
    "사용자의 음성 턴을 정상적으로 받았습니다."
)


@dataclass(frozen=True)
class TurnResult:
    turn_id: str
    text: str
    user_text: str | None
    audio_path: Path | None
    used_mock_omni: bool
    used_tts: bool
    timings: dict[str, float] = field(default_factory=dict)


class OmniServerClient(Protocol):
    async def generate(
        self,
        audio_path: Path,
        history: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> AgentGeneration: ...


async def run_turn_pipeline(
    input_audio: bytes,
    settings: Settings,
    session: ConversationSession | None = None,
    omni_client: OmniServerClient | None = None,
    max_tokens: int | None = None,
    tts_num_steps: int | None = None,
    observability: AiObservability | None = None,
    trace_source: str = "turn",
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
    observer = observability or NoopObservability()

    with observer.start_turn(
        turn_id=turn_id,
        session_id=session.id if session else None,
        audio_path=input_path,
        history=history,
        metadata={
            "source": trace_source,
            "omni_backend": settings.omni_backend,
            "tts_backend": settings.tts_backend,
            "max_tokens": max_tokens or settings.llama_n_predict,
            "tts_num_steps": tts_num_steps or settings.omnivoice_num_steps,
        },
    ) as turn_observation:
        generation: AgentGeneration | None = None
        parsed_result: ParsedModelTurn | None = None
        try:
            omni_started_at = time.perf_counter()
            if settings.omni_backend == "server":
                if omni_client is None:
                    raise RuntimeError("llama-server client is not initialized.")
                generation = await omni_client.generate(
                    input_path,
                    history,
                    max_tokens=max_tokens,
                )
                omni_result = _parse_omni_output(
                    generation.output,
                    parsed=generation.parsed,
                )
                if generation.tool_trace:
                    (turn_dir / "tool-trace.json").write_text(
                        json.dumps(
                            {"calls": generation.tool_trace},
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
            else:
                with observer.start_observation(
                    name="omni-command",
                    as_type="generation",
                    input={"audio": "[see voice-turn input audio]", "history": history},
                    model="external-command" if settings.omni_command else "mock",
                ) as command_observation:
                    try:
                        omni_result = await asyncio.to_thread(
                            run_omni,
                            input_path,
                            settings,
                            history=history,
                            max_tokens=max_tokens,
                        )
                    except Exception as exc:
                        command_observation.fail(exc)
                        raise
                    command_observation.update(
                        output={
                            "raw_content": omni_result.raw_output,
                            "transcript": omni_result.user_text,
                            "response": omni_result.text,
                        }
                    )
                if omni_result.parsed is not None:
                    with observer.start_observation(
                        name="parse-model-turn",
                        input={"raw_content": omni_result.raw_output},
                    ) as parse_observation:
                        parse_observation.update(
                            output=omni_result.parsed.trace(),
                            level=(
                                "DEFAULT"
                                if omni_result.parsed.structured_output_valid
                                else "WARNING"
                            ),
                            status_message=omni_result.parsed.fallback_reason,
                        )
            omni_elapsed = time.perf_counter() - omni_started_at
            parsed_result = omni_result.parsed or _parse_model_turn(
                json.dumps(
                    {
                        "user_text": omni_result.user_text,
                        "text": omni_result.text,
                    },
                    ensure_ascii=False,
                )
            )

            if session:
                session.add_user_audio_turn(omni_result.user_text)
                session.add_assistant_message(omni_result.text)

            tts_started_at = time.perf_counter()
            with observer.start_observation(
                name="tts",
                input={"text": omni_result.text},
                metadata={
                    "backend": settings.tts_backend,
                    "num_steps": tts_num_steps or settings.omnivoice_num_steps,
                },
                model=(
                    settings.omnivoice_model_id
                    if settings.tts_backend == "omnivoice"
                    else "external-command"
                ),
            ) as tts_observation:
                try:
                    audio_path = await asyncio.to_thread(
                        run_tts,
                        omni_result.text,
                        turn_dir,
                        settings,
                        num_steps=tts_num_steps,
                    )
                except Exception as exc:
                    tts_observation.fail(exc)
                    raise
                tts_observation.update(output={"audio_created": audio_path is not None})
            tts_elapsed = time.perf_counter() - tts_started_at

            timings = {
                "omni_seconds": round(omni_elapsed, 3),
                "tts_seconds": round(tts_elapsed, 3),
                "total_seconds": round(time.perf_counter() - started_at, 3),
            }
            result = TurnResult(
                turn_id=turn_id,
                text=omni_result.text,
                user_text=omni_result.user_text,
                audio_path=audio_path,
                used_mock_omni=omni_result.used_mock,
                used_tts=audio_path is not None,
                timings=timings,
            )
            turn_observation.update(
                output={
                    "transcript": result.user_text,
                    "response": result.text,
                    "used_mock_omni": result.used_mock_omni,
                    "used_tts": result.used_tts,
                    "timings": timings,
                    "parse": parsed_result.trace(),
                }
            )
            turn_observation.score_trace("turn_succeeded", True)
            turn_observation.score_trace(
                "structured_output_valid",
                parsed_result.structured_output_valid,
            )
            turn_observation.score_trace(
                "transcript_present",
                omni_result.user_text is not None,
            )
            turn_observation.score_trace(
                "tool_loop_limit_hit",
                bool(generation and generation.tool_loop_limit_hit),
            )
            if generation and generation.tool_attempted:
                turn_observation.score_trace(
                    "tool_calls_succeeded",
                    generation.tool_succeeded,
                )
            return result
        except Exception as exc:
            turn_observation.fail(exc)
            turn_observation.score_trace("turn_succeeded", False)
            if parsed_result is not None:
                turn_observation.score_trace(
                    "structured_output_valid",
                    parsed_result.structured_output_valid,
                )
                turn_observation.score_trace(
                    "transcript_present",
                    parsed_result.transcript is not None,
                )
            if generation is not None:
                turn_observation.score_trace(
                    "tool_loop_limit_hit",
                    generation.tool_loop_limit_hit,
                )
                if generation.tool_attempted:
                    turn_observation.score_trace(
                        "tool_calls_succeeded",
                        generation.tool_succeeded,
                    )
            raise


async def run_demo_pipeline(
    input_audio: bytes,
    settings: Settings,
    omni_client: OmniServerClient | None = None,
    max_tokens: int | None = None,
    tts_num_steps: int | None = None,
    observability: AiObservability | None = None,
) -> TurnResult:
    return await run_turn_pipeline(
        input_audio,
        settings,
        omni_client=omni_client,
        max_tokens=max_tokens,
        tts_num_steps=tts_num_steps,
        observability=observability,
        trace_source="demo",
    )


@dataclass(frozen=True)
class OmniResult:
    text: str
    user_text: str | None
    used_mock: bool
    parsed: ParsedModelTurn | None = None
    raw_output: str | None = None


def run_omni(
    input_path: Path,
    settings: Settings,
    history: list[dict[str, str]] | None = None,
    omni_client: OmniServerClient | None = None,
    max_tokens: int | None = None,
) -> OmniResult:
    if settings.omni_backend == "server":
        raise RuntimeError(
            "The server OMNI backend must run through the asynchronous turn pipeline."
        )

    if not settings.omni_command:
        return OmniResult(
            text=MOCK_OMNI_RESPONSE,
            user_text=None,
            used_mock=True,
            parsed=ParsedModelTurn(
                transcript=None,
                response=MOCK_OMNI_RESPONSE,
                fallback_reason="The mock OMNI backend does not create transcripts.",
            ),
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


def _parse_omni_output(
    output: str,
    parsed: ParsedModelTurn | None = None,
) -> OmniResult:
    parsed = parsed or _parse_model_turn(output)
    return OmniResult(
        text=parsed.response,
        user_text=parsed.transcript,
        used_mock=False,
        parsed=parsed,
        raw_output=output,
    )


def run_tts(
    text: str,
    turn_dir: Path,
    settings: Settings,
    num_steps: int | None = None,
) -> Path | None:
    if settings.tts_backend == "omnivoice":
        from .tts_omnivoice import synthesize

        output_path = turn_dir / "reply.wav"
        synthesize(
            text=text,
            output_path=output_path,
            language=settings.omnivoice_language,
            instruct=settings.omnivoice_instruct,
            speed=settings.omnivoice_speed,
            model_id=settings.omnivoice_model_id,
            num_steps=(
                num_steps if num_steps is not None else settings.omnivoice_num_steps
            ),
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
    quoted = {key: shlex.quote(str(value)) for key, value in values.items()}
    return shlex.split(template.format(**quoted))
