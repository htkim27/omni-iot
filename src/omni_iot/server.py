from __future__ import annotations

import argparse
import asyncio
import io
import json
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .conversation import ConversationSession, ConversationStore
from .mcp_client import McpManager
from .omni_agent import OmniAgent
from .omni_llama import LlamaServer
from .pipeline import TurnResult, run_demo_pipeline, run_turn_pipeline
from .wakeword import SAMPLE_RATE, WakeWordDetector

settings = get_settings()
omni_service = LlamaServer(settings) if settings.omni_backend == "server" else None
mcp_service = McpManager(
    settings.mcp_config,
    catalog_max_chars=settings.mcp_tool_catalog_max_chars,
    result_max_chars=settings.mcp_tool_result_max_chars,
    brave_search_max_results=settings.mcp_brave_search_max_results,
)
omni_agent = (
    OmniAgent(omni_service, mcp_service, settings) if omni_service is not None else None
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    tts_loaded = False
    try:
        if omni_service:
            await asyncio.to_thread(omni_service.start)
        await mcp_service.start()
        if settings.tts_backend == "omnivoice":
            from .tts_omnivoice import load_model, warmup_model

            await asyncio.to_thread(load_model, settings.omnivoice_model_id)
            tts_loaded = True
            if settings.omnivoice_warmup:
                await asyncio.to_thread(
                    warmup_model,
                    settings.omnivoice_model_id,
                    settings.omnivoice_language,
                    settings.omnivoice_instruct,
                    settings.omnivoice_speed,
                    settings.omnivoice_num_steps,
                )
        yield
    finally:
        await mcp_service.stop()
        if tts_loaded:
            from .tts_omnivoice import clear_model_cache

            clear_model_cache()
        if omni_service:
            await asyncio.to_thread(omni_service.stop)


app = FastAPI(title="omni-iot demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
conversations = ConversationStore()
wakeword_detector_factory = WakeWordDetector


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(settings.static_dir / "index.html")


@app.post("/api/demo")
async def demo(request: Request) -> JSONResponse:
    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=400, detail="No audio bytes received.")
    max_tokens = _int_header(request, "x-response-token-limit", 16, 512)
    tts_num_steps = _int_header(request, "x-tts-num-steps", 4, 64)

    try:
        result = await run_demo_pipeline(
            audio,
            settings,
            omni_agent,
            max_tokens,
            tts_num_steps,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    audio_url = None
    if result.audio_path:
        audio_url = f"/api/audio/{result.audio_path.parent.name}/{result.audio_path.name}"

    return JSONResponse(
        _turn_payload(result, audio_url=audio_url)
    )


@app.post("/api/turn")
async def turn(
    request: Request,
    x_session_id: str | None = Header(default=None),
) -> JSONResponse:
    audio_bytes = await request.body()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio bytes received.")
    max_tokens = _int_header(request, "x-response-token-limit", 16, 512)
    tts_num_steps = _int_header(request, "x-tts-num-steps", 4, 64)

    session = conversations.get(x_session_id)
    try:
        result = await run_turn_pipeline(
            audio_bytes,
            settings,
            session,
            omni_agent,
            max_tokens,
            tts_num_steps,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    audio_url = None
    if result.audio_path:
        audio_url = f"/api/audio/{result.audio_path.parent.name}/{result.audio_path.name}"

    payload = _turn_payload(result, audio_url=audio_url)
    payload["session_id"] = session.id
    payload["transcript"] = session.transcript()
    return JSONResponse(payload)


@app.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket) -> None:
    """Stream 16 kHz mono Int16 PCM through wake, turn, and follow-up states."""
    await websocket.accept()
    try:
        detector = await asyncio.to_thread(
            wakeword_detector_factory,
            settings.wakeword_model,
            settings.wakeword_threshold,
            settings.wakeword_inference_framework,
        )
    except Exception as exc:  # noqa: BLE001 - report model initialization failures
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=1011)
        return

    state = "sleeping"
    session = conversations.get(None)
    audio_buffer = bytearray()
    wake_buffer = bytearray()
    wake_buffer_bytes = SAMPLE_RATE * 2 * settings.vad_pre_roll_ms // 1_000
    max_audio_bytes = SAMPLE_RATE * 2 * settings.vad_max_turn_ms // 1_000
    max_tokens: int | None = None
    tts_num_steps: int | None = None

    await websocket.send_json(
        {
            "type": "ready",
            "state": state,
            "sample_rate": SAMPLE_RATE,
            "session_id": session.id,
        }
    )

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            pcm = message.get("bytes")
            if pcm is not None:
                if len(pcm) % 2:
                    await websocket.send_json(
                        {"type": "error", "message": "Invalid 16-bit PCM chunk."}
                    )
                    continue

                if state == "sleeping":
                    wake_buffer.extend(pcm)
                    if wake_buffer_bytes and len(wake_buffer) > wake_buffer_bytes:
                        del wake_buffer[:-wake_buffer_bytes]
                    elif not wake_buffer_bytes:
                        wake_buffer = bytearray(pcm)
                    detection = await asyncio.to_thread(detector.process, pcm)
                    if detection is not None:
                        state = "recording"
                        audio_buffer = bytearray(wake_buffer)
                        wake_buffer.clear()
                        await websocket.send_json(
                            {
                                "type": "wake_detected",
                                "model": detection.model,
                                "score": detection.score,
                                "state": state,
                            }
                        )
                elif state == "recording":
                    audio_buffer.extend(pcm)
                    if len(audio_buffer) >= max_audio_bytes:
                        state = await _finish_websocket_turn(
                            websocket,
                            audio_buffer,
                            session,
                            max_tokens,
                            tts_num_steps,
                        )
                        audio_buffer.clear()
                continue

            raw_text = message.get("text")
            if raw_text is None:
                continue
            try:
                event = json.loads(raw_text)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "message": "Invalid JSON event."}
                )
                continue
            if not isinstance(event, dict):
                await websocket.send_json(
                    {"type": "error", "message": "JSON event must be an object."}
                )
                continue

            event_type = event.get("type")
            if event_type == "start":
                if event.get("sample_rate") != SAMPLE_RATE:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": f"Audio must be {SAMPLE_RATE} Hz mono Int16 PCM.",
                        }
                    )
                    continue
                session = conversations.get(_optional_string(event.get("session_id")))
                try:
                    max_tokens = _optional_bounded_int(
                        event.get("max_response_tokens"), 16, 512
                    )
                    tts_num_steps = _optional_bounded_int(
                        event.get("tts_num_steps"), 4, 64
                    )
                except (TypeError, ValueError) as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    continue
                await websocket.send_json(
                    {
                        "type": "started",
                        "state": state,
                        "session_id": session.id,
                    }
                )
            elif event_type == "speech_started" and state in {"follow_up", "speaking"}:
                state = "recording"
                audio_buffer.clear()
                await websocket.send_json({"type": "state", "state": state})
            elif event_type == "speech_ended" and state == "recording":
                if audio_buffer:
                    state = await _finish_websocket_turn(
                        websocket,
                        audio_buffer,
                        session,
                        max_tokens,
                        tts_num_steps,
                    )
                    audio_buffer.clear()
                else:
                    state = "follow_up"
                    await websocket.send_json({"type": "state", "state": state})
            elif event_type == "reply_ended" and state == "speaking":
                state = "follow_up"
                await websocket.send_json({"type": "state", "state": state})
            elif event_type == "playback_started":
                _record_browser_playback_timing(event)
            elif event_type == "sleep":
                state = "sleeping"
                audio_buffer.clear()
                wake_buffer.clear()
                await asyncio.to_thread(detector.reset)
                await websocket.send_json({"type": "state", "state": state})
            elif event_type == "reset_session":
                session = conversations.reset(session.id)
                await websocket.send_json(
                    {
                        "type": "session_reset",
                        "session_id": session.id,
                        "transcript": [],
                    }
                )
    except (WebSocketDisconnect, RuntimeError):
        return


async def _finish_websocket_turn(
    websocket: WebSocket,
    pcm: bytearray,
    session: ConversationSession,
    max_tokens: int | None,
    tts_num_steps: int | None,
) -> str:
    await websocket.send_json({"type": "state", "state": "processing"})
    try:
        result = await run_turn_pipeline(
            _pcm_to_wav(bytes(pcm)),
            settings,
            session,
            omni_agent,
            max_tokens,
            tts_num_steps,
        )
    except Exception as exc:  # noqa: BLE001 - surface pipeline failures to the client
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.send_json({"type": "state", "state": "follow_up"})
        return "follow_up"

    audio_url = None
    if result.audio_path:
        audio_url = f"/api/audio/{result.audio_path.parent.name}/{result.audio_path.name}"
    payload = _turn_payload(result, audio_url=audio_url)
    payload.update(
        {
            "type": "reply_audio",
            "url": audio_url,
            "session_id": session.id,
            "transcript": session.transcript(),
            "state": "speaking",
        }
    )
    await websocket.send_json(payload)
    return "speaking"


def _pcm_to_wav(pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm)
    return output.getvalue()


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_bounded_int(value: object, minimum: int, maximum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("WebSocket generation options must be integers.")
    if not minimum <= value <= maximum:
        raise ValueError(f"WebSocket option must be between {minimum} and {maximum}.")
    return value


@app.post("/api/session/reset")
async def reset_session(x_session_id: str | None = Header(default=None)) -> JSONResponse:
    session = conversations.reset(x_session_id)
    return JSONResponse({"session_id": session.id, "transcript": []})


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "omni_backend": settings.omni_backend,
            "omni_configured": (
                omni_service.ready if omni_service else bool(settings.omni_command)
            ),
            "tts_backend": settings.tts_backend,
            "tts_configured": settings.tts_backend == "omnivoice" or bool(settings.tts_command),
            "mcp": mcp_service.health(),
        }
    )


@app.get("/api/config")
def client_config() -> JSONResponse:
    """Return non-secret, browser-adjustable defaults and fixed VAD settings."""
    return JSONResponse(
        {
            "vad": {
                "threshold": settings.vad_threshold,
                "silence_end_ms": settings.vad_silence_end_ms,
                "pre_roll_ms": settings.vad_pre_roll_ms,
                "max_turn_ms": settings.vad_max_turn_ms,
                "barge_in_multiplier": settings.vad_barge_in_multiplier,
                "continue_multiplier": settings.vad_continue_multiplier,
            },
            "generation": {
                "max_response_tokens": settings.llama_n_predict,
                "tts_num_steps": settings.omnivoice_num_steps,
            },
            "features": {
                "sentence_tts_pipelining": False,
                "wake_word": True,
            },
            "audio": {
                "sample_rate": SAMPLE_RATE,
                "encoding": "pcm_s16le",
            },
            "wakeword": {
                "label": settings.wakeword_label,
                "threshold": settings.wakeword_threshold,
                "follow_up_timeout_ms": settings.follow_up_timeout_ms,
            },
        }
    )


@app.get("/api/audio/{turn_id}/{filename}")
def audio(turn_id: str, filename: str) -> FileResponse:
    audio_path = settings.runtime_dir / turn_id / filename
    if not _is_runtime_child(audio_path):
        raise HTTPException(status_code=404, detail="Audio not found.")
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found.")
    return FileResponse(audio_path, media_type="audio/wav")


def _is_runtime_child(path: Path) -> bool:
    try:
        path.resolve().relative_to(settings.runtime_dir.resolve())
    except ValueError:
        return False
    return True


def _record_browser_playback_timing(event: dict[str, object]) -> None:
    turn_id = event.get("turn_id")
    seconds = event.get("browser_speech_end_to_audio_start_seconds")
    if (
        not isinstance(turn_id, str)
        or len(turn_id) != 32
        or not all(character in "0123456789abcdef" for character in turn_id)
        or isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not 0 <= seconds <= settings.omni_timeout_seconds + settings.tts_timeout_seconds
    ):
        return
    timing_path = settings.runtime_dir / turn_id / "timings.json"
    if not _is_runtime_child(timing_path) or not timing_path.exists():
        return
    try:
        timings = json.loads(timing_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(timings, dict):
        return
    timings["browser_speech_end_to_audio_start_seconds"] = round(float(seconds), 3)
    timing_path.write_text(
        json.dumps(timings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _turn_payload(result: TurnResult, audio_url: str | None) -> dict[str, object]:
    return {
        "turn_id": result.turn_id,
        "text": result.text,
        "user_text": result.user_text,
        "audio_url": audio_url,
        "used_mock_omni": result.used_mock_omni,
        "used_tts": result.used_tts,
        "timings": result.timings,
    }


def _int_header(request: Request, name: str, minimum: int, maximum: int) -> int | None:
    value = request.headers.get(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {name} header.") from exc
    if not minimum <= parsed <= maximum:
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be between {minimum} and {maximum}.",
        )
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("omni_iot.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
