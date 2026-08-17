from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .config import get_settings
from .conversation import ConversationStore
from .omni_llama import LlamaServer
from .pipeline import TurnResult, run_demo_pipeline, run_turn_pipeline


settings = get_settings()
omni_service = LlamaServer(settings) if settings.omni_backend == "server" else None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    tts_loaded = False
    try:
        if omni_service:
            await asyncio.to_thread(omni_service.start)
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
        if tts_loaded:
            from .tts_omnivoice import clear_model_cache

            clear_model_cache()
        if omni_service:
            await asyncio.to_thread(omni_service.stop)


app = FastAPI(title="omni-iot demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
conversations = ConversationStore()


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
        result = await run_in_threadpool(
            run_demo_pipeline,
            audio,
            settings,
            omni_service,
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
        result = await run_in_threadpool(
            run_turn_pipeline,
            audio_bytes,
            settings,
            session,
            omni_service,
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
