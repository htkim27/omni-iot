from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .conversation import ConversationStore
from .pipeline import TurnResult, run_demo_pipeline, run_turn_pipeline


settings = get_settings()
app = FastAPI(title="omni-iot demo")
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

    try:
        result = run_demo_pipeline(audio, settings)
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

    session = conversations.get(x_session_id)
    try:
        result = run_turn_pipeline(audio_bytes, settings, session=session)
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
            "omni_configured": bool(settings.omni_command),
            "tts_backend": settings.tts_backend,
            "tts_configured": settings.tts_backend == "omnivoice" or bool(settings.tts_command),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("omni_iot.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
