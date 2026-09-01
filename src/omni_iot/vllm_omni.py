from __future__ import annotations

import asyncio
import base64
import io
import json
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import Thread

import numpy as np
import soundfile as sf

from .config import Settings


@dataclass(frozen=True)
class OmniStreamChunk:
    modality: str
    content: str | bytes


@dataclass(frozen=True)
class PcmChunk:
    data: bytes
    sample_rate: int
    channels: int


class VllmOmniClient:
    """Stream Qwen3-Omni text/audio from vLLM-Omni's OpenAI endpoint."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def ready(self) -> bool:
        url = self.settings.vllm_omni_base_url.removesuffix("/v1") + "/health"
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError):
            return False

    async def stream(
        self,
        audio_path: Path,
        history: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> AsyncIterator[OmniStreamChunk]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[OmniStreamChunk | BaseException | None] = asyncio.Queue()

        def produce() -> None:
            try:
                for chunk in self._stream_sync(audio_path, history, max_tokens):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except BaseException as exc:  # noqa: BLE001 - cross thread boundary
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        Thread(target=produce, name="vllm-omni-sse", daemon=True).start()
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

    def _stream_sync(
        self,
        audio_path: Path,
        history: list[dict[str, str]],
        max_tokens: int | None,
    ) -> Iterator[OmniStreamChunk]:
        audio_data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": (
                    "You are a private Korean voice assistant. Reply naturally and "
                    "concisely in Korean using only content that should be spoken aloud."
                ),
            }
        ]
        messages.extend(
            {"role": item["role"], "content": item["content"]}
            for item in history
            if item.get("role") in {"user", "assistant"} and item.get("content")
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": f"data:audio/wav;base64,{audio_data}",
                        },
                    }
                ],
            }
        )
        payload: dict[str, object] = {
            "model": self.settings.vllm_omni_model,
            "messages": messages,
            "modalities": ["text", "audio"],
            "stream": True,
            "temperature": self.settings.llama_temperature,
            "max_tokens": max_tokens or self.settings.llama_n_predict,
            "speaker": self.settings.vllm_omni_speaker,
        }
        request = urllib.request.Request(
            f"{self.settings.vllm_omni_base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.vllm_omni_api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.settings.omni_timeout_seconds,
            ) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    event = json.loads(data)
                    yield from _parse_stream_event(event)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM-Omni request failed ({exc.code}): {detail}"
            ) from exc
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"vLLM-Omni streaming failed: {exc}") from exc


def _parse_stream_event(event: object) -> Iterator[OmniStreamChunk]:
    if not isinstance(event, dict):
        return
    modality = event.get("modality")
    choices = event.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if modality == "audio" and isinstance(content, str) and content:
            yield OmniStreamChunk("audio", base64.b64decode(content))
        elif modality == "text" and isinstance(content, str) and content:
            yield OmniStreamChunk("text", content)
        audio = delta.get("audio")
        if isinstance(audio, dict) and isinstance(audio.get("data"), str):
            yield OmniStreamChunk("audio", base64.b64decode(audio["data"]))


def decode_audio_chunk(data: bytes) -> PcmChunk:
    """Decode one vLLM-Omni WAV chunk to interleaved signed 16-bit PCM."""
    audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1, keepdims=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = np.where(clipped < 0, clipped * 32768.0, clipped * 32767.0).astype(
        "<i2"
    )
    return PcmChunk(
        data=pcm.tobytes(),
        sample_rate=int(sample_rate),
        channels=int(pcm.shape[1]),
    )
