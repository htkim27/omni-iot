from __future__ import annotations

import json
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import soundfile as sf
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from omni_iot.pipeline import TurnResult
from omni_iot.server import (
    _int_header,
    _record_browser_playback_timing,
    app,
    client_config,
    settings as server_settings,
)
from omni_iot.wakeword import WakeDetection
from omni_iot.vllm_omni import OmniStreamChunk


def _request(**headers: str) -> Request:
    raw_headers = [
        (name.replace("_", "-").encode(), value.encode())
        for name, value in headers.items()
    ]
    return Request({"type": "http", "headers": raw_headers})


class ClientConfigTest(unittest.TestCase):
    def test_records_browser_audio_start_against_the_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_dir = Path(temp_dir)
            turn_id = "a" * 32
            turn_dir = runtime_dir / turn_id
            turn_dir.mkdir()
            timing_path = turn_dir / "timings.json"
            timing_path.write_text('{"total_seconds": 2.0}', encoding="utf-8")
            benchmark_settings = replace(server_settings, runtime_dir=runtime_dir)

            with patch("omni_iot.server.settings", benchmark_settings):
                _record_browser_playback_timing(
                    {
                        "turn_id": turn_id,
                        "browser_speech_end_to_audio_start_seconds": 2.643,
                    }
                )

            timings = json.loads(timing_path.read_text(encoding="utf-8"))
            self.assertEqual(
                timings["browser_speech_end_to_audio_start_seconds"], 2.643
            )

    def test_exposes_only_browser_safe_defaults(self) -> None:
        payload = json.loads(client_config().body)

        self.assertEqual(payload["vad"]["silence_end_ms"], 600)
        self.assertEqual(payload["generation"]["max_response_tokens"], 192)
        self.assertEqual(payload["wakeword"]["label"], "오둥아")
        self.assertEqual(payload["wakeword"]["threshold"], 0.5)
        self.assertNotIn("model", payload)
        self.assertNotIn("command", payload)

    def test_integer_header_accepts_valid_override(self) -> None:
        request = _request(x_response_token_limit="96")

        self.assertEqual(
            _int_header(request, "x-response-token-limit", 16, 512),
            96,
        )

    def test_integer_header_rejects_out_of_range_override(self) -> None:
        request = _request(x_tts_num_steps="2")

        with self.assertRaises(HTTPException) as raised:
            _int_header(request, "x-tts-num-steps", 4, 64)

        self.assertEqual(raised.exception.status_code, 400)


class FakeDetector:
    def __init__(self, *_args: object) -> None:
        self.detected = False

    def process(self, _pcm: bytes) -> WakeDetection | None:
        if self.detected:
            return None
        self.detected = True
        return WakeDetection(model="hey_jarvis_v0.1", score=0.91)

    def reset(self) -> None:
        self.detected = False


class AudioWebSocketTest(unittest.TestCase):
    def test_wake_turn_follow_up_and_sleep_flow(self) -> None:
        result = TurnResult(
            turn_id="a" * 32,
            text="안녕하세요",
            user_text="오늘 날씨 알려줘",
            audio_path=None,
            used_mock_omni=False,
            used_tts=False,
            timings={"total_seconds": 0.1},
        )

        with (
            patch("omni_iot.server.wakeword_detector_factory", FakeDetector),
            patch("omni_iot.server.omni_service", None),
            patch(
                "omni_iot.server.settings",
                replace(
                    server_settings,
                    omni_backend="command",
                    tts_backend="command",
                ),
            ),
            patch(
                "omni_iot.server.run_turn_pipeline",
                new=AsyncMock(return_value=result),
            ),
            TestClient(app) as client,
            client.websocket_connect("/ws/audio") as websocket,
        ):
            self.assertEqual(websocket.receive_json()["type"], "ready")
            websocket.send_json({"type": "start", "sample_rate": 16_000})
            started = websocket.receive_json()
            self.assertEqual(started["type"], "started")

            websocket.send_bytes(bytes(2_560))
            self.assertEqual(websocket.receive_json()["type"], "wake_detected")
            websocket.send_bytes(bytes(640))
            websocket.send_json({"type": "speech_ended"})
            self.assertEqual(websocket.receive_json()["state"], "processing")
            reply = websocket.receive_json()
            self.assertEqual(reply["type"], "reply_audio")
            self.assertEqual(reply["text"], "안녕하세요")

            websocket.send_json({"type": "reply_ended"})
            self.assertEqual(websocket.receive_json()["state"], "follow_up")
            websocket.send_json({"type": "sleep"})
            self.assertEqual(websocket.receive_json()["state"], "sleeping")

    def test_vllm_omni_streams_pcm_between_start_and_end_events(self) -> None:
        wav = io.BytesIO()
        sf.write(
            wav,
            np.zeros(240, dtype=np.float32),
            24_000,
            format="WAV",
            subtype="PCM_16",
        )

        class FakeVllmOmni:
            async def stream(self, *_args: object, **_kwargs: object):
                yield OmniStreamChunk("text", "안녕하세요")
                yield OmniStreamChunk("audio", wav.getvalue())

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("omni_iot.server.wakeword_detector_factory", FakeDetector),
                patch("omni_iot.server.omni_service", None),
                patch("omni_iot.server.vllm_omni_service", FakeVllmOmni()),
                patch(
                    "omni_iot.server.settings",
                    replace(
                        server_settings,
                        omni_backend="vllm_omni",
                        runtime_dir=Path(temp_dir),
                    ),
                ),
                TestClient(app) as client,
                client.websocket_connect("/ws/audio") as websocket,
            ):
                self.assertEqual(websocket.receive_json()["type"], "ready")
                websocket.send_json({"type": "start", "sample_rate": 16_000})
                self.assertEqual(websocket.receive_json()["type"], "started")
                websocket.send_bytes(bytes(2_560))
                self.assertEqual(websocket.receive_json()["type"], "wake_detected")
                websocket.send_bytes(bytes(640))
                websocket.send_json({"type": "speech_ended"})
                self.assertEqual(websocket.receive_json()["state"], "processing")
                stream_start = websocket.receive_json()
                self.assertEqual(stream_start["type"], "audio_stream_start")
                self.assertEqual(stream_start["sample_rate"], 24_000)
                self.assertEqual(len(websocket.receive_bytes()), 480)
                stream_end = websocket.receive_json()
                self.assertEqual(stream_end["type"], "audio_stream_end")
                self.assertEqual(stream_end["text"], "안녕하세요")


if __name__ == "__main__":
    unittest.main()
