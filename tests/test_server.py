from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from omni_iot.observability import NoopObservability
from omni_iot.pipeline import TurnResult
from omni_iot.server import (
    _int_header,
    app,
    client_config,
    health,
)
from omni_iot.server import (
    settings as server_settings,
)
from omni_iot.wakeword import WakeDetection


def _request(**headers: str) -> Request:
    raw_headers = [
        (name.replace("_", "-").encode(), value.encode())
        for name, value in headers.items()
    ]
    return Request({"type": "http", "headers": raw_headers})


class ClientConfigTest(unittest.TestCase):
    def test_exposes_only_browser_safe_defaults(self) -> None:
        payload = json.loads(client_config().body)

        self.assertEqual(payload["vad"]["silence_end_ms"], 600)
        self.assertEqual(payload["generation"]["max_response_tokens"], 192)
        self.assertEqual(payload["wakeword"]["label"], "Hey Jarvis")
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

    def test_health_exposes_non_secret_observability_state(self) -> None:
        with patch("omni_iot.server.ai_observability", NoopObservability()):
            payload = json.loads(health().body)

        self.assertEqual(payload["observability"]["backend"], "none")
        self.assertFalse(payload["observability"]["enabled"])
        self.assertNotIn("secret_key", payload["observability"])


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


if __name__ == "__main__":
    unittest.main()
