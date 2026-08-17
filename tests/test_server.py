from __future__ import annotations

import json
import unittest

from fastapi import HTTPException, Request

from omni_iot.server import _int_header, client_config


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


if __name__ == "__main__":
    unittest.main()
