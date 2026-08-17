from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omni_iot.config import Settings
from omni_iot.omni_llama import (
    LlamaServer,
    _build_chat_messages,
    _build_turn_prompt,
    _load_history,
    _parse_model_turn,
)


class OmniLlamaTest(unittest.TestCase):
    def test_chat_messages_keep_history_before_current_audio(self) -> None:
        messages = _build_chat_messages(
            audio_data="d2F2",
            history=[
                {"role": "user", "content": "내 이름은 민수야"},
                {"role": "assistant", "content": "반가워요."},
            ],
        )

        self.assertEqual(
            [item["role"] for item in messages],
            ["system", "user", "assistant", "user"],
        )
        current_content = messages[-1]["content"]
        self.assertIsInstance(current_content, list)
        self.assertEqual(current_content[-1]["type"], "input_audio")
        self.assertEqual(current_content[-1]["input_audio"]["data"], "d2F2")

    def test_server_client_parses_model_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_dir = Path(temp_dir)
            audio_path = runtime_dir / "input.wav"
            audio_path.write_bytes(b"wav")
            server = LlamaServer(Settings(runtime_dir=runtime_dir))
            response = {
                "choices": [
                    {
                        "message": {
                            "content": '{"transcript":"안녕","response":"반가워요."}'
                        }
                    }
                ]
            }

            with patch.object(server, "_request_json", return_value=response) as request:
                result = json.loads(server.generate(audio_path, []))

            self.assertEqual(result, {"user_text": "안녕", "text": "반가워요."})
            payload = request.call_args.args[1]
            self.assertTrue(payload["cache_prompt"])

    def test_build_turn_prompt_includes_previous_conversation(self) -> None:
        prompt = _build_turn_prompt(
            "짧게 대답해줘.",
            [
                {"role": "user", "content": "내 이름은 민수야"},
                {"role": "assistant", "content": "반가워요, 민수님."},
            ],
        )

        self.assertIn("conversation_history", prompt)
        self.assertIn("내 이름은 민수야", prompt)
        self.assertIn('"transcript"', prompt)
        self.assertIn('"response"', prompt)

    def test_parse_model_turn_accepts_fenced_json(self) -> None:
        transcript, response = _parse_model_turn(
            '```json\n{"transcript":"기억해?","response":"네, 기억해요."}\n```'
        )

        self.assertEqual(transcript, "기억해?")
        self.assertEqual(response, "네, 기억해요.")

    def test_parse_model_turn_falls_back_to_plain_text(self) -> None:
        transcript, response = _parse_model_turn("그대로 사용할 응답")

        self.assertIsNone(transcript)
        self.assertEqual(response, "그대로 사용할 응답")

    def test_load_history_filters_invalid_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "history.json"
            path.write_text(
                json.dumps(
                    [
                        {"role": "user", "content": "안녕"},
                        {"role": "system", "content": "무시"},
                        {"role": "assistant", "content": "반가워요"},
                        "invalid",
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                _load_history(path),
                [
                    {"role": "user", "content": "안녕"},
                    {"role": "assistant", "content": "반가워요"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
