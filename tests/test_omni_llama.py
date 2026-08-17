from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from omni_iot.omni_llama import _build_turn_prompt, _load_history, _parse_model_turn


class OmniLlamaTest(unittest.TestCase):
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
