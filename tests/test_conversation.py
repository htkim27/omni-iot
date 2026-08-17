from __future__ import annotations

import unittest

from omni_iot.conversation import ConversationSession


class ConversationSessionTest(unittest.TestCase):
    def test_prompt_history_keeps_real_transcript_and_recent_messages(self) -> None:
        session = ConversationSession(id="session")
        session.add_user_audio_turn("내 이름은 민수야")
        session.add_assistant_message("반가워요, 민수님.")
        session.add_user_audio_turn("내 이름을 기억해?")
        session.add_assistant_message("네, 민수님이에요.")

        self.assertEqual(
            session.prompt_history(max_messages=2),
            [
                {"role": "user", "content": "내 이름을 기억해?"},
                {"role": "assistant", "content": "네, 민수님이에요."},
            ],
        )

    def test_zero_history_limit_disables_context(self) -> None:
        session = ConversationSession(id="session")
        session.add_user_audio_turn("안녕")

        self.assertEqual(session.prompt_history(max_messages=0), [])


if __name__ == "__main__":
    unittest.main()
