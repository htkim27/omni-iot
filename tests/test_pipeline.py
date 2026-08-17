from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omni_iot.config import Settings
from omni_iot.conversation import ConversationSession
from omni_iot.pipeline import (
    OmniResult,
    _parse_omni_output,
    run_omni,
    run_tts,
    run_turn_pipeline,
)


def _settings(runtime_dir: Path, omni_command: str | None = None) -> Settings:
    return Settings(
        project_root=runtime_dir,
        runtime_dir=runtime_dir,
        omni_backend="command",
        omni_command=omni_command,
        tts_backend="command",
        tts_command=None,
    )


class PipelineMultiTurnTest(unittest.TestCase):
    def test_external_backend_transcript_placeholder_is_discarded(self) -> None:
        result = _parse_omni_output(
            '{"user_text":"사용자가 실제로 말한 내용","text":"괜찮아요."}'
        )

        self.assertIsNone(result.user_text)
        self.assertEqual(result.text, "괜찮아요.")

    def test_omnivoice_runs_in_the_server_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            turn_dir = Path(temp_dir)
            settings = Settings(
                project_root=turn_dir,
                runtime_dir=turn_dir,
                tts_backend="omnivoice",
            )

            def fake_synthesize(**kwargs: object) -> None:
                Path(kwargs["output_path"]).write_bytes(b"wav")

            with patch(
                "omni_iot.tts_omnivoice.synthesize",
                side_effect=fake_synthesize,
            ) as synthesize:
                output = run_tts("안녕하세요.", turn_dir, settings)

            self.assertEqual(output, turn_dir / "reply.wav")
            synthesize.assert_called_once()
            self.assertEqual(synthesize.call_args.kwargs["num_steps"], 32)

    def test_omnivoice_accepts_per_turn_step_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            turn_dir = Path(temp_dir)
            settings = Settings(
                project_root=turn_dir,
                runtime_dir=turn_dir,
                tts_backend="omnivoice",
            )

            def fake_synthesize(**kwargs: object) -> None:
                Path(kwargs["output_path"]).write_bytes(b"wav")

            with patch(
                "omni_iot.tts_omnivoice.synthesize",
                side_effect=fake_synthesize,
            ) as synthesize:
                run_tts("안녕하세요.", turn_dir, settings, num_steps=16)

            self.assertEqual(synthesize.call_args.kwargs["num_steps"], 16)

    def test_second_turn_receives_first_turn_transcript_and_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = _settings(Path(temp_dir))
            session = ConversationSession(id="session")
            histories: list[list[dict[str, str]]] = []
            results = iter(
                [
                    OmniResult("반가워요, 민수님.", "내 이름은 민수야", False),
                    OmniResult("네, 민수님이에요.", "내 이름 기억해?", False),
                ]
            )

            def fake_omni(
                _input_path: Path,
                _settings: Settings,
                history: list[dict[str, str]] | None = None,
                max_tokens: int | None = None,
            ) -> OmniResult:
                histories.append(history or [])
                return next(results)

            with (
                patch("omni_iot.pipeline.run_omni", side_effect=fake_omni),
                patch("omni_iot.pipeline.run_tts", return_value=None),
            ):
                run_turn_pipeline(b"first", settings, session=session)
                second = run_turn_pipeline(b"second", settings, session=session)

            self.assertEqual(histories[0], [])
            self.assertEqual(
                histories[1],
                [
                    {"role": "user", "content": "내 이름은 민수야"},
                    {"role": "assistant", "content": "반가워요, 민수님."},
                ],
            )
            self.assertEqual(second.user_text, "내 이름 기억해?")
            self.assertEqual(len(session.messages), 4)

    def test_command_receives_history_file_and_parses_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_dir = Path(temp_dir)
            input_path = runtime_dir / "input.wav"
            input_path.write_bytes(b"wav")
            settings = _settings(
                runtime_dir,
                "runner --audio {audio} --history-file {history_file}",
            )
            history = [{"role": "user", "content": "내 이름은 민수야"}]
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(
                    {"user_text": "기억해?", "text": "네, 민수님."},
                    ensure_ascii=False,
                ),
                stderr="",
            )

            with patch("omni_iot.pipeline.subprocess.run", return_value=completed) as run:
                result = run_omni(input_path, settings, history=history)

            command = run.call_args.args[0]
            history_path = Path(command[command.index("--history-file") + 1])
            self.assertEqual(
                json.loads(history_path.read_text(encoding="utf-8")),
                history,
            )
            self.assertEqual(result.user_text, "기억해?")
            self.assertEqual(result.text, "네, 민수님.")


if __name__ == "__main__":
    unittest.main()
