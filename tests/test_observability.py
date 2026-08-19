from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omni_iot.config import Settings
from omni_iot.eval_cli import _setup_resources
from omni_iot.mcp_client import McpCallResult
from omni_iot.observability import (
    NoopObservability,
    create_observability,
    prepare_trace_value,
    redact,
)
from omni_iot.omni_agent import AgentGeneration, OmniAgent
from omni_iot.omni_llama import ChatCompletion, ParsedModelTurn, ToolCall
from omni_iot.pipeline import run_turn_pipeline


class _RecordedObservation:
    def __init__(self, name: str, parent: str | None) -> None:
        self.name = name
        self.parent = parent
        self.updates: list[dict[str, object]] = []
        self.scores: dict[str, bool] = {}

    def update(self, **values: object) -> None:
        self.updates.append(values)

    def fail(self, exc: BaseException) -> None:
        self.updates.append({"failed": type(exc).__name__})

    def score_trace(self, name: str, value: bool) -> None:
        self.scores[name] = value


class _RecordingObservability:
    enabled = True

    def __init__(self) -> None:
        self.stack: list[str] = []
        self.records: list[_RecordedObservation] = []

    def authenticate(self) -> bool:
        return True

    def shutdown(self) -> None:
        return None

    def health(self) -> dict[str, object]:
        return {"enabled": True}

    @contextmanager
    def start_turn(self, **_values: object) -> Iterator[_RecordedObservation]:
        with self._record("voice-turn") as observation:
            yield observation

    @contextmanager
    def start_observation(
        self,
        *,
        name: str,
        **_values: object,
    ) -> Iterator[_RecordedObservation]:
        with self._record(name) as observation:
            yield observation

    @contextmanager
    def _record(self, name: str) -> Iterator[_RecordedObservation]:
        observation = _RecordedObservation(
            name,
            self.stack[-1] if self.stack else None,
        )
        self.records.append(observation)
        self.stack.append(name)
        try:
            yield observation
        finally:
            self.stack.pop()

    def trace_value(self, value: object) -> object:
        return prepare_trace_value(value)


class ObservabilityTest(unittest.TestCase):
    def test_redacts_nested_credentials_without_masking_usage_tokens(self) -> None:
        value = redact(
            {
                "authorization": "Bearer secret",
                "nested": {
                    "client_secret": "secret",
                    "input_tokens": 12,
                    "command": "turnOn",
                },
                "json": '{"api_key":"secret","device":"lamp"}',
            }
        )

        self.assertEqual(value["authorization"], "[REDACTED]")
        self.assertEqual(value["nested"]["client_secret"], "[REDACTED]")
        self.assertEqual(value["nested"]["input_tokens"], 12)
        self.assertEqual(value["nested"]["command"], "turnOn")
        self.assertEqual(
            json.loads(value["json"]),
            {"api_key": "[REDACTED]", "device": "lamp"},
        )

    def test_trace_value_replaces_audio_base64_without_mutating_request(self) -> None:
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "d2F2", "format": "wav"},
                        }
                    ],
                }
            ]
        }

        traced = prepare_trace_value(request)

        self.assertEqual(
            traced["messages"][0]["content"][0]["input_audio"]["data"],
            "[see voice-turn input audio]",
        )
        self.assertEqual(
            request["messages"][0]["content"][0]["input_audio"]["data"],
            "d2F2",
        )

    def test_disabled_mode_constructs_only_noop_observability(self) -> None:
        with patch("omni_iot.observability.LangfuseObservability") as langfuse:
            result = create_observability(Settings(ai_eval_enabled=False))

        self.assertIsInstance(result, NoopObservability)
        langfuse.assert_not_called()

    def test_enabled_mode_requires_both_langfuse_keys(self) -> None:
        from omni_iot import config

        with tempfile.TemporaryDirectory() as temp_dir:
            invalid = Settings(
                runtime_dir=Path(temp_dir),
                ai_eval_enabled=True,
                langfuse_public_key=None,
                langfuse_secret_key=None,
            )
            with (
                patch("omni_iot.config.Settings", return_value=invalid),
                self.assertRaisesRegex(ValueError, "LANGFUSE_PUBLIC_KEY"),
            ):
                config.get_settings()

    def test_evaluation_setup_is_idempotent(self) -> None:
        class ScoreConfigs:
            def __init__(self) -> None:
                self.data: list[SimpleNamespace] = []

            def get(self, **_kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(data=self.data)

            def create(self, *, name: str, **_kwargs: object) -> SimpleNamespace:
                config = SimpleNamespace(id=f"config-{name}", name=name)
                self.data.append(config)
                return config

        class Queues:
            def __init__(self) -> None:
                self.data: list[SimpleNamespace] = []

            def list_queues(self, **_kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(data=self.data)

            def create_queue(self, *, name: str, **_kwargs: object) -> None:
                self.data.append(SimpleNamespace(name=name))

        api = SimpleNamespace(score_configs=ScoreConfigs(), annotation_queues=Queues())
        observability = SimpleNamespace(api=api)

        _setup_resources(observability)  # type: ignore[arg-type]
        _setup_resources(observability)  # type: ignore[arg-type]

        self.assertEqual(len(api.score_configs.data), 3)
        self.assertEqual(
            [queue.name for queue in api.annotation_queues.data],
            ["voice-transcript-review"],
        )

    def test_pipeline_records_root_children_and_deterministic_scores(self) -> None:
        class FakeAgent:
            async def generate(
                self, *_args: object, **_kwargs: object
            ) -> AgentGeneration:
                parsed = ParsedModelTurn(
                    transcript="불 켜줘",
                    response="불을 켰습니다.",
                    raw_transcript="불 켜줘",
                    parse_status="valid_json",
                    structured_output_valid=True,
                    transcript_normalization="accepted",
                )
                return AgentGeneration(
                    output=parsed.output_json(),
                    parsed=parsed,
                    tool_attempted=True,
                    tool_succeeded=True,
                )

        observer = _RecordingObservability()
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                project_root=Path(temp_dir),
                runtime_dir=Path(temp_dir),
                omni_backend="server",
                tts_backend="command",
                tts_command=None,
            )
            result = asyncio.run(
                run_turn_pipeline(
                    b"wav",
                    settings,
                    omni_client=FakeAgent(),
                    observability=observer,
                )
            )

        self.assertEqual(result.user_text, "불 켜줘")
        self.assertEqual(
            [(item.name, item.parent) for item in observer.records],
            [("voice-turn", None), ("tts", "voice-turn")],
        )
        root = observer.records[0]
        self.assertEqual(
            root.scores,
            {
                "turn_succeeded": True,
                "structured_output_valid": True,
                "transcript_present": True,
                "tool_loop_limit_hit": False,
                "tool_calls_succeeded": True,
            },
        )

    def test_pipeline_records_failure_score_and_reraises(self) -> None:
        class FailingAgent:
            async def generate(
                self, *_args: object, **_kwargs: object
            ) -> AgentGeneration:
                raise RuntimeError("model failed")

        observer = _RecordingObservability()
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                project_root=Path(temp_dir),
                runtime_dir=Path(temp_dir),
                omni_backend="server",
                tts_backend="command",
            )
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                asyncio.run(
                    run_turn_pipeline(
                        b"wav",
                        settings,
                        omni_client=FailingAgent(),
                        observability=observer,
                    )
                )

        root = observer.records[0]
        self.assertEqual(root.scores, {"turn_succeeded": False})
        self.assertIn({"failed": "RuntimeError"}, root.updates)

    def test_tts_failure_keeps_model_quality_scores(self) -> None:
        class FakeAgent:
            async def generate(
                self, *_args: object, **_kwargs: object
            ) -> AgentGeneration:
                parsed = ParsedModelTurn(
                    transcript="안녕",
                    response="반가워요.",
                    parse_status="valid_json",
                    structured_output_valid=True,
                    transcript_normalization="accepted",
                )
                return AgentGeneration(output=parsed.output_json(), parsed=parsed)

        observer = _RecordingObservability()
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                project_root=Path(temp_dir),
                runtime_dir=Path(temp_dir),
                omni_backend="server",
                tts_backend="command",
            )
            with (
                patch("omni_iot.pipeline.run_tts", side_effect=RuntimeError("tts")),
                self.assertRaisesRegex(RuntimeError, "tts"),
            ):
                asyncio.run(
                    run_turn_pipeline(
                        b"wav",
                        settings,
                        omni_client=FakeAgent(),
                        observability=observer,
                    )
                )

        root = observer.records[0]
        self.assertFalse(root.scores["turn_succeeded"])
        self.assertTrue(root.scores["structured_output_valid"])
        self.assertTrue(root.scores["transcript_present"])
        tts = next(item for item in observer.records if item.name == "tts")
        self.assertIn({"failed": "RuntimeError"}, tts.updates)

    def test_agent_records_generation_tool_and_parse_observations(self) -> None:
        class FakeLlama:
            def __init__(self) -> None:
                self.completions = iter(
                    [
                        ChatCompletion(
                            content=None,
                            tool_calls=(ToolCall("call-1", "fixture__echo", "{}"),),
                        ),
                        ChatCompletion(
                            content=('{"transcript":"안녕","response":"반가워요."}'),
                            usage={"prompt_tokens": 10, "completion_tokens": 5},
                        ),
                    ]
                )

            def chat(self, *_args: object) -> ChatCompletion:
                return next(self.completions)

        class FakeMcp:
            async def prepare_turn(self) -> list[dict[str, object]]:
                return [
                    {
                        "type": "function",
                        "function": {
                            "name": "fixture__echo",
                            "parameters": {"type": "object"},
                        },
                    }
                ]

            async def call_tool(
                self,
                _name: str,
                _arguments: dict[str, object],
            ) -> McpCallResult:
                return McpCallResult(
                    content='{"content":"ok"}',
                    is_error=False,
                    server_name="fixture",
                    tool_name="echo",
                    seconds=0.01,
                )

        observer = _RecordingObservability()
        with tempfile.TemporaryDirectory() as temp_dir:
            audio = Path(temp_dir) / "input.wav"
            audio.write_bytes(b"wav")
            settings = Settings(runtime_dir=Path(temp_dir))
            agent = OmniAgent(
                FakeLlama(),  # type: ignore[arg-type]
                FakeMcp(),  # type: ignore[arg-type]
                settings,
                observer,
            )
            with observer.start_turn(
                turn_id="turn",
                session_id="session",
                audio_path=audio,
                history=[],
                metadata={},
            ):
                result = asyncio.run(agent.generate(audio, []))

        self.assertEqual(result.parsed.transcript, "안녕")
        self.assertEqual(
            [(item.name, item.parent) for item in observer.records],
            [
                ("voice-turn", None),
                ("omni-generation-1", "voice-turn"),
                ("mcp-tool", "voice-turn"),
                ("omni-generation-2", "voice-turn"),
                ("parse-model-turn", "voice-turn"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
