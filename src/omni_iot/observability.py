from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from .config import Settings

ObservationType = Literal["span", "agent", "tool", "chain", "generation"]


class Observation(Protocol):
    def update(
        self,
        *,
        output: object | None = None,
        metadata: dict[str, object] | None = None,
        usage_details: dict[str, int] | None = None,
        level: Literal["DEFAULT", "WARNING", "ERROR"] | None = None,
        status_message: str | None = None,
    ) -> None: ...

    def fail(self, exc: BaseException) -> None: ...

    def score_trace(self, name: str, value: bool) -> None: ...


class AiObservability(Protocol):
    @property
    def enabled(self) -> bool: ...

    def authenticate(self) -> bool: ...

    def shutdown(self) -> None: ...

    def health(self) -> dict[str, object]: ...

    def start_turn(
        self,
        *,
        turn_id: str,
        session_id: str | None,
        audio_path: Path,
        history: list[dict[str, str]],
        metadata: dict[str, object],
    ) -> AbstractContextManager[Observation]: ...

    def start_observation(
        self,
        *,
        name: str,
        as_type: ObservationType = "span",
        input: object | None = None,
        metadata: dict[str, object] | None = None,
        model: str | None = None,
        model_parameters: dict[str, object] | None = None,
    ) -> AbstractContextManager[Observation]: ...

    def trace_value(self, value: object) -> object: ...


class _NoopObservation:
    def update(
        self,
        *,
        output: object | None = None,
        metadata: dict[str, object] | None = None,
        usage_details: dict[str, int] | None = None,
        level: Literal["DEFAULT", "WARNING", "ERROR"] | None = None,
        status_message: str | None = None,
    ) -> None:
        return None

    def fail(self, exc: BaseException) -> None:
        return None

    def score_trace(self, name: str, value: bool) -> None:
        return None


class NoopObservability:
    enabled = False

    def authenticate(self) -> bool:
        return True

    def shutdown(self) -> None:
        return None

    def health(self) -> dict[str, object]:
        return {
            "enabled": False,
            "configured": False,
            "backend": "none",
            "base_url": None,
            "environment": None,
        }

    @contextmanager
    def start_turn(
        self,
        *,
        turn_id: str,
        session_id: str | None,
        audio_path: Path,
        history: list[dict[str, str]],
        metadata: dict[str, object],
    ) -> Iterator[Observation]:
        yield _NoopObservation()

    @contextmanager
    def start_observation(
        self,
        *,
        name: str,
        as_type: ObservationType = "span",
        input: object | None = None,
        metadata: dict[str, object] | None = None,
        model: str | None = None,
        model_parameters: dict[str, object] | None = None,
    ) -> Iterator[Observation]:
        yield _NoopObservation()

    def trace_value(self, value: object) -> object:
        return value


class _LangfuseObservation:
    def __init__(self, observation: Any) -> None:
        self._observation = observation

    def update(
        self,
        *,
        output: object | None = None,
        metadata: dict[str, object] | None = None,
        usage_details: dict[str, int] | None = None,
        level: Literal["DEFAULT", "WARNING", "ERROR"] | None = None,
        status_message: str | None = None,
    ) -> None:
        values: dict[str, object] = {}
        if output is not None:
            values["output"] = redact(output)
        if metadata is not None:
            values["metadata"] = redact(metadata)
        if usage_details is not None:
            values["usage_details"] = usage_details
        if level is not None:
            values["level"] = level
        if status_message is not None:
            values["status_message"] = status_message
        self._observation.update(**values)

    def fail(self, exc: BaseException) -> None:
        self._observation.update(
            output={"error_type": type(exc).__name__},
            level="ERROR",
            status_message=f"{type(exc).__name__}: operation failed",
        )

    def score_trace(self, name: str, value: bool) -> None:
        self._observation.score_trace(
            name=name,
            value=1.0 if value else 0.0,
            data_type="BOOLEAN",
        )


class LangfuseObservability:
    def __init__(self, settings: Settings) -> None:
        from langfuse import Langfuse
        from opentelemetry.sdk.trace import TracerProvider

        self._base_url = settings.langfuse_base_url
        self._environment = settings.langfuse_environment
        self._tracer_provider = TracerProvider()
        self._client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url,
            environment=settings.langfuse_environment,
            sample_rate=1.0,
            tracer_provider=self._tracer_provider,
        )

    @property
    def enabled(self) -> bool:
        return True

    @property
    def api(self) -> Any:
        return self._client.api

    def authenticate(self) -> bool:
        return bool(self._client.auth_check())

    def shutdown(self) -> None:
        self._client.shutdown()

    def health(self) -> dict[str, object]:
        return {
            "enabled": True,
            "configured": True,
            "backend": "langfuse",
            "base_url": self._base_url,
            "environment": self._environment,
        }

    @contextmanager
    def start_turn(
        self,
        *,
        turn_id: str,
        session_id: str | None,
        audio_path: Path,
        history: list[dict[str, str]],
        metadata: dict[str, object],
    ) -> Iterator[Observation]:
        from langfuse import propagate_attributes
        from langfuse.api.media.types import MediaContentType
        from langfuse.media import LangfuseMedia

        trace_id = self._client.create_trace_id(seed=turn_id)
        root_input = {
            "turn_id": turn_id,
            "audio": LangfuseMedia(
                content_bytes=audio_path.read_bytes(),
                content_type=MediaContentType.AUDIO_WAV,
            ),
            "history": redact(history),
        }
        client = cast(Any, self._client)
        with client.start_as_current_observation(
            trace_context={"trace_id": trace_id},
            name="voice-turn",
            as_type="agent",
            input=root_input,
            metadata=redact(metadata),
        ) as span:
            with propagate_attributes(
                session_id=session_id,
                trace_name="voice-turn",
                environment=self._environment,
                metadata={"turn_id": turn_id},
            ):
                yield _LangfuseObservation(span)

    @contextmanager
    def start_observation(
        self,
        *,
        name: str,
        as_type: ObservationType = "span",
        input: object | None = None,
        metadata: dict[str, object] | None = None,
        model: str | None = None,
        model_parameters: dict[str, object] | None = None,
    ) -> Iterator[Observation]:
        client = cast(Any, self._client)
        with client.start_as_current_observation(
            name=name,
            as_type=as_type,
            input=redact(input) if input is not None else None,
            metadata=redact(metadata) if metadata is not None else None,
            model=model,
            model_parameters=cast(Any, model_parameters),
        ) as span:
            yield _LangfuseObservation(span)

    def trace_value(self, value: object) -> object:
        return prepare_trace_value(value)


def create_observability(settings: Settings) -> AiObservability:
    if not settings.ai_eval_enabled:
        return NoopObservability()
    return LangfuseObservability(settings)


def redact(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _is_sensitive_key(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
        return json.dumps(redact(decoded), ensure_ascii=False)
    return value


def prepare_trace_value(value: object) -> object:
    copied = copy.deepcopy(value)
    _replace_audio_data(copied)
    return redact(copied)


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().casefold().replace("-", "_")
    exact = {
        "authorization",
        "proxy_authorization",
        "token",
        "access_token",
        "refresh_token",
        "api_token",
        "secret",
        "client_secret",
        "password",
        "passwd",
        "api_key",
        "apikey",
    }
    return normalized in exact or normalized.endswith(
        ("_password", "_secret", "_api_key", "_access_token", "_refresh_token")
    )


def _replace_audio_data(value: object) -> None:
    if isinstance(value, dict):
        if value.get("type") == "input_audio":
            audio = value.get("input_audio")
            if isinstance(audio, dict) and "data" in audio:
                audio["data"] = "[see voice-turn input audio]"
        for item in value.values():
            _replace_audio_data(item)
    elif isinstance(value, list):
        for item in value:
            _replace_audio_data(item)
