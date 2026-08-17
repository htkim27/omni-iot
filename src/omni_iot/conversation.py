from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


_TRANSCRIPT_PLACEHOLDERS = frozenset(
    {
        "사용자가 실제로 말한 내용",
        "사용자의 실제 발화 내용",
        "actual words spoken by the user",
        "user's actual spoken words",
    }
)


def normalize_user_transcript(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or normalized.casefold() in _TRANSCRIPT_PLACEHOLDERS:
        return None
    return normalized


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    created_at: float = field(default_factory=time.time)


@dataclass
class ConversationSession:
    id: str
    messages: list[Message] = field(default_factory=list)

    def add_user_audio_turn(self, transcript: str | None = None) -> None:
        transcript = normalize_user_transcript(transcript)
        self.messages.append(
            Message(role="user", content=transcript or "[voice input]")
        )

    def add_assistant_message(self, text: str) -> None:
        self.messages.append(Message(role="assistant", content=text))

    def transcript(self) -> list[dict[str, str | float]]:
        return [
            {
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at,
            }
            for message in self.messages
        ]

    def prompt_history(self, max_messages: int) -> list[dict[str, str]]:
        if max_messages <= 0:
            return []
        return [
            {"role": message.role, "content": message.content}
            for message in self.messages[-max_messages:]
        ]


class ConversationStore:
    def __init__(self) -> None:
        self._sessions: dict[str, ConversationSession] = {}

    def get(self, session_id: str | None) -> ConversationSession:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        new_session = ConversationSession(id=session_id or uuid.uuid4().hex)
        self._sessions[new_session.id] = new_session
        return new_session

    def reset(self, session_id: str | None) -> ConversationSession:
        new_session = ConversationSession(id=session_id or uuid.uuid4().hex)
        self._sessions[new_session.id] = new_session
        return new_session
