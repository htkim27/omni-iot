from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    created_at: float = field(default_factory=time.time)


@dataclass
class ConversationSession:
    id: str
    messages: list[Message] = field(default_factory=list)

    def add_user_audio_turn(self) -> None:
        self.messages.append(Message(role="user", content="[voice input]"))

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
