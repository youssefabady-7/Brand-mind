"""Session Memory Manager.

Tracks:
1. Last N conversational turns per session_id (configurable via .env, default 5).
2. Entity memory for resolving contextual co-references (e.g. 'the black one', 'it', previously mentioned order_id).
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

DEFAULT_MAX_TURNS = int(os.getenv("MAX_MEMORY_TURNS", "5"))


class ConversationTurn(BaseModel):
    """Single conversational turn in a session."""
    user_message: str
    assistant_reply: str
    intent: str
    entities: Dict[str, Any] = Field(default_factory=dict)
    tool_called: Optional[str] = None
    escalated: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionMemory(BaseModel):
    """Short-term conversational state for a user session."""
    session_id: str
    brand_id: str
    turns: List[ConversationTurn] = Field(default_factory=list)
    last_mentioned_entities: Dict[str, Any] = Field(default_factory=dict)

    def add_turn(self, turn: ConversationTurn, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        """Appends a turn and preserves the last N turns."""
        self.turns.append(turn)
        if len(self.turns) > max_turns:
            self.turns = self.turns[-max_turns:]

        # Update last-mentioned entities
        for key, value in turn.entities.items():
            if value is not None and value != "":
                self.last_mentioned_entities[key] = value

    def get_recent_history(self, n: Optional[int] = None) -> List[Dict[str, str]]:
        """Returns simplified chat history suitable for prompt injection."""
        limit = n or len(self.turns)
        selected_turns = self.turns[-limit:] if limit > 0 else []
        return [
            {"user": t.user_message, "assistant": t.assistant_reply}
            for t in selected_turns
        ]

    def resolve_entity(self, key: str, current_value: Optional[Any] = None) -> Optional[Any]:
        """Resolves an entity: returns current_value if present, else recalls from memory."""
        if current_value is not None and current_value != "":
            return current_value
        return self.last_mentioned_entities.get(key)


class MemoryManager:
    """In-memory multi-tenant session storage."""

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS):
        self.max_turns = max_turns
        self._sessions: Dict[str, SessionMemory] = {}

    def get_or_create_session(self, session_id: str, brand_id: str) -> SessionMemory:
        """Retrieves active session or initializes a fresh session state."""
        key = f"{brand_id.lower()}_{session_id}"
        if key not in self._sessions:
            self._sessions[key] = SessionMemory(session_id=session_id, brand_id=brand_id)
        return self._sessions[key]

    def clear_session(self, session_id: str, brand_id: str) -> None:
        """Clears memory for a specific session."""
        key = f"{brand_id.lower()}_{session_id}"
        if key in self._sessions:
            del self._sessions[key]
