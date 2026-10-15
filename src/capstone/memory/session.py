"""Session memory: what the assistant holds while handling ONE request. M3.

Design (decided in M3 phase 2):
- A session is one request run (one intake record through the M2 agent). It holds the messages
  the agent exchanged with the LLM and its tools, in order.
- While open it lives in the program; `close()` saves it to
  <sessions_dir>/<patient_ref>/<session_id>.json so it survives a restart and can be audited.
  A closed session is read-only, and a session is never reused for another request.
- Isolation: sessions sit in a folder per patient, and loading checks the patient matches.
  `forget_patient` deletes the patient's folder (pairs with PatientMemory.forget_patient).
- Size: an approximate token count is kept; going over the budget is reported (`over_budget`),
  never silently cut (A3: truncation deletes what the user assumes you still know). M2 runs are
  at most 8 rounds, so no LLM summary is needed.
- Defence in depth: member numbers are masked in any message before it is stored. The M2 tools
  never return them, so this should never fire; `redactions` counts it if it does.
"""
import json
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from capstone.memory.long_term import PATIENT_REF
from capstone.schemas.intake import MEMBER_NUMBER
from capstone.settings import get_settings

SESSION_ID = re.compile(r"S-[0-9a-f]{12}")
ROLES = {"system", "user", "assistant", "tool"}
CHARS_PER_TOKEN = 4          # rough estimate (A3); good enough for a budget warning
MASK = "[MEMBER-NUMBER]"


class SessionClosedError(RuntimeError):
    """Raised when something tries to change a session that was already closed."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _mask(value: Any) -> tuple[Any, int]:
    """Mask member numbers anywhere in a message (strings inside dicts/lists too)."""
    if isinstance(value, str):
        return MEMBER_NUMBER.subn(MASK, value)
    if isinstance(value, dict):
        out, n = {}, 0
        for k, v in value.items():
            out[k], c = _mask(v)
            n += c
        return out, n
    if isinstance(value, list):
        items = [_mask(v) for v in value]
        return [i for i, _ in items], sum(c for _, c in items)
    return value, 0


class Session(BaseModel):
    session_id: str
    patient_ref: str
    record_id: str
    started_at: str
    closed_at: str | None = None
    token_budget: int
    messages: list[dict[str, Any]] = []
    redactions: int = 0

    @property
    def is_closed(self) -> bool:
        return self.closed_at is not None

    @property
    def approx_tokens(self) -> int:
        return len(json.dumps(self.messages)) // CHARS_PER_TOKEN

    @property
    def over_budget(self) -> bool:
        return self.approx_tokens > self.token_budget

    def add(self, message: dict[str, Any]) -> None:
        """Append one message (a copy, with member numbers masked)."""
        if self.is_closed:
            raise SessionClosedError(f"session {self.session_id} is closed")
        if message.get("role") not in ROLES:
            raise ValueError(f"unknown message role: {message.get('role')!r}")
        masked, n = _mask(message)
        self.messages.append(masked)
        self.redactions += n

    def extend(self, messages: list[dict[str, Any]]) -> None:
        for m in messages:
            self.add(m)


class SessionStore:
    """Saved sessions, one folder per patient; tests point it at a temporary folder."""

    def __init__(self, sessions_dir: Path, token_budget: int = 8000):
        self.sessions_dir = Path(sessions_dir)
        self.token_budget = token_budget

    def _patient_dir(self, patient_ref: str) -> Path:
        if not PATIENT_REF.fullmatch(patient_ref or ""):
            raise ValueError(f"not a patient_ref: {patient_ref!r}")
        return self.sessions_dir / patient_ref

    def _path(self, patient_ref: str, session_id: str) -> Path:
        if not SESSION_ID.fullmatch(session_id or ""):
            raise ValueError(f"not a session_id: {session_id!r}")
        return self._patient_dir(patient_ref) / f"{session_id}.json"

    def open(self, patient_ref: str, record_id: str) -> Session:
        """Start a new, empty session for one request. Nothing is written until close()."""
        self._patient_dir(patient_ref)          # validate early
        return Session(session_id=f"S-{uuid.uuid4().hex[:12]}", patient_ref=patient_ref,
                       record_id=record_id, started_at=_now(), token_budget=self.token_budget)

    def close(self, session: Session) -> Path:
        """Mark the session closed and save it. Closing twice is an error."""
        if session.is_closed:
            raise SessionClosedError(f"session {session.session_id} is already closed")
        path = self._path(session.patient_ref, session.session_id)
        session.closed_at = _now()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(session.model_dump_json(indent=1), encoding="utf-8")
        tmp.replace(path)                       # all-or-nothing: no half-written file
        return path

    def load(self, patient_ref: str, session_id: str) -> Session:
        """Load one closed session of this patient. Another patient's session id is not found."""
        path = self._path(patient_ref, session_id)
        if not path.exists():
            raise FileNotFoundError(f"no session {session_id} for {patient_ref}")
        session = Session.model_validate_json(path.read_text(encoding="utf-8"))
        if session.patient_ref != patient_ref or session.session_id != session_id:
            raise ValueError(f"{path.name} does not belong to {patient_ref}")
        return session

    def list_for_patient(self, patient_ref: str) -> list[str]:
        """This patient's saved session ids, oldest first."""
        folder = self._patient_dir(patient_ref)
        if not folder.exists():
            return []
        sessions = [self.load(patient_ref, p.stem) for p in folder.glob("*.json")
                    if SESSION_ID.fullmatch(p.stem)]
        return [s.session_id for s in sorted(sessions, key=lambda s: s.started_at)]

    def forget_patient(self, patient_ref: str) -> int:
        """Delete all saved sessions of this patient. Returns how many there were."""
        folder = self._patient_dir(patient_ref)
        if not folder.exists():
            return 0
        count = len(list(folder.glob("*.json")))
        shutil.rmtree(folder)
        return count


def default_sessions() -> SessionStore:
    """The real store, at runtime/memory/sessions."""
    s = get_settings()
    return SessionStore(s.sessions_dir, s.session_token_budget)

