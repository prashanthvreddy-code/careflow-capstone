"""Long-term patient memory: what the assistant should still know about a patient next time. M3.

Design (decided in M3 phase 2):
- One JSON-lines file per patient: <memory_dir>/<patient_ref>.jsonl. Isolation is physical: a
  read for patient A only ever opens A's file. `patient_ref` is checked against a strict pattern
  before it becomes a file name, so a bad value can't reach another file.
- Only code writes memory, as structured rows (topic, outcome, short code-built detail). No LLM
  free text, names, member numbers or raw patient messages. Every row says where it came from
  (`source` + `source_id`), and writing the same source twice is a no-op.
- Recall returns the whole (short) history, newest first, so nothing is dropped by a top-k cut.
- Consolidation happens when reading, never by rewriting the file: a scheduled callback that was
  later resolved on the same topic is shown as "superseded" (kept for audit).
- `forget_patient` deletes the patient's file (data-handling policy: revocation within 24 h).
- The seed import runs once per store (`ensure_seeded` leaves a marker file), so a forgotten
  patient is not brought back by a later import.
- Coverage, plan and appointments are NOT remembered here; they are always looked up (M2 tools).
"""
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from capstone.schemas.intake import MEMBER_NUMBER
from capstone.settings import get_settings
from capstone.tools.repository import SeedTables

PATIENT_REF = re.compile(r"MHP-P-\d{5}")

Source = Literal["seed_interaction", "agent_booking", "agent_flag", "agent_outcome",
                 "workflow_outcome"]                     # M5: one row per workflow request
Status = Literal["open", "closed", "superseded"]

# Topics and outcomes: the 5 / 4 values in the seed interactions, plus what the M2 agent produces.
Topic = Literal[
    "appointment_booking", "eligibility_check", "referral_status", "billing_question",
    "clinical_question", "staff_review", "care_request",
]
Outcome = Literal[
    "resolved", "callback_scheduled", "escalated_to_clinician", "routed_to_billing",
    "booked", "flagged_for_staff", "slots_suggested", "no_action", "referral_routed",
]


class MemoryItem(BaseModel):
    """One remembered event about one patient."""

    memory_id: str                       # "<source>:<source_id>", so a re-import is a no-op
    patient_ref: str
    kind: Literal["episodic"] = "episodic"
    occurred_at: str                     # ISO date-time, as in the source
    topic: Topic
    outcome: Outcome
    detail: str | None = Field(None, max_length=200)   # short, code-built, never LLM text
    channel: str | None = None
    referral_id: str | None = None
    source: Source
    source_id: str
    created_at: str

    @field_validator("patient_ref")
    @classmethod
    def _ref_format(cls, v: str) -> str:
        if not PATIENT_REF.fullmatch(v):
            raise ValueError(f"not a patient_ref: {v!r}")
        return v

    @field_validator("detail")
    @classmethod
    def _no_member_number(cls, v: str | None) -> str | None:
        if v and MEMBER_NUMBER.search(v):
            raise ValueError("detail must not contain a member number")
        return v


class HistoryEntry(MemoryItem):
    """A MemoryItem as returned by `history()`, with its read-time status."""

    status: Status


def make_item(
    *, patient_ref: str, occurred_at: str, topic: str, outcome: str, source: str, source_id: str,
    detail: str | None = None, channel: str | None = None, referral_id: str | None = None,
) -> MemoryItem:
    """Build a MemoryItem with its deterministic id and a creation time."""
    return MemoryItem(
        memory_id=f"{source}:{source_id}", patient_ref=patient_ref, occurred_at=occurred_at,
        topic=topic, outcome=outcome, detail=detail, channel=channel, referral_id=referral_id,
        source=source, source_id=source_id, created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


class PatientMemory:
    """Per-patient memory files under one folder; tests point it at a temporary folder."""

    def __init__(self, memory_dir: Path):
        self.memory_dir = Path(memory_dir)

    def _path(self, patient_ref: str) -> Path:
        if not PATIENT_REF.fullmatch(patient_ref or ""):
            raise ValueError(f"not a patient_ref: {patient_ref!r}")
        return self.memory_dir / f"{patient_ref}.jsonl"

    def _read(self, patient_ref: str) -> list[MemoryItem]:
        path = self._path(patient_ref)
        if not path.exists():
            return []
        items = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            item = MemoryItem.model_validate_json(line)   # a corrupt line is an error, not skipped
            if item.patient_ref != patient_ref:
                raise ValueError(f"{path.name} line {n} belongs to {item.patient_ref}")
            items.append(item)
        return items

    def add(self, item: MemoryItem) -> bool:
        """Append one item. Returns False (and writes nothing) if its memory_id is already there."""
        if any(existing.memory_id == item.memory_id for existing in self._read(item.patient_ref)):
            return False
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        with open(self._path(item.patient_ref), "a", encoding="utf-8") as fh:
            fh.write(item.model_dump_json() + "\n")
        return True

    def history(self, patient_ref: str, limit: int | None = None) -> list[HistoryEntry]:
        """This patient's memory, newest first, each with status open / closed / superseded."""
        items = sorted(self._read(patient_ref), key=lambda i: i.occurred_at)
        entries = []
        for idx, item in enumerate(items):
            if item.outcome != "callback_scheduled":
                status = "closed"
            elif any(later.topic == item.topic and later.outcome == "resolved"
                     for later in items[idx + 1:]):
                status = "superseded"
            else:
                status = "open"
            entries.append(HistoryEntry(**item.model_dump(), status=status))
        entries.reverse()
        return entries[:limit] if limit else entries

    def patients(self) -> list[str]:
        """patient_refs that have a memory file."""
        if not self.memory_dir.exists():
            return []
        return sorted(p.stem for p in self.memory_dir.glob("*.jsonl") if PATIENT_REF.fullmatch(p.stem))

    def forget_patient(self, patient_ref: str) -> bool:
        """Delete everything remembered about this patient. Returns True if there was anything."""
        path = self._path(patient_ref)
        if path.exists():
            path.unlink()
            return True
        return False

    def import_seed(self, seed: SeedTables) -> dict[str, int]:
        """Load the seed `patient_interactions` table once. Safe to re-run (already-present rows skip).

        The seed `summary` field is not copied: it holds only 5 template sentences that repeat the
        topic, so it adds nothing.
        """
        added = skipped = 0
        for row in seed.rows("patient_interactions"):
            item = make_item(
                patient_ref=row["patient_ref"], occurred_at=row["occurred_at"], topic=row["topic"],
                outcome=row["outcome"], channel=row["channel"], referral_id=row.get("referral_id"),
                source="seed_interaction", source_id=row["interaction_id"],
            )
            if self.add(item):
                added += 1
            else:
                skipped += 1
        return {"added": added, "skipped": skipped}


SEED_MARKER = "_seed_imported.json"


def ensure_seeded(memory: PatientMemory, seed: SeedTables) -> dict[str, int] | None:
    """Import the seed interactions the first time only. Returns the counts, or None if done before."""
    marker = memory.memory_dir / SEED_MARKER
    if marker.exists():
        return None
    counts = memory.import_seed(seed)
    memory.memory_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({**counts, "at": datetime.now(UTC).isoformat(timespec="seconds")}))
    return counts


def default_memory() -> PatientMemory:
    """The real store, at runtime/memory/patients."""
    return PatientMemory(get_settings().memory_dir)
