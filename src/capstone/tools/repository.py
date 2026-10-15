"""Data access for the M2 tools. Two separate stores:

- `SeedTables`: the 14 mock tables in data/careflow/mock_api/*.json. READ-ONLY. Rows come back as
  read-only mappings, so no tool can change the seed data by accident.
- `StateStore`: what the agent changes (bookings, flags), appended to JSON-lines files in
  data/careflow/state/. Seed tables are never rewritten; tests point this at a temporary folder.
"""
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from capstone.settings import get_settings

TABLES = (
    "appointment_slots", "appointments", "patient_interactions", "patients", "plan_copays",
    "plans", "preauth_rules", "procedures", "providers", "referral_details", "referrals",
    "sites", "specialties", "urgency_bands",
)

StateKind = Literal["bookings", "flags", "redirects"]
STATE_KINDS: tuple[str, ...] = ("bookings", "flags", "redirects")      # redirects: improvement #3


class SeedTables:
    """Loads each table once on first use; every row is a read-only mapping."""

    def __init__(self, mock_api_dir: Path):
        self.mock_api_dir = Path(mock_api_dir)
        self._cache: dict[str, tuple[Mapping, ...]] = {}

    def rows(self, table: str) -> tuple[Mapping, ...]:
        """All rows of a table. Unknown table names are an error, not an empty result."""
        if table not in TABLES:
            raise KeyError(f"unknown table: {table}")
        if table not in self._cache:
            data = json.loads((self.mock_api_dir / f"{table}.json").read_text())
            self._cache[table] = tuple(MappingProxyType(row) for row in data)
        return self._cache[table]

    def find(self, table: str, **match) -> list[Mapping]:
        """Rows whose fields equal every given value, e.g. find("appointment_slots", specialty="ENT")."""
        return [row for row in self.rows(table) if all(row.get(k) == v for k, v in match.items())]

    def get(self, table: str, **match) -> Mapping | None:
        """The first matching row, or None if there is none."""
        found = self.find(table, **match)
        return found[0] if found else None


class StateStore:
    """Append-only JSON-lines files, one per kind: <state_dir>/bookings.jsonl, flags.jsonl."""

    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)

    def _path(self, kind: str) -> Path:
        if kind not in STATE_KINDS:
            raise KeyError(f"unknown state kind: {kind}")
        return self.state_dir / f"{kind}.jsonl"

    def append(self, kind: StateKind, record: dict) -> dict:
        """Add one record (one line) and return it."""
        path = self._path(kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def read(self, kind: StateKind) -> list[dict]:
        """All records of a kind, oldest first; empty if nothing was written yet."""
        path = self._path(kind)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def reset(self) -> None:
        """Delete all state files (bookings and flags). Seed tables are untouched."""
        for kind in STATE_KINDS:
            self._path(kind).unlink(missing_ok=True)


def default_seed() -> SeedTables:
    return SeedTables(get_settings().data_root / "mock_api")


def default_state() -> StateStore:
    return StateStore(get_settings().data_root / "state")
