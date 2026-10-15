"""Durable checkpoints for the M5 workflow: a SQLite file that survives a restart. M5.

Decided in M5 phase 2 / A5 Lab B:
- SqliteSaver at runtime/checkpoints/workflow.sqlite (runtime/ is gitignored). The in-memory
  saver dies with the process, so it cannot meet "resume after a restart".
- One thread per request: thread_id = record_id (CAREFLOW-NNNNN). A thread is never shared, so
  one reviewer's answer can never resume another patient's request.
- Strict loading: the file is read back into objects on resume, so anyone who can write to it
  could influence what is rebuilt (A5 B7). The saver only accepts LangGraph's built-in safe
  types; the state holds plain values only (schemas/state.py), so nothing else is needed.
- A checkpoint saves STATE, never code. If the state's fields change after checkpoints were
  written, old paused runs no longer fit. A small sidecar file records the state fields the
  checkpoints were written with, and opening refuses a mismatch with a clear message (the same
  idea as the M3 index refusing a different embedding model).
- Data handling: `forget_patient` deletes every thread of one patient (pairs with
  PatientMemory.forget_patient and SessionStore.forget_patient).
"""
import json
import re
import sqlite3
from pathlib import Path
from typing import Self

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from capstone.schemas.state import WorkflowState
from capstone.settings import get_settings

RECORD_ID = re.compile(r"CAREFLOW-\d{5}")
FILE_NAME = "workflow.sqlite"
META_NAME = "workflow.meta.json"


def default_checkpoint_dir() -> Path:
    return get_settings().runtime_root / "checkpoints"


def state_fields() -> list[str]:
    return sorted(WorkflowState.__annotations__)


def thread_config(record_id: str) -> dict:
    """The LangGraph config for one request. The thread id IS the record id."""
    if not RECORD_ID.fullmatch(record_id or ""):
        raise ValueError(f"not a record id: {record_id!r}")
    return {"configurable": {"thread_id": record_id}}


class SchemaMismatchError(RuntimeError):
    """Checkpoints on disk were written with different state fields; old runs cannot resume."""


class Checkpoints:
    """The checkpoint file. Use as `with Checkpoints() as cp: graph = build_graph(cp.saver, ...)`."""

    def __init__(self, folder: Path | None = None):
        self.folder = Path(folder) if folder else default_checkpoint_dir()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / FILE_NAME
        self._check_schema()
        # check_same_thread=False: LangGraph may use the connection from a worker thread (A5 B7)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.saver = SqliteSaver(self.conn, serde=JsonPlusSerializer(allowed_msgpack_modules=None))
        self.saver.setup()                       # creates the tables; safe to repeat

    def _check_schema(self) -> None:
        meta = self.folder / META_NAME
        fields = state_fields()
        if meta.exists():
            written = json.loads(meta.read_text()).get("state_fields")
            if written != fields:
                raise SchemaMismatchError(
                    "the checkpoints were written with different state fields; finish or clear the "
                    f"paused runs before changing the state (added {sorted(set(fields) - set(written or []))}, "
                    f"removed {sorted(set(written or []) - set(fields))})")
        else:
            meta.write_text(json.dumps({"state_fields": fields}, indent=2))

    def thread_ids(self) -> list[str]:
        """Every request with at least one checkpoint, sorted."""
        rows = self.conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
        return sorted(r[0] for r in rows)

    def delete(self, record_id: str) -> None:
        self.saver.delete_thread(thread_config(record_id)["configurable"]["thread_id"])

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def paused_reviews(graph, checkpoints: Checkpoints) -> list[dict]:
    """The review payloads of every request waiting for a clinical reviewer (feed overdue_reviews)."""
    out = []
    for record_id in checkpoints.thread_ids():
        snap = graph.get_state(thread_config(record_id))
        if "review" in snap.next and snap.values.get("review"):
            out.append(snap.values["review"])
    return out


def forget_patient(graph, checkpoints: Checkpoints, patient_ref: str) -> list[str]:
    """Delete every thread of one patient. Returns the record ids deleted."""
    gone = []
    for record_id in checkpoints.thread_ids():
        if graph.get_state(thread_config(record_id)).values.get("patient_ref") == patient_ref:
            checkpoints.delete(record_id)
            gone.append(record_id)
    return gone
