"""M5 - LangGraph workflow with a human pause that survives a restart.

Each request runs through a graph: intake -> eligibility -> route (code decides: emergency,
clinical review, referral or staff queue). A clinical request PAUSES at the review step; its
state is saved to an on-disk checkpoint (SQLite). A different process later reopens the
checkpoint and resumes it with a reviewer's decision - nothing is lost across the restart.
Live: 3 records, about 1-2 LLM calls each (intake only; resuming needs no LLM).
Run:  python milestones/run_m5.py          (start here, then resume in a separate process)
      python milestones/run_m5.py start | resume | status
Demo files go to runtime/demo_m5/ (ignored by git), cleared on every full run.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

try:                                   # as a module (tests) or as a script (python milestones/...)
    from milestones import _common
except ImportError:
    import _common

DEMO_IDS = ("CAREFLOW-00068",       # asks a clinical question -> pauses for a clinician
            "CAREFLOW-00042",       # Routine specialist request -> referral path
            "CAREFLOW-00196")       # emergency signs -> emergency notice, never booked
WORKDIR = _common.ROOT / "runtime" / "demo_m5"
TRACES = _common.ROOT / "runtime" / "demo_traces" / "m5"


def _short(row: dict) -> dict:
    return {k: row.get(k) for k in ("record_id", "patient_ref", "status", "route", "expected_route",
                                    "waiting_for_review", "flags", "actions_done")} | (
        {"decision": row["answer"]["decision"]} if row.get("answer") else {})


def start(llm, workdir: Path = WORKDIR, ids: tuple[str, ...] = DEMO_IDS) -> list[dict]:
    from evaluation.runner import run_m5_start
    return [_short(r) for r in run_m5_start(llm, ids, workdir)]


def resume(workdir: Path = WORKDIR) -> list[dict]:
    from evaluation.runner import run_m5_resume
    return [_short(r) for r in run_m5_resume(workdir)]


def status(workdir: Path = WORKDIR) -> list[str]:
    from capstone.workflows.checkpoints import Checkpoints, paused_reviews
    from capstone.workflows.graph import load_intake_records
    from evaluation.runner import m5_workflow

    with Checkpoints(workdir / "checkpoints") as cp:
        return [p["record_id"] for p in paused_reviews(m5_workflow(None, cp, load_intake_records(), workdir), cp)]


def show(title: str, rows: list[dict]) -> None:
    _common.header(title)
    for r in rows:
        extra = f" reviewer decision={r['decision']}" if r.get("decision") else ""
        print(f"  {r['record_id']} {r['patient_ref']} status={r['status']:8} route={r['route']!s:15} "
              f"(key: {r['expected_route']}) waiting={r['waiting_for_review']}{extra}\n"
              f"     flags {r['flags'] or 'none'}; actions {r['actions_done'] or 'none'}")


def main(argv: list[str]) -> None:
    print(f"traces: {_common.cloud_tracing(TRACES)}")
    cmd = argv[1] if len(argv) > 1 else "all"
    if cmd == "resume":
        show(f"M5 resume in a separate process (waiting before: {status()})", resume())
        print(f"  still waiting after: {status() or 'none'}")
    elif cmd == "status":
        print(json.dumps({"waiting": status()}))
    else:
        if cmd == "all":
            shutil.rmtree(WORKDIR, ignore_errors=True)
        show("M5 start (live intake)", start(_common.live_llm()))
        if cmd == "all":
            sys.stdout.flush()                 # keep the two processes' output in order
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "resume"], check=True)


if __name__ == "__main__":
    main(sys.argv)
