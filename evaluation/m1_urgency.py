"""Improvement #2: M1 urgency on fixed sets, before vs after a prompt change (intake step only).

Sets (fixed, no draw at run time):
- unseen  = the 76 intake records no earlier test used or inspected (used_records.json) and not in
            the M8 random-30 sample: 45 Routine, 23 Priority, 8 Urgent by the answer key;
- safety  = every Urgent / Emergent record in the data (33): the under-triage gate.
One LLM call per record (intake extraction only; urgency = the code's highest band per message).
Over-triage = a more urgent band than the key; under-triage = a less urgent one (the dangerous way).
"""
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from capstone.settings import PROJECT_ROOT

ORDER = {"Routine": 0, "Priority": 1, "Urgent": 2, "Emergent": 3}
OUT_DIR = PROJECT_ROOT / "evaluation" / "reports" / "m1_urgency"
USED = PROJECT_ROOT / "evaluation" / "datasets" / "used_records.json"


def random_patient_ids() -> list[str]:
    from capstone.workflows.graph import load_intake_records
    from evaluation.m8_random import draw_patients
    used = set(json.loads(USED.read_text())["record_ids"])
    return draw_patients(load_intake_records(), used)


def eval_sets(records: dict) -> tuple[list[str], list[str]]:
    used = set(json.loads(USED.read_text())["record_ids"]) | set(random_patient_ids())
    unseen = sorted(r for r in records if r not in used)
    safety = sorted(r for r, rec in records.items() if rec["ground_truth"]["urgency"] in ("Urgent", "Emergent"))
    return unseen, safety


def row(record_id: str, truth: str, predicted: str) -> dict:
    t, p = ORDER.get(truth), ORDER.get(predicted)
    return {"record_id": record_id, "truth": truth, "predicted": predicted,
            "result": "unknown" if p is None else "exact" if p == t else "over" if p > t else "under"}


def summarise(rows: list[dict]) -> dict:
    c = Counter(r["result"] for r in rows)
    confusion: dict[str, dict[str, int]] = {}
    for r in rows:
        confusion.setdefault(r["truth"], Counter())[r["predicted"]] += 1
    return {"n": len(rows), "exact": c["exact"], "over": c["over"], "under": c["under"], "unknown": c["unknown"],
            "under_urgent_or_emergent": sum(r["result"] in ("under", "unknown") and r["truth"] in ("Urgent", "Emergent")
                                            for r in rows),
            "routine_marked_higher": sum(r["truth"] == "Routine" and r["result"] == "over" for r in rows),
            "confusion": {k: dict(v) for k, v in confusion.items()}}


def run(label: str, out_dir: Path = OUT_DIR) -> dict:
    from capstone.agents.intake_agent import parse_intake
    from capstone.llm.client import LLMClient
    from capstone.settings import get_settings
    from capstone.workflows.graph import load_intake_records

    records = load_intake_records()
    unseen, safety = eval_sets(records)
    llm, cache, report = LLMClient(get_settings()), {}, {}
    for name, ids in (("unseen", unseen), ("safety", safety)):
        rows = []
        for rid in ids:
            if rid not in cache:
                try:
                    cache[rid] = parse_intake(records[rid], llm).urgency.value
                except Exception as e:  # noqa: BLE001 - a failed extraction is reported, never guessed
                    cache[rid] = f"error:{type(e).__name__}"
            rows.append(row(rid, records[rid]["ground_truth"]["urgency"], cache[rid]))
        report[name] = {"summary": summarise(rows), "rows": rows}
    report |= {"label": label, "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
               "model": get_settings().llm_model, "llm_calls": len(cache)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"m1_urgency_{label}.json").write_text(json.dumps(report, indent=1))
    return report
