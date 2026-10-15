"""M2 - Tool agent + co-pay calculator.

Part 1 (no LLM): the co-pay calculator. Money is computed by code from the plan tables, never by
the LLM; unknown plans or services raise an error instead of guessing.
Part 2 (live): one intake record through the tool-calling agent. The LLM decides which tools to
call (eligibility, appointments, open slots, co-pay, booking, flag for a human); code enforces the
booking rules, and the staff summary is reviewed before it is kept.
Live: 1 record, about 5-8 LLM calls. Writes only to a temporary folder.
Run:  python milestones/run_m2.py            (both parts)
      python milestones/run_m2.py --offline  (calculator only, no keys needed)
"""
import sys
import tempfile
from pathlib import Path

try:                                   # as a module (tests) or as a script (python milestones/...)
    from milestones import _common
except ImportError:
    import _common

QUOTES = (("MERIDIAN-SILVER", "consultation", 0),     # flat co-pay
          ("MERIDIAN-SILVER", "CARD-ECHO", 0),        # procedure billed as flat imaging
          ("MERIDIAN-BRONZE", "ORTH-ARTH", 0),        # deductible, then coinsurance
          ("MERIDIAN-BRONZE", "ORTH-ARTH", 3000))     # same, deductible already met
DEMO_ID = "CAREFLOW-00001"                            # Orthopaedics, routine, wants a booking


def quotes() -> list[dict]:
    from capstone.tools.calculators import calculate_copay
    from capstone.tools.repository import default_seed

    seed = default_seed()
    rows = [calculate_copay(seed, plan, service, met).model_dump() | {"deductible_met_usd": met}
            for plan, service, met in QUOTES]
    try:
        calculate_copay(seed, "NO-SUCH-PLAN", "consultation")
    except ValueError as e:
        rows.append({"refused": str(e)})
    return rows


def agent(llm, workdir: Path, record_id: str = DEMO_ID) -> dict:
    from capstone.agents.action_agent import process_record
    from capstone.memory.long_term import PatientMemory
    from capstone.memory.session import SessionStore
    from capstone.tools.repository import StateStore
    from capstone.workflows.graph import load_intake_records

    result = process_record(load_intake_records()[record_id], llm,          # every store in the temp folder
                            state=StateStore(workdir / "state"), memory=PatientMemory(workdir / "memory"),
                            sessions=SessionStore(workdir / "sessions"))
    return {"record_id": result.record_id, "patient_ref": result.patient_ref, "urgency": str(getattr(result.urgency, "value", result.urgency)),
            "tools": [c.tool for c in result.tool_calls],
            "eligible": result.eligibility.is_active if result.eligibility else None,
            "copay": result.copay.amount_usd if result.copay else None,
            "slots_offered": len(result.suggested_slots),
            "booked": result.booking is not None,
            "flags": [str(getattr(f.reason, "value", f.reason)) for f in result.flags],
            "summary": result.staff_summary}


def show_quotes(rows: list[dict]) -> None:
    _common.header("M2 Part 1: co-pay calculator (no LLM)")
    for r in rows:
        if "refused" in r:
            print(f"  unknown plan -> refused: {r['refused']}")
        else:
            print(f"  {r['plan_code']:16} {r['service']:13} deductible met ${r['deductible_met_usd']:<5} "
                  f"-> ${r['amount_usd']:.2f}  ({r['working']})")


def show_agent(r: dict) -> None:
    _common.header("M2 Part 2: tool agent on one record (live)", f"{r['record_id']} {r['patient_ref']}")
    print(f"  tools called  {', '.join(r['tools']) or 'none'}\n"
          f"  eligible {r['eligible']}  co-pay {r['copay']}  slots offered {r['slots_offered']}  "
          f"booked {r['booked']}\n  flags {r['flags'] or 'none'}\n  staff summary: {r['summary']}")


if __name__ == "__main__":
    show_quotes(quotes())
    if "--offline" not in sys.argv:
        with tempfile.TemporaryDirectory() as tmp:
            show_agent(agent(_common.live_llm(), Path(tmp)))
