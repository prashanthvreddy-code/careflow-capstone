"""M6 - Agent team + MCP.

Part 1 (no LLM): the EHR / scheduling system as an MCP server, started as a SEPARATE process
(stdio). The client lists its tools and checks one patient's coverage through it; the workflow
reaches patient data only through these tools.
Part 2 (live): two requests through the agent team inside the M5 graph -
  Intake agent -> Insurance agent (coverage over MCP, co-pay by the calculator, cited policy answer
  from the M4 PolicyAgent) -> Referral tracking agent (sign-off, booking rules, overdue referrals)
  -> Clinical Safety Reviewer (checks any patient-facing text and every booking; clinical content
  goes to a clinician). A clinical request pauses for a person, then resumes in a new process.
Live: 2 records, about 8-12 LLM calls.
Run:  python milestones/run_m6.py            (both parts; needs the index for the policy answer)
      python milestones/run_m6.py --offline  (Part 1 only, no keys needed)
Demo files go to runtime/demo_m6/ (ignored by git), cleared on every full run.
"""
import shutil
import subprocess
import sys
from pathlib import Path

try:                                   # as a module (tests) or as a script (python milestones/...)
    from milestones import _common
except ImportError:
    import _common

DEMO_IDS = ("CAREFLOW-00090",       # Dermatology, open access -> insurance answer, safety check, booked
            "CAREFLOW-00155")       # asks a clinical question -> safety reviewer escalates, pauses
WORKDIR = _common.ROOT / "runtime" / "demo_m6"
TRACES = _common.ROOT / "runtime" / "demo_traces" / "m6"
MCP_PATIENT = "MHP-P-00236"


def mcp_demo(patient_ref: str = MCP_PATIENT) -> dict:
    from capstone.mcp.client import EHRClient, stdio_params

    client = EHRClient(stdio=stdio_params())
    e = client.check_eligibility(patient_ref)
    return {"transport": client.transport, "tools": client.tool_names(), "patient_ref": patient_ref,
            "plan": e.plan_code, "status": e.eligibility_status, "active": e.is_active}


def _short(row: dict) -> dict:
    ins, tr, sf = row.get("insurance") or {}, row.get("tracking") or {}, row.get("safety") or {}
    return {"record_id": row["record_id"], "patient_ref": row["patient_ref"], "status": row["status"],
            "route": row["route"], "expected_route": row["expected_route"],
            "plan": ins.get("plan"), "copay": ins.get("copay"), "policy_route": ins.get("policy_route"),
            "policy_answer": ins.get("policy_answer"), "citations": ins.get("citations", []),
            "booking": tr.get("booking"), "safety": sf.get("reviews"), "booked": row.get("booking"),
            "flags": row.get("flags"), "decision": (row.get("answer") or {}).get("decision")}


def team(command: str, llm, workdir: Path = WORKDIR, ids: tuple[str, ...] = DEMO_IDS) -> list[dict]:
    from evaluation.runner import run_m6
    return [_short(r) for r in run_m6(command, llm, workdir, ids)]


def show_mcp(r: dict) -> None:
    _common.header("M6 Part 1: EHR / scheduling MCP server in a separate process (no LLM)")
    print(f"  transport {r['transport']}; tools: {', '.join(r['tools'])}\n"
          f"  check_eligibility({r['patient_ref']}) -> plan {r['plan']}, status {r['status']}, covered {r['active']}")


def show_team(title: str, rows: list[dict]) -> None:
    _common.header(title)
    for r in rows:
        print(f"  {r['record_id']} {r['patient_ref']} status={r['status']} route={r['route']} (key: {r['expected_route']})"
              + (f" reviewer decision={r['decision']}" if r["decision"] else ""))
        print(f"     insurance: plan {r['plan']}, co-pay {r['copay']}, policy answer {r['policy_route']}")
        if r["policy_answer"]:
            print(f"       \"{r['policy_answer']}\"\n       cites {r['citations']}")
        print(f"     tracking: booking {r['booking']}; booked {r['booked']}\n"
              f"     safety reviews: {r['safety']}\n     flags: {r['flags'] or 'none'}")


def main(argv: list[str]) -> None:
    print(f"traces: {_common.cloud_tracing(TRACES)}")
    if "resume" in argv:
        show_team("M6 resume in a separate process", team("m6-resume", _common.live_llm()))
        return
    show_mcp(mcp_demo())
    if "--offline" in argv:
        return
    shutil.rmtree(WORKDIR, ignore_errors=True)
    show_team("M6 Part 2: agent team (live)", team("m6-start", _common.live_llm()))
    sys.stdout.flush()                 # keep the two processes' output in order
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "resume"], check=True)


if __name__ == "__main__":
    main(sys.argv)
