"""M7 - Tracing + reliability under failure. No real LLM or API calls (runs without keys).

The same records run through the full workflow under seeded faults, BEFORE (M6 behaviour) and
AFTER the M7 hardening:
  none             - no faults (the baseline)
  flaky_ehr        - the EHR/scheduling MCP server fails some calls or times out
  ehr_down         - the EHR is down for the whole run
  llm_primary_down - the primary LLM provider is down (the fallback model must answer)
AFTER adds retries with backoff, circuit breakers, strict reply checks, a staff fallback with a
dependency flag, and booking verification. Every step and LLM call is a trace span (metadata
only, no patient text). The LLM is replaced by a stand-in that answers from the answer key, so
only the reliability behaviour is measured. "silent" = a result that changed with NOTHING visible
(no crash, no flag) - the confident wrong answer M7 must remove.
Run:  python milestones/run_m7.py          (10 records, about 10 seconds)
      python milestones/run_m7.py --full   (the measured 50 records, as in the M7 report)
"""
import sys

try:                                   # as a module (tests) or as a script (python milestones/...)
    from milestones import _common
except ImportError:
    import _common

DEMO_N = 10


def scenarios(n: int = DEMO_N, names: tuple[str, ...] | None = None) -> list[dict]:
    from capstone.tools.repository import default_seed
    from capstone.workflows.graph import load_intake_records
    from evaluation import m7

    records, seed = load_intake_records(), default_seed()
    ids = m7.offline_sample(records)[:n]
    baseline = m7.run_offline("none", "after", ids, records, seed)
    rows = []
    for scenario in names or m7.SCENARIO_NAMES:
        for variant in ("before", "after"):
            run = baseline if (scenario, variant) == ("none", "after") else \
                m7.run_offline(scenario, variant, ids, records, seed)
            t, o = run["totals"], run["totals"]["outcomes"]
            rows.append({"scenario": scenario, "variant": variant, "n": len(ids),
                         "done": o.get("done", 0), "paused": o.get("paused", 0), "crashed": o.get("crashed", 0),
                         "to_staff": t["staff_dependency_flags"], "booked": t["booked"],
                         "duplicate_bookings": t["duplicate_bookings"],
                         "silent_wrong": len(m7.silent_wrong(baseline, run)),
                         "short_circuited": t["short_circuited"], "fallback_llm_answers": t["fallback_llm_answers"]})
    return rows


def show(rows: list[dict]) -> None:
    _common.header(f"M7 failure scenarios on {rows[0]['n']} records (no real LLM calls)")
    print(f"  {'scenario':17} {'variant':7} {'done':>4} {'paused':>6} {'crash':>5} {'staff':>5} {'booked':>6} "
          f"{'dup':>3} {'silent':>6} {'breaker':>7} {'fallback':>8}")
    for r in rows:
        print(f"  {r['scenario']:17} {r['variant']:7} {r['done']:>4} {r['paused']:>6} {r['crashed']:>5} "
              f"{r['to_staff']:>5} {r['booked']:>6} {r['duplicate_bookings']:>3} {r['silent_wrong']:>6} "
              f"{r['short_circuited']:>7} {r['fallback_llm_answers']:>8}")
    print("  staff = handed to staff with a dependency flag; breaker = calls skipped by an open circuit breaker;\n"
          "  fallback = answers from the fallback model; silent = changed result with nothing visible")


if __name__ == "__main__":
    show(scenarios(50 if "--full" in sys.argv else DEMO_N))
