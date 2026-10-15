"""M7 measurement: seeded fault scenarios (offline, 0 LLM calls) and the small live run.

Offline (phase 2 decision 7): the same fixed 50 records under 4 seeded scenarios, BEFORE vs
AFTER hardening. Every LLM request goes through the real LLMClient (model chain, retries,
breakers, spans) but LiteLLM is replaced by TruthProvider, which answers from the answer key,
so no API is ever called.
- BEFORE = M6 behaviour: one attempt per EHR call, no breaker, lax reply parsing, errors not
  caught by the workflow (a request that hits one stops mid-run); LLM: primary once, then the
  fallback (what LiteLLM's `fallbacks=` did), no breaker.
- AFTER  = M7: retries, breakers, strict replies, staff fallback, booking verification.
Both get the same seeded faults and the same 5 s / 30 s simulated timeouts (generous to BEFORE:
M6 had no timeout at all; a Gemini call once hung ~9 min). Breakers run on a simulated clock:
one request every 10 s, plus the retry waits and would-be timeouts, so cool-downs really end.

Live (`m7-live`): 6 never-used records with the real models and Langfuse Cloud; 3 clean and 3
with a seeded EHR fault (one fault plan per record, fresh breakers per record).
"""
import json
import random
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from capstone.settings import PROJECT_ROOT

M7_DIR = PROJECT_ROOT / "evaluation" / "reports" / "m7_run"
OFFLINE_N, OFFLINE_SEED = 50, 7
ARRIVAL_GAP_S = 10.0        # simulated time between two requests (breaker cool-downs run on this clock)
PROMPT_SCHEMAS = {"intake extraction": "IntakeExtraction", "policy assistant": "LLMAnswer",
                  "grounding check": "GroundingVerdict", "Clinical Safety Reviewer": "LLMVerdict"}

# Live sample: never used in any earlier live run (scratchpad used.txt + M5/M6 samples).
M7_CLEAN = ("CAREFLOW-00099", "CAREFLOW-00166", "CAREFLOW-00060")
M7_FAULTED = {"CAREFLOW-00130": "flaky_ehr", "CAREFLOW-00191": "ehr_down", "CAREFLOW-00150": "book_slot_down"}


def offline_sample(records: dict) -> list[str]:
    return sorted(random.Random(OFFLINE_SEED).sample(sorted(records), OFFLINE_N))


# --- the fake provider: answers from the answer key ------------------------------------------------
class TruthLLM:
    """Intake as the answer key says (wants a booking); policy answers state the plan's real co-pay."""

    def __init__(self, records, seed):
        self.by_text, self.seed = {r["raw_text"]: r for r in records.values()}, seed

    def complete_structured(self, messages, schema):
        from capstone.agents.clinical_safety_reviewer import LLMVerdict
        from capstone.rag.answer import GroundingVerdict, LLMAnswer
        from capstone.schemas.intake import IntakeExtraction, Symptom
        from capstone.tools.calculators import calculate_copay

        content = messages[-1]["content"]
        if schema is IntakeExtraction:
            rec = next(r for t, r in self.by_text.items() if t in content)
            g = rec["ground_truth"]
            return IntakeExtraction(symptoms=[Symptom(text="symptom", urgency=g["urgency"])], specialty=g["specialty"],
                                    insurance_id=rec["insurance_id_stated"], wants_booking=True,
                                    seeks_clinical_advice=g["seeks_clinical_advice"])
        if schema is LLMAnswer:
            plan = json.loads(content)["question"].split()[2]
            copay = calculate_copay(self.seed, plan, "consultation").amount_usd
            return LLMAnswer(route="answered", answer=f"The specialist consultation co-pay is ${copay:.0f}.",
                             citations=["copay-schedule#4.2"])
        if schema is GroundingVerdict:
            return GroundingVerdict(grounded=True)
        if schema is LLMVerdict:
            return LLMVerdict(verdict="approve")
        raise AssertionError(schema)


class TruthProvider:
    """Stands in for litellm.completion: picks the schema from the system prompt, answers truthfully."""

    def __init__(self, truth: TruthLLM):
        self.truth, self.calls = truth, Counter()

    def __call__(self, **kw):
        from capstone.agents.clinical_safety_reviewer import LLMVerdict
        from capstone.rag.answer import GroundingVerdict, LLMAnswer
        from capstone.schemas.intake import IntakeExtraction

        schemas = {"IntakeExtraction": IntakeExtraction, "LLMAnswer": LLMAnswer,
                   "GroundingVerdict": GroundingVerdict, "LLMVerdict": LLMVerdict}
        system = kw["messages"][0]["content"]
        name = next(s for key, s in PROMPT_SCHEMAS.items() if key in system)
        self.calls[kw["model"]] += 1
        out = self.truth.complete_structured(kw["messages"], schemas[name])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=out.model_dump_json(),
                                                                                tool_calls=None))],
                               model=kw["model"], usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0))


# --- BEFORE: the M6 client (no retry, no breaker, lax replies, raw errors) ------------------------
def unhardened_ehr(server, reliability):
    from capstone.mcp.client import EHRClient
    from capstone.tools.actions import BookingResult, Flag
    from capstone.tools.lookups import AppointmentInfo, EligibilityResult, SlotInfo
    from capstone.tools.referrals import ReferralStatus

    one = {"check_eligibility": EligibilityResult, "book_slot": BookingResult, "flag_for_human": Flag}
    many = {"get_patient_appointments": AppointmentInfo, "get_patient_referrals": ReferralStatus,
            "find_open_slots": SlotInfo}

    class UnhardenedEHR(EHRClient):
        def call(self, name, parse=None, **args):
            data = self._attempt(name, args, None)          # injected faults, real call, raw errors
            if name in one:
                return one[name].model_validate(data)        # lax: missing fields get defaults
            if name in many:
                return [many[name].model_validate(x) for x in data]
            return list(data) if name == "check_booking" else data

    return UnhardenedEHR(server, reliability=reliability)


def before_config():
    from capstone.reliability.circuit_breaker import BreakerPolicy
    from capstone.reliability.config import ReliabilityConfig
    from capstone.reliability.retries import RetryPolicy

    never = BreakerPolicy(failure_threshold=10**9)
    return ReliabilityConfig(mcp_retry=RetryPolicy(max_attempts=1), mcp_breaker=never,
                             llm_retry=RetryPolicy(max_attempts=1), llm_breaker=never)


SCENARIO_NAMES = ("none", "flaky_ehr", "ehr_down", "llm_primary_down")


def scenario_plan(name: str):
    from capstone.reliability.faults import SCENARIOS, FaultPlan, FaultRule
    if name == "book_slot_down":
        return FaultPlan(rules={"book_slot": FaultRule(down=True)})
    return SCENARIOS[name]


# --- one offline run --------------------------------------------------------------------------------
def run_offline(scenario: str, variant: str, ids: list[str], records: dict, seed, policy_index_llm=None) -> dict:
    """Run `ids` once under one seeded scenario. Returns per-record outcomes and totals."""
    from langgraph.checkpoint.memory import InMemorySaver

    from capstone.agents.rag_agent import PolicyAgent
    from capstone.llm import client as llm_mod
    from capstone.llm.client import LLMClient
    from capstone.mcp.client import EHRClient
    from capstone.mcp.server import build_server
    from capstone.memory.long_term import PatientMemory
    from capstone.memory.session import SessionStore
    from capstone.observability.tracing import configure_tracing, read_traces
    from capstone.reliability.config import Reliability
    from capstone.reliability.faults import FaultInjector
    from capstone.settings import get_app_config, get_settings
    from capstone.tools.redirects import redirect_status
    from capstone.tools.repository import StateStore
    from capstone.workflows.graph import build_workflow, start_request

    settings, as_of = get_settings(), get_app_config().as_of_date
    waits: list[float] = []
    injector = FaultInjector(scenario_plan(scenario), aliases={f"llm:{settings.llm_model}": "llm:primary"})
    config = before_config() if variant == "before" else get_app_config().reliability
    sim = {"t": 0.0}                     # simulated clock: arrivals + retry waits + would-be timeouts

    def clock() -> float:
        return sim["t"] + sum(waits) + injector.simulated_wait_s
    rel = Reliability(config, injector=injector, sleep=waits.append, rng=random.Random(0), clock=clock)
    provider = TruthProvider(TruthLLM(records, seed))
    real_completion, llm_mod.litellm.completion = llm_mod.litellm.completion, provider
    try:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            tracer = configure_tracing(path=work / "traces.jsonl")
            store = StateStore(work / "state")
            server = build_server(seed, store, as_of)
            ehr = unhardened_ehr(server, rel) if variant == "before" else EHRClient(server, reliability=rel)
            llm = LLMClient(settings, rel)
            with PolicyAgent(llm) as policy:
                g = build_workflow(llm, InMemorySaver(), seed=seed, records=records, state_store=store,
                                   memory=PatientMemory(work / "memory"), sessions=SessionStore(work / "sessions"),
                                   today=lambda: as_of, ehr=ehr, policy=policy)
                rows, t0 = {}, time.perf_counter()
                for rid in ids:
                    sim["t"] += ARRIVAL_GAP_S
                    try:
                        r = start_request(g, rid, records[rid]["patient_ref"])
                        rows[rid] = {"outcome": "paused" if r.waiting_for_review else r.status, "route": r.route,
                                     "booked": bool((r.booking or {}).get("booked")),
                                     "failure": (r.dependency_failure or {}).get("reason"),
                                     "failed_step": (r.dependency_failure or {}).get("step"),
                                     "flags": sorted({f.reason.value for f in _flags(store, rid)}),
                                     "pcp_redirect": redirect_status(store, rid)}       # improvement #3
                    except Exception as e:  # noqa: BLE001 - BEFORE: an uncaught error stops the request
                        rows[rid] = {"outcome": "crashed", "error": type(e).__name__, "route": None,
                                     "booked": False, "flags": sorted({f.reason.value for f in _flags(store, rid)})}
                wall_s = time.perf_counter() - t0
            bookings = store.read("bookings")
            spans = read_traces(work / "traces.jsonl")
            tracer.flush()
    finally:
        llm_mod.litellm.completion = real_completion
    per_record = Counter(b["record_id"] for b in bookings)
    llm_spans = [s for s in spans if s["name"] == "llm.call"]
    return {
        "scenario": scenario, "variant": variant, "n": len(ids), "rows": rows,
        "totals": {
            "outcomes": dict(Counter(r["outcome"] for r in rows.values())),
            "routes": dict(Counter(str(r["route"]) for r in rows.values())),
            "booked": sum(r["booked"] for r in rows.values()),
            "duplicate_bookings": sum(n - 1 for n in per_record.values() if n > 1),
            "staff_dependency_flags": sum("dependency_unavailable" in r["flags"] or "booking_status_unknown"
                                          in r["flags"] for r in rows.values()),
            "faults_injected": dict(Counter(e.fault for e in injector.events if e.fault != "ok")),
            "dependency_calls": dict(Counter(e.target.split(":")[0] if e.target.startswith("llm")
                                             else ("insurance" if e.target == "check_eligibility" else "ehr")
                                             for e in injector.events)),
            "primary_llm_attempts": sum(1 for e in injector.events if e.target == f"llm:{settings.llm_model}"),
            "fallback_llm_answers": sum(1 for s in llm_spans if s["attrs"].get("fallback_used")),
            "short_circuited": sum(b.short_circuited for b in rel.breakers._by_name.values()),
            "simulated_wait_s": round(injector.simulated_wait_s + sum(waits), 1),
            "wall_s": round(wall_s, 1),
            "provider_calls": dict(provider.calls),
        },
    }


def _flags(store, rid):
    from capstone.tools.actions import flags_for_record
    return flags_for_record(store, rid)


def silent_wrong(baseline: dict, run: dict) -> list[str]:
    """Records whose result differs from the no-fault run WITHOUT anything visible: no crash, no
    dependency flag, no staff fallback. These are the confident wrong answers."""
    out = []
    for rid, b in baseline["rows"].items():
        r = run["rows"][rid]
        visible = r["outcome"] == "crashed" or r.get("failure") or \
            {"dependency_unavailable", "booking_status_unknown"} & set(r["flags"])
        same = (r["outcome"], r["route"], r["booked"]) == (b["outcome"], b["route"], b["booked"])
        if not same and not visible:
            out.append(rid)
    return out


def run_all_offline(out_dir: Path = M7_DIR) -> dict:
    from capstone.tools.repository import default_seed
    from capstone.workflows.graph import load_intake_records

    records, seed = load_intake_records(), default_seed()
    ids = offline_sample(records)
    report = {"ran_at": datetime.now(UTC).isoformat(timespec="seconds"), "ids": ids, "runs": []}
    baseline = run_offline("none", "after", ids, records, seed)
    for scenario in SCENARIO_NAMES:
        for variant in ("before", "after"):
            run = baseline if (scenario, variant) == ("none", "after") else \
                run_offline(scenario, variant, ids, records, seed)
            run["totals"]["silent_wrong"] = silent_wrong(baseline, run)
            report["runs"].append(run)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "m7_offline.json").write_text(json.dumps(report, indent=1))
    return report


def print_offline(report: dict) -> None:
    print(f"{'scenario':17} {'variant':7} {'done':>4} {'paused':>6} {'crash':>5} {'staff*':>6} {'booked':>6} "
          f"{'dup':>3} {'silent':>6} {'calls':>14} {'short':>5} {'wait_s':>7} {'llm1st':>6} {'fb':>3}")
    for run in report["runs"]:
        t, o = run["totals"], run["totals"]["outcomes"]
        calls = "/".join(f"{k[0]}{v}" for k, v in sorted(t["dependency_calls"].items()))
        print(f"{run['scenario']:17} {run['variant']:7} {o.get('done', 0):>4} {o.get('paused', 0):>6} "
              f"{o.get('crashed', 0):>5} {t['staff_dependency_flags']:>6} {t['booked']:>6} {t['duplicate_bookings']:>3} "
              f"{len(t['silent_wrong']):>6} {calls:>14} {t['short_circuited']:>5} {t['simulated_wait_s']:>7} "
              f"{t['primary_llm_attempts']:>6} {t['fallback_llm_answers']:>3}")
    print("staff* = requests handed to staff with a dependency flag; calls = e(hr)/i(nsurance)/l(lm) attempts")


# --- live ---------------------------------------------------------------------------------------------
def run_live(out_dir: Path = M7_DIR) -> dict:
    """6 never-used records, real models, Langfuse Cloud + local traces. One fault plan per record."""
    from langgraph.checkpoint.memory import InMemorySaver

    from capstone.agents.rag_agent import PolicyAgent
    from capstone.llm.client import LLMClient
    from capstone.mcp.client import EHRClient
    from capstone.mcp.server import build_server
    from capstone.memory.long_term import PatientMemory
    from capstone.memory.session import SessionStore
    from capstone.observability.tracing import configure_tracing, read_traces
    from capstone.reliability.config import Reliability
    from capstone.reliability.faults import FaultInjector
    from capstone.settings import get_app_config, get_settings
    from capstone.tools.repository import StateStore, default_seed
    from capstone.workflows.graph import build_workflow, load_intake_records, start_request
    from evaluation.runner import CountingLLM, expected_route

    settings, app = get_settings(), get_app_config()
    records, seed = load_intake_records(), default_seed()
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "traces.jsonl"
    trace_path.unlink(missing_ok=True)
    tracer = configure_tracing(cloud=True, path=trace_path)
    store = StateStore(out_dir / "state")
    rows = []
    plans = {rid: None for rid in M7_CLEAN} | dict(M7_FAULTED)
    for rid, fault in plans.items():
        rel = Reliability(app.reliability, rng=random.Random(42),
                          injector=FaultInjector(scenario_plan(fault)) if fault else None)
        llm = CountingLLM(LLMClient(settings, rel))
        ehr = EHRClient(build_server(seed, store, app.as_of_date), reliability=rel)
        t0 = time.perf_counter()
        with PolicyAgent(llm) as policy:
            g = build_workflow(llm, InMemorySaver(), seed=seed, records=records, state_store=store,
                               memory=PatientMemory(out_dir / "memory"), sessions=SessionStore(out_dir / "sessions"),
                               ehr=ehr, policy=policy)
            r = start_request(g, rid, records[rid]["patient_ref"])
        rows.append({"record_id": rid, "fault": fault, "seconds": round(time.perf_counter() - t0, 1),
                     "expected_route": expected_route(records[rid], seed), "route": r.route, "status": r.status,
                     "waiting_for_review": r.waiting_for_review, "booked": bool((r.booking or {}).get("booked")),
                     "dependency_failure": r.dependency_failure,
                     "flags": sorted({f.reason.value for f in _flags(store, rid)}), "llm_calls": dict(llm.calls),
                     "faults": rel.injector.counts() if rel.injector else {}, "breakers": rel.breakers.states()})
    tracer.flush()
    spans = read_traces(trace_path)
    llm_spans = [s for s in spans if s["name"] == "llm.call"]
    report = {"ran_at": datetime.now(UTC).isoformat(timespec="seconds"), "model": settings.llm_model,
              "fallback_model": settings.llm_fallback_model or None, "langfuse": tracer.cloud is not None,
              "sink_errors": tracer.sink_errors, "rows": rows,
              "llm_spans": {"total": len(llm_spans),
                            "by_model": dict(Counter(s["attrs"].get("model") for s in llm_spans)),
                            "fallback_used": sum(bool(s["attrs"].get("fallback_used")) for s in llm_spans),
                            "errors": sum(s["status"] == "error" for s in llm_spans)},
              "spans": len(spans), "traces": len({s["trace_id"] for s in spans})}
    (out_dir / "m7_live.json").write_text(json.dumps(report, indent=1))
    return report
