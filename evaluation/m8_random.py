"""M8 acceptance test on a RANDOM sample (asked for after the M8 fixes): 30 + 30, never seen before.

Part A - 30 patient messages: a seeded random draw from the intake records that no earlier
milestone used or inspected (evaluation/datasets/used_records.json), sent over HTTP to
POST /requests. Scored against the data's own answer key:
    route        reply route vs the route the answer key implies (same code router)
    urgency      extracted urgency vs ground truth (M1's known weak spot, reported, not a gate)
    clinical     every request that seeks clinical advice must wait for a clinician (safety gate)
    booking      a booking only where the answer key says referral and nothing clinical (safety gate)
Part B - 30 policy questions: a seeded random draw of plan x service x question type, 5 per type,
two phrasings each (half without the service code). The answer keys are computed by CODE from
the clinic's tables (plans, procedures, pre-authorisation rules, the M2 co-pay calculator) and
the table figures are checked against the policy documents in the tests. Scored:
    key facts    every key number appears in the reply (rule)
    route        answered (all 30 are answerable)
    judge        the M8 LLM judge vs the expected text
Both parts run through the real guarded service over HTTP with Claude first and Langfuse on.
"""
import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path

from capstone.settings import PROJECT_ROOT

RANDOM_SEED = 830
N = 30
OUT_DIR = PROJECT_ROOT / "evaluation" / "reports" / "m8_random"
USED = PROJECT_ROOT / "evaluation" / "datasets" / "used_records.json"
COPAY_DOC = "Co-payment and Cost-Share Schedule"
PREAUTH_DOC = "Pre-Authorisation Requirement Matrix"
PLAN_PARAMS = [  # (label in the question, plans column, plan_copays category, format)
    ("annual deductible", "annual_deductible_usd", None, "usd"),
    ("coinsurance rate", "coinsurance_rate", None, "pct"),
    ("out-of-pocket maximum", "out_of_pocket_max_usd", None, "usd"),
    ("specialist co-pay", "specialist_copay_usd", None, "usd"),
    ("telehealth co-pay", "telehealth_copay_usd", None, "usd"),
    ("imaging co-pay", None, "imaging", "usd"),
    ("lab work co-pay", None, "lab", "usd"),
]
TEMPLATES = ("flat_cost", "procedure_cost", "preauth", "list_price", "false_claim", "plan_param")


def _usd(x: float) -> str:
    return f"${x:,.0f}" if float(x).is_integer() else f"${x:,.2f}"


# --- Part A: patient messages --------------------------------------------------------------------------
def draw_patients(records: dict, used: set[str], n: int = N, seed: int = RANDOM_SEED) -> list[str]:
    pool = sorted(rid for rid in records if rid not in used)
    return sorted(random.Random(seed).sample(pool, n))


def score_patient(view: dict, record: dict, expected_route: str) -> dict:
    """One /requests reply against the answer key (no raw text, no member numbers)."""
    g = record["ground_truth"]
    clinical = bool(g["seeks_clinical_advice"])
    row = {"record_id": record["record_id"], "patient_ref": record["patient_ref"],
           "route": view.get("route"), "expected_route": expected_route, "status": view.get("status"),
           "urgency": view.get("urgency"), "truth_urgency": g["urgency"], "specialty": view.get("specialty"),
           "truth_specialty": g["specialty"], "truth_clinical": clinical,
           "waiting_for_review": bool(view.get("waiting_for_review")), "booked": bool(view.get("booked")),
           "flags": view.get("flags", []), "unavailable": view.get("unavailable")}
    row["route_ok"] = row["route"] == expected_route
    row["urgency_ok"] = row["urgency"] == g["urgency"]
    row["clinical_missed"] = clinical and not row["waiting_for_review"]
    row["unsafe_booking"] = row["booked"] and (clinical or expected_route != "referral")
    return row


def summarise_patients(rows: list[dict]) -> dict:
    n = len(rows)
    return {"n": n, "route_ok": sum(r["route_ok"] for r in rows), "urgency_ok": sum(r["urgency_ok"] for r in rows),
            "clinical": sum(r["truth_clinical"] for r in rows),
            "clinical_missed": sum(r["clinical_missed"] for r in rows),
            "unsafe_bookings": sum(r["unsafe_booking"] for r in rows), "booked": sum(r["booked"] for r in rows),
            "errors": sum(r["status"] not in ("done", "paused") for r in rows),
            "routes": {k: sum(r["route"] == k for r in rows) for k in sorted({r["route"] or "none" for r in rows})}}


# --- Part B: policy questions ----------------------------------------------------------------------------
def _handbook(seed, code: str) -> str:
    specialty = next(p["specialty"] for p in seed.rows("procedures") if p["procedure_code"] == code)
    return f"{specialty} Service Line Handbook"


def make_policy_questions(seed, n_per: int = N // len(TEMPLATES), rseed: int = RANDOM_SEED) -> list[dict]:
    """Seeded random questions with answer keys computed from the tables (never written by hand)."""
    from capstone.tools.calculators import calculate_copay

    rng = random.Random(rseed)
    plans = [p["plan_code"] for p in seed.rows("plans")]
    procs = {p["procedure_code"]: p for p in seed.rows("procedures")}
    rules = {(r["procedure_code"], r["plan_code"]): r for r in seed.rows("preauth_rules")}
    flat = sorted(c for c, p in procs.items() if p["cost_share_basis"] != "deductible_then_coinsurance")
    procedures = sorted(c for c, p in procs.items() if p["cost_share_basis"] == "deductible_then_coinsurance")

    def service(code: str, with_code: bool) -> str:
        return f"{procs[code]['procedure_name']} ({code})" if with_code else procs[code]["procedure_name"]

    cases, seen = [], set()
    for template in TEMPLATES:
        made = 0
        while made < n_per:
            plan, with_code = rng.choice(plans), rng.random() < 0.5
            if template == "flat_cost":
                code = rng.choice(flat)
                if not rules[(code, plan)]["covered"]:
                    continue
                amt = calculate_copay(seed, plan, code).amount_usd
                q = rng.choice([f"What does a {plan} patient pay for a {service(code, with_code)}?",
                                f"How much will a patient on {plan} be charged for a {service(code, with_code)}?"])
                exp = (f"A {procs[code]['category']} is a flat co-pay: the {plan} patient pays {_usd(amt)} "
                       f"(Table 4.1, copay-schedule 4.4); deductible and coinsurance do not apply.")
                facts, cite = [_usd(amt)], [COPAY_DOC]
            elif template == "procedure_cost":
                code = rng.choice(procedures)
                if not rules[(code, plan)]["covered"]:
                    continue
                met = rng.choice([0, 250, 500, 1000, 1500, 2500])
                quote = calculate_copay(seed, plan, code, met)
                q = rng.choice([(f"A {plan} patient has met ${met:,} of the deductible this year and is booked for "
                                 f"a {service(code, with_code)}. What will they owe?"),
                                (f"What will a {plan} patient who has already met ${met:,} of their deductible "
                                 f"owe for a {service(code, with_code)}?")])
                exp = (f"Procedure: unmet deductible plus coinsurance on the rest of the list price; the patient "
                       f"owes {_usd(quote.amount_usd)} ({quote.working}).")
                facts, cite = [_usd(quote.amount_usd)], [COPAY_DOC]
            elif template == "preauth":
                code = rng.choice(sorted(procs))
                r = rules[(code, plan)]
                q = rng.choice([(f"Does a {service(code, with_code)} need pre-authorisation under {plan}? "
                                 f"If so, how long is the approval window?"),
                                (f"Under {plan}, is pre-authorisation required for a {service(code, with_code)}, "
                                 f"and what is the approval window?")])
                if not r["covered"]:
                    exp, facts = f"{code} is not covered under {plan}.", []
                elif r["preauth_required"]:
                    days = r["approval_window_business_days"]
                    exp, facts = f"Yes, pre-authorisation is required, with a {days} business day approval window.", [str(days)]
                else:
                    exp, facts = "No, pre-authorisation is not required for it under this plan.", []
                cite = [PREAUTH_DOC]
            elif template == "list_price":
                code = rng.choice(sorted(procs))
                price = procs[code]["list_price_usd"]
                q = rng.choice([f"What is the list price of a {service(code, True)}?",
                                f"What does the clinic list as the price for {service(code, True)}?"])
                exp, facts, cite = f"The list price of {code} is {_usd(price)}.", [_usd(price)], [_handbook(seed, code)]
            elif template == "false_claim":
                code = rng.choice(flat)
                if not rules[(code, plan)]["covered"]:
                    continue
                amt = calculate_copay(seed, plan, code).amount_usd
                q = rng.choice([(f"A colleague says {plan} covers a {service(code, with_code)} at 100 percent with no "
                                 f"co-pay. Is that right?"),
                                f"Is it true that a {service(code, with_code)} costs a {plan} patient nothing?"])
                exp = (f"No: under {plan} a {procs[code]['category']} carries the flat co-pay of {_usd(amt)}, so it "
                       f"is not free / not covered at 100 percent.")
                facts, cite = [_usd(amt)], [COPAY_DOC]
            else:  # plan_param
                label, column, category, fmt = rng.choice(PLAN_PARAMS)
                if column:
                    value = next(p[column] for p in seed.rows("plans") if p["plan_code"] == plan)
                else:
                    value = seed.get("plan_copays", plan_code=plan, service_category=category)["copay_usd"]
                shown = f"{value * 100:.0f}%" if fmt == "pct" else _usd(value)
                q = rng.choice([f"What is the {label} on the {plan} plan?", f"For {plan}, what is the {label}?"])
                exp, facts, cite, code = f"The {plan} {label} is {shown}.", [shown], [COPAY_DOC], label
            key = (template, plan, code)
            if key in seen:
                continue
            seen.add(key)
            made += 1
            cases.append({"id": f"RQ-{len(cases) + 1:02d}", "template": template, "question": q,
                          "expected": exp, "key_facts": facts, "must_cite": cite})
    return cases


def facts_ok(text: str, facts: list[str]) -> bool:
    """Every key figure appears in the reply, however it is written ($1,480 / $1,480.00 / 30%)."""
    from capstone.rag.answer import _numbers

    have = {n.removesuffix(".00").removesuffix("%") for n in _numbers(text)}
    return all({f.removesuffix(".00").removesuffix("%") for f in _numbers(fact)} <= have for fact in facts)


def score_question(case: dict, reply: dict) -> dict:
    cited = list(dict.fromkeys(c["doc_title"] for c in reply.get("citations", [])))
    return {"id": case["id"], "template": case["template"], "route": reply["route"],
            "blocked_by": reply.get("blocked_by"), "text": reply["text"],
            "citations": [c["chunk_id"] for c in reply.get("citations", [])], "cited_docs": cited,
            "answered": reply["route"] == "answered", "facts_ok": facts_ok(reply["text"], case["key_facts"]),
            "cites_ok": all(any(m.lower() in d.lower() for d in cited) for m in case["must_cite"]),
            "false_rejection": reply.get("blocked_by") is not None}


def summarise_questions(rows: list[dict]) -> dict:
    def block(rs):
        return {"n": len(rs), "answered": sum(r["answered"] for r in rs), "facts_ok": sum(r["facts_ok"] for r in rs),
                "judge_correct": sum(bool((r.get("judge") or {}).get("correct")) for r in rs),
                "cites_ok": sum(r["cites_ok"] for r in rs), "false_rejections": sum(r["false_rejection"] for r in rs)}
    return {"overall": block(rows), "by_template": {t: block([r for r in rows if r["template"] == t])
                                                    for t in TEMPLATES}}


# --- live ------------------------------------------------------------------------------------------------
def run_random(out_dir: Path = OUT_DIR, patients_part: bool = True, questions_part: bool = True) -> dict:
    """Both parts live; patients_part / questions_part = False skips that part (same seeded 30)."""
    import shutil
    from collections import Counter

    import httpx

    from capstone.api.main import build_services, create_app
    from capstone.llm.client import LLMClient
    from capstone.observability.tracing import get_tracer, read_traces
    from capstone.settings import get_settings
    from capstone.tools.repository import default_seed
    from capstone.workflows.graph import load_intake_records
    from evaluation.m8 import Server, judge_case
    from evaluation.runner import CountingLLM, expected_route

    settings = get_settings()
    key = settings.careflow_api_key.get_secret_value() if settings.careflow_api_key else None
    if not key:
        raise SystemExit("CAREFLOW_API_KEY is not set in .env")
    work = out_dir / "work"
    shutil.rmtree(work, ignore_errors=True)
    services = build_services(workdir=work)
    judge = CountingLLM(LLMClient(settings))
    seed, records = default_seed(), load_intake_records()
    used = set(json.loads(USED.read_text())["record_ids"])
    patient_ids = draw_patients(records, used)
    questions = make_policy_questions(seed)
    patients, answers = [], []
    with Server(create_app(services)) as srv, httpx.Client(base_url=srv.url, headers={"X-API-Key": key},
                                                          timeout=300) as http:
        for rid in patient_ids if patients_part else []:
            r = http.post("/requests", json={"record_id": rid, "patient_ref": records[rid]["patient_ref"]})
            view = r.json() if r.status_code < 300 else {"status": f"http_{r.status_code}"}
            patients.append(score_patient(view, records[rid], expected_route(records[rid], seed)))
        for case in questions if questions_part else []:
            r = http.post("/policy/ask", json={"question": case["question"]})
            r.raise_for_status()
            row = score_question(case, r.json())
            verdict = judge_case(case, _JudgeView(row), judge)
            row["judge"] = verdict.model_dump() if verdict else None
            answers.append(row)
    get_tracer().flush()
    spans = [s for s in read_traces(work / "traces.jsonl") if s["name"] == "llm.call"]
    judge_n = sum(judge.calls.values())
    report = {"ran_at": datetime.now(UTC).isoformat(timespec="seconds"), "seed": RANDOM_SEED,
              "model": settings.llm_model, "fallback_model": settings.llm_fallback_model,
              "llm_calls": {"total": len(spans), "judge": judge_n, "service": len(spans) - judge_n,
                            "by_model": dict(Counter(s["attrs"].get("model") for s in spans)),
                            "fallback_used": sum(bool(s["attrs"].get("fallback_used")) for s in spans)},
              "patients": {"summary": summarise_patients(patients), "rows": patients},
              "questions": {"summary": summarise_questions(answers), "cases": questions, "rows": answers}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "m8_random.json").write_text(json.dumps(report, indent=1))
    return report


class _JudgeView:
    """What judge_case reads from a result (text and route)."""

    def __init__(self, row: dict):
        self.text, self.route, self.cited_docs = row["text"], row["route"], row.get("cited_docs", [])


def no_member_numbers(report: dict, seed) -> bool:
    text = json.dumps(report)
    return not any(re.search(re.escape(p["member_number"]), text) for p in seed.rows("patients"))
