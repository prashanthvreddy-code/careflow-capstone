"""Run evaluation cases against the real application and compare with labels.

M3: retrieval baseline. Every golden question that names documents to cite is searched in the
policy index (dense only, as built in M3); scores are document-level recall@k, all-found@k and
MRR. No LLM is called. The golden set is only READ here; it is never indexed (two locks in
settings.py and rag/index.py). Run:  .venv/bin/python -m evaluation.runner

M4: the same scoring for three retrieval modes (dense / hybrid / hybrid_rerank) on two sets: the
golden set (used for tuning) and the held-out set (evaluation/datasets/m4_heldout.json, never
used for tuning). Still no LLM. Run:  .venv/bin/python -m evaluation.runner m4

M4 answers (LIVE LLM calls, opt-in, ~2-3 calls per question): every question goes through the
PolicyAgent and is scored by evaluation.metrics.score_answer (route, citations, expected facts,
forbidden phrases, unsafe count, latency). Run:
    .venv/bin/python -m evaluation.runner m4-answers golden|heldout|fresh|copay [--label NAME] [ID ...]
--label keeps runs of different models apart (m4_answers_<set>_<label>.json); every report
records the model and fallback model it ran with.

M5 workflow (LIVE LLM calls, opt-in, ~1 intake call per record + repairs): the approved sample of
10 records goes through the LangGraph workflow in one process (`m5-start`); clinical ones pause.
A SEPARATE process (`m5-resume`) reopens the checkpoint file and answers each paused review by a
fixed plan, so the human pause is shown to survive a real restart. `m5-status` lists what is
still waiting and what is overdue. Everything (checkpoints, flags, memory, sessions) lives in
evaluation/reports/m5_run/, never in runtime/. Run:
    .venv/bin/python -m evaluation.runner m5-start | m5-resume | m5-status

M6 agent team (LIVE LLM calls, opt-in, ~30-35 for the approved 10): the same start / resume /
status flow, with the Insurance agent's cited policy answer (M4 PolicyAgent) and the Clinical
Safety Reviewer switched on, the EHR/scheduling system reached over MCP, and booking enabled.
Everything lives in evaluation/reports/m6_run/. LLM calls are counted per process. Run:
    .venv/bin/python -m evaluation.runner m6-start | m6-resume | m6-status
"""
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from capstone.rag.index import PolicyIndex, open_policy_index
from capstone.rag.retrieval import PolicyRetriever, Retriever
from capstone.settings import PROJECT_ROOT
from evaluation.metrics import (
    all_found_at_k,
    doc_recall_at_k,
    mean,
    reciprocal_rank,
    score_answer,
    summarise_answers,
)

GOLDEN = PROJECT_ROOT / "data" / "careflow" / "eval" / "golden_set.json"
HELDOUT = PROJECT_ROOT / "evaluation" / "datasets" / "m4_heldout.json"
FRESH = PROJECT_ROOT / "evaluation" / "datasets" / "m4_fresh.json"
COPAY_CHECK = PROJECT_ROOT / "evaluation" / "datasets" / "m4_copay_check.json"
ANSWER_SETS = {"golden": GOLDEN, "heldout": HELDOUT, "fresh": FRESH, "copay": COPAY_CHECK}
REPORT = PROJECT_ROOT / "evaluation" / "reports" / "m3_retrieval_baseline.json"
M4_REPORT = PROJECT_ROOT / "evaluation" / "reports" / "m4_retrieval_variants.json"
MODES = ("dense", "hybrid", "hybrid_rerank")
KS = (1, 3, 5)
DEPTH = 10


Search = Callable[[str], list[tuple[str, str]]]      # question -> [(chunk_id, doc_title)] best first


def score_cases(cases: list[dict], search: Search) -> dict:
    """Document-level retrieval scores for every case that names documents to cite."""
    scored, skipped = [], []
    for case in cases:
        required = case.get("must_cite") or []
        if not required:
            skipped.append({"id": case["id"], "category": case["category"]})
            continue
        hits = search(case["question"])
        docs = [title for _, title in hits]
        row = {
            "id": case["id"], "category": case["category"], "required": required,
            "rr": reciprocal_rank(docs, required),
            "top_passages": [cid for cid, _ in hits[:5]],
            "first_required_rank": next((i for i, d in enumerate(docs, 1) if d in required), None),
        }
        for k in KS:
            row[f"recall@{k}"] = doc_recall_at_k(docs, required, k)
            row[f"all_found@{k}"] = all_found_at_k(docs, required, k)
        scored.append(row)

    def summary(rows: list[dict]) -> dict:
        out = {"n": len(rows), "mrr": mean([r["rr"] for r in rows])}
        for k in KS:
            out[f"recall@{k}"] = mean([r[f"recall@{k}"] for r in rows])
            out[f"all_found@{k}"] = mean([r[f"all_found@{k}"] for r in rows])
        return out

    categories = sorted({r["category"] for r in scored})
    return {
        "overall": summary(scored),
        "by_category": {c: summary([r for r in scored if r["category"] == c]) for c in categories},
        "cases": scored, "skipped_no_citation": skipped,
    }


def run_retrieval_eval(index: PolicyIndex, golden_path: Path = GOLDEN, depth: int = DEPTH) -> dict:
    """M3: dense search straight on the index."""
    cases = json.loads(Path(golden_path).read_text())
    scores = score_cases(cases, lambda q: [(h.chunk_id, h.doc_title) for h in index.search(q, top_k=depth)])
    return {"run_at": datetime.now(UTC).isoformat(timespec="seconds"), "index": index.meta(),
            "search": "dense only (M3)", "depth": depth, **scores}


def run_variant_eval(retriever: Retriever | PolicyRetriever, sets: dict[str, Path],
                     modes: tuple[str, ...] = MODES, depth: int = DEPTH) -> dict:
    """M4: every mode on every question set, same questions and same depth for a fair comparison."""
    results: dict = {}
    for set_name, path in sets.items():
        cases = json.loads(Path(path).read_text())
        results[set_name] = {}
        for mode in modes:
            def search(q, mode=mode):
                return [(h.chunk_id, h.doc_title) for h in retriever.retrieve(q, mode, top_k=depth).hits]
            results[set_name][mode] = score_cases(cases, search)
    return {"run_at": datetime.now(UTC).isoformat(timespec="seconds"), "depth": depth, "results": results}


def run_answer_eval(agent, cases: list[dict]) -> dict:
    """LIVE: ask every question, keep the full answer record (incl. draft) and its scores."""
    rows, records = [], []
    for case in cases:
        result = agent.ask(case["question"]).model_dump(mode="json")
        records.append(result)
        rows.append(score_answer(case, result))
    categories = sorted({r["category"] for r in rows})
    return {"run_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "overall": summarise_answers(rows),
            "by_category": {c: summarise_answers([r for r in rows if r["category"] == c]) for c in categories},
            "rows": rows, "answers": records}


def report_path(set_name: str, label: str | None = None) -> Path:
    """Where an answer report is saved; a label (e.g. the model) keeps runs from overwriting each other."""
    name = f"m4_answers_{set_name}"
    if label:
        name += "_" + "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in label)
    return PROJECT_ROOT / "evaluation" / "reports" / f"{name}.json"


def split_label(args: list[str]) -> tuple[str | None, list[str]]:
    """Take `--label NAME` out of the arguments; the rest are case IDs."""
    if "--label" not in args:
        return None, args
    i = args.index("--label")
    if i + 1 >= len(args):
        raise SystemExit("--label needs a name")
    return args[i + 1], args[:i] + args[i + 2:]


def main_answers(set_name: str, args: list[str]) -> None:
    from capstone.agents.rag_agent import PolicyAgent
    from capstone.llm.client import LLMClient
    from capstone.settings import get_settings

    label, ids = split_label(args)
    cases = json.loads(ANSWER_SETS[set_name].read_text())
    if ids:
        cases = [c for c in cases if c["id"] in ids]
    settings = get_settings()
    with PolicyAgent(LLMClient(settings)) as agent:
        report = run_answer_eval(agent, cases)
    report = {"model": settings.llm_model, "fallback_model": settings.llm_fallback_model or None, **report}
    out = report_path(set_name, label)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["overall"], indent=1))
    for r in report["rows"]:
        flags = "" if r["correct"] else " <-- " + ", ".join(
            k for k in ("route_ok", "citations_ok", "facts_ok") if not r[k]) + (" FORBIDDEN" if r["forbidden_shown"] else "")
        print(f"{r['id']:18} expected={r['expected_route']:16} got={r['route']:12} "
              f"withheld={r['withheld']!s:5}{' UNSAFE' if r['unsafe'] else ''}{flags}")
    print(f"saved {out.relative_to(PROJECT_ROOT)}")


def main_m4() -> None:
    with PolicyRetriever() as retriever:
        report = run_variant_eval(retriever, {"golden (tuning)": GOLDEN, "held-out": HELDOUT})
    M4_REPORT.parent.mkdir(parents=True, exist_ok=True)
    M4_REPORT.write_text(json.dumps(report, indent=1))
    for set_name, by_mode in report["results"].items():
        print(f"\n{set_name}")
        for mode, res in by_mode.items():
            o = res["overall"]
            misses = [r["id"] for r in res["cases"] if r["recall@5"] < 1]
            print(f"  {mode:14} n={o['n']:2} recall@3={o['recall@3']:.3f} recall@5={o['recall@5']:.3f} "
                  f"all@5={o['all_found@5']:.3f} MRR={o['mrr']:.3f} misses@5={misses}")
    print(f"saved {M4_REPORT.relative_to(PROJECT_ROOT)}")


M5_SAMPLE = ("CAREFLOW-00068", "CAREFLOW-00041", "CAREFLOW-00105", "CAREFLOW-00042", "CAREFLOW-00055",
             "CAREFLOW-00045", "CAREFLOW-00088", "CAREFLOW-00076", "CAREFLOW-00083", "CAREFLOW-00196")
M5_HANDLED = ("CAREFLOW-00041",)        # the plan answers 'handled' here, so that path runs live too
M5_DIR = PROJECT_ROOT / "evaluation" / "reports" / "m5_run"
M5_REVIEWER = "eval.reviewer"


def expected_route(record: dict, seed) -> str:
    """The route the answer key implies (ground-truth fields through the same code router)."""
    from capstone.schemas.intake import PatientRequest
    from capstone.tools.lookups import check_eligibility
    from capstone.workflows.routing import choose_route

    g = record["ground_truth"]
    req = PatientRequest(record_id=record["record_id"], patient_ref=record["patient_ref"],
                         channel=record["channel"], symptoms=[], specialty=g["specialty"],
                         insurance_id=record["insurance_id_stated"], urgency=g["urgency"],
                         seeks_clinical_advice=g["seeks_clinical_advice"], missing_fields=g["missing_fields"])
    return choose_route(req, check_eligibility(seed, req.patient_ref, record["insurance_id_stated"]))


def review_plan(record: dict, payload: dict) -> dict:
    """The simulated reviewer: follows the answer key (escalate a clinical request, clear a
    non-clinical one and correct its urgency), except 'handled' for the records in M5_HANDLED."""
    g = record["ground_truth"]
    if record["record_id"] in M5_HANDLED:
        return {"decision": "handled", "reviewer_id": M5_REVIEWER, "note": "patient called back by staff"}
    if g["seeks_clinical_advice"]:
        return {"decision": "escalate", "reviewer_id": M5_REVIEWER, "note": "clinical question"}
    answer = {"decision": "not_clinical", "reviewer_id": M5_REVIEWER, "note": "no clinical question"}
    if g["urgency"] != payload["urgency"]:
        answer["corrected_urgency"] = g["urgency"]
    return answer


def m5_row(result, record: dict, seed) -> dict:
    """One record's outcome next to the answer key (no raw text, no member numbers)."""
    g = record["ground_truth"]
    review = result.review or {}
    return {"record_id": result.record_id, "patient_ref": result.patient_ref, "status": result.status,
            "waiting_for_review": result.waiting_for_review, "route": result.route,
            "expected_route": expected_route(record, seed),
            "truth": {"urgency": g["urgency"], "specialty": g["specialty"], "clinical": g["seeks_clinical_advice"]},
            "review_urgency": review.get("urgency"), "flags": [f["reason"] for f in result.flags],
            "referral": result.referral.model_dump(mode="json") if result.referral else None,
            "approval": result.approval.model_dump(mode="json") if result.approval else None,
            "actions_done": result.actions_done, "error": result.error, "audit": result.audit}


def m5_workflow(llm, checkpoints, records: dict, workdir: Path = M5_DIR, policy=None):
    from capstone.memory.long_term import PatientMemory
    from capstone.memory.session import SessionStore
    from capstone.tools.repository import StateStore, default_seed
    from capstone.workflows.graph import build_workflow

    return build_workflow(llm, checkpoints.saver, seed=default_seed(), records=records,
                          state_store=StateStore(workdir / "state"), memory=PatientMemory(workdir / "memory"),
                          sessions=SessionStore(workdir / "sessions"), policy=policy)


def run_m5_start(llm, ids: tuple[str, ...] = M5_SAMPLE, workdir: Path = M5_DIR) -> list[dict]:
    from capstone.tools.repository import default_seed
    from capstone.workflows.checkpoints import Checkpoints
    from capstone.workflows.graph import load_intake_records, start_request

    records, seed = load_intake_records(), default_seed()
    with Checkpoints(workdir / "checkpoints") as cp:
        graph = m5_workflow(llm, cp, records, workdir)
        return [m5_row(start_request(graph, rid, records[rid]["patient_ref"]), records[rid], seed) for rid in ids]


def run_m5_resume(workdir: Path = M5_DIR) -> list[dict]:
    """Answer every paused review by the plan. Needs no LLM: the requests are already parsed."""
    from capstone.tools.repository import default_seed
    from capstone.workflows.checkpoints import Checkpoints, paused_reviews
    from capstone.workflows.graph import load_intake_records, resume_request

    records, seed = load_intake_records(), default_seed()
    with Checkpoints(workdir / "checkpoints") as cp:
        graph = m5_workflow(None, cp, records, workdir)
        rows = []
        for payload in paused_reviews(graph, cp):
            record = records[payload["record_id"]]
            answer = review_plan(record, payload)
            row = m5_row(resume_request(graph, record["record_id"], answer), record, seed)
            rows.append(row | {"answer": answer})
        return rows


M6_SAMPLE = ("CAREFLOW-00090", "CAREFLOW-00182", "CAREFLOW-00086", "CAREFLOW-00079", "CAREFLOW-00155",
             "CAREFLOW-00044", "CAREFLOW-00046", "CAREFLOW-00050", "CAREFLOW-00047", "CAREFLOW-00101")
M6_DIR = PROJECT_ROOT / "evaluation" / "reports" / "m6_run"


class CountingLLM:
    """Counts LLM requests by kind (a repair inside the client counts once). Delegates everything."""

    def __init__(self, llm):
        self.llm, self.calls = llm, {}

    def complete_structured(self, messages, schema):
        self.calls[schema.__name__] = self.calls.get(schema.__name__, 0) + 1
        return self.llm.complete_structured(messages, schema)

    def __getattr__(self, name):
        return getattr(self.llm, name)


def m6_row(result, record: dict, seed) -> dict:
    """An M5 row plus what each agent produced (no raw text, no member numbers)."""
    row = m5_row(result, record, seed)
    ins, track, safety = result.insurance, result.tracking, result.safety or {}
    row.update({
        "insurance": None if ins is None else {
            "plan": ins.eligibility.plan_code, "copay": ins.copay.amount_usd if ins.copay else None,
            "policy_route": ins.policy.route if ins.policy else None,
            "policy_answer": ins.policy.answer if ins.policy else None,
            "citations": [c.chunk_id for c in ins.policy.citations] if ins.policy else [],
            "copay_agrees": ins.copay_agrees, "notes": ins.notes},
        "tracking": None if track is None else {
            "booking": track.booking.status, "slot": track.booking.slot.slot_id if track.booking.slot else None,
            "refusals": track.booking.refusals, "breached": [r.referral_id for r in track.referrals if r.breached],
            "breach_flags": [f.reason for f in track.breach_flags]},
        "safety": {"reviews": [f"{r['item']}:{r['verdict']}:{r['by']}" for r in safety.get("reviews", [])],
                   "text_withheld": safety.get("text_withheld"), "booking_approved": safety.get("booking_approved"),
                   "escalate_clinical": safety.get("escalate_clinical")} if safety else None,
        "booking": (result.booking or {}).get("booking", {}).get("booking_id") if result.booking else None,
    })
    return row


def run_m6(command: str, llm, workdir: Path = M6_DIR, ids: tuple[str, ...] = M6_SAMPLE) -> list[dict]:
    """m6-start: run each request until done or paused. m6-resume: answer every paused review."""
    from capstone.agents.rag_agent import PolicyAgent
    from capstone.tools.repository import default_seed
    from capstone.workflows.checkpoints import Checkpoints, paused_reviews
    from capstone.workflows.graph import load_intake_records, resume_request, start_request

    records, seed = load_intake_records(), default_seed()
    with Checkpoints(workdir / "checkpoints") as cp, PolicyAgent(llm) as policy:
        graph = m5_workflow(llm, cp, records, workdir, policy=policy)
        if command == "m6-start":
            return [m6_row(start_request(graph, rid, records[rid]["patient_ref"]), records[rid], seed) for rid in ids]
        rows = []
        for payload in paused_reviews(graph, cp):
            record = records[payload["record_id"]]
            answer = review_plan(record, payload)
            rows.append(m6_row(resume_request(graph, record["record_id"], answer), record, seed) | {"answer": answer})
        return rows


def main_m6(command: str) -> None:
    from capstone.llm.client import LLMClient
    from capstone.settings import get_settings
    from capstone.workflows.approvals import overdue_reviews
    from capstone.workflows.checkpoints import Checkpoints, paused_reviews
    from capstone.workflows.graph import load_intake_records

    settings = get_settings()
    if command == "m6-status":
        with Checkpoints(M6_DIR / "checkpoints") as cp:
            waiting = paused_reviews(m5_workflow(None, cp, load_intake_records(), M6_DIR), cp)
        print(json.dumps({"waiting": [p["record_id"] for p in waiting],
                          "overdue": [o.record_id for o in overdue_reviews(waiting, datetime.now(UTC))]}, indent=1))
        return
    llm = CountingLLM(LLMClient(settings))
    rows = run_m6(command, llm)
    out = M6_DIR / f"{command.replace('-', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": settings.llm_model, "fallback_model": settings.llm_fallback_model or None,
                               "ran_at": datetime.now(UTC).isoformat(timespec="seconds"), "llm_calls": llm.calls,
                               "rows": rows}, indent=1))
    for r in rows:
        ins, tr, sf = r["insurance"] or {}, r["tracking"] or {}, r["safety"] or {}
        print(f"{r['record_id']} {r['status']:7} route={r['route']!s:15} expected={r['expected_route']:15} "
              f"policy={ins.get('policy_route')} agrees={ins.get('copay_agrees')} booking={tr.get('booking')} "
              f"safety={sf.get('reviews')} booked={r['booking']}"
              + (f" answer={r['answer']['decision']}" if r.get("answer") else ""))
    print("LLM calls:", llm.calls)
    print(f"saved {out.relative_to(PROJECT_ROOT)}")


def main_m5(command: str) -> None:
    from capstone.llm.client import LLMClient
    from capstone.settings import get_settings
    from capstone.workflows.approvals import overdue_reviews
    from capstone.workflows.checkpoints import Checkpoints, paused_reviews
    from capstone.workflows.graph import load_intake_records

    settings = get_settings()
    if command == "m5-start":
        rows = run_m5_start(LLMClient(settings))
    elif command == "m5-resume":
        rows = run_m5_resume()
    else:
        with Checkpoints(M5_DIR / "checkpoints") as cp:
            waiting = paused_reviews(m5_workflow(None, cp, load_intake_records()), cp)
        overdue = overdue_reviews(waiting, datetime.now(UTC))
        print(json.dumps({"waiting": [p["record_id"] for p in waiting],
                          "overdue": [o.model_dump() for o in overdue]}, indent=1))
        return
    out = M5_DIR / f"{command.replace('-', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": settings.llm_model, "fallback_model": settings.llm_fallback_model or None,
                               "ran_at": datetime.now(UTC).isoformat(timespec="seconds"), "rows": rows}, indent=1))
    for r in rows:
        mark = "" if r["route"] == r["expected_route"] or r.get("answer") else "  <-- differs from answer key"
        extra = f" answer={r['answer']['decision']}" if r.get("answer") else ""
        print(f"{r['record_id']} status={r['status']:7} route={r['route']!s:15} expected={r['expected_route']:15}"
              f"{extra}{mark}")
    print(f"saved {out.relative_to(PROJECT_ROOT)}")


def main_m7(command: str) -> None:
    from evaluation import m7

    if command == "m7-scenarios":
        report = m7.run_all_offline()
        m7.print_offline(report)
        print(f"saved {(m7.M7_DIR / 'm7_offline.json').relative_to(PROJECT_ROOT)}")
        return
    report = m7.run_live()
    for r in report["rows"]:
        print(f"{r['record_id']} fault={r['fault']!s:15} {r['seconds']:>5}s route={r['route']!s:15} "
              f"expected={r['expected_route']:15} booked={r['booked']} failure={r['dependency_failure']} "
              f"flags={r['flags']} llm={r['llm_calls']}")
    print("langfuse:", report["langfuse"], "| llm spans:", report["llm_spans"], "| sink errors:", report["sink_errors"])
    print(f"saved {(m7.M7_DIR / 'm7_live.json').relative_to(PROJECT_ROOT)}")


def main_m8_random(args: list[str]) -> None:
    """m8-random [--questions-only OUT_DIR_NAME]: both parts, or only the 30 policy questions again."""
    from evaluation import m8_random

    if args[:1] == ["--patients-only"]:
        out = m8_random.OUT_DIR.parent / (args[1] if len(args) > 1 else "m8_random_patients")
        report = m8_random.run_random(out, questions_part=False)
        m8_random.OUT_DIR = out
    elif args[:1] == ["--questions-only"]:
        out = m8_random.OUT_DIR.parent / (args[1] if len(args) > 1 else "m8_random_questions")
        report = m8_random.run_random(out, patients_part=False)
        m8_random.OUT_DIR = out
    else:
        report = m8_random.run_random()
    for r in report["patients"]["rows"]:
        print(f"{r['record_id']} route={r['route']!s:15} expected={r['expected_route']:15} urgency={r['urgency']!s:8}"
              f" truth={r['truth_urgency']:8} clinical={r['truth_clinical']!s:5} paused={r['waiting_for_review']!s:5}"
              f" booked={r['booked']!s:5}{' MISSED-CLINICAL' if r['clinical_missed'] else ''}"
              f"{' UNSAFE-BOOKING' if r['unsafe_booking'] else ''}")
    for r in report["questions"]["rows"]:
        j = r["judge"] or {}
        print(f"{r['id']} {r['template']:15} route={r['route']:12} facts_ok={r['facts_ok']!s:5} judge={j.get('correct')}")
    print(json.dumps({"patients": report["patients"]["summary"], "questions": report["questions"]["summary"],
                      "llm_calls": report["llm_calls"]}, indent=1))
    print(f"saved {(m8_random.OUT_DIR / 'm8_random.json').relative_to(PROJECT_ROOT)}")


def main_m8(command: str) -> None:
    from evaluation import m8

    if command == "m8-rejudge":                       # M8 #4: stored replies, judged again by EVAL_JUDGE_MODEL
        from capstone.llm.client import LLMClient
        from capstone.settings import get_settings
        report = json.loads((m8.M8_DIR / "m8_eval.json").read_text())
        labels = json.loads((m8.M8_DIR / "labels.json").read_text())
        out = m8.rejudge(report, m8.load_golden(), labels, LLMClient(m8.judge_settings(get_settings())))
        (m8.M8_DIR / "m8_rejudge.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        return
    if command == "m8-eval" and "--runs" in sys.argv:  # M8 #5: m8-eval --runs N
        print(json.dumps(m8.run_many(int(sys.argv[sys.argv.index("--runs") + 1])), indent=1))
        return
    if command == "m8-agreement":
        report = json.loads((m8.M8_DIR / "m8_eval.json").read_text())
        labels = json.loads((m8.M8_DIR / "labels.json").read_text())
        result = m8.agreement(report["results"], labels)
        report["judge_agreement"] = result
        (m8.M8_DIR / "m8_eval.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(result, indent=1))
        return
    report, demo = m8.run_live()
    for r in report["results"]:
        j = r["judge"] or {}
        print(f"{r['id']:16} {r['category']:12} expected={r['expected_route']:12} route={r['route']:12} "
              f"blocked={r['blocked_by']!s:6} cites={r['scores']['citations_must_cite']} "
              f"forbidden_ok={r['scores']['no_forbidden_phrase']} judge={j.get('correct')}")
    print(json.dumps(report["summary"], indent=1))
    print("llm calls:", report["llm_calls"], "| langfuse:", report["langfuse"])
    for step, out in demo.items():
        body = out.get("body", {}) if isinstance(out, dict) else {}
        print(step, "->", out.get("status_code") if isinstance(out, dict) else out,
              {k: body.get(k) for k in ("route", "status", "waiting_for_review", "booked", "flags")} if body else "")
    print(f"saved {(m8.M8_DIR / 'm8_eval.json').relative_to(PROJECT_ROOT)}; label page: "
          f"{(m8.M8_DIR / 'label.html').relative_to(PROJECT_ROOT)}")


def main() -> None:
    if sys.argv[1:2] and sys.argv[1] in ("m8-eval", "m8-agreement", "m8-rejudge"):
        main_m8(sys.argv[1])
        return
    if sys.argv[1:2] and sys.argv[1] in ("m7-scenarios", "m7-live"):
        main_m7(sys.argv[1])
        return
    if sys.argv[1:2] and sys.argv[1] in ("m6-start", "m6-resume", "m6-status"):
        main_m6(sys.argv[1])
        return
    if sys.argv[1:2] and sys.argv[1] in ("m5-start", "m5-resume", "m5-status"):
        main_m5(sys.argv[1])
        return
    if sys.argv[1:] == ["m4"]:
        main_m4()
        return
    if sys.argv[1:2] == ["m1-urgency"]:              # improvement #2: m1-urgency LABEL
        from evaluation import m1_urgency
        rep = m1_urgency.run(sys.argv[2] if len(sys.argv) > 2 else "run")
        print(json.dumps({k: rep[k]["summary"] for k in ("unseen", "safety")} | {"llm_calls": rep["llm_calls"]}, indent=1))
        return
    if sys.argv[1:2] == ["m8-random"]:
        main_m8_random(sys.argv[2:])
        return
    if sys.argv[1:2] == ["m4-answers"]:
        main_answers(sys.argv[2], sys.argv[3:])
        return
    with open_policy_index() as index:
        report = run_retrieval_eval(index)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))
    print(json.dumps({"overall": report["overall"], "by_category": report["by_category"]}, indent=1))
    for r in report["cases"]:
        print(f"{r['id']:18} {r['category']:10} rank={r['first_required_rank']} "
              f"recall@3={r['recall@3']:.2f} top={r['top_passages'][:3]}")
    print(f"saved {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
