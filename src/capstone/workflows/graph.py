"""The CareFlow workflow: intake -> eligibility -> referral routing, with a human pause before any
clinical escalation (M5), run by the M6 agent team.

    START -> intake -> eligibility -> route
      route: emergency       -> emergency_notice -> prepare_review -> review
             clinical_review -> prepare_review -> review
             staff_queue     -> staff_queue -> record
             referral        -> insurance -> tracking -> safety_review
                                  safety_review: escalate -> prepare_review -> review
                                                 continue -> book -> record
      review: ask_again      -> review          (invalid answer: pause again)
              escalate       -> escalate_clinical -> record
              not_clinical   -> route           (routed again without the clinical label)
              handled        -> record
    record -> END

Rules (M5 phase 2 + A5):
- Only `intake` calls the LLM (M1 parse_intake). Every decision is code (workflows/routing.py)
  and every edge only reads a field a step wrote (decide in a step, route with an edge).
- The raw message is read inside the intake step and never stored; the insurance ID is
  compared there and only the result is kept (schemas/state.py).
- Side effects (emergency notice, escalation, staff flags, the memory row and session) each have
  their own step AFTER any pause, and each checks its idempotency key in `actions_done` first, so
  a resume or a repeated run never repeats an action (README M5 acceptance). The stores behind
  them are idempotent too: M2 flags (same record + reason), M3 memory rows (same source id).
- Staff-facing actions use the M2 flag store (runtime state) with M2's flag reasons; the
  emergency notice is a fixed, policy-based message (clinical-escalation 5.4), never advice. It is
  sent on intake's Emergent rating, which the M5 live run showed is sometimes wrong (kept by
  decision: a conditional notice to a non-emergency is far less harmful than a missed emergency;
  the note says so, and M8 measures the false-emergency rate).
- Escalate and handled also queue the request's non-clinical flags (coverage, insurance,
  sign-off): they are separate staff tasks (M5 fix F1; live: a lapsed coverage was dropped).
- A request that already has a thread is never started again (`start_request`).

M6 (agents as steps; the code router still decides every handoff, so they are bounded):
- Intake agent = `intake`. Insurance agent = `eligibility` (coverage over MCP) + `insurance`
  (co-pay quote, one cited policy answer). Referral Tracker = `tracking` (decision, referrals,
  breach flags, booking plan) + `book`. Clinical Safety Reviewer = `safety_review`.
- The EHR / scheduling system is reached only through MCP (an in-memory session on the same
  stores), so the workflow and any other MCP client share one set of rules.
- Booking runs in its own step, after the reviewer approved it, and BEFORE the request's flags
  are queued (open flags block booking). Clinical content found by the reviewer goes to the M5
  pause; if the reviewer there says "not clinical", the text is withheld (never re-escalated) and
  the earlier insurance result is reused (no repeated LLM calls).
- "Today" is the configured as-of date (configs/app.yaml) unless a test passes its own.

M7 (observability + reliability; phase 2 decisions):
- Every request run is one trace: a root span "careflow.request" (record_id, patient_ref, phase,
  final route / status / outcome) with one span per step ("step.<name>") and, below them, the
  MCP tool and LLM spans from the clients. A step that pauses for a person is marked paused,
  not failed.
- An EHR / insurance step that cannot get its answer (DependencyUnavailable: retries used up,
  breaker open, malformed reply) records `dependency_failure` and the request goes to the staff
  queue with a `dependency_unavailable` flag (reliability/fallbacks.py). Emergent and clinical
  requests keep their own path. Results already found are kept; nothing is guessed or booked.
- A failed booking is looked up before anything else happens (booked / not booked / unknown);
  the workflow never books again by itself after a failure.
"""
import json
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path

from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel

from capstone.agents.clinical_safety_reviewer import SAFETY_REJECTED, run_review
from capstone.agents.insurance_agent import InsuranceResult, run_insurance
from capstone.agents.intake_agent import parse_intake
from capstone.agents.referral_agent import BookingPlan, TrackingResult, execute_booking, run_tracker
from capstone.mcp.client import DependencyUnavailable, EHRClient
from capstone.mcp.server import build_server
from capstone.memory.long_term import PATIENT_REF, PatientMemory, default_memory, make_item
from capstone.memory.session import SessionStore, default_sessions
from capstone.observability.tracing import get_tracer
from capstone.reliability.fallbacks import DependencyFailure, fallback_route, verify_booking
from capstone.schemas.intake import PatientRequest
from capstone.schemas.state import (
    ApprovalRecord,
    FlagItem,
    ReferralDecision,
    ReviewerAnswer,
    WorkflowState,
    action_key,
    initial_state,
    mask_request,
    now_iso,
)
from capstone.settings import get_app_config, get_settings
from capstone.tools.actions import FlagReason, flag_for_human
from capstone.tools.lookups import EligibilityResult, check_eligibility
from capstone.tools.redirects import prepare_pcp_redirect
from capstone.tools.repository import SeedTables, StateStore, default_seed, default_state
from capstone.workflows.approvals import (
    after_review,
    prepare_review_step,
    review_step,
    validate_answer,
)
from capstone.workflows.checkpoints import thread_config
from capstone.workflows.routing import NEEDS_SIGNOFF, choose_route, current_flags

EMERGENCY_MESSAGE = (
    "If this is a medical emergency, call 911 now or go to the nearest emergency department. "
    "Your message has been passed to our clinical team.")
M2_REASONS = {r.value for r in FlagReason}
SAFETY_ESCALATION = "safety_escalation"
OUTCOMES = {"booked": "booked", "referral": "referral_routed", "staff_queue": "flagged_for_staff",
            "escalate": "escalated_to_clinician", "handled": "resolved"}


def load_intake_records(path: Path | None = None) -> dict[str, dict]:
    """record_id -> intake record (the raw messages stay here, outside the workflow state)."""
    path = path or get_settings().data_root / "intake" / "records.jsonl"
    return {r["record_id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}


def _request(state: WorkflowState) -> PatientRequest | None:
    return PatientRequest.model_validate(state["request"]) if state.get("request") else None


def _eligibility(state: WorkflowState) -> EligibilityResult | None:
    return EligibilityResult.model_validate(state["eligibility"]) if state.get("eligibility") else None


def _flag_reason(reason: str) -> FlagReason:
    try:
        return FlagReason(reason)
    except ValueError:
        return FlagReason.OTHER                  # any reason without its own value goes in as "other"


def _traced_step(name: str, fn):
    """Run a step inside its own span. A pause for a person is 'paused', not an error (M7)."""
    def run(state: WorkflowState) -> dict:
        pause = None
        with get_tracer().span(f"step.{name}", kind="chain") as span:
            try:
                out = fn(state)
            except GraphInterrupt as e:
                span.set(paused=True)
                pause, out = e, None
            else:
                if isinstance(out, dict):
                    span.set(route=out.get("route"), failure=(out.get("dependency_failure") or {}).get("reason"))
        if pause is not None:
            raise pause
        return out
    run.__name__ = name
    return run


def build_workflow(llm, checkpointer, *, seed: SeedTables | None = None,
                   records: Mapping[str, dict] | None = None, state_store: StateStore | None = None,
                   memory: PatientMemory | None = None, sessions: SessionStore | None = None,
                   clock: Callable[[], str] = now_iso, today: Callable[[], date] | None = None,
                   ehr: EHRClient | None = None, policy=None, safety_llm=None):
    """Compile the workflow graph. `llm` needs complete_structured (the M1 LLMClient, or a fake).

    M6: `ehr` is the EHR/scheduling MCP client (default: an in-memory MCP server on the same
    seed and state store); `policy` answers the Insurance agent's question (M4 PolicyAgent; None =
    no policy answer); `safety_llm` runs the safety check (default: `llm`)."""
    seed = seed or default_seed()
    records = records if records is not None else load_intake_records()
    store = state_store or default_state()
    memory = memory or default_memory()
    sessions = sessions or default_sessions()
    as_of = get_app_config().as_of_date
    today = today or (lambda: as_of)
    ehr = ehr or EHRClient(build_server(seed, store, today()))
    safety_llm = safety_llm or llm

    def done(state: WorkflowState, kind: str) -> bool:
        return action_key(kind, state["record_id"]) in state.get("actions_done", [])

    def queue_flags(state: WorkflowState, flags: list[dict]) -> None:
        for f in flags:
            note = f["note"] if f["reason"] in M2_REASONS else f"{f['reason']}: {f['note']}"
            flag_for_human(store, state["record_id"], state["patient_ref"], _flag_reason(f["reason"]),
                           note, raised_by="code")

    def queue_other_flags(state: WorkflowState) -> int:
        """Non-clinical flags (coverage, insurance, sign-off) are separate staff tasks (M5 fix F1)."""
        others = [f for f in state.get("flags", []) if f["reason"] != FlagReason.CLINICAL_QUESTION.value]
        queue_flags(state, others)
        return len(others)

    def failed(state: WorkflowState, step: str, error: DependencyUnavailable) -> dict:
        """The update for an EHR / insurance step that could not get its answer (M7)."""
        failure = DependencyFailure.from_error(step, error)
        route = fallback_route(state.get("route") or "referral", failure)
        return {"dependency_failure": failure.model_dump(), "route": route,
                "flags": list(state.get("flags", [])) + [failure.flag().model_dump()],
                "audit": [f"{step}: {failure.dependency} unavailable ({failure.tool}: {failure.reason}) -> {route}"]}

    # --- stage 1: intake (the only LLM call) ------------------------------------------------
    def intake(state: WorkflowState) -> dict:
        record = records.get(state["record_id"])
        if record is None or record.get("patient_ref") != state["patient_ref"]:
            return {"error": "no intake record for this request", "audit": ["intake: record not found"]}
        try:
            request = parse_intake(record, llm)
        except Exception as e:  # noqa: BLE001 - any LLM / parsing failure -> staff queue, never a crash
            return {"error": f"intake failed: {type(e).__name__}", "audit": [f"intake: failed ({type(e).__name__})"]}
        check = check_eligibility(seed, request.patient_ref, request.insurance_id).insurance_check
        return {"request": mask_request(request), "insurance_check": check,
                "audit": [(f"intake: urgency {request.urgency.value}, clinical {request.seeks_clinical_advice}, "
                          f"specialty {request.specialty.value if request.specialty else 'none'}")]}

    # --- stage 2: eligibility (code) --------------------------------------------------------
    def eligibility(state: WorkflowState) -> dict:
        if state.get("error"):
            return {"audit": ["eligibility: skipped (no request)"]}
        try:
            e = ehr.check_eligibility(state["patient_ref"])      # M6: the Insurance agent asks the EHR over MCP
        except DependencyUnavailable as err:                      # M7: route decides; the flag says why
            f = DependencyFailure.from_error("eligibility", err)
            return {"dependency_failure": f.model_dump(),
                    "audit": [f"eligibility: {f.dependency} unavailable ({f.tool}: {f.reason})"]}
        e = e.model_copy(update={"insurance_check": state.get("insurance_check", "not_checked")}) if e.found else e
        return {"eligibility": e.model_dump(mode="json"),
                "audit": [f"eligibility: found {e.found}, status {e.eligibility_status}, insurance {e.insurance_check}"]}

    # --- the router step: decide, write the decision; the edge only reads it ------------------
    def route(state: WorkflowState) -> dict:
        request, elig = _request(state), _eligibility(state)
        cleared, error = state.get("clinical_cleared", False), state.get("error")
        r = choose_route(request, elig, clinical_cleared=cleared, error=error)
        flags = current_flags(request, elig, clinical_cleared=cleared, error=error)
        failure = DependencyFailure.model_validate(state["dependency_failure"]) if state.get("dependency_failure") else None
        if failure is not None:                   # M7: emergency / clinical keep their path; the rest go to staff
            r = fallback_route(r, failure)
            flags.append(failure.flag())
        return {"route": r, "flags": [f.model_dump() for f in flags], "audit": [f"route: {r}"]}

    # --- stage 3 branches ---------------------------------------------------------------------
    def emergency_notice(state: WorkflowState) -> dict:
        if done(state, "emergency_notice"):
            return {}
        flag_for_human(store, state["record_id"], state["patient_ref"], FlagReason.URGENT_CASE,
                       "EMERGENT (on intake's urgency rating): patient told to call 911 / go to the ED; "
                       "site clinic manager notified (clinical-escalation 5.4)", raised_by="code")
        return {"actions_done": [action_key("emergency_notice", state["record_id"])],
                "audit": [f"emergency_notice: sent on intake's Emergent rating: {EMERGENCY_MESSAGE}"]}

    def escalate_clinical(state: WorkflowState) -> dict:
        if done(state, "escalate"):
            return {}
        a = ApprovalRecord.model_validate(state["approval"])
        flag_for_human(store, state["record_id"], state["patient_ref"], FlagReason.CLINICAL_QUESTION,
                       f"escalated to the clinical triage pool by {a.reviewer_id}", raised_by="code")
        n = queue_other_flags(state)              # after the clinical flag, so the reviewer note is kept
        return {"actions_done": [action_key("escalate", state["record_id"])],
                "audit": [(f"escalate_clinical: sent to clinical triage pool (approved by {a.reviewer_id}); "
                          f"{n} other flag(s) queued for staff")]}

    def staff_queue(state: WorkflowState) -> dict:
        if done(state, "staff_queue"):
            return {}
        queue_flags(state, state.get("flags", []))
        return {"actions_done": [action_key("staff_queue", state["record_id"])],
                "audit": [f"staff_queue: {len(state.get('flags', []))} flag(s) queued for staff"]}

    # --- M6: the referral path, run by the agent team ------------------------------------------
    def insurance(state: WorkflowState) -> dict:
        if state.get("insurance"):                 # already done (e.g. after a "not clinical" answer)
            return {}
        try:
            result = run_insurance(_request(state), ehr, seed, policy)
        except DependencyUnavailable as err:
            return failed(state, "insurance", err)
        p = result.policy
        return {"insurance": result.model_dump(mode="json"),
                "audit": [(f"insurance: plan {result.eligibility.plan_code}, co-pay "
                           f"{result.copay.amount_usd if result.copay else 'n/a'}, policy "
                           f"{p.route if p else 'not asked'}, agrees {result.copay_agrees}")]}

    def tracking(state: WorkflowState) -> dict:
        request = _request(state)
        try:
            result = run_tracker(request, ehr, seed, today())
        except DependencyUnavailable as err:
            return failed(state, "tracking", err)
        flags = current_flags(request, _eligibility(state), clinical_cleared=state.get("clinical_cleared", False),
                              referral=result.decision) + result.breach_flags
        notes = (state.get("insurance") or {}).get("notes", [])
        flags += [FlagItem(reason=n.split(":")[0], note=n) for n in notes if n.startswith("copay_mismatch")]
        d = result.decision
        return {"referral": d.model_dump(mode="json"), "tracking": result.model_dump(mode="json"),
                "flags": [f.model_dump() for f in flags],
                "audit": [(f"tracking: {d.specialty.value}, due {d.due_on}, sign-off {d.signoff_status}, "
                           f"{sum(r.breached for r in result.referrals)} breached referral(s), "
                           f"booking {result.booking.status}")]}

    def safety_review(state: WorkflowState) -> dict:
        request = _request(state)
        ins = InsuranceResult.model_validate(state["insurance"])
        track = TrackingResult.model_validate(state["tracking"])
        answer, cleared = ins.policy, state.get("clinical_cleared", False)
        if cleared and (state.get("safety") or {}).get("escalate_clinical"):
            answer = None                           # a person said "not clinical": withhold, never re-escalate
        retry = (lambda: policy.ask(ins.question)) if (policy and ins.question) else None
        try:
            review = run_review(request, answer, track.booking, track.decision, safety_llm, ehr, retry)
        except DependencyUnavailable as err:      # the booking re-check could not reach the EHR
            return failed(state, "safety_review", err)
        flags = list(state.get("flags", [])) + [f.model_dump() for f in review.flags]
        if answer is None and ins.policy is not None and cleared:
            flags.append(FlagItem(reason=SAFETY_REJECTED,
                                  note="policy answer withheld: it had clinical wording").model_dump())
        if review.escalate_clinical:
            flags.append(FlagItem(reason=SAFETY_ESCALATION,
                                  note="the safety reviewer found clinical content in the policy answer").model_dump())
        return {"safety": review.model_dump(mode="json"), "flags": flags,
                "route": "clinical_review" if review.escalate_clinical else state["route"],
                "audit": [(f"safety_review: {[f'{r.item}:{r.verdict}' for r in review.reviews]}, "
                           f"text {'withheld' if review.text_withheld else 'kept'}, booking "
                           f"{'approved' if review.booking_approved else 'not approved'}"
                           f"{', ESCALATE to clinical review' if review.escalate_clinical else ''}")]}

    def book(state: WorkflowState) -> dict:
        update: dict = {"actions_done": [], "audit": []}
        flags = list(state.get("flags", []))
        if (state.get("safety") or {}).get("booking_approved") and not done(state, "book"):
            plan = BookingPlan.model_validate(state["tracking"]["booking"])
            try:
                result = execute_booking(_request(state), plan, ehr)
                update["booking"] = result.model_dump(mode="json") if result else None
                update["audit"].append(f"book: {'booked ' + result.booking.booking_id if result and result.booked else 'refused ' + str(result.refused_reasons if result else [])}")
            except DependencyUnavailable as err:  # M7: did it happen? look before doing anything else
                check = verify_booking(ehr, state["patient_ref"], plan.slot, err)
                update["dependency_failure"] = DependencyFailure.from_error("book", err).model_dump()
                update["booking"] = ({"booked": True, "booking": None, "refused_reasons": [],
                                      "verified_after_failure": True, "appointment_id": check.appointment_id}
                                     if check.status == "booked" else None)
                if check.flag is not None:
                    flags.append(check.flag.model_dump())
                    update["flags"] = flags
                update["audit"].append(f"book: failed ({err.reason}); schedule check: {check.status}")
            update["actions_done"].append(action_key("book", state["record_id"]))   # never booked again by itself
        if flags and not done(state, "referral_flags"):     # after booking: open flags block it
            queue_flags(state, flags)
            update["actions_done"].append(action_key("referral_flags", state["record_id"]))
            update["audit"].append(f"book: {len(flags)} flag(s) queued for staff")
        signoff_missing = any(f.get("reason") == NEEDS_SIGNOFF for f in flags)
        booked = bool((update.get("booking") or state.get("booking") or {}).get("booked"))
        if signoff_missing and not booked and not done(state, "pcp_redirect"):
            specialty = (state.get("referral") or {}).get("specialty") or "the specialist"
            prepare_pcp_redirect(store, state["record_id"], state["patient_ref"], specialty)
            update["actions_done"].append(action_key("pcp_redirect", state["record_id"]))
            update["audit"].append("book: primary-care redirect prepared (referral-policy 3.2.3); "
                                   "a staff member must approve it before it is sent")
        return update

    # --- the end: remember the outcome once ----------------------------------------------------
    def record(state: WorkflowState) -> dict:
        if done(state, "record"):
            return {"status": "done"}
        decision = (state.get("approval") or {}).get("decision")
        booked = bool((state.get("booking") or {}).get("booked"))
        key = decision if decision in ("escalate", "handled") else ("booked" if booked else state.get("route"))
        outcome = OUTCOMES.get(key, "flagged_for_staff")
        request = _request(state)
        specialty = request.specialty.value if request and request.specialty else "none"
        update: dict = {"status": "done", "actions_done": [action_key("record", state["record_id"])]}
        lines = [f"record: outcome {outcome}"]
        if decision == "handled" and not done(state, "handled_flags"):
            n = queue_other_flags(state)
            update["actions_done"].append(action_key("handled_flags", state["record_id"]))
            lines.append(f"record: {n} other flag(s) queued for staff")
        elig = _eligibility(state)
        if elig and elig.found:                   # nothing is remembered for an unknown patient (M3)
            item = make_item(patient_ref=state["patient_ref"], occurred_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M"),
                             topic="care_request", outcome=outcome, source="workflow_outcome",
                             source_id=state["record_id"],
                             detail=f"request {state['record_id']}; route {state.get('route')}; specialty {specialty}")
            memory.add(item)
            lines.append(f"record: memory {item.memory_id}")
        if PATIENT_REF.fullmatch(state["patient_ref"]):
            session = sessions.open(state["patient_ref"], state["record_id"])
            session.extend([{"role": "system", "content": "workflow audit (M5/M6)"}]
                           + [{"role": "tool", "content": line} for line in state.get("audit", []) + lines]
                           + ([{"role": "tool", "content": json.dumps({"approval": state["approval"]})}]
                              if state.get("approval") else []))
            sessions.close(session)
            update["session_id"] = session.session_id
        update["audit"] = lines
        return update

    # --- edges ---------------------------------------------------------------------------------
    def after_route(state: WorkflowState) -> str:
        return state["route"]

    def after_safety(state: WorkflowState) -> str:
        if state.get("route") == "staff_queue":   # M7: the booking re-check could not reach the EHR
            return "staff_queue"
        return "escalate" if (state.get("safety") or {}).get("escalate_clinical") and state["route"] == "clinical_review" \
            else "continue"

    def after_team_step(state: WorkflowState) -> str:
        return "staff_queue" if state.get("route") == "staff_queue" else "continue"

    def after_answer(state: WorkflowState) -> str:
        if after_review(state) == "ask_again":
            return "ask_again"
        return state["approval"]["decision"]

    g = StateGraph(WorkflowState)
    for name, fn in [("intake", intake), ("eligibility", eligibility), ("route", route),
                     ("emergency_notice", emergency_notice),
                     ("prepare_review", prepare_review_step(seed, clock)), ("review", review_step),
                     ("escalate_clinical", escalate_clinical), ("staff_queue", staff_queue),
                     ("insurance", insurance), ("tracking", tracking), ("safety_review", safety_review),
                     ("book", book), ("record", record)]:
        g.add_node(name, _traced_step(name, fn))
    g.add_edge(START, "intake")
    g.add_edge("intake", "eligibility")
    g.add_edge("eligibility", "route")
    g.add_conditional_edges("route", after_route, {"emergency": "emergency_notice",
                                                   "clinical_review": "prepare_review",
                                                   "staff_queue": "staff_queue", "referral": "insurance"})
    g.add_edge("emergency_notice", "prepare_review")
    g.add_edge("prepare_review", "review")
    g.add_conditional_edges("review", after_answer, {"ask_again": "review", "escalate": "escalate_clinical",
                                                      "not_clinical": "route", "handled": "record"})
    g.add_edge("escalate_clinical", "record")
    g.add_edge("staff_queue", "record")
    g.add_conditional_edges("insurance", after_team_step, {"staff_queue": "staff_queue", "continue": "tracking"})
    g.add_conditional_edges("tracking", after_team_step, {"staff_queue": "staff_queue", "continue": "safety_review"})
    g.add_conditional_edges("safety_review", after_safety, {"escalate": "prepare_review", "continue": "book",
                                                            "staff_queue": "staff_queue"})
    g.add_edge("book", "record")
    g.add_edge("record", END)
    return g.compile(checkpointer=checkpointer)


# --- the API callers use (runner, later FastAPI) -----------------------------------------------

class WorkflowResult(BaseModel):
    record_id: str
    patient_ref: str
    status: str
    waiting_for_review: bool
    route: str | None = None
    flags: list[dict] = []
    referral: ReferralDecision | None = None
    insurance: InsuranceResult | None = None
    tracking: TrackingResult | None = None
    safety: dict | None = None
    booking: dict | None = None
    review: dict | None = None
    approval: ApprovalRecord | None = None
    actions_done: list[str] = []
    audit: list[str] = []
    error: str | None = None
    dependency_failure: dict | None = None
    session_id: str | None = None


class NotWaitingError(RuntimeError):
    """Resume was called for a request that is not waiting for a clinical reviewer."""


def get_result(graph, record_id: str) -> WorkflowResult | None:
    snap = graph.get_state(thread_config(record_id))
    v = snap.values
    if not v:
        return None
    return WorkflowResult(record_id=v["record_id"], patient_ref=v["patient_ref"], status=v.get("status", "running"),
                          waiting_for_review="review" in snap.next, route=v.get("route"),
                          flags=v.get("flags", []), referral=v.get("referral"), review=v.get("review"),
                          insurance=v.get("insurance"), tracking=v.get("tracking"), safety=v.get("safety"),
                          booking=v.get("booking"),
                          approval=v.get("approval"), actions_done=v.get("actions_done", []),
                          audit=v.get("audit", []), error=v.get("error"),
                          dependency_failure=v.get("dependency_failure"), session_id=v.get("session_id"))


def start_request(graph, record_id: str, patient_ref: str) -> WorkflowResult:
    """Run a request until it finishes or pauses. A finished or paused request is NOT run again;
    one interrupted mid-run (crash, kill, timeout) continues from its last checkpoint (M6 fix F1)."""
    existing = get_result(graph, record_id)
    if existing is not None:
        snap = graph.get_state(thread_config(record_id))
        if snap.next and "review" not in snap.next:      # stopped between steps, not paused for a person
            with _request_span(record_id, patient_ref, "continue") as span:
                graph.invoke(None, thread_config(record_id))  # LangGraph resumes at the step that didn't finish
                return _finish(span, get_result(graph, record_id))
        return existing
    with _request_span(record_id, patient_ref, "start") as span:
        graph.invoke(initial_state(record_id, patient_ref), thread_config(record_id))
        return _finish(span, get_result(graph, record_id))


def resume_request(graph, record_id: str, answer: dict | ReviewerAnswer) -> WorkflowResult:
    """Give a reviewer's answer to a paused request. The answer is checked before resuming."""
    checked = validate_answer(answer)                       # raises InvalidAnswerError
    current = get_result(graph, record_id)
    if current is None or not current.waiting_for_review:
        raise NotWaitingError(f"{record_id} is not waiting for a clinical reviewer")
    with _request_span(record_id, current.patient_ref, "resume") as span:
        graph.invoke(Command(resume=checked.model_dump(mode="json")), thread_config(record_id))
        return _finish(span, get_result(graph, record_id))


def _request_span(record_id: str, patient_ref: str, phase: str):
    """The root span of one run of a request (M7). session_id = record_id: start and resume group."""
    return get_tracer().span("careflow.request", kind="chain", record_id=record_id, patient_ref=patient_ref,
                             session_id=record_id, phase=phase)


def _finish(span, result: WorkflowResult | None) -> WorkflowResult | None:
    if result is not None:
        failure = result.dependency_failure or {}
        span.set(route=result.route, status=result.status, waiting_for_review=result.waiting_for_review,
                 booked=bool((result.booking or {}).get("booked")), flags=len(result.flags),
                 failed_step=failure.get("step"), failure_reason=failure.get("reason"))
    return result
