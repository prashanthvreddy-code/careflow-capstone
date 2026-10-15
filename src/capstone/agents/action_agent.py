"""Scheduling/insurance agent: PatientRequest (M1) -> tool-calling loop -> AgentResult. M2.

Rules agreed for M2:
- The LLM chooses tools; CODE fills in who/what they act on (patient_ref, plan, specialty), so a
  tool can only ever touch this one request (least privilege, patient isolation).
- Code raises ALL rule-based flags AFTER the loop (request rules, eligibility rules, tool
  failures, LLM error, round cap). The LLM is never told about code flags: it flags on its own
  judgement (an independent second layer, measurable as LLM-vs-code agreement). Duplicates merge.
- Booking is decided by the guard in tools/actions.py, which checks the rules directly.
- The staff summary is the LLM's text plus a code-written "Flags for staff" line.
- The loop is capped at MAX_ROUNDS; every tool call is logged.

M3 adds memory:
- get_patient_history: a read-only tool returning this patient's remembered contacts (code fills
  in the patient_ref). It never changes a safety decision: the booking guard and flags ignore it.
- After the run, CODE writes memory rows (booking, each flag reason, the request outcome); no
  LLM text is stored. Nothing is written for a patient who is not in the records.
- The whole conversation of the run is saved as one session (audit), and its id is returned.
"""
import json
from datetime import UTC, datetime
from typing import Any, Protocol

from capstone.agents.summary_review import review_summary
from capstone.llm.client import AssistantTurn, ToolCall, tool_result_message
from capstone.memory.long_term import (
    PATIENT_REF,
    HistoryEntry,
    MemoryItem,
    PatientMemory,
    default_memory,
    ensure_seeded,
    make_item,
)
from capstone.memory.session import SessionStore, default_sessions
from capstone.schemas.intake import PatientRequest
from capstone.schemas.response import AgentResult, StopReason, ToolCallRecord
from capstone.settings import PROJECT_ROOT
from capstone.tools import actions, calculators, lookups
from capstone.tools.actions import (
    M2_AGENT_REASONS,
    FlagReason,
    eligibility_rule_flags,
    request_rule_flags,
)
from capstone.tools.repository import SeedTables, StateStore, default_seed, default_state

PROMPT_PATH = PROJECT_ROOT / "prompts" / "action.md"
MAX_ROUNDS = 8
HISTORY_LIMIT = 10          # most patients have 1-7 remembered contacts


class ToolLLM(Protocol):
    """The real LLMClient, or a fake in tests: tool calls, plus the review/rewrite calls."""

    def complete_with_tools(self, messages: list[dict], tools: list[dict]) -> AssistantTurn: ...

    def complete(self, messages: list[dict], *, json_mode: bool = False) -> str: ...

    def complete_structured(self, messages: list[dict], schema: type) -> Any: ...


def _fn(name: str, description: str, properties: dict | None = None,
        required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {}, "required": required or []}}}


# What the LLM sees. No tool takes a patient, plan or specialty: code supplies those.
TOOL_SCHEMAS = [
    _fn("check_eligibility", "Insurance eligibility of this request's patient: plan, coverage "
        "status, deductible met, and whether the stated insurance ID matches the record."),
    _fn("get_patient_appointments", "This patient's existing appointments."),
    _fn("get_patient_history", "This patient's past contacts with the clinic (newest first): "
        "topic, outcome, date and status open/closed/superseded. Context only."),
    _fn("find_open_slots", "Up to 3 earliest open slots for this request's specialty.",
        {"mode": {"type": "string", "enum": ["in_person", "telehealth"]}}),
    _fn("calculate_copay", "What this patient pays for a service under their plan.",
        {"service": {"type": "string", "description": "consultation, telehealth, imaging, lab, "
                     "or a procedure code such as ORTH-ARTH"}}, ["service"]),
    _fn("book_slot", "Book one slot for this request. Refused unless every booking rule is met.",
        {"slot_id": {"type": "string"}}, ["slot_id"]),
    _fn("flag_for_human", "Queue this request for clinic staff.",
        {"reason": {"type": "string", "enum": list(M2_AGENT_REASONS)},
         "note": {"type": "string", "description": "one short factual sentence"}}, ["reason"]),
]
ALLOWED_ARGS = {t["function"]["name"]: set(t["function"]["parameters"]["properties"])
                for t in TOOL_SCHEMAS}


def build_agent_input(request: PatientRequest) -> dict:
    """What the LLM is told about the request: facts only, no code flags, no member number."""
    return {
        "record_id": request.record_id,
        "patient_ref": request.patient_ref,
        "specialty": request.specialty.value if request.specialty else None,
        "urgency": request.urgency.value,
        "wants_booking": request.wants_booking,
        "seeks_clinical_advice": request.seeks_clinical_advice,
        "insurance_stated": request.insurance_id is not None,
        "missing_fields": request.missing_fields,
    }


class _Run:
    """State of one agent run: executes tool calls for exactly one request."""

    def __init__(self, request: PatientRequest, seed: SeedTables, state: StateStore,
                 memory: PatientMemory):
        self.request, self.seed, self.state, self.memory = request, seed, state, memory
        self.history: list[HistoryEntry] = []
        self.eligibility: lookups.EligibilityResult | None = None
        self.copay: calculators.CopayQuote | None = None
        self.slots: list[lookups.SlotInfo] = []
        self.appointments: list[lookups.AppointmentInfo] = []
        self.booking: actions.Booking | None = None
        self.refusals: list[str] = []
        self.log: list[ToolCallRecord] = []

    def flag(self, reason: FlagReason | str, note: str, raised_by: str) -> actions.Flag:
        return actions.flag_for_human(self.state, self.request.record_id, self.request.patient_ref,
                                      reason, note, raised_by)

    def flags(self) -> list[actions.Flag]:
        return actions.flags_for_record(self.state, self.request.record_id)

    def ensure_eligibility(self) -> lookups.EligibilityResult:
        """Check once and reuse (flags from it are raised after the loop)."""
        if self.eligibility is None:
            self.eligibility = lookups.check_eligibility(self.seed, self.request.patient_ref,
                                                         self.request.insurance_id)
        return self.eligibility

    # ---- tool handlers: each returns something JSON-serialisable, or raises ValueError
    def check_eligibility(self) -> Any:
        return self.ensure_eligibility().model_dump(mode="json")

    def get_patient_appointments(self) -> Any:
        self.appointments = lookups.get_patient_appointments(self.seed, self.state,
                                                             self.request.patient_ref)
        return [a.model_dump(mode="json") for a in self.appointments]

    def get_patient_history(self) -> Any:
        self.history = self.memory.history(self.request.patient_ref, limit=HISTORY_LIMIT)
        return [h.model_dump(mode="json", include={"occurred_at", "topic", "outcome", "status",
                                                   "channel", "referral_id", "detail"})
                for h in self.history]

    def find_open_slots(self, mode: str = "in_person") -> Any:
        if self.request.specialty is None:
            raise ValueError("this request has no specialty")
        self.slots = lookups.find_open_slots(self.seed, self.state, self.request.specialty.value, mode)
        return [s.model_dump(mode="json") for s in self.slots]

    def calculate_copay(self, service: str) -> Any:
        e = self.ensure_eligibility()
        if not (e.found and e.is_active):
            raise ValueError("coverage is not active: no co-pay is quoted")
        self.copay = calculators.calculate_copay(self.seed, e.plan_code, service,
                                                 e.deductible_met_usd or 0)
        return self.copay.model_dump(mode="json")

    def book_slot(self, slot_id: str) -> Any:
        result = actions.book_slot(self.seed, self.state, self.request, self.ensure_eligibility(),
                                   slot_id)
        if result.booked:
            self.booking = result.booking
        else:
            self.refusals = result.refused_reasons
        return result.model_dump(mode="json")

    def flag_for_human(self, reason: str, note: str = "") -> Any:
        return self.flag(reason, note, "llm").model_dump(mode="json")

    # ---- dispatch
    def execute(self, call: ToolCall, round_no: int) -> str:
        """Run one tool call, log it, and return the text sent back to the model."""
        try:
            if call.arguments_error:
                raise ValueError(call.arguments_error)
            if call.name not in ALLOWED_ARGS:
                raise ValueError(f"unknown tool: {call.name}")
            extra = set(call.arguments) - ALLOWED_ARGS[call.name]
            if extra:
                raise ValueError(f"unexpected argument(s) {sorted(extra)}: this tool acts on the "
                                 "current request only")
            result = getattr(self, call.name)(**call.arguments)
        except (ValueError, TypeError, KeyError) as e:
            self.log.append(ToolCallRecord(round=round_no, tool=call.name, arguments=call.arguments,
                                           ok=False, error=str(e)))
            return json.dumps({"error": str(e)})
        self.log.append(ToolCallRecord(round=round_no, tool=call.name, arguments=call.arguments,
                                       ok=True, result=result))
        return json.dumps(result)

    def failed_tools(self) -> list[str]:
        """Tools that failed and never succeeded later in the run."""
        ok = {c.tool for c in self.log if c.ok}
        return sorted({c.tool for c in self.log if not c.ok} - ok)


def fallback_summary(run: _Run) -> str:
    """Code-written staff note, used when the LLM gave no final answer."""
    parts = []
    e = run.eligibility
    if e is not None:
        parts.append(f"Coverage: {e.eligibility_status or 'patient not found'}"
                     + (f" ({e.plan_code})." if e.plan_code else "."))
    if run.copay:
        parts.append(f"Co-pay: ${run.copay.amount_usd:.2f}.")
    if run.booking:
        parts.append(f"Booked {run.booking.slot_id} at {run.booking.starts_at}.")
    elif run.slots:
        parts.append("Suggested slots: " + ", ".join(f"{s.starts_at} ({s.site_name})" for s in run.slots) + ".")
    return " ".join(parts) or "No result: needs staff review."


def review_facts(run: _Run) -> dict:
    """Everything the tools returned, for the summary reviewer (no member number, no names)."""
    e = run.eligibility
    return {
        "patient_ref": run.request.patient_ref,
        "specialty": run.request.specialty.value if run.request.specialty else None,
        "urgency": run.request.urgency.value,
        "wants_booking": run.request.wants_booking,
        "eligibility": e.model_dump(mode="json") if e else None,
        "deductible_met_usd": e.deductible_met_usd if e else None,
        "copay_usd": run.copay.amount_usd if run.copay else None,
        "copay": run.copay.model_dump(mode="json") if run.copay else None,
        "suggested_slots": [s.model_dump(mode="json") for s in run.slots],
        "booking": run.booking.model_dump(mode="json") if run.booking else None,
        "booking_refusals": run.refusals,
        # why the booking rules would not book (worked out by code, even if no booking was tried)
        "booking_blockers": [] if run.booking else actions.booking_blockers(run.state, run.request, e),
        "appointments": [a.model_dump(mode="json") for a in run.appointments],
        "patient_history": [h.model_dump(mode="json", include={"occurred_at", "topic", "outcome",
                                                                "status", "referral_id", "detail"})
                            for h in run.history],
        "flags": [{"reason": f.reason.value, "note": f.note} for f in run.flags()],
    }


def flags_line(flags: list[actions.Flag]) -> str:
    """Code-written, so the summary always lists every flag exactly."""
    if not flags:
        return "Flags for staff: none."
    return "Flags for staff: " + "; ".join(f"{f.reason.value} ({f.raised_by}: {f.note})" if f.note
                                           else f"{f.reason.value} ({f.raised_by})" for f in flags) + "."


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M")        # same shape as the seed dates


def memory_rows(run: _Run, stop: StopReason) -> list[MemoryItem]:
    """What code remembers about this request: booking, each flag reason, and the outcome.

    Only fixed fields; no LLM text (flag notes are not copied). Nothing for an unknown patient.
    """
    if not (run.eligibility and run.eligibility.found):
        return []
    req, now, rows = run.request, _stamp(), []
    if run.booking:
        b = run.booking
        rows.append(make_item(patient_ref=req.patient_ref, occurred_at=now, topic="appointment_booking",
                              outcome="booked", source="agent_booking", source_id=b.booking_id,
                              detail=f"{b.specialty} slot {b.slot_id} at {b.starts_at}"))
    flags = run.flags()
    for f in flags:
        rows.append(make_item(patient_ref=req.patient_ref, occurred_at=now, topic="staff_review",
                              outcome="flagged_for_staff", source="agent_flag",
                              source_id=f"{req.record_id}:{f.reason.value}",
                              detail=f"reason {f.reason.value}, raised by {f.raised_by}"))
    outcome = ("booked" if run.booking else "flagged_for_staff" if flags
               else "slots_suggested" if run.slots else "no_action")
    specialty = req.specialty.value if req.specialty else "none"
    rows.append(make_item(patient_ref=req.patient_ref, occurred_at=now, topic="care_request",
                          outcome=outcome, source="agent_outcome", source_id=req.record_id,
                          detail=f"request {req.record_id}; specialty {specialty}; stop {stop}"))
    return rows


def run_agent(request: PatientRequest, llm: ToolLLM, seed: SeedTables | None = None,
              state: StateStore | None = None, max_rounds: int = MAX_ROUNDS,
              review: bool = True, memory: PatientMemory | None = None,
              sessions: SessionStore | None = None) -> AgentResult:
    """Run the tool-calling loop for one PatientRequest and return the AgentResult."""
    run = _Run(request, seed or default_seed(), state or default_state(), memory or default_memory())
    sessions = sessions or default_sessions()

    messages = [
        {"role": "system", "content": PROMPT_PATH.read_text()},
        {"role": "user", "content": json.dumps(build_agent_input(request))},
    ]

    # 1. the bounded tool loop (the LLM flags on its own judgement only)
    stop: StopReason = "max_rounds"
    summary = ""
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        try:
            turn = llm.complete_with_tools(messages, TOOL_SCHEMAS)
        except Exception as e:  # noqa: BLE001 - any LLM failure (quota, network, all fallbacks down) -> flag, never crash
            stop = "llm_error"
            run.flag(FlagReason.TOOL_FAILURE, f"agent LLM unavailable: {type(e).__name__}", "code")
            break
        messages.append(turn.as_message())
        if not turn.tool_calls:
            stop, summary = "final_answer", turn.content.strip()
            break
        for call in turn.tool_calls:
            messages.append(tool_result_message(call, run.execute(call, rounds)))

    # 2. after the loop: the code safety net raises every rule-based flag
    eligibility = run.ensure_eligibility()   # even if the LLM never checked it
    for reason, note in request_rule_flags(request) + eligibility_rule_flags(eligibility):
        run.flag(reason, note, "code")
    for tool in run.failed_tools():
        run.flag(FlagReason.TOOL_FAILURE, f"tool {tool} failed", "code")
    if stop == "max_rounds":
        run.flag(FlagReason.OTHER, f"agent stopped after {max_rounds} rounds without finishing", "code")

    # 3. critique the LLM's staff summary (Reflection): code checks + LLM review, max one rewrite
    summary_review = None
    if summary and review:
        summary, summary_review = review_summary(summary, review_facts(run), llm)

    # 4. M3: code writes long-term memory, then the session is saved for audit
    written: list[str] = []
    try:
        written = [row.memory_id for row in memory_rows(run, stop) if run.memory.add(row)]
    except (OSError, ValueError) as e:
        run.flag(FlagReason.TOOL_FAILURE, f"memory write failed: {type(e).__name__}", "code")
    session_id = None
    if PATIENT_REF.fullmatch(request.patient_ref or ""):
        try:
            session = sessions.open(request.patient_ref, request.record_id)
            session.extend(messages)
            sessions.close(session)
            session_id = session.session_id
        except (OSError, ValueError) as e:
            run.flag(FlagReason.TOOL_FAILURE, f"session save failed: {type(e).__name__}", "code")

    return AgentResult(
        record_id=request.record_id, patient_ref=request.patient_ref, urgency=request.urgency,
        eligibility=run.eligibility, copay=run.copay, suggested_slots=run.slots,
        booking=run.booking, booking_refusals=run.refusals, flags=run.flags(),
        tool_calls=run.log,
        staff_summary=f"{summary or fallback_summary(run)}\n{flags_line(run.flags())}",
        summary_review=summary_review, stop_reason=stop, rounds=rounds,
        patient_history=run.history, memory_written=written, session_id=session_id,
    )


def process_record(record: dict, llm, seed: SeedTables | None = None,
                   state: StateStore | None = None, memory: PatientMemory | None = None,
                   sessions: SessionStore | None = None) -> AgentResult:
    """Full M1 + M2 + M3 pipeline for one intake record: parse_intake, then run_agent.

    The seed interactions are imported into memory the first time (no-op afterwards).
    """
    from capstone.agents.intake_agent import parse_intake

    seed = seed or default_seed()
    memory = memory or default_memory()
    ensure_seeded(memory, seed)
    return run_agent(parse_intake(record, llm), llm, seed, state, memory=memory, sessions=sessions)
