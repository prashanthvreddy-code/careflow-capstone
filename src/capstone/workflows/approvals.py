"""The human checkpoint before clinical escalation: pause, validate the answer, record it. M5.

Decided in M5 phase 2:
- Only clinical and Emergent requests pause. The reviewer sees the ReviewPayload (urgency,
  specialty, the symptom phrases, flags, the response window): never raw text or member numbers.
- The reviewer answers escalate | not_clinical | handled, with a reviewer_id (required), an
  optional note and an optional corrected urgency. The answer is checked BEFORE the graph is
  resumed (`validate_answer`) and again inside the step, so a bad answer can never move the
  request on.
- The approval record says who decided what, when the pause started and when it was answered.
- Overdue: a pause older than its urgency band's first_response_hours is listed with an
  "review_overdue" flag for a supervisor. Nothing is decided automatically.

A5 B8 rule: on resume the paused step runs again from the top, so `review_step` does nothing
before interrupt() except read the state. Every action (escalate, record) is a later step.
"""
from collections.abc import Callable
from datetime import datetime

from langgraph.types import interrupt
from pydantic import BaseModel, ValidationError

from capstone.schemas.intake import PatientRequest
from capstone.schemas.state import (
    ApprovalRecord,
    FlagItem,
    ReviewerAnswer,
    ReviewPayload,
    WorkflowState,
    action_key,
    now_iso,
)
from capstone.tools.repository import SeedTables
from capstone.workflows.routing import first_response_hours

REVIEW_OVERDUE = "review_overdue"


class InvalidAnswerError(ValueError):
    """The reviewer's answer is not a valid ReviewerAnswer; the request stays paused."""


def validate_answer(raw: dict | ReviewerAnswer) -> ReviewerAnswer:
    """Check a reviewer's answer before resuming the graph with it."""
    if isinstance(raw, ReviewerAnswer):
        return raw
    try:
        return ReviewerAnswer.model_validate(raw)
    except ValidationError as e:
        raise InvalidAnswerError(f"invalid reviewer answer: {e.errors()[0]['msg']}") from e


def build_review_payload(state: WorkflowState, seed: SeedTables, paused_at: str) -> ReviewPayload:
    """What the reviewer sees. Built from the state only (no raw text is ever in the state)."""
    request = PatientRequest.model_validate(state["request"])
    return ReviewPayload(
        record_id=state["record_id"],
        patient_ref=state["patient_ref"],
        urgency=request.urgency,
        specialty=request.specialty,
        symptoms=[s.text for s in request.symptoms],
        flags=[f["reason"] for f in state.get("flags", [])],
        emergency_notice_sent=action_key("emergency_notice", state["record_id"]) in state.get("actions_done", []),
        respond_within_hours=first_response_hours(seed, request.urgency),
        paused_at=paused_at,
    )


def apply_answer(state: WorkflowState, answer: ReviewerAnswer, paused_at: str,
                 decided_at: str | None = None) -> dict:
    """The state update after a valid answer: the approval record, plus what it changes."""
    record = ApprovalRecord.from_answer(answer, paused_at=paused_at, decided_at=decided_at)
    update: dict = {
        "approval": record.model_dump(mode="json"),
        "clinical_cleared": answer.decision == "not_clinical",
        "status": "running",
        "audit": [f"review: {answer.decision} by {answer.reviewer_id} at {record.decided_at}"],
    }
    if answer.corrected_urgency is not None:
        request = dict(state["request"])
        old = request["urgency"]
        request["urgency"] = answer.corrected_urgency.value
        update["request"] = request
        update["audit"].append(f"review: urgency corrected {old} -> {answer.corrected_urgency.value}")
    return update


def prepare_review_step(seed: SeedTables, clock: Callable[[], str] = now_iso,
                        ) -> Callable[[WorkflowState], dict]:
    """Graph step BEFORE the pause: store the payload (with its pause time) in the state.

    Two steps, not one: on resume the paused step re-runs from the top, so a pause time taken
    inside it would become the resume time. Stored here, it is checkpointed before the pause.
    """
    def step(state: WorkflowState) -> dict:
        payload = build_review_payload(state, seed, clock())
        return {"review": payload.model_dump(mode="json"), "status": "paused",
                "audit": [f"review: paused at {payload.paused_at}"]}
    return step


def review_step(state: WorkflowState) -> dict:
    """Graph step that pauses: reads the stored payload, waits for the reviewer, records the answer.

    An invalid answer is NOT raised: LangGraph stores a resume value even when the step fails and
    replays it on the next resume, so one bad answer would block the request for good. Instead
    the step ends with `review_error` set and the graph loops back here (`after_review`), which
    pauses again with the error shown to the reviewer (A5: one interrupt per step; ask again by
    looping back through an edge). Callers should still check answers first (validate_answer).
    """
    payload = state["review"]
    error = state.get("review_error")
    raw = interrupt(payload | {"previous_answer_error": error} if error else payload)
    try:
        answer = validate_answer(raw)
    except InvalidAnswerError as e:
        return {"review_error": str(e), "audit": [f"review: invalid answer refused ({e})"]}
    return apply_answer(state, answer, payload["paused_at"]) | {"review_error": None}


def after_review(state: WorkflowState) -> str:
    """Edge after the review step: ask again after an invalid answer, otherwise go on."""
    return "ask_again" if state.get("review_error") else "answered"


class OverdueReview(BaseModel):
    record_id: str
    patient_ref: str
    urgency: str
    paused_at: str
    waited_hours: float
    limit_hours: int
    flag: FlagItem


def overdue_reviews(payloads: list[dict], now: datetime) -> list[OverdueReview]:
    """Paused reviews waiting longer than their band's first-response hours. Flags only; no action."""
    out = []
    for p in payloads:
        payload = ReviewPayload.model_validate(p)
        waited = (now - datetime.fromisoformat(payload.paused_at)).total_seconds() / 3600
        limit = payload.respond_within_hours
        if limit is not None and waited > limit:
            note = f"clinical review waiting {waited:.1f} h (limit {limit} h for {payload.urgency.value})"
            out.append(OverdueReview(record_id=payload.record_id, patient_ref=payload.patient_ref,
                                     urgency=payload.urgency.value, paused_at=payload.paused_at,
                                     waited_hours=round(waited, 2), limit_hours=limit,
                                     flag=FlagItem(reason=REVIEW_OVERDUE, note=note)))
    return sorted(out, key=lambda o: o.waited_hours - o.limit_hours, reverse=True)
