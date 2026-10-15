"""Typed workflow state for the M5 intake -> eligibility -> referral-routing graph. M5.

Decided in M5 phase 2:
- The state is what LangGraph checkpoints after every step, so it holds only plain JSON values
  (strings, numbers, lists, dicts): checkpoints stay readable and nothing exotic is rebuilt on
  resume (A5 B7 security note). The Pydantic models below check those values on the way in.
- No raw message text and no member numbers, ever: the request is stored with member numbers
  masked. The insurance ID is compared with the patient record inside the intake step (which
  already has the message) and only the result (match / mismatch / not_stated) is kept.
- Control fields describe the request NOW and are overwritten (route, flags, status); audit
  fields only grow (audit, actions_done). A5 B1: an accumulating control field never empties.
- actions_done holds one idempotency key per side effect ("escalate:<record_id>"). A step checks
  it before acting, so a resume never repeats an action (README M5 acceptance).
- The pause payload (what the reviewer sees) and the reviewer's answer are validated models:
  the answer must name the reviewer, and the record of the decision says who, when and what.
"""
from datetime import UTC, datetime
from operator import add
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator

from capstone.memory.session import MASK
from capstone.schemas.intake import MEMBER_NUMBER, PatientRequest, Specialty, Urgency
from capstone.tools.lookups import InsuranceCheck

Route = Literal["emergency", "clinical_review", "staff_queue", "referral"]
Status = Literal["running", "paused", "done"]
ReviewDecision = Literal["escalate", "not_clinical", "handled"]
SignoffStatus = Literal["not_required", "live_referral", "none"]
Preauth = Literal["not_required", "approved", "pending", "denied", "unknown"]


def mask_numbers(value: Any) -> Any:
    """Replace member numbers anywhere in a value (strings inside dicts and lists too)."""
    if isinstance(value, str):
        return MEMBER_NUMBER.sub(MASK, value)
    if isinstance(value, dict):
        return {k: mask_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [mask_numbers(v) for v in value]
    return value


def _has_number(value: Any) -> bool:
    return mask_numbers(value) != value


def mask_request(request: PatientRequest) -> dict:
    """The PatientRequest as stored in the state: plain values, member numbers masked."""
    return mask_numbers(request.model_dump(mode="json"))


def action_key(kind: str, record_id: str) -> str:
    """Idempotency key of one side effect for one request, e.g. 'escalate:CAREFLOW-00004'."""
    return f"{kind}:{record_id}"


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class FlagItem(BaseModel):
    """One reason staff must look at this request (current flags: a control field)."""

    reason: str
    note: str
    raised_by: Literal["code", "reviewer"] = "code"


class ReferralDecision(BaseModel):
    """What referral routing decides for a routable request (code only, no booking)."""

    specialty: Specialty
    urgency: Urgency
    signoff_required: bool                        # policy 3.2: every specialty but Dermatology
    signoff_status: SignoffStatus
    signoff_referral_id: str | None = None        # the live referral that is the evidence
    referral_status: str | None = None            # open / accepted / scheduled
    sla_business_days: int                        # urgency_bands (Routine 5, Priority 2, Urgent 1)
    due_on: str                                   # ISO date
    preauth: Preauth = "unknown"

    @property
    def needs_signoff(self) -> bool:
        """Sign-off is required and no live referral shows it: staff must get it (never rejected)."""
        return self.signoff_required and self.signoff_status == "none"

    @model_validator(mode="after")
    def _consistent(self) -> "ReferralDecision":
        if self.signoff_status == "live_referral" and not self.signoff_referral_id:
            raise ValueError("a live-referral sign-off needs the referral id")
        if self.signoff_status == "not_required" and self.signoff_required:
            raise ValueError("sign-off cannot be both required and not required")
        if self.urgency in (Urgency.EMERGENT, Urgency.UNKNOWN):
            raise ValueError(f"urgency {self.urgency.value} is never referred")
        return self


class ReviewPayload(BaseModel):
    """What the reviewer is shown at the clinical pause. No raw text, no member numbers."""

    question: str = "Clinical escalation: escalate, not_clinical or handled?"
    record_id: str
    patient_ref: str
    urgency: Urgency
    specialty: Specialty | None
    symptoms: list[str]                           # the patient's symptom phrases (from M1)
    flags: list[str]
    emergency_notice_sent: bool = False
    respond_within_hours: int | None              # urgency_bands first_response_hours
    paused_at: str

    @model_validator(mode="after")
    def _no_member_numbers(self) -> "ReviewPayload":
        if _has_number(self.model_dump(mode="json")):
            raise ValueError("the review payload must not contain a member number")
        return self


class ReviewerAnswer(BaseModel):
    """The value a reviewer resumes the graph with (Command(resume=...))."""

    decision: ReviewDecision
    reviewer_id: str = Field(pattern=r"^[A-Za-z0-9._-]{2,40}$")
    note: str = Field(default="", max_length=500)
    corrected_urgency: Urgency | None = None

    @field_validator("note")
    @classmethod
    def _mask_note(cls, v: str) -> str:
        return mask_numbers(v.strip())

    @field_validator("corrected_urgency")
    @classmethod
    def _known_urgency(cls, v: Urgency | None) -> Urgency | None:
        if v == Urgency.UNKNOWN:
            raise ValueError("a corrected urgency must be a real band, not Unknown")
        return v


class ApprovalRecord(BaseModel):
    """Who decided what, and when: stored in the state and in the session (audit)."""

    decision: ReviewDecision
    reviewer_id: str
    note: str = ""
    corrected_urgency: Urgency | None = None
    paused_at: str
    decided_at: str

    @classmethod
    def from_answer(cls, answer: ReviewerAnswer, paused_at: str, decided_at: str | None = None,
                    ) -> "ApprovalRecord":
        return cls(decision=answer.decision, reviewer_id=answer.reviewer_id, note=answer.note,
                   corrected_urgency=answer.corrected_urgency, paused_at=paused_at,
                   decided_at=decided_at or now_iso())


class WorkflowState(TypedDict, total=False):
    """The LangGraph state. Only plain values; models are dumped with model_dump(mode="json")."""

    # identity (set once)
    record_id: str
    patient_ref: str
    # results of each stage
    request: dict | None                          # masked PatientRequest
    insurance_check: InsuranceCheck               # compared inside the intake step
    eligibility: dict | None                      # EligibilityResult
    referral: dict | None                         # ReferralDecision
    review: dict | None                           # ReviewPayload (set when the graph pauses)
    approval: dict | None                         # ApprovalRecord
    # M6: the team's results (plain values)
    insurance: dict | None                        # InsuranceResult (coverage, co-pay, cited policy answer)
    tracking: dict | None                         # TrackingResult (referrals, breach flags, booking plan)
    safety: dict | None                           # SafetyReview (verdicts, kept/withheld, booking approved)
    booking: dict | None                          # BookingResult once booked
    # control fields: describe the request now, overwritten
    route: Route
    flags: list[dict]                             # FlagItem dicts
    clinical_cleared: bool                        # the reviewer said not_clinical
    review_error: str | None                      # the last answer was invalid: ask again
    status: Status
    error: str | None                             # intake failure -> staff queue
    dependency_failure: dict | None               # M7: DependencyFailure (EHR / insurance step that failed)
    session_id: str | None                        # the M3 session the record step saved
    # audit fields: only grow
    actions_done: Annotated[list[str], add]
    audit: Annotated[list[str], add]


def initial_state(record_id: str, patient_ref: str) -> WorkflowState:
    """The input a new run starts from (the intake step fills in the rest)."""
    return WorkflowState(record_id=record_id, patient_ref=patient_ref, request=None,
                         insurance_check="not_checked", eligibility=None, referral=None,
                         review=None, approval=None, insurance=None, tracking=None, safety=None,
                         booking=None, flags=[], clinical_cleared=False, review_error=None,
                         status="running", error=None, dependency_failure=None, session_id=None, actions_done=[],
                         audit=[])
