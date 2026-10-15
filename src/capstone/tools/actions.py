"""Action tools for the M2 agent: flag a case for a human, and book a slot. M2.

Both write to the StateStore (never to the seed tables) and are idempotent: repeating the same
call returns the existing flag/booking instead of creating a duplicate.

The clinic's flag RULES live here too (`request_rule_flags`, `eligibility_rule_flags`): one source
of truth used both by the agent (to raise flags after its loop) and by the booking guard.

Booking is guarded by CODE. The LLM only chooses a slot_id; everything else the guard checks comes
from the PatientRequest (M1) and the eligibility lookup, not from the LLM. The guard checks the
rules DIRECTLY, so it does not depend on when flags were written. Booking needs ALL of:
  1. the patient explicitly asked to book (wants_booking)
  2. coverage is active
  3. no rule would flag this request (clinical question, insurance missing or mismatched, ...)
     and no flag is already open for it
  4. urgency is Priority or Routine (Emergent / Urgent / Unknown -> a human first)
  5. the slot exists, is open, matches the requested specialty, its provider accepts new
     patients, and nobody has booked it yet
"""
from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel

from capstone.schemas.intake import PatientRequest, Urgency
from capstone.tools.lookups import EligibilityResult, booked_slot_ids
from capstone.tools.referrals import signoff_check
from capstone.tools.repository import SeedTables, StateStore

BOOKABLE_URGENCY = (Urgency.PRIORITY, Urgency.ROUTINE)
URGENT_BANDS = (Urgency.EMERGENT, Urgency.URGENT, Urgency.UNKNOWN)
MAX_NOTE_CHARS = 300


class FlagReason(str, Enum):
    CLINICAL_QUESTION = "clinical_question"
    PATIENT_NOT_FOUND = "patient_not_found"
    COVERAGE_NOT_ACTIVE = "coverage_not_active"
    INSURANCE_MISSING = "insurance_missing"
    INSURANCE_MISMATCH = "insurance_mismatch"
    URGENT_CASE = "urgent_case"
    SPECIALTY_MISSING = "specialty_missing"
    TOOL_FAILURE = "tool_failure"
    OTHER = "other"
    # M6: staff tasks raised by the agent team. Each has its own reason so the store (one flag per
    # record and reason) keeps them apart; as "other" a second one was silently dropped.
    NEEDS_PCP_SIGNOFF = "needs_pcp_signoff"
    REFERRAL_SLA_BREACHED = "referral_sla_breached"
    NO_SLOT_BEFORE_DEADLINE = "no_slot_before_deadline"
    COPAY_MISMATCH = "copay_mismatch"
    SAFETY_REVIEW_REJECTED = "safety_review_rejected"
    SAFETY_ESCALATION = "safety_escalation"
    # M7: the EHR / insurance APIs failed; staff finish the request by hand
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    BOOKING_STATUS_UNKNOWN = "booking_status_unknown"


# The reasons the M2 agent's flag tool offers the LLM (unchanged by M6).
M2_AGENT_REASONS = tuple(r.value for r in FlagReason)[:9]


class Flag(BaseModel):
    flag_id: str
    record_id: str
    patient_ref: str
    reason: FlagReason
    note: str
    raised_by: Literal["llm", "code"]
    status: Literal["queued"] = "queued"
    created_at: str


class Booking(BaseModel):
    booking_id: str
    record_id: str
    patient_ref: str
    slot_id: str
    specialty: str
    starts_at: str
    site_id: str
    provider_id: str
    created_at: str


class BookingResult(BaseModel):
    booked: bool
    booking: Booking | None = None
    refused_reasons: list[str] = []


def request_rule_flags(request: PatientRequest) -> list[tuple[FlagReason, str]]:
    """Flags the clinic's rules raise from the PatientRequest alone."""
    flags = []
    if request.seeks_clinical_advice:
        flags.append((FlagReason.CLINICAL_QUESTION, "patient asked a clinical question"))
    if request.urgency in URGENT_BANDS:
        flags.append((FlagReason.URGENT_CASE, f"urgency {request.urgency.value}"))
    if request.specialty is None:
        flags.append((FlagReason.SPECIALTY_MISSING, "no specialty identified"))
    if request.insurance_id is None:
        flags.append((FlagReason.INSURANCE_MISSING, "no insurance ID in the message"))
    return flags


def eligibility_rule_flags(e: EligibilityResult) -> list[tuple[FlagReason, str]]:
    """Flags the clinic's rules raise once eligibility has been checked."""
    if not e.found:
        return [(FlagReason.PATIENT_NOT_FOUND, "no patient record for this reference")]
    flags = []
    if not e.is_active:
        flags.append((FlagReason.COVERAGE_NOT_ACTIVE, f"coverage status {e.eligibility_status}"))
    if e.insurance_check == "mismatch":
        flags.append((FlagReason.INSURANCE_MISMATCH, "stated insurance ID does not match the record"))
    return flags


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def flags_for_record(state: StateStore, record_id: str) -> list[Flag]:
    return [Flag(**f) for f in state.read("flags") if f["record_id"] == record_id]


def flag_for_human(state: StateStore, record_id: str, patient_ref: str, reason: FlagReason | str,
                   note: str = "", raised_by: Literal["llm", "code"] = "llm") -> Flag:
    """Queue a case for staff. Same record + same reason twice -> the existing flag is returned."""
    reason = FlagReason(reason)                     # unknown reason -> ValueError
    for existing in flags_for_record(state, record_id):
        if existing.reason == reason:
            return existing
    flag = Flag(flag_id=f"MHP-F-{len(state.read('flags')) + 1:05d}", record_id=record_id,
                patient_ref=patient_ref, reason=reason, note=note.strip()[:MAX_NOTE_CHARS],
                raised_by=raised_by, created_at=_now())
    state.append("flags", flag.model_dump(mode="json"))
    return flag


def booking_blockers(state: StateStore, request: PatientRequest,
                     eligibility: EligibilityResult) -> list[str]:
    """Every request-level reason the booking rules say no (any slot). Empty = may book."""
    reasons = []
    if not request.wants_booking:
        reasons.append("patient did not ask to book")
    if not (eligibility.found and eligibility.is_active):
        reasons.append("coverage is not active")
    if request.urgency not in BOOKABLE_URGENCY:
        reasons.append(f"urgency {request.urgency.value} needs a human first")
    # rules not already covered by the checks above or the slot checks
    covered = {FlagReason.URGENT_CASE, FlagReason.SPECIALTY_MISSING,
               FlagReason.COVERAGE_NOT_ACTIVE, FlagReason.PATIENT_NOT_FOUND}
    for reason, note in request_rule_flags(request) + eligibility_rule_flags(eligibility):
        if reason not in covered:
            reasons.append(f"{note}: needs a human first")
    if flags_for_record(state, request.record_id):
        reasons.append("this request has open flags for a human")
    return reasons


def booking_refusals(seed: SeedTables, state: StateStore, request: PatientRequest,
                     eligibility: EligibilityResult, slot_id: str) -> list[str]:
    """Every reason the guard says no to this slot. Empty list = booking allowed."""
    reasons = booking_blockers(state, request, eligibility)

    slot = seed.get("appointment_slots", slot_id=slot_id)
    if slot is None:
        reasons.append(f"slot {slot_id} does not exist")
        return reasons
    provider = seed.get("providers", provider_id=slot["provider_id"])
    if slot["status"] != "open" or slot_id in booked_slot_ids(state):
        reasons.append(f"slot {slot_id} is already taken")
    if request.specialty is None or slot["specialty"] != request.specialty.value:
        reasons.append("slot specialty does not match the request")
    if provider is None or not provider["accepts_new_patients"]:
        reasons.append("provider is not accepting new patients")
    # M6: referral-policy 3.2 - no booking with a sign-off specialty without a live referral
    if request.specialty is not None and signoff_check(seed, request.patient_ref, request.specialty) == "none":
        reasons.append(f"no primary care sign-off on record for {request.specialty.value} (referral policy 3.2)")
    return reasons


def book_slot(seed: SeedTables, state: StateStore, request: PatientRequest,
              eligibility: EligibilityResult, slot_id: str) -> BookingResult:
    """Book one slot for this request if the guard allows it. One booking per request."""
    for b in state.read("bookings"):
        if b["record_id"] == request.record_id:     # already booked for this request
            return BookingResult(booked=True, booking=Booking(**b))
    refusals = booking_refusals(seed, state, request, eligibility, slot_id)
    if refusals:
        return BookingResult(booked=False, refused_reasons=refusals)
    slot = seed.get("appointment_slots", slot_id=slot_id)
    booking = Booking(booking_id=f"MHP-B-{len(state.read('bookings')) + 1:05d}",
                      record_id=request.record_id, patient_ref=request.patient_ref,
                      slot_id=slot_id, specialty=slot["specialty"], starts_at=slot["starts_at"],
                      site_id=slot["site_id"], provider_id=slot["provider_id"], created_at=_now())
    state.append("bookings", booking.model_dump(mode="json"))
    return BookingResult(booked=True, booking=booking)
