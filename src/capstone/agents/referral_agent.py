"""Referral Tracker agent: the routing decision, the patient's referrals, and a booking plan. M6.

Decided in M6 phase 2 (code only; this agent makes no LLM call):
- The routing decision is M5's (workflows/routing.referral_decision) with "today" = the
  configured as-of date.
- The patient's referrals come from the EHR over MCP with deadline, business days late and the
  policy 3.5 escalation level. All are reported to staff; a flag is raised ONLY when an
  open/accepted referral for the SAME specialty as this request is past its deadline (the other
  breaches are old, unrelated work that would otherwise flag a third of all requests).
- Booking is planned here and executed later, in its own step, after the Clinical Safety
  Reviewer approves (an irreversible action never runs before its check). Code picks the
  earliest open slot that starts on or before the referral deadline; the clinic's booking rules
  are asked through MCP (check_booking, a dry run). No slot in time -> the earliest 3 are
  suggested and staff are told.
"""
from datetime import date
from typing import Literal, Protocol

from pydantic import BaseModel

from capstone.mcp.server import BookingRequest
from capstone.schemas.intake import PatientRequest
from capstone.schemas.state import FlagItem, ReferralDecision
from capstone.tools.actions import BookingResult
from capstone.tools.lookups import SlotInfo
from capstone.tools.referrals import ReferralStatus
from capstone.tools.repository import SeedTables
from capstone.workflows.routing import referral_decision

BREACH_FLAG = "referral_sla_breached"
NO_SLOT_FLAG = "no_slot_before_deadline"
BookingStatus = Literal["not_requested", "not_allowed", "proposed", "no_slot_before_deadline"]


class SchedulingSystem(Protocol):
    """What the tracker needs from the EHR / scheduling MCP client."""

    def get_patient_referrals(self, patient_ref: str) -> list[ReferralStatus]: ...

    def find_open_slots(self, specialty: str, mode: str = "in_person", limit: int = 3,
                        before: str | None = None) -> list[SlotInfo]: ...

    def check_booking(self, request: BookingRequest) -> list[str]: ...

    def book_slot(self, request: BookingRequest) -> BookingResult: ...


class BookingPlan(BaseModel):
    status: BookingStatus
    slot: SlotInfo | None = None                  # the slot to book (status "proposed")
    refusals: list[str] = []                      # the booking rules' reasons (status "not_allowed")
    suggestions: list[SlotInfo] = []              # earliest slots when none fits the deadline


class TrackingResult(BaseModel):
    decision: ReferralDecision
    referrals: list[ReferralStatus]
    breach_flags: list[FlagItem] = []
    booking: BookingPlan


def booking_request(request: PatientRequest, slot_id: str) -> BookingRequest:
    """The summary sent over MCP to check or book a slot (no raw text, no member number)."""
    return BookingRequest(record_id=request.record_id, patient_ref=request.patient_ref,
                          specialty=request.specialty, urgency=request.urgency,
                          wants_booking=request.wants_booking,
                          seeks_clinical_advice=request.seeks_clinical_advice,
                          insurance_stated=request.insurance_id is not None, slot_id=slot_id)


def breach_flags(request: PatientRequest, referrals: list[ReferralStatus]) -> list[FlagItem]:
    """A flag per same-specialty referral past its deadline (policy 3.5 level in the note)."""
    return [FlagItem(reason=BREACH_FLAG,
                     note=(f"{r.referral_id} to {r.to_specialty} is {r.business_days_late} business days "
                           f"past its deadline {r.sla_due_on} (policy 3.5 level {r.escalation_level})"))
            for r in referrals
            if r.breached and request.specialty is not None and r.to_specialty == request.specialty.value]


def plan_booking(request: PatientRequest, decision: ReferralDecision, ehr: SchedulingSystem) -> BookingPlan:
    if not request.wants_booking:
        return BookingPlan(status="not_requested")
    in_time = ehr.find_open_slots(decision.specialty.value, limit=1, before=decision.due_on)
    if not in_time:
        return BookingPlan(status="no_slot_before_deadline",
                           suggestions=ehr.find_open_slots(decision.specialty.value, limit=3))
    slot = in_time[0]
    refusals = ehr.check_booking(booking_request(request, slot.slot_id))
    if refusals:
        return BookingPlan(status="not_allowed", slot=slot, refusals=refusals)
    return BookingPlan(status="proposed", slot=slot)


def run_tracker(request: PatientRequest, ehr: SchedulingSystem, seed: SeedTables, as_of: date) -> TrackingResult:
    """The tracker's work for one routable request (call only when routing said 'referral')."""
    decision = referral_decision(request, seed, as_of)
    referrals = ehr.get_patient_referrals(request.patient_ref)
    plan = plan_booking(request, decision, ehr)
    flags = breach_flags(request, referrals)
    if plan.status == "no_slot_before_deadline":
        flags.append(FlagItem(reason=NO_SLOT_FLAG,
                              note=f"no open {decision.specialty.value} slot on or before {decision.due_on}"))
    return TrackingResult(decision=decision, referrals=referrals, breach_flags=flags, booking=plan)


def execute_booking(request: PatientRequest, plan: BookingPlan, ehr: SchedulingSystem) -> BookingResult | None:
    """Book the proposed slot (the graph calls this only after the reviewer approved it).
    The server re-checks every rule and returns the same booking if it already exists."""
    if plan.status != "proposed" or plan.slot is None:
        return None
    return ehr.book_slot(booking_request(request, plan.slot.slot_id))
