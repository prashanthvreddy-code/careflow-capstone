"""Safe fallbacks: what CareFlow does when the EHR / insurance APIs stay unavailable. M7.

Decided in M7 phase 2 (the fallback is a PERSON, never a guess):
- Emergent and clinical requests take their own path whatever happens to the EHR: the emergency
  notice and the clinical pause never wait on an API.
- A referral request whose EHR / insurance step fails (retries used up, breaker open, malformed
  reply) stops at that step and goes to the staff queue with a `dependency_unavailable` flag
  naming the step, tool and reason. What was already found (e.g. the co-pay quote) is kept;
  nothing is guessed and nothing is booked.
- Booking is irreversible. book_slot is idempotent per request on the server, so it is retried
  within the cap like any call. If it still fails, look the booking up:
    the patient's appointments show it  -> it happened: record it, no flag;
    the lookup works and it isn't there -> it did not happen: staff may book it;
    the lookup fails too                -> status UNKNOWN: staff must check before rebooking.
  The workflow never books again by itself after a failure (the bank-transfer lesson).
"""
from typing import Literal

from pydantic import BaseModel

from capstone.mcp.client import DependencyUnavailable
from capstone.schemas.state import FlagItem
from capstone.tools.actions import FlagReason
from capstone.tools.lookups import SlotInfo

SAFE_PATHS = ("emergency", "clinical_review")
BookingCheck = Literal["booked", "not_booked", "unknown"]


class DependencyFailure(BaseModel):
    """What the workflow keeps about a failed EHR / insurance step (plain values, no text)."""

    step: str
    tool: str
    dependency: str
    reason: str
    detail: str = ""

    @classmethod
    def from_error(cls, step: str, error: DependencyUnavailable) -> "DependencyFailure":
        return cls(step=step, tool=error.tool, dependency=error.dependency, reason=error.reason, detail=error.detail)

    def flag(self) -> FlagItem:
        return FlagItem(reason=FlagReason.DEPENDENCY_UNAVAILABLE.value,
                        note=(f"{self.dependency} unavailable at step '{self.step}' ({self.tool}: {self.reason}"
                              f"{', ' + self.detail if self.detail else ''}); finish this request by hand"))


def fallback_route(route: str, failure: DependencyFailure | None) -> str:
    """The route after a failure: safe clinical paths are kept, everything else goes to staff."""
    if failure is None or route in SAFE_PATHS:
        return route
    return "staff_queue"


class BookingVerification(BaseModel):
    status: BookingCheck
    appointment_id: str | None = None              # the booking found in the schedule ("booked")
    flag: FlagItem | None = None


def verify_booking(ehr, patient_ref: str, slot: SlotInfo, error: DependencyUnavailable) -> BookingVerification:
    """After book_slot failed: did the booking happen? Ask the EHR once (it has its own retries)."""
    try:
        appointments = ehr.get_patient_appointments(patient_ref)
    except DependencyUnavailable as e:
        return BookingVerification(status="unknown", flag=FlagItem(
            reason=FlagReason.BOOKING_STATUS_UNKNOWN.value,
            note=(f"booking of slot {slot.slot_id} ({slot.specialty}, {slot.starts_at}) failed ({error.reason}) and "
                  f"could not be checked ({e.reason}): CHECK THE SCHEDULE BEFORE REBOOKING")))
    match = next((a for a in appointments if a.source == "booked_by_agent" and a.specialty == slot.specialty
                  and a.scheduled_for == slot.starts_at), None)
    if match is not None:
        return BookingVerification(status="booked", appointment_id=match.appointment_id)
    return BookingVerification(status="not_booked", flag=FlagItem(
        reason=FlagReason.DEPENDENCY_UNAVAILABLE.value,
        note=(f"booking of slot {slot.slot_id} ({slot.specialty}, {slot.starts_at}) failed ({error.reason}); "
              "the schedule shows it was NOT booked: book it by hand")))
