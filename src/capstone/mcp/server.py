"""The mock EHR / scheduling system as an MCP server (official MCP Python SDK, FastMCP). M6.

Decided in M6 phase 2:
- Every tool is a thin wrapper around an existing function in tools/ (README: MCP exposes the same
  tool functions used locally; it does not duplicate business rules). The booking guard, the
  sign-off rule and the referral ladder therefore run identically with or without MCP.
- EHR side: check_eligibility, get_patient_appointments, get_patient_referrals.
  Scheduling side: find_open_slots, check_booking (dry run), book_slot, flag_for_human. The co-pay calculator and policy
  search are not EHR/scheduling, so they stay local.
- Nothing sensitive crosses MCP: no tool takes or returns a member number, a name or a raw
  message. Eligibility here checks coverage only (the insurance-ID comparison happens in the
  intake step, which already has the message). Booking sends a request SUMMARY; the server
  re-reads eligibility itself, because the system that owns the data decides (A6: the seller
  owns state).
- Patient references are validated; a bad argument comes back as a tool error, never a crash.
- "Today" is the configured as-of date (configs/app.yaml).

Run as a stdio server (for any MCP client):  .venv/bin/python -m capstone.mcp.server
"""
from datetime import date

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from capstone.memory.long_term import PATIENT_REF
from capstone.schemas.intake import PatientRequest, Specialty, Urgency
from capstone.settings import get_app_config
from capstone.tools import actions, lookups, referrals
from capstone.tools.actions import BookingResult, Flag, FlagReason
from capstone.tools.lookups import AppointmentInfo, EligibilityResult, SlotInfo
from capstone.tools.referrals import ReferralStatus
from capstone.tools.repository import SeedTables, StateStore, default_seed, default_state

SERVER_NAME = "careflow-ehr-scheduling"
RECORD_ID = r"^CAREFLOW-\d{5}$"


class BookingRequest(BaseModel):
    """What the caller sends to book: a summary of the request, never the raw message."""

    record_id: str = Field(pattern=RECORD_ID)
    patient_ref: str = Field(pattern=r"^MHP-P-\d{5}$")
    specialty: Specialty
    urgency: Urgency
    wants_booking: bool
    seeks_clinical_advice: bool
    insurance_stated: bool
    slot_id: str


def _check_ref(patient_ref: str) -> str:
    if not PATIENT_REF.fullmatch(patient_ref or ""):
        raise ValueError(f"not a patient reference: {patient_ref!r}")
    return patient_ref


def build_server(seed: SeedTables | None = None, state: StateStore | None = None,
                 as_of: date | None = None) -> FastMCP:
    """One server definition, used in-process by the workflow and over stdio by other clients."""
    seed = seed or default_seed()
    state = state or default_state()
    as_of = as_of or get_app_config().as_of_date
    mcp = FastMCP(SERVER_NAME, log_level="WARNING", instructions=(
        "Mock EHR and scheduling system of a multi-specialty clinic (synthetic data). Tools return "
        "facts only; they never give clinical advice. Booking is checked by the clinic's rules."))

    # --- EHR -----------------------------------------------------------------------------------
    @mcp.tool()
    def check_eligibility(patient_ref: str) -> EligibilityResult:
        """Coverage for one patient: found, plan, status (active / pending_verification / lapsed).
        Never returns names or member numbers."""
        return lookups.check_eligibility(seed, _check_ref(patient_ref))

    @mcp.tool()
    def get_patient_appointments(patient_ref: str) -> list[AppointmentInfo]:
        """This patient's EHR appointments plus any booked through this system, oldest first."""
        return lookups.get_patient_appointments(seed, state, _check_ref(patient_ref))

    @mcp.tool()
    def get_patient_referrals(patient_ref: str) -> list[ReferralStatus]:
        """This patient's referrals with deadline, business days late and the referral-policy 3.5
        escalation level, as of the system date. Newest first."""
        return referrals.patient_referrals(seed, _check_ref(patient_ref), as_of)

    # --- scheduling ------------------------------------------------------------------------------
    @mcp.tool()
    def find_open_slots(specialty: Specialty, mode: str = "in_person", limit: int = 3,
                        before: str | None = None) -> list[SlotInfo]:
        """Earliest open slots for a specialty (providers accepting new patients). `before` is an
        ISO date: only slots starting on or before that day (e.g. a referral deadline)."""
        return lookups.find_open_slots(seed, state, specialty.value, mode=mode, limit=limit, before=before)

    def _as_request(request: BookingRequest) -> PatientRequest:
        return PatientRequest(
            record_id=request.record_id, patient_ref=request.patient_ref, channel="web_form",
            symptoms=[], specialty=request.specialty, urgency=request.urgency,
            insurance_id="[STATED]" if request.insurance_stated else None,
            seeks_clinical_advice=request.seeks_clinical_advice, wants_booking=request.wants_booking,
            missing_fields=[])

    @mcp.tool()
    def check_booking(request: BookingRequest) -> list[str]:
        """Dry run of book_slot: every reason the clinic's booking rules would refuse this slot
        (empty = allowed). Books nothing."""
        eligibility = lookups.check_eligibility(seed, request.patient_ref)
        return actions.booking_refusals(seed, state, _as_request(request), eligibility, request.slot_id)

    @mcp.tool()
    def book_slot(request: BookingRequest) -> BookingResult:
        """Book one slot for one request if the clinic's booking rules allow it (coverage active,
        the patient asked, Routine/Priority, no open flags, slot free, primary-care sign-off on
        record). One booking per request: repeating the call returns the same booking."""
        eligibility = lookups.check_eligibility(seed, request.patient_ref)    # the owner decides
        return actions.book_slot(seed, state, _as_request(request), eligibility, request.slot_id)

    @mcp.tool()
    def flag_for_human(record_id: str, patient_ref: str, reason: FlagReason, note: str = "") -> Flag:
        """Queue a request for staff. The same record and reason twice returns the existing flag."""
        return actions.flag_for_human(state, record_id, _check_ref(patient_ref), reason, note, raised_by="code")

    return mcp


if __name__ == "__main__":
    build_server().run("stdio")
