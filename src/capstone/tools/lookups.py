"""Read-only lookup tools for the M2 agent: insurance eligibility, a patient's appointments, open slots.

Entity-scoped: every patient lookup takes one patient_ref and returns only that patient's data.
Never returned: names, birth dates, member numbers (the eligibility check compares the member
number internally and reports only match / mismatch / not_stated).
"""
from typing import Literal

from pydantic import BaseModel

from capstone.schemas.intake import MEMBER_NUMBER, Specialty
from capstone.tools.repository import SeedTables, StateStore

InsuranceCheck = Literal["match", "mismatch", "not_stated", "not_checked"]


class EligibilityResult(BaseModel):
    patient_ref: str
    found: bool
    plan_code: str | None = None
    plan_name: str | None = None
    eligibility_status: str | None = None          # active / pending_verification / lapsed
    is_active: bool = False                        # only "active" counts as covered
    deductible_met_usd: float | None = None
    insurance_check: InsuranceCheck = "not_checked"


class AppointmentInfo(BaseModel):
    appointment_id: str
    specialty: str
    scheduled_for: str
    status: str
    source: Literal["ehr", "booked_by_agent"]


class SlotInfo(BaseModel):
    slot_id: str
    specialty: str
    starts_at: str
    duration_minutes: int
    mode: str
    site_id: str
    site_name: str
    provider_id: str


def compare_insurance(stated: str | None, plan_code: str, member_number: str) -> InsuranceCheck:
    """Does the insurance ID from the message belong to this patient's record?

    not_stated: nothing given. match: the member number in the text equals the record's and, if a
    plan name is given, it is the record's plan. Anything else (other number, other plan, no
    recognisable member number) is a mismatch -> a human checks it.
    """
    if not stated or not stated.strip():
        return "not_stated"
    found = MEMBER_NUMBER.search(stated)
    if found is None or found.group() != member_number:
        return "mismatch"
    plan_part = stated.replace(found.group(), "").strip()
    if plan_part and plan_part.upper() != plan_code.upper():
        return "mismatch"
    return "match"


def check_eligibility(seed: SeedTables, patient_ref: str,
                      stated_insurance_id: str | None = None) -> EligibilityResult:
    """Coverage for one patient. Unknown patient -> found=False (never a guess, never 'active')."""
    patient = seed.get("patients", patient_ref=patient_ref)
    if patient is None:
        return EligibilityResult(patient_ref=patient_ref, found=False)
    plan = seed.get("plans", plan_code=patient["plan_code"])
    return EligibilityResult(
        patient_ref=patient_ref,
        found=True,
        plan_code=patient["plan_code"],
        plan_name=plan["plan_name"] if plan else None,
        eligibility_status=patient["eligibility_status"],
        is_active=patient["eligibility_status"] == "active",
        deductible_met_usd=patient["deductible_met_usd"],
        insurance_check=compare_insurance(stated_insurance_id, patient["plan_code"],
                                          patient["member_number"]),
    )


def get_patient_appointments(seed: SeedTables, state: StateStore,
                             patient_ref: str) -> list[AppointmentInfo]:
    """This patient's EHR appointments plus any the agent booked, oldest first."""
    ehr = [AppointmentInfo(appointment_id=a["appointment_id"], specialty=a["specialty"],
                           scheduled_for=a["scheduled_for"], status=a["status"], source="ehr")
           for a in seed.find("appointments", patient_ref=patient_ref)]
    booked = [AppointmentInfo(appointment_id=b["booking_id"], specialty=b["specialty"],
                              scheduled_for=b["starts_at"], status="scheduled",
                              source="booked_by_agent")
              for b in state.read("bookings") if b["patient_ref"] == patient_ref]
    return sorted(ehr + booked, key=lambda a: a.scheduled_for)


def booked_slot_ids(state: StateStore) -> set[str]:
    return {b["slot_id"] for b in state.read("bookings")}


def find_open_slots(seed: SeedTables, state: StateStore, specialty: str,
                    mode: str = "in_person", limit: int = 3, before: str | None = None) -> list[SlotInfo]:
    """Earliest open slots for a specialty, from providers accepting new patients,
    leaving out slots the agent already booked. Unknown specialty -> ValueError.
    M6: `before` (an ISO date) keeps only slots starting on or before that day (a deadline)."""
    if specialty not in {s.value for s in Specialty}:
        raise ValueError(f"unknown specialty: {specialty}")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    taken = booked_slot_ids(state)
    sites = {s["site_id"]: s["site_name"] for s in seed.rows("sites")}
    accepting = {p["provider_id"] for p in seed.rows("providers") if p["accepts_new_patients"]}
    slots = [s for s in seed.find("appointment_slots", specialty=specialty, mode=mode, status="open")
             if s["provider_id"] in accepting and s["slot_id"] not in taken
             and (before is None or s["starts_at"][:10] <= before)]
    slots.sort(key=lambda s: s["starts_at"])
    return [SlotInfo(slot_id=s["slot_id"], specialty=s["specialty"], starts_at=s["starts_at"],
                     duration_minutes=s["duration_minutes"], mode=s["mode"], site_id=s["site_id"],
                     site_name=sites.get(s["site_id"], s["site_id"]), provider_id=s["provider_id"])
            for s in slots[:limit]]
