"""Validated intake models for CareFlow. M1.

Two layers:
- `IntakeExtraction`: what the LLM returns (JSON mode). Only things readable from `raw_text`.
- `PatientRequest`: the final object, built by *our code* from the record metadata + extraction.
  The LLM never sets `urgency` (final) or `missing_fields` directly.
"""
import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel


class Urgency(str, Enum):
    """SOP §3 urgency bands plus UNKNOWN (no symptom could be banded -> a human decides)."""

    EMERGENT = "Emergent"
    URGENT = "Urgent"
    PRIORITY = "Priority"
    ROUTINE = "Routine"
    UNKNOWN = "Unknown"


# Severity order used to pick the highest band. Higher number = more severe.
URGENCY_RANK: dict[Urgency, int] = {
    Urgency.EMERGENT: 4,
    Urgency.URGENT: 3,
    Urgency.PRIORITY: 2,
    Urgency.ROUTINE: 1,
    Urgency.UNKNOWN: 0,
}


class Specialty(str, Enum):
    """The 8 specialties present in the CareFlow data."""

    CARDIOLOGY = "Cardiology"
    DERMATOLOGY = "Dermatology"
    ENT = "ENT"
    ENDOCRINOLOGY = "Endocrinology"
    GASTROENTEROLOGY = "Gastroenterology"
    NEUROLOGY = "Neurology"
    ORTHOPAEDICS = "Orthopaedics"
    PULMONOLOGY = "Pulmonology"


# Member numbers in CareFlow data: MER or CIV + 8 digits (e.g. MER12345678). Data-specific.
MEMBER_NUMBER = re.compile(r"\b(?:MER|CIV)\d{8}\b")

Channel = Literal["phone_transcript", "email", "web_form", "sms"]


class Symptom(BaseModel):
    """One symptom in the patient's own words, with the band the LLM assigned to it."""

    text: str
    urgency: Urgency


class IntakeExtraction(BaseModel):
    """Exactly what the LLM must return. Missing information -> None, never invented."""

    symptoms: list[Symptom]
    specialty: Specialty | None = None
    insurance_id: str | None = None
    seeks_clinical_advice: bool
    # M2: True only if the patient explicitly asks to book/schedule an appointment.
    # Defaults to False (the safe value: no booking without a clear request).
    wants_booking: bool = False


class PatientRequest(BaseModel):
    """Final validated intake object handed to later milestones."""

    record_id: str
    patient_ref: str
    channel: Channel
    symptoms: list[Symptom]
    specialty: Specialty | None
    insurance_id: str | None
    seeks_clinical_advice: bool
    wants_booking: bool = False
    urgency: Urgency
    missing_fields: list[str]
