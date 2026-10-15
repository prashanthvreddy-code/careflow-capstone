"""Intake: raw CareFlow record -> validated PatientRequest. M1 (M2 adds requesting missing info).

Rules agreed for M1:
- The LLM sees ONLY `raw_text`. Never send `ground_truth` or `insurance_id_stated` to the model.
- The LLM assigns a band per symptom; the final urgency is the HIGHEST band (code, never lowered).
- Missing values are None and listed in `missing_fields`; nothing is invented.
"""
from typing import Protocol

from capstone.schemas.intake import (
    MEMBER_NUMBER,
    URGENCY_RANK,
    IntakeExtraction,
    PatientRequest,
    Symptom,
    Urgency,
)
from capstone.settings import PROJECT_ROOT

PROMPT_PATH = PROJECT_ROOT / "prompts" / "intake.md"

class StructuredLLM(Protocol):
    """Anything with complete_structured(): the real LLMClient, or a fake in tests."""

    def complete_structured(self, messages: list[dict], schema: type) -> IntakeExtraction: ...


def highest_urgency(symptoms: list[Symptom]) -> Urgency:
    """Most severe band across symptoms; UNKNOWN if there are none or all are UNKNOWN.

    Use URGENCY_RANK; do not compare enum strings alphabetically.
    """
    if not symptoms:
        return Urgency.UNKNOWN
    return max((s.urgency for s in symptoms), key=lambda band: URGENCY_RANK[band])


def find_missing_fields(extraction: IntakeExtraction) -> list[str]:
    """Names of required-but-absent fields, in this order: "insurance_id", "specialty"."""
    missing = []
    if extraction.insurance_id is None:
        missing.append("insurance_id")
    if extraction.specialty is None:
        missing.append("specialty")
    return missing


def verify_insurance_id(insurance_id: str | None, raw_text: str) -> str | None:
    """Keep the LLM's insurance_id only if it contains a member number that appears in raw_text.

    Rejects patient references (MHP-P-...), invented IDs and IDs not present in the message.
    Never reformats: a kept value is returned exactly as the LLM gave it.
    """
    if not insurance_id:
        return None
    match = MEMBER_NUMBER.search(insurance_id)
    if match is None or match.group() not in raw_text:
        return None
    return insurance_id


def build_messages(raw_text: str) -> list[dict]:
    """[system (from prompts/intake.md), user (the raw_text)]. Nothing else goes to the model."""
    return [
        {"role": "system", "content": PROMPT_PATH.read_text()},
        {"role": "user", "content": raw_text},
    ]


def parse_intake(record: dict, llm: StructuredLLM) -> PatientRequest:
    """One intake record -> PatientRequest.

    1. extraction = llm.complete_structured(build_messages(record["raw_text"]), IntakeExtraction)
    2. copy record_id, patient_ref, channel from the record (not from the LLM)
    3. urgency = highest_urgency(...); missing_fields = find_missing_fields(...)
    """
    extraction = llm.complete_structured(build_messages(record["raw_text"]), IntakeExtraction)
    insurance_id = verify_insurance_id(extraction.insurance_id, record["raw_text"])
    extraction = extraction.model_copy(update={"insurance_id": insurance_id})
    return PatientRequest(
        record_id=record["record_id"],
        patient_ref=record["patient_ref"],
        channel=record["channel"],
        symptoms=extraction.symptoms,
        specialty=extraction.specialty,
        insurance_id=extraction.insurance_id,
        seeks_clinical_advice=extraction.seeks_clinical_advice,
        wants_booking=extraction.wants_booking,
        urgency=highest_urgency(extraction.symptoms),
        missing_fields=find_missing_fields(extraction),
    )
