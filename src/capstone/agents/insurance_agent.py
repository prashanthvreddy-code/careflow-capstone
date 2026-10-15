"""Insurance agent: coverage, the co-pay quote, and one cited policy answer. M6.

Decided in M6 phase 2 ("facts + cited policy"), for routable requests only:
1. Coverage comes from the EHR over MCP (check_eligibility); never from the model.
2. The co-pay for a specialist consultation comes from the M2 calculator (code, Decimal money).
3. ONE policy question is built by code from the plan and specialty (never from the patient's
   words, so nothing the patient wrote can steer it) and answered by the M4 PolicyAgent: hybrid
   retrieval, citations, code gates, grounding check; a failed check means the answer is withheld.
4. Numbers from code, rules from cited policy (closes M4's "calculated amounts" item): if the
   cited answer states dollar amounts and none equals the calculator's co-pay, the two sources
   disagree and staff get a "copay_mismatch" note. Disagreement never changes the quote.
The policy answer is staff-facing text; the Clinical Safety Reviewer checks it before it is kept.
"""
import re
from typing import Protocol

from pydantic import BaseModel

from capstone.rag.answer import PolicyAnswer
from capstone.schemas.intake import PatientRequest
from capstone.tools.calculators import CopayQuote, calculate_copay
from capstone.tools.lookups import EligibilityResult
from capstone.tools.repository import SeedTables

SERVICE = "consultation"                 # a referral's first visit is a specialist consultation
DOLLARS = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{1,2})?)")
COPAY_MISMATCH = "copay_mismatch"


class Coverage(Protocol):
    def check_eligibility(self, patient_ref: str) -> EligibilityResult: ...


class PolicyAsker(Protocol):
    def ask(self, question: str) -> PolicyAnswer: ...


class InsuranceResult(BaseModel):
    eligibility: EligibilityResult
    copay: CopayQuote | None = None
    question: str | None = None
    policy: PolicyAnswer | None = None
    copay_agrees: bool | None = None          # None = nothing to compare (no answer / no amount)
    notes: list[str] = []


def policy_question(plan_code: str, specialty: str) -> str:
    """The one question asked of the policies. Built by code: plan code + specialty only."""
    return (f"Under the {plan_code} plan, what is the co-pay for a specialist consultation in "
            f"{specialty}, and does that consultation need pre-authorisation?")


def amounts_in(text: str) -> set[float]:
    return {float(m.replace(",", "")) for m in DOLLARS.findall(text or "")}


def copay_agreement(quote: CopayQuote | None, answer: PolicyAnswer | None) -> bool | None:
    """Does the cited policy answer state the calculator's amount? None if there is nothing to compare."""
    if quote is None or answer is None or answer.route != "answered":
        return None
    stated = amounts_in(answer.answer)
    if not stated:
        return None
    return quote.amount_usd in stated


def run_insurance(request: PatientRequest, ehr: Coverage, seed: SeedTables,
                  policy: PolicyAsker | None) -> InsuranceResult:
    """The Insurance agent's work for one routable request."""
    eligibility = ehr.check_eligibility(request.patient_ref)
    notes: list[str] = []
    if not (eligibility.found and eligibility.plan_code):
        return InsuranceResult(eligibility=eligibility, notes=["no plan on record: no quote, no policy question"])
    quote = None
    try:
        quote = calculate_copay(seed, eligibility.plan_code, SERVICE)
    except ValueError as e:
        notes.append(f"co-pay not calculated: {e}")
    question = answer = None
    if policy is not None and request.specialty is not None:
        question = policy_question(eligibility.plan_code, request.specialty.value)
        answer = policy.ask(question)
    agrees = copay_agreement(quote, answer)
    if agrees is False:
        notes.append(f"{COPAY_MISMATCH}: calculator ${quote.amount_usd:.2f} not stated in the cited policy answer")
    return InsuranceResult(eligibility=eligibility, copay=quote, question=question, policy=answer,
                           copay_agrees=agrees, notes=notes)
