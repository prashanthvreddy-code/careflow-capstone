"""Clinical Safety Reviewer: the central check before anything leaves the team. M6.

Decided in M6 phase 2 ("texts + booking, 1 retry"):
- Reviews every LLM-written text (the Insurance agent's policy answer) and every booking BEFORE
  it runs. Code-written text (refusal templates, the fixed emergency notice) needs no review.
- Code gates first (free, exact), then one LLM safety check on text (prompts/reviewer.md):
    member number in the text                  -> reject
    clinical wording (dose, diagnosis, ...)    -> escalate (clinical content)
    LLM: escalate (clinical) / reject / approve
- Reject -> the producing agent tries ONCE more (a fresh answer); still rejected -> the text is
  withheld and staff are flagged. Escalate -> the request goes to the M5 clinical pause.
  "Until it is good" is not a stop condition (slides): the cap is 1 retry, always.
- A booking is re-checked through MCP (check_booking, the same rules as booking) plus the
  referral deadline; any refusal -> not booked + staff flag. A clinical request is never booked.
- Fail safe: if the LLM check fails, the text is withheld (never shown unchecked), as in M4.
"""
import json
import re
from collections.abc import Callable
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from capstone.agents.referral_agent import BookingPlan, booking_request
from capstone.mcp.server import BookingRequest
from capstone.rag.answer import PolicyAnswer
from capstone.schemas.intake import MEMBER_NUMBER, PatientRequest
from capstone.schemas.state import FlagItem, ReferralDecision
from capstone.settings import PROJECT_ROOT

PROMPT = PROJECT_ROOT / "prompts" / "reviewer.md"
MAX_RETRIES = 1
SAFETY_REJECTED = "safety_review_rejected"
# "diagnostic" (a procedure type in the policies) is not a diagnosis (M8 F4)
CLINICAL_WORDS = re.compile(
    r"\b(dos(e|es|age|ing)|\d+\s?mg|tablets?|pills?|diagnos(e|es|ed|is|ing)|prognos\w*|side[- ]effects?|"
    r"prescri\w*|you (may|might|probably) have|sounds like|stop taking|take (one|two|\d))\b", re.IGNORECASE)

Verdict = Literal["approve", "reject", "escalate"]


class LLMVerdict(BaseModel):
    """What prompts/reviewer.md returns."""

    verdict: Verdict
    category: Literal["none", "clinical_advice", "other_unsafe"] = "none"
    reason: str = ""


class ItemReview(BaseModel):
    item: Literal["policy_answer", "booking"]
    verdict: Verdict
    by: Literal["code", "llm", "llm_error"]
    reasons: list[str] = []
    attempt: int = 1


class SafetyReview(BaseModel):
    reviews: list[ItemReview] = []
    policy_answer: PolicyAnswer | None = None     # the answer to keep (None = withheld or none)
    text_withheld: bool = False
    booking_approved: bool = False
    escalate_clinical: bool = False
    flags: list[FlagItem] = []


class SafetyLLM(Protocol):
    def complete_structured(self, messages: list[dict], schema: type) -> Any: ...


class BookingChecker(Protocol):
    def check_booking(self, request: BookingRequest) -> list[str]: ...


def text_gates(text: str) -> tuple[Verdict, list[str]] | None:
    """Exact checks before any LLM. None = passed (go on to the LLM check)."""
    if MEMBER_NUMBER.search(text):
        return "reject", ["contains a member number"]
    found = sorted({m.group(0).lower() for m in CLINICAL_WORDS.finditer(text)})
    if found:
        return "escalate", [f"clinical wording: {', '.join(found)}"]
    return None


def review_text(text: str, purpose: str, llm: SafetyLLM, attempt: int = 1) -> ItemReview:
    gate = text_gates(text)
    if gate:
        return ItemReview(item="policy_answer", verdict=gate[0], by="code", reasons=gate[1], attempt=attempt)
    messages = [{"role": "system", "content": PROMPT.read_text()},
                {"role": "user", "content": json.dumps({"purpose": purpose, "text": text})}]
    try:
        v: LLMVerdict = llm.complete_structured(messages, LLMVerdict)
    except Exception as e:  # noqa: BLE001 - reviewer down -> withhold, never show unchecked text
        return ItemReview(item="policy_answer", verdict="reject", by="llm_error",
                          reasons=[f"safety check failed: {type(e).__name__}"], attempt=attempt)
    verdict = "escalate" if v.category == "clinical_advice" else v.verdict
    return ItemReview(item="policy_answer", verdict=verdict, by="llm",
                      reasons=[v.reason] if v.reason else [], attempt=attempt)


def review_policy_answer(answer: PolicyAnswer | None, llm: SafetyLLM,
                         retry: Callable[[], PolicyAnswer] | None = None) -> tuple[PolicyAnswer | None, list[ItemReview]]:
    """Review the policy answer; one fresh attempt after a reject. Returns (answer to keep, reviews)."""
    if answer is None or answer.route != "answered":
        return answer, []                     # refusals / withheld: code-written text, nothing to review
    purpose = "policy answer about co-pay and pre-authorisation, shown to clinic staff"
    reviews = [review_text(answer.answer, purpose, llm)]
    attempt = 1
    while reviews[-1].verdict == "reject" and attempt <= MAX_RETRIES and retry is not None:
        attempt += 1
        answer = retry()
        if answer is None or answer.route != "answered":
            return answer, reviews
        reviews.append(review_text(answer.answer, purpose, llm, attempt))
    return (answer if reviews[-1].verdict == "approve" else None), reviews


def review_booking(request: PatientRequest, plan: BookingPlan, decision: ReferralDecision,
                   ehr: BookingChecker) -> ItemReview | None:
    """Second check of a proposed booking, before it runs. None = no booking proposed."""
    if plan.status != "proposed" or plan.slot is None:
        return None
    reasons = []
    if request.seeks_clinical_advice:
        reasons.append("clinical request: a clinician decides first")
    if plan.slot.starts_at[:10] > decision.due_on:
        reasons.append(f"slot {plan.slot.starts_at} is after the referral deadline {decision.due_on}")
    reasons += ehr.check_booking(booking_request(request, plan.slot.slot_id))
    return ItemReview(item="booking", verdict="reject" if reasons else "approve", by="code", reasons=reasons)


def run_review(request: PatientRequest, answer: PolicyAnswer | None, plan: BookingPlan | None,
               decision: ReferralDecision | None, llm: SafetyLLM, ehr: BookingChecker,
               retry: Callable[[], PolicyAnswer] | None = None) -> SafetyReview:
    """The reviewer's verdict on everything a routable request would send out or do."""
    kept, reviews = review_policy_answer(answer, llm, retry)
    flags: list[FlagItem] = []
    escalate = any(r.verdict == "escalate" for r in reviews)
    withheld = bool(reviews) and reviews[-1].verdict != "approve"
    if withheld and not escalate:
        flags.append(FlagItem(reason=SAFETY_REJECTED,
                              note=f"policy answer withheld after {len(reviews)} review(s): {reviews[-1].reasons}"))
    booking_ok = False
    if plan is not None and decision is not None and not escalate:
        b = review_booking(request, plan, decision, ehr)
        if b is not None:
            reviews.append(b)
            booking_ok = b.verdict == "approve"
            if not booking_ok:
                flags.append(FlagItem(reason=SAFETY_REJECTED, note=f"booking not made: {b.reasons}"))
    return SafetyReview(reviews=reviews, policy_answer=kept if not withheld else None, text_withheld=withheld,
                        booking_approved=booking_ok, escalate_clinical=escalate, flags=flags)
