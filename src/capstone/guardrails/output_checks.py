"""Output gate: nothing an LLM wrote leaves CareFlow unless it is free of clinical advice. M8.

Decided in M8 phase 2 (the second, independent layer after the input gate):
- One entry point over the checks built earlier, so the rule lives in one place:
    M4 policy assistant's own routes (a "clinical_question" refusal, grounding failures -> withheld)
    M6 Clinical Safety Reviewer: code gates (member number -> reject, clinical wording ->
    escalate) then one LLM check, one fresh attempt after a reject, withheld if still rejected.
  They stay where they are (other milestones import them); this module composes them.
- Code-written texts (the assistant's fixed refusals, "not in the sources", "unavailable") need
  no review and pass unchanged.
- A blocked answer is replaced by CODE-WRITTEN text only: the clinic's written deflection script
  (Clinical Escalation standard 4.3, which must be used verbatim), with a fixed citation of the
  standard; if the question showed emergency signs, the 911 / emergency-department line of 5.4
  comes first. The caller never sees the model's draft.
"""
from typing import Literal

from pydantic import BaseModel

from capstone.agents.clinical_safety_reviewer import (  # noqa: F401 - re-exported: one place for output rules
    CLINICAL_WORDS,
    ItemReview,
    SafetyLLM,
    review_policy_answer,
    text_gates,
)
from capstone.rag.answer import Citation, PolicyAnswer

STANDARD = "Clinical Escalation and Scope-of-Practice Standard"
# 4.3 Written/Digital Deflection Script (Patient Portal and Automated Chatbots), verbatim
DEFLECTION_SCRIPT = ("I am an automated assistant / non-clinical staff member and cannot provide medical advice or "
                     "answer clinical questions. Your message has been flagged and forwarded to your provider's "
                     "clinical team for review. A clinical staff member will respond via the portal or phone within "
                     "the standard triage window.")
# 5.4 Emergency exceptions: instruct to dial 911 / go to the nearest emergency department
EMERGENCY_LINE = "If this is a medical emergency, call 911 now or go to the nearest emergency department."
INJECTION_NOTE = "This request tries to change the assistant's rules, so it is not answered."
UNCHECKED_NOTE = ("This question could not be checked safely right now, so it is not answered automatically. "
                  "It has been forwarded to a staff member.")
CITATIONS = {
    "categories": Citation(chunk_id="clinical-escalation#3", doc_title=STANDARD, section="3. MANDATORY ROUTING CATEGORIES"),
    "script": Citation(chunk_id="clinical-escalation#4", doc_title=STANDARD,
                       section="4. MANDATORY DEFLECTION PROTOCOL AND SCRIPTS"),
    "no_exceptions": Citation(chunk_id="clinical-escalation#5", doc_title=STANDARD,
                              section="5. ABSOLUTE ZERO-EXCEPTION RULE"),
}

BlockKind = Literal["clinical", "injection", "unchecked"]
OutVerdict = Literal["pass", "block_clinical", "block_unsafe"]


class GuardedReply(BaseModel):
    """What the caller receives when a guardrail blocks: code-written text and fixed citations."""

    text: str
    citations: list[Citation]
    emergency_line: bool = False


class OutputDecision(BaseModel):
    verdict: OutVerdict
    by: Literal["template", "policy_agent", "code", "llm", "llm_error"]
    reasons: list[str] = []
    reviews: list[ItemReview] = []
    answer: PolicyAnswer | None = None          # the answer to send when verdict == "pass"

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


def blocked_reply(kind: BlockKind, emergency: bool = False) -> GuardedReply:
    """The fixed reply for a blocked question or answer (never LLM text)."""
    parts = [EMERGENCY_LINE] if emergency else []
    if kind == "injection":
        parts.append(INJECTION_NOTE)
    if kind == "unchecked":
        parts.append(UNCHECKED_NOTE)
        return GuardedReply(text=" ".join(parts), citations=[], emergency_line=emergency)
    parts.append(DEFLECTION_SCRIPT)
    cites = [CITATIONS["categories"], CITATIONS["script"]] + ([CITATIONS["no_exceptions"]] if emergency else [])
    return GuardedReply(text=" ".join(parts), citations=cites, emergency_line=emergency)


def check_output(answer: PolicyAnswer, llm: SafetyLLM, retry=None) -> OutputDecision:
    """The output gate for one policy answer. `retry` asks the assistant again (one fresh attempt)."""
    if answer.route == "refused" and answer.refusal_reason == "clinical_question":
        return OutputDecision(verdict="block_clinical", by="policy_agent",
                              reasons=["the policy assistant itself classed the question as clinical"])
    if answer.route == "refused" and answer.refusal_reason == "unsafe_request":
        return OutputDecision(verdict="block_unsafe", by="policy_agent",
                              reasons=["the policy assistant itself classed the request as unsafe"])
    if answer.route != "answered":
        return OutputDecision(verdict="pass", by="template", answer=answer)
    kept, reviews = review_policy_answer(answer, llm, retry)
    last = reviews[-1]
    if kept is not None and kept.route != "answered":     # the fresh attempt came back as a fixed template
        return OutputDecision(verdict="pass", by="template", reviews=reviews, answer=kept)
    if last.verdict == "approve" and kept is not None:
        return OutputDecision(verdict="pass", by=last.by, reviews=reviews, answer=kept)
    if any(r.verdict == "escalate" for r in reviews):
        return OutputDecision(verdict="block_clinical", by=next(r.by for r in reviews if r.verdict == "escalate"),
                              reviews=reviews, reasons=[x for r in reviews for x in r.reasons])
    return OutputDecision(verdict="block_unsafe", by=last.by, reviews=reviews,
                          reasons=[x for r in reviews for x in r.reasons])
