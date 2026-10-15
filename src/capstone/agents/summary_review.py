"""Summary critique for the M2 staff summary: Reflection (A2 Lab B), plus code checks. M2.

1. Code checks (free, exact): complete (ends with . ! or ?), at most 4 sentences, no member
   number, every $ amount is one the tools returned (the co-pay or the deductible met).
2. LLM review (1 call): is every statement supported by the facts, and is there no clinical
   advice? Returns {"approved": bool, "problems": [...]}.
3. If either finds a problem: ONE rewrite call from the facts. The rewrite is code-checked again;
   if it still fails, the caller falls back to the code-written summary. Capped at one revision.
If the reviewer itself fails, the draft is kept only if the code checks pass.
"""
import json
import re
from typing import Any, Protocol

from pydantic import BaseModel

from capstone.schemas.intake import MEMBER_NUMBER
from capstone.schemas.response import SummaryReview
from capstone.settings import PROJECT_ROOT

REVIEW_PROMPT_PATH = PROJECT_ROOT / "prompts" / "summary_review.md"
REWRITE_PROMPT_PATH = PROJECT_ROOT / "prompts" / "summary_rewrite.md"
MAX_SENTENCES = 4
MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{1,2})?)")
SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


class SummaryVerdict(BaseModel):
    """What the reviewer LLM returns."""

    approved: bool
    problems: list[str] = []


class ReviewLLM(Protocol):
    def complete(self, messages: list[dict], *, json_mode: bool = False) -> str: ...

    def complete_structured(self, messages: list[dict], schema: type) -> Any: ...


def code_checks(summary: str, facts: dict) -> list[str]:
    """Exact checks that need no LLM."""
    problems = []
    if not summary.strip().endswith((".", "!", "?")):
        problems.append("incomplete: does not end with a full sentence")
    sentences = len(SENTENCE_END.findall(summary.strip())) or 1
    if sentences > MAX_SENTENCES:
        problems.append(f"too long: {sentences} sentences (max {MAX_SENTENCES})")
    if MEMBER_NUMBER.search(summary):
        problems.append("contains a member number")
    allowed = {float(v) for v in (facts.get("copay_usd"), facts.get("deductible_met_usd")) if v is not None}
    for amount in MONEY.findall(summary):
        if float(amount.replace(",", "")) not in allowed:
            problems.append(f"amount ${amount} is not a value returned by the tools")
    return problems


def _ask(llm: ReviewLLM, prompt_path, payload: dict) -> list[dict]:
    return [{"role": "system", "content": prompt_path.read_text()},
            {"role": "user", "content": json.dumps(payload)}]


def review_summary(summary: str, facts: dict, llm: ReviewLLM) -> tuple[str | None, SummaryReview]:
    """Return (summary to use, review record). None = use the code-written fallback summary."""
    problems = code_checks(summary, facts)
    approved_by_llm, reviewer_error = False, None
    try:
        verdict = llm.complete_structured(_ask(llm, REVIEW_PROMPT_PATH,
                                               {"facts": facts, "summary": summary}), SummaryVerdict)
        approved_by_llm = verdict.approved
        problems += [p for p in verdict.problems if not verdict.approved]
    except Exception as e:  # noqa: BLE001 - reviewer down -> rely on the code checks
        reviewer_error = type(e).__name__
        approved_by_llm = True

    if not problems and approved_by_llm:
        return summary, SummaryReview(approved=True, problems=[], revised=False, final_source="llm",
                                      reviewer_error=reviewer_error, draft=summary)

    try:
        rewritten = llm.complete(_ask(llm, REWRITE_PROMPT_PATH,
                                      {"facts": facts, "draft": summary, "problems": problems})).strip()
    except Exception as e:  # noqa: BLE001
        rewritten, reviewer_error = "", reviewer_error or type(e).__name__
    if rewritten and not code_checks(rewritten, facts):
        return rewritten, SummaryReview(approved=False, problems=problems, revised=True,
                                        final_source="llm_revised", reviewer_error=reviewer_error,
                                        draft=summary)
    return None, SummaryReview(approved=False, problems=problems, revised=bool(rewritten),
                               final_source="code", reviewer_error=reviewer_error, draft=summary)
