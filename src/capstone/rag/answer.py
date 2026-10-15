"""Answer a policy question from retrieved evidence, or refuse; never show an unsupported answer. M4.

Flow for one question (decided in phase 2):
1. LLM answer call (prompts/rag.md): the passages go in as an id-tagged, untrusted data block;
   the model returns json {route, answer, citations, refusal_reason}.
2. Refusals are written by CODE from fixed templates (the model only picks the reason), so a
   refusal can never carry clinical wording. A clinical refusal always cites the clinical-
   escalation policy's routing section, whether or not search found it.
3. Code gates on an answer (free, exact): it cites something; every cited id was retrieved; every
   number and every code / plan name in the answer appears in a cited passage; no member number;
   not over-long.
4. LLM grounding check (prompts/rag_grounding.md), only if the code gates pass: is every claim
   supported by the cited passages?
5. Any failure, including either LLM call failing, WITHHOLDS the answer: route "human_review",
   nothing shown, the draft and the reasons kept for audit. Fail safe, as in M2.
Amounts for a specific patient still come from the M2 co-pay calculator, never from here.
"""
import json
import re
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from capstone.rag.keyword import tokenize
from capstone.rag.retrieval import Evidence, EvidenceHit
from capstone.schemas.intake import MEMBER_NUMBER
from capstone.settings import PROJECT_ROOT

ANSWER_PROMPT = PROJECT_ROOT / "prompts" / "rag.md"
GROUNDING_PROMPT = PROJECT_ROOT / "prompts" / "rag_grounding.md"
MAX_ANSWER_WORDS = 120
CLINICAL_POLICY = ("clinical-escalation#3", "Clinical Escalation and Scope-of-Practice Standard",
                   "3. MANDATORY ROUTING CATEGORIES")

NUMBER = re.compile(r"(?<![\w-])\$?\d[\d,]*(?:\.\d+)?%?")
CODE = re.compile(r"(?<![\w-])[A-Z]{2,}(?:-[A-Z0-9]+)+(?![\w-])")

Route = Literal["answered", "refused", "human_review"]
RefusalReason = Literal["clinical_question", "not_in_sources", "unsafe_request"]

TEMPLATES = {
    "clinical_question": "This is a clinical question. Under the Clinical Escalation and Scope-of-Practice "
                         "Standard it must be routed to a licensed clinician; no clinical answer is given.",
    "not_in_sources": "The retrieved policy sections do not answer this, so no answer is given. Please "
                      "route the question to a staff member.",
    "unsafe_request": "This request tries to override the clinic policies, so it is not answered.",
    "human_review": "The policies only partly cover this or appear to conflict; a staff member should "
                    "review the cited sections.",
    "withheld": "An answer was drafted but did not pass the grounding checks, so it is withheld for "
                "staff review.",
    "unavailable": "The policy assistant is unavailable right now; please route the question to a staff member.",
}


class LLMAnswer(BaseModel):
    """What the answer prompt returns."""

    route: Route
    answer: str = ""
    citations: list[str] = []
    refusal_reason: RefusalReason | None = None


class GroundingVerdict(BaseModel):
    """What the grounding prompt returns."""

    grounded: bool
    unsupported_claims: list[str] = []


class Citation(BaseModel):
    chunk_id: str
    doc_title: str
    section: str


class GroundingReport(BaseModel):
    code_problems: list[str] = []
    llm_grounded: bool | None = None          # None = not checked (refusal, code gate failed, error)
    unsupported_claims: list[str] = []
    checker_error: str | None = None
    redrafted: bool = False                   # M8 #1 G2: this is the one redraft after a failed check
    first_attempt: dict | None = None         # what the first draft failed on (code_problems / unsupported)


class PolicyAnswer(BaseModel):
    question: str
    route: Route
    answer: str                               # what staff see (an answer, or a fixed message)
    citations: list[Citation] = []
    refusal_reason: RefusalReason | None = None
    withheld: bool = False                    # True: a drafted answer failed a check and is hidden
    draft: str | None = None                  # the model's own text, kept for audit
    grounding: GroundingReport = GroundingReport()
    evidence_ids: list[str] = []
    mode: str | None = None
    latency_ms: int | None = None             # filled in by the agent (retrieval + answer + check)


class AnswerLLM(Protocol):
    def complete_structured(self, messages: list[dict], schema: type) -> Any: ...


def _numbers(text: str) -> set[str]:
    return {t for m in NUMBER.findall(text) for t in tokenize(m)}


def code_gates(answer: str, citations: list[str], evidence: dict[str, EvidenceHit],
               question: str = "") -> list[str]:
    """Exact checks that need no LLM. Returns the problems found (empty = passed).

    A number the answer repeats from the question (e.g. refuting "100 percent") is not a new fact,
    so it is exempt here; the LLM grounding check still judges whether it is used correctly."""
    problems = []
    if not citations:
        problems.append("no citations")
    unknown = [c for c in citations if c not in evidence]
    if unknown:
        problems.append(f"cites passages that were not retrieved: {unknown}")
    cited_text = " ".join(evidence[c].text + " " + evidence[c].section for c in citations if c in evidence)
    cited_numbers = _numbers(cited_text)
    for n in sorted(_numbers(answer) - cited_numbers - _numbers(question)):
        problems.append(f"number {n} is not in the cited passages")
    for code in sorted(set(CODE.findall(answer))):
        if code.lower() not in cited_text.lower():
            problems.append(f"code {code} is not in the cited passages")
    if MEMBER_NUMBER.search(answer):
        problems.append("contains a member number")
    if len(answer.split()) > MAX_ANSWER_WORDS:
        problems.append(f"too long: {len(answer.split())} words (max {MAX_ANSWER_WORDS})")
    if not answer.strip():
        problems.append("empty answer")
    return problems


def _sources(hits: list[EvidenceHit]) -> list[dict]:
    return [{"id": h.chunk_id, "document": h.doc_title, "section": h.section,
             "effective_date": h.effective_date, "text": h.text} for h in hits]


def _messages(prompt_path, payload: dict) -> list[dict]:
    return [{"role": "system", "content": prompt_path.read_text()},
            {"role": "user", "content": json.dumps(payload)}]


REDRAFT_NOTE = ("A check found problems in your answer:\n{problems}\n\nWrite the answer again using only facts "
                "the sources state for this exact service and plan: correct or drop each problem claim and "
                "cite every source you use. Same json format.")


def answer_question(question: str, evidence: Evidence, llm: AnswerLLM) -> PolicyAnswer:
    """Answer call -> code gates -> LLM grounding check. M8 #1 G2: if the draft fails a check, ONE
    redraft is asked for (told what failed) and goes through the same checks; still failing ->
    withheld. A first draft that passes costs no extra call."""
    by_id = {h.chunk_id: h for h in evidence.hits}
    base = {"question": question, "evidence_ids": list(by_id), "mode": evidence.mode}
    messages = _messages(ANSWER_PROMPT, {"question": question, "sources": _sources(evidence.hits)})

    # 1. the answer call
    try:
        out: LLMAnswer = llm.complete_structured(messages, LLMAnswer)
    except Exception as e:  # noqa: BLE001 - any LLM failure -> safe fixed message, never a crash
        return PolicyAnswer(**base, route="human_review", answer=TEMPLATES["unavailable"],
                            grounding=GroundingReport(checker_error=f"answer call failed: {type(e).__name__}"))
    first, failed = _check(question, out, by_id, base, llm)
    if failed is None:
        return first

    # one redraft, told what failed
    problems = "\n".join(f"- {p}" for p in failed["code_problems"] + failed["unsupported_claims"])
    retry = messages + [{"role": "assistant", "content": out.model_dump_json()},
                        {"role": "user", "content": REDRAFT_NOTE.format(problems=problems)}]
    try:
        out2: LLMAnswer = llm.complete_structured(retry, LLMAnswer)
    except Exception:  # noqa: BLE001 - redraft failed -> the first result (withheld) stands
        return first
    second, _ = _check(question, out2, by_id, base, llm)
    second.grounding = second.grounding.model_copy(update={"redrafted": True, "first_attempt": failed})
    return second


def _check(question: str, out: LLMAnswer, by_id: dict, base: dict, llm: AnswerLLM
           ) -> tuple[PolicyAnswer, dict | None]:
    """Route one draft through the checks. Returns the answer and, when it was WITHHELD by a check
    (code gate or grounding verdict, not a checker error), what it failed on."""
    cited = [c for c in dict.fromkeys(out.citations) if c in by_id]
    citations = [Citation(chunk_id=c, doc_title=by_id[c].doc_title, section=by_id[c].section) for c in cited]

    # 2. refusals and human review: fixed, code-written text
    if out.route == "refused" or (out.route == "answered" and out.refusal_reason):
        reason = out.refusal_reason or "not_in_sources"
        if reason == "clinical_question":
            citations = [Citation(chunk_id=CLINICAL_POLICY[0], doc_title=CLINICAL_POLICY[1],
                                  section=CLINICAL_POLICY[2])]
        else:
            citations = []
        return PolicyAnswer(**base, route="refused", answer=TEMPLATES[reason], refusal_reason=reason,
                            citations=citations, draft=out.answer or None), None
    if out.route == "human_review":
        return PolicyAnswer(**base, route="human_review", answer=TEMPLATES["human_review"],
                            citations=citations, draft=out.answer or None), None

    # 3. code gates on a real answer
    problems = code_gates(out.answer, out.citations, by_id, question=question)
    if problems:
        return (PolicyAnswer(**base, route="human_review", answer=TEMPLATES["withheld"], withheld=True,
                             citations=citations, draft=out.answer,
                             grounding=GroundingReport(code_problems=problems)),
                {"code_problems": problems, "unsupported_claims": []})

    # 4. LLM grounding check
    try:
        verdict: GroundingVerdict = llm.complete_structured(
            _messages(GROUNDING_PROMPT, {"question": question, "answer": out.answer,
                                         "cited_sources": _sources([by_id[c] for c in cited])}),
            GroundingVerdict)
    except Exception as e:  # noqa: BLE001 - checker down -> withhold (never show an unchecked answer)
        return PolicyAnswer(**base, route="human_review", answer=TEMPLATES["withheld"], withheld=True,
                            citations=citations, draft=out.answer,
                            grounding=GroundingReport(checker_error=f"grounding check failed: {type(e).__name__}")), None
    report = GroundingReport(llm_grounded=verdict.grounded, unsupported_claims=verdict.unsupported_claims)
    if not verdict.grounded:
        return (PolicyAnswer(**base, route="human_review", answer=TEMPLATES["withheld"], withheld=True,
                             citations=citations, draft=out.answer, grounding=report),
                {"code_problems": [], "unsupported_claims": verdict.unsupported_claims})

    # 5. shown
    return PolicyAnswer(**base, route="answered", answer=out.answer, citations=citations,
                        draft=out.answer, grounding=report), None
