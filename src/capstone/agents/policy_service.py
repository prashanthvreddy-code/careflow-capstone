"""The guarded policy question: input gate -> policy assistant -> output gate -> reply. M8.

What POST /policy/ask runs, and what the 20 golden questions are evaluated on.

    question --input gate--> refused?  -> fixed reply + clinician flag (no LLM ever answers it)
             --M4 policy assistant (retrieval, cited answer, grounding check)-->
             --output gate (M6 reviewer: code gates + LLM, 1 fresh attempt)--> blocked? -> fixed reply + flag
             --> the answer, with its citations

Decided in M8 phase 2:
- A blocked question or answer gets CODE-WRITTEN text only (the clinic's verbatim deflection
  script, fixed citations, the 911 line first when emergency signs were seen) and a staff flag:
  clinical / injection -> "clinical_question" (the clinician queue), unchecked -> "tool_failure",
  an answer rejected for other reasons -> "safety_review_rejected". A policy answer that needs a
  person ("human_review") also becomes a staff task ("other"), so "route to a human" is an action.
- Routes match the golden set: "answered" (auto), "refuse", "human_review".
- The question id is a hash of the question unless the caller gives one, so asking the same
  question twice never queues a second flag (flag store: one flag per id and reason).
- The flag note holds the question member-number masked (staff must see what was asked); the
  reply and the trace hold no question text.
"""
import hashlib
from typing import Literal, Protocol

from pydantic import BaseModel

from capstone.guardrails.input_checks import GateLLM, InputDecision, check_input
from capstone.guardrails.output_checks import SafetyLLM, blocked_reply, check_output
from capstone.observability.tracing import get_tracer
from capstone.rag.answer import Citation, PolicyAnswer
from capstone.schemas.state import mask_numbers
from capstone.tools.actions import FlagReason, flag_for_human
from capstone.tools.repository import StateStore

NO_PATIENT = "staff-question"                  # policy questions are not about a stored patient
Route = Literal["answered", "refuse", "human_review"]
ROUTE_OF = {"answered": "answered", "refused": "refuse", "human_review": "human_review"}


class InvalidQuestion(ValueError):
    """Empty or oversized question (the API answers 422)."""


class Policy(Protocol):
    def ask(self, question: str) -> PolicyAnswer: ...


class GuardedAnswer(BaseModel):
    question_id: str
    route: Route
    text: str
    citations: list[Citation] = []
    blocked_by: Literal["input", "output"] | None = None
    input_gate: InputDecision
    output_verdict: str | None = None              # pass / block_clinical / block_unsafe (None: never answered)
    emergency_line: bool = False
    flag_id: str | None = None
    evidence_ids: list[str] = []                   # what retrieval returned (for the evaluation)


def question_id(question: str) -> str:
    return "PQ-" + hashlib.sha256(question.strip().lower().encode()).hexdigest()[:10]


def _flag(store: StateStore, qid: str, reason: FlagReason, why: str, question: str) -> str:
    note = f"{why}. Question: {mask_numbers(question.strip())}"
    return flag_for_human(store, qid, NO_PATIENT, reason, note, raised_by="code").flag_id


def ask_guarded(question: str, *, gate_llm: GateLLM | None, policy: Policy, safety_llm: SafetyLLM,
                store: StateStore, qid: str | None = None) -> GuardedAnswer:
    qid = qid or question_id(question)
    with get_tracer().span("policy.ask_guarded", kind="chain", question_id=qid) as span:
        gate = check_input(question, gate_llm)
        span.set(input_verdict=gate.verdict, input_by=gate.by, categories=gate.categories, emergency=gate.emergency)
        if gate.verdict == "invalid":
            raise InvalidQuestion("the question is empty or too long")
        if not gate.allowed:
            kind = {"refuse_clinical": "clinical", "refuse_injection": "injection"}.get(gate.verdict, "unchecked")
            reply = blocked_reply(kind, gate.emergency)
            reason = FlagReason.TOOL_FAILURE if kind == "unchecked" else FlagReason.CLINICAL_QUESTION
            why = (f"blocked by the input gate ({gate.verdict}, {gate.by}"
                   f"{', ' + ', '.join(gate.categories) if gate.categories else ''})")
            out = GuardedAnswer(question_id=qid, route="human_review" if kind == "unchecked" else "refuse",
                                text=reply.text, citations=reply.citations, blocked_by="input", input_gate=gate,
                                emergency_line=reply.emergency_line, flag_id=_flag(store, qid, reason, why, question))
            span.set(route=out.route, blocked_by="input")
            return out

        answer = policy.ask(question)
        decision = check_output(answer, safety_llm, retry=lambda: policy.ask(question))
        evidence = answer.evidence_ids
        if not decision.passed:
            clinical = decision.verdict == "block_clinical"
            reply = blocked_reply("clinical" if clinical else "unchecked", gate.emergency)
            reason = FlagReason.CLINICAL_QUESTION if clinical else FlagReason.SAFETY_REVIEW_REJECTED
            why = f"answer blocked by the output gate ({decision.verdict}, {decision.by})"
            out = GuardedAnswer(question_id=qid, route="refuse" if clinical else "human_review", text=reply.text,
                                citations=reply.citations, blocked_by="output", input_gate=gate,
                                output_verdict=decision.verdict, emergency_line=reply.emergency_line,
                                flag_id=_flag(store, qid, reason, why, question), evidence_ids=evidence)
            span.set(route=out.route, blocked_by="output")
            return out

        final = decision.answer
        route: Route = ROUTE_OF[final.route]
        flag_id = None
        if route == "human_review":
            flag_id = _flag(store, qid, FlagReason.OTHER, "the policy assistant needs a person to review this", question)
        span.set(route=route, output_verdict=decision.verdict)
        return GuardedAnswer(question_id=qid, route=route, text=final.answer, citations=final.citations,
                             input_gate=gate, output_verdict=decision.verdict, flag_id=flag_id,
                             evidence_ids=evidence)
