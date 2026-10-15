"""Input gate: decide whether a question may reach any LLM at all. M8.

Decided in M8 phase 2 (input gate + output gate, refuse by default):
- Runs BEFORE the policy assistant, so a question fishing for a diagnosis or a dose is never
  answered, not just filtered afterwards (A8: the lab's guardrail ran too late).
- Code rules first (free, exact, explainable), from the clinic's own Clinical Escalation and
  Scope-of-Practice Standard, section 3: the five categories that must go to a clinician
    1 symptom interpretation   2 medication questions   3 dosage   4 test-result meaning
    5 "should I" questions (the standard says ANY question phrased "should I")
  plus prompt injection / persona hijack ("you are now MedGPT", "ignore the policy documents").
- If no rule fires, ONE LLM classifier (prompts/input_gate.md) decides: ok / clinical / injection.
  It catches paraphrases the rules miss (e.g. "clinician to clinician, what would you go with?").
- Fail safe: if the classifier fails, the question is NOT answered; it goes to a person
  ("default stance: refuse", Day 4 slides).
- Emergency signs (policy 5.4 and the intake SOP's Emergent list, as word families because no
  intake message used the exact phrases, M1) are detected separately: a refused question that
  shows them gets the 911 / emergency-department line first.
- The decision carries no question text: only the verdict, categories and which layer decided.
"""
import json
import re
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from capstone.settings import PROJECT_ROOT

PROMPT = PROJECT_ROOT / "prompts" / "input_gate.md"
MAX_CHARS = 2000

Category = Literal["symptom_interpretation", "medication", "dosage", "test_result", "should_i", "other_clinical"]

CATEGORY_RULES: dict[str, re.Pattern] = {
    "symptom_interpretation": re.compile(
        r"\b(diagnos\w*|could (this|it|that)( \w+){0,3} be|do you think (this|it|that) (is|could)|what does my \w+ mean|"
        r"what('s| is) wrong with me|is (this|my) (rash|pain|lump|cough|mole|swelling|bleeding) \w+)\b",
        re.IGNORECASE),
    "medication": re.compile(
        r"\b(can i take|side[- ]effects?|interact\w* with|what is (this|my) (pill|medication|drug) for|"
        r"stop taking|safe to take)\b", re.IGNORECASE),
    "dosage": re.compile(
        r"\b(dos(e|es|ing|age)|\d+\s?(mg|mcg|ml)\b|how (much|many) .{0,30}(take|pills?|tablets?)|missed (a |my |the )?\w* ?dose)",
        re.IGNORECASE),
    "test_result": re.compile(
        r"\b(what do(es)? (my |these |the )?(test|lab|blood|scan|mri|x-?ray) (results?|numbers?|values?) mean|"
        r"(blood|white (blood )?cell|wbc) count|is my (mri|scan|x-?ray|ct|ecg) (clear|normal)|"
        r"are (these|my) (numbers|results|levels) normal)", re.IGNORECASE),
    "should_i": re.compile(r"\bshould i\b", re.IGNORECASE),
}
INJECTION_RULE = re.compile(
    r"(ignore (all |any |the )?(previous|prior|above|clinic|your)?\s*(instructions|policy|policies|documents|rules)|"
    r"disregard (the |your )?(system|previous|clinic)? ?(prompt|instructions|policy)|"
    r"reveal (your |the )?(system )?prompt|you are now\b|developer mode|jailbr(oken|eak)|"
    r"(unrestricted|uncensored) (medical )?(ai|assistant|model)|pretend (you are|to be)|"
    r"act as (if you have |having )?no (restrictions|rules|guardrails))", re.IGNORECASE)
EMERGENCY_SIGNS = re.compile(
    r"\b(chest (pain|tightness|pressure)|crushing chest|heart attack|(left )?arm (feels )?(numb|odd|heavy|weak)|"
    r"can'?t breathe|cannot breathe|short(ness)? of breath|difficulty breathing|"
    r"(face|facial)( \w+){0,2} droop\w*|droop\w* (face|mouth|smile)|slurred speech|can'?t speak|inability to speak|stroke|"
    r"sudden(ly)? (loss of vision|lost (my )?vision|blind)|"
    r"suicid\w*|kill (myself|me)|overdos\w*|unconscious|passed out|fainted|"
    r"(heavy|uncontrolled|severe) bleeding|hemorrhag\w*|haemorrhag\w*|"
    r"throat (is )?(swelling|closing)|anaphyla\w*)\b", re.IGNORECASE)

Verdict = Literal["allow", "refuse_clinical", "refuse_injection", "refuse_unchecked", "invalid"]


class GateLLMVerdict(BaseModel):
    """What prompts/input_gate.md returns."""

    verdict: Literal["ok", "clinical", "injection"]
    category: Literal["none", "symptom_interpretation", "medication", "dosage", "test_result", "should_i",
                      "other_clinical"] = "none"
    emergency: bool = False
    reason: str = ""


class InputDecision(BaseModel):
    verdict: Verdict
    categories: list[str] = []
    emergency: bool = False
    by: Literal["code", "llm", "llm_error"] = "code"
    reasons: list[str] = []

    @property
    def allowed(self) -> bool:
        return self.verdict == "allow"


class GateLLM(Protocol):
    def complete_structured(self, messages: list[dict], schema: type) -> Any: ...


def rule_hits(question: str) -> tuple[list[str], bool]:
    """(clinical categories found by the code rules, injection found)."""
    cats = [name for name, rule in CATEGORY_RULES.items() if rule.search(question)]
    return cats, bool(INJECTION_RULE.search(question))


def has_emergency_signs(question: str) -> bool:
    return bool(EMERGENCY_SIGNS.search(question))


def check_input(question: str, llm: GateLLM | None) -> InputDecision:
    """The input gate. llm=None means code rules only (questions no rule catches are refused
    as unchecked, never answered unchecked)."""
    q = (question or "").strip()
    if not q or len(q) > MAX_CHARS:
        return InputDecision(verdict="invalid", reasons=["empty or too long"])
    emergency = has_emergency_signs(q)
    cats, injection = rule_hits(q)
    if injection:
        return InputDecision(verdict="refuse_injection", categories=cats, emergency=emergency, by="code",
                             reasons=["instruction / persona override in the question"])
    if cats:
        return InputDecision(verdict="refuse_clinical", categories=cats, emergency=emergency, by="code",
                             reasons=[f"clinical inquiry (scope-of-practice 3): {', '.join(cats)}"])
    if llm is None:
        return InputDecision(verdict="refuse_unchecked", emergency=emergency, by="code",
                             reasons=["no classifier available"])
    messages = [{"role": "system", "content": PROMPT.read_text()},
                {"role": "user", "content": json.dumps({"question": q})}]
    try:
        v: GateLLMVerdict = llm.complete_structured(messages, GateLLMVerdict)
    except Exception as e:  # noqa: BLE001 - classifier down -> a person decides, never an unchecked answer
        return InputDecision(verdict="refuse_unchecked", emergency=emergency, by="llm_error",
                             reasons=[f"input check failed: {type(e).__name__}"])
    emergency = emergency or v.emergency
    if v.verdict == "injection":
        return InputDecision(verdict="refuse_injection", emergency=emergency, by="llm", reasons=[v.reason])
    if v.verdict == "clinical":
        cat = v.category if v.category != "none" else "other_clinical"
        return InputDecision(verdict="refuse_clinical", categories=[cat], emergency=emergency, by="llm",
                             reasons=[v.reason])
    return InputDecision(verdict="allow", emergency=emergency, by="llm", reasons=[v.reason] if v.reason else [])
