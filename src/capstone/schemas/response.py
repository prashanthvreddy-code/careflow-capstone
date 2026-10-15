"""What the M2 scheduling/insurance agent returns. M2.

`AgentResult` is the one object handed to staff (and to later milestones): what was checked,
what it costs, which slots were offered or booked, which flags were raised, and a log of every
tool call. The tool result models themselves live next to their tools (tools/lookups.py,
tools/calculators.py, tools/actions.py) and are reused here.

M3 adds: the patient history the agent looked at, the memory rows code wrote after the run, and
the id of the saved session (runtime/memory/sessions/<patient_ref>/<session_id>.json).
"""
from typing import Any, Literal

from pydantic import BaseModel, computed_field

from capstone.memory.long_term import HistoryEntry
from capstone.schemas.intake import Urgency
from capstone.tools.actions import Booking, Flag
from capstone.tools.calculators import CopayQuote
from capstone.tools.lookups import EligibilityResult, SlotInfo

StopReason = Literal["final_answer", "max_rounds", "llm_error"]


class ToolCallRecord(BaseModel):
    """One tool call made during the loop: what was asked, and what came back (or the error)."""

    round: int
    tool: str
    arguments: dict[str, Any]
    ok: bool
    result: Any = None
    error: str | None = None


class SummaryReview(BaseModel):
    """Outcome of the summary critique (code checks + LLM review + at most one rewrite)."""

    approved: bool
    problems: list[str] = []
    revised: bool = False
    final_source: Literal["llm", "llm_revised", "code"]
    reviewer_error: str | None = None
    draft: str | None = None          # the LLM's original summary, kept for audit


class AgentResult(BaseModel):
    record_id: str
    patient_ref: str
    urgency: Urgency
    eligibility: EligibilityResult | None = None
    copay: CopayQuote | None = None
    suggested_slots: list[SlotInfo] = []
    booking: Booking | None = None
    booking_refusals: list[str] = []
    flags: list[Flag] = []
    tool_calls: list[ToolCallRecord] = []
    staff_summary: str = ""
    summary_review: SummaryReview | None = None
    stop_reason: StopReason
    rounds: int = 0
    patient_history: list[HistoryEntry] = []    # M3: what get_patient_history returned (if called)
    memory_written: list[str] = []              # M3: memory_ids added after this run
    session_id: str | None = None               # M3: the saved session for this run

    @computed_field
    @property
    def needs_human(self) -> bool:
        """True whenever any flag was raised: staff must look at this request."""
        return bool(self.flags)
