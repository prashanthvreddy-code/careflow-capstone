"""Action checks: every side effect CareFlow can cause, and what must hold before it runs. M8.

Decided in M8 phase 2: gather the action rules in one documented place WITHOUT changing them
(the "tool gateway" of the Day 4 secure-agent flow). The rules themselves were built where they
are used, from M2 on, and stay there; this module names each side effect, its guards, how a
repeat is prevented and whether a person must approve it, and re-exports the guard functions.
Tests check that every side effect the workflow performs is listed here and that every guard
named here exists, so the list cannot silently fall out of date.

Rules that hold for all of them:
- No LLM chooses an MCP tool or a side effect; code decides from checked state (M6).
- Patient / plan identity is filled in by code, never taken from model output (M2).
- Every side effect is idempotent: a repeat or a resume never does it twice (M5).
- Irreversible actions run after their check, and are never retried blindly after a failure (M7).
"""
from importlib import import_module

from pydantic import BaseModel

from capstone.agents.clinical_safety_reviewer import review_booking  # noqa: F401 - re-exported
from capstone.tools.actions import booking_blockers, booking_refusals  # noqa: F401 - re-exported
from capstone.tools.referrals import signoff_check  # noqa: F401 - re-exported
from capstone.workflows.approvals import validate_answer  # noqa: F401 - re-exported


class SideEffect(BaseModel):
    name: str                      # the workflow's action key kind (or the M8 service's)
    does: str
    guards: list[str]              # dotted paths of the functions that must pass first
    repeat_safe: str               # how a second run is prevented
    needs_person: bool             # a human approves before it runs
    since: str


SIDE_EFFECTS: dict[str, SideEffect] = {s.name: s for s in [
    SideEffect(name="book", does="book one appointment slot through MCP book_slot",
               guards=["capstone.tools.actions.booking_refusals",          # coverage, urgency, open flags, slot, sign-off
                       "capstone.agents.clinical_safety_reviewer.review_booking",   # deadline + MCP re-check
                       "capstone.tools.referrals.signoff_check"],
               repeat_safe="one booking per request on the server; action key 'book'; after a failure the "
                           "schedule is looked up and the workflow never books again (M7)",
               needs_person=False, since="M2/M6/M7"),
    SideEffect(name="emergency_notice", does="fixed 911 / emergency-department notice + site manager flag",
               guards=["capstone.workflows.routing.choose_route"],
               repeat_safe="action key 'emergency_notice'", needs_person=False, since="M5"),
    SideEffect(name="pcp_redirect", does="prepare the fixed primary-care redirect (referral-policy 3.2.3) when "
                                         "sign-off evidence is missing; SENT only when a named staff member approves it",
               guards=["capstone.tools.referrals.signoff_check", "capstone.tools.redirects.send_pcp_redirect"],
               repeat_safe="action key 'pcp_redirect'; one redirect per request; sending twice is refused",
               needs_person=True, since="M8 #3"),
    SideEffect(name="escalate", does="send the request to the clinical triage pool",
               guards=["capstone.workflows.approvals.validate_answer"],    # a named reviewer decided
               repeat_safe="action key 'escalate'; flag store keeps one flag per record and reason",
               needs_person=True, since="M5"),
    SideEffect(name="staff_queue", does="queue the request's flags for staff",
               guards=["capstone.workflows.routing.current_flags"],
               repeat_safe="action key 'staff_queue'; one flag per record and reason", needs_person=False,
               since="M5"),
    SideEffect(name="referral_flags", does="queue the referral path's flags after booking",
               guards=["capstone.workflows.routing.current_flags"],
               repeat_safe="action key 'referral_flags'", needs_person=False, since="M6"),
    SideEffect(name="handled_flags", does="queue the other flags after a reviewer said 'handled'",
               guards=["capstone.workflows.approvals.validate_answer"],
               repeat_safe="action key 'handled_flags'", needs_person=True, since="M5"),
    SideEffect(name="record", does="write the outcome to patient memory and save the session",
               guards=["capstone.memory.long_term.make_item"],
               repeat_safe="action key 'record'; memory rows keyed by source id", needs_person=False, since="M3/M5"),
    SideEffect(name="clinician_flag", does="M8: put a blocked policy question in the clinician queue",
               guards=["capstone.guardrails.input_checks.check_input",
                       "capstone.guardrails.output_checks.check_output"],
               repeat_safe="flag store keeps one flag per question id and reason", needs_person=False, since="M8"),
]}


def resolve(path: str):
    """Import a guard by its dotted path (used by the tests and the docs)."""
    module, _, name = path.rpartition(".")
    return getattr(import_module(module), name)


def guards_for(name: str) -> list:
    return [resolve(p) for p in SIDE_EFFECTS[name].guards]
