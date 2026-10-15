"""Referral tracking rules (referral-policy.md), as plain code over the mock EHR tables. M6.

Used by the Referral Tracker agent, the MCP server and the booking guard, so each rule lives in
exactly one place (README: MCP must not duplicate business rules).

- 3.2 sign-off: every specialty except Dermatology needs primary care sign-off. The evidence in
  the data is a live referral (open / accepted / scheduled) from primary care to that specialty.
- 3.3 SLAs: referral_details.sla_due_on is the deadline; only open / accepted referrals still
  need action (scheduled = actioned; closed / rejected = finished).
- 3.5 escalation ladder, counted in business hours after the deadline: Level 1 at +4 h, Level 2
  at +24 h, Level 3 at +48 h, Level 4 at +72 h. The data has dates only, so one business day
  late = 8 business hours (a clinic day). A 75%-of-window warning ("at risk") comes before that.
- "Today" is the configured as-of date (configs/app.yaml), so results are the same every run.
"""
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel

from capstone.schemas.intake import Specialty
from capstone.tools.repository import SeedTables

OPEN_ACCESS = {Specialty.DERMATOLOGY}                      # referral-policy 3.2.1 (b)
LIVE_REFERRAL = ("open", "accepted", "scheduled")
NEEDS_ACTION = ("open", "accepted")
HOURS_PER_BUSINESS_DAY = 8
LADDER = ((72, 4), (48, 3), (24, 2), (4, 1))               # (hours late, level), highest first


def business_days_between(start: date, end: date) -> int:
    """Weekdays after `start` up to and including `end` (0 if end <= start)."""
    days, current = 0, start
    while current < end:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days += 1
    return days


def escalation_level(hours_late: int) -> int:
    """Policy 3.5 level for a referral this many business hours past its deadline (0 = none)."""
    return next((level for hours, level in LADDER if hours_late >= hours), 0)


class ReferralStatus(BaseModel):
    """One referral as the tracker reports it (no names, no member numbers)."""

    referral_id: str
    to_specialty: str
    status: str                                   # open / accepted / scheduled / closed / rejected
    urgency: str
    raised_on: str
    sla_due_on: str
    preauth_status: str
    needs_action: bool                            # open / accepted
    business_days_late: int = 0
    escalation_level: int = 0                     # policy 3.5 ladder, 0 = not breached
    at_risk: bool = False                         # >= 75% of the SLA window used, not yet late

    @property
    def breached(self) -> bool:
        return self.needs_action and self.business_days_late > 0


def referral_status(referral: dict, details: dict | None, as_of: date) -> ReferralStatus:
    due = date.fromisoformat(details["sla_due_on"]) if details else None
    needs_action = referral["status"] in NEEDS_ACTION
    late = business_days_between(due, as_of) if (due and needs_action) else 0
    at_risk = False
    if due and needs_action and not late:
        raised = date.fromisoformat(referral["raised_on"])
        window = max(business_days_between(raised, due), 1)
        at_risk = business_days_between(raised, as_of) >= 0.75 * window
    return ReferralStatus(
        referral_id=referral["referral_id"], to_specialty=referral["to_specialty"],
        status=referral["status"], urgency=referral["urgency"], raised_on=referral["raised_on"],
        sla_due_on=details["sla_due_on"] if details else "", preauth_status=details["preauth_status"] if details else "unknown",
        needs_action=needs_action, business_days_late=late,
        escalation_level=escalation_level(late * HOURS_PER_BUSINESS_DAY), at_risk=at_risk)


def patient_referrals(seed: SeedTables, patient_ref: str, as_of: date) -> list[ReferralStatus]:
    """All of one patient's referrals with deadline status, newest first."""
    out = [referral_status(dict(r), seed.get("referral_details", referral_id=r["referral_id"]), as_of)
           for r in seed.find("referrals", patient_ref=patient_ref)]
    return sorted(out, key=lambda s: (s.raised_on, s.referral_id), reverse=True)


def live_referral(seed: SeedTables, patient_ref: str, specialty: Specialty) -> dict | None:
    """This patient's newest live referral to the specialty (the sign-off evidence), if any."""
    rows = [r for r in seed.find("referrals", patient_ref=patient_ref, to_specialty=specialty.value)
            if r["status"] in LIVE_REFERRAL]
    if not rows:
        return None
    return dict(max(rows, key=lambda r: (r["raised_on"], r["referral_id"])))


SignoffCheck = Literal["not_required", "live_referral", "none"]


def signoff_check(seed: SeedTables, patient_ref: str, specialty: Specialty) -> SignoffCheck:
    """Policy 3.2: is primary care sign-off needed, and is there evidence of it?"""
    if specialty in OPEN_ACCESS:
        return "not_required"
    return "live_referral" if live_referral(seed, patient_ref, specialty) else "none"
