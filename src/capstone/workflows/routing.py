"""Code routers and the referral decision for the M5 workflow. M5.

Every decision here is a plain function of the request, the eligibility result and the seed
tables: no LLM, no graph. That is what makes the routing testable in one line (A5: decide in a
node, route in an edge). The graph (workflows/graph.py) only calls these functions.

Route order (a safety rule, agreed in phase 2):
1. Emergent                          -> emergency  (act first, then the clinical pause)
2. clinical question                 -> clinical_review  (pause before any escalation)
3. cannot proceed (see `blockers`)   -> staff_queue
4. otherwise                         -> referral  (a routing decision; no booking in M5)
After a reviewer answers "not_clinical", the request is routed again without the clinical label;
an Emergent request is never referred (referral-policy 3.3.5), so it goes to the staff queue.

Referral decision (referral-policy.md):
- 3.2 primary care sign-off is required for every specialty except Dermatology. The evidence is
  a live referral (open / accepted / scheduled) from primary care to that specialty for this
  patient. None found -> "needs_pcp_signoff" flag for staff, never an automatic rejection.
- 3.3 + urgency_bands: Routine 5, Priority 2, Urgent 1 business days; due date counts weekdays.
- Pre-authorisation status comes from the live referral's details; without one it is "unknown".
"""
from datetime import date, timedelta

from capstone.schemas.intake import PatientRequest, Urgency
from capstone.schemas.state import FlagItem, Preauth, ReferralDecision, Route
from capstone.tools.actions import FlagReason, eligibility_rule_flags, request_rule_flags
from capstone.tools.lookups import EligibilityResult
from capstone.tools.referrals import OPEN_ACCESS, live_referral
from capstone.tools.repository import SeedTables

NEEDS_SIGNOFF = "needs_pcp_signoff"
PREAUTH_VALUES = ("not_required", "approved", "pending", "denied")


def blockers(request: PatientRequest, eligibility: EligibilityResult | None) -> list[str]:
    """Why this request cannot get a referral decision yet (empty = routable)."""
    reasons = []
    if eligibility is None:
        reasons.append("eligibility not checked")
    elif not eligibility.found:
        reasons.append("patient not found")
    else:
        if not eligibility.is_active:
            reasons.append(f"coverage {eligibility.eligibility_status}")
        if eligibility.insurance_check == "mismatch":
            reasons.append("stated insurance ID does not match the record")
    if request.specialty is None:
        reasons.append("no specialty")
    if request.insurance_id is None:
        reasons.append("no insurance ID")           # SOP 4: a coordinator runs the decline protocol
    if request.urgency == Urgency.UNKNOWN:
        reasons.append("urgency unknown")
    return reasons


def choose_route(request: PatientRequest | None, eligibility: EligibilityResult | None, *,
                 clinical_cleared: bool = False, error: str | None = None) -> Route:
    """Where the request goes next. Pure function: same inputs, same route."""
    if error or request is None:
        return "staff_queue"
    if request.urgency == Urgency.EMERGENT:
        return "staff_queue" if clinical_cleared else "emergency"
    if request.seeks_clinical_advice and not clinical_cleared:
        return "clinical_review"
    if blockers(request, eligibility):
        return "staff_queue"
    return "referral"


def current_flags(request: PatientRequest | None, eligibility: EligibilityResult | None, *,
                  clinical_cleared: bool = False, error: str | None = None,
                  referral: ReferralDecision | None = None) -> list[FlagItem]:
    """The flags that describe the request now (the M2 code rules, plus M5's sign-off flag)."""
    pairs: list[tuple[str, str]] = []
    if error or request is None:
        pairs.append((FlagReason.OTHER.value, f"intake failed: {error or 'no request'}"))
    else:
        pairs += [(r.value, n) for r, n in request_rule_flags(request)]
        if eligibility is not None:
            pairs += [(r.value, n) for r, n in eligibility_rule_flags(eligibility)]
        if clinical_cleared:
            pairs = [(r, n) for r, n in pairs if r != FlagReason.CLINICAL_QUESTION.value]
    if referral is not None and referral.needs_signoff:
        pairs.append((NEEDS_SIGNOFF, f"no live primary care referral to {referral.specialty.value}"))
    seen, flags = set(), []
    for reason, note in pairs:
        if reason not in seen:
            seen.add(reason)
            flags.append(FlagItem(reason=reason, note=note))
    return flags


def add_business_days(start: date, days: int) -> date:
    """start + `days` weekdays (Saturday and Sunday skipped; the data has no holiday calendar)."""
    current, left = start, days
    while left > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            left -= 1
    return current


def first_response_hours(seed: SeedTables, urgency: Urgency) -> int:
    """urgency_bands first_response_hours. Unknown is treated as Urgent until a person decides."""
    band = Urgency.URGENT if urgency == Urgency.UNKNOWN else urgency
    row = seed.get("urgency_bands", urgency=band.value)
    if row is None:
        raise ValueError(f"no urgency band for {band.value}")
    return int(row["first_response_hours"])


def referral_decision(request: PatientRequest, seed: SeedTables, today: date) -> ReferralDecision:
    """The routing decision for a routable request (call only when choose_route said 'referral')."""
    if request.specialty is None:
        raise ValueError("a referral decision needs a specialty")
    band = seed.get("urgency_bands", urgency=request.urgency.value)
    if band is None or band["referral_sla_business_days"] is None:
        raise ValueError(f"urgency {request.urgency.value} is never referred")
    sla = int(band["referral_sla_business_days"])
    required = request.specialty not in OPEN_ACCESS
    ref = live_referral(seed, request.patient_ref, request.specialty) if required else None
    preauth: Preauth = "unknown"
    if ref is not None:
        details = seed.get("referral_details", referral_id=ref["referral_id"])
        if details and details["preauth_status"] in PREAUTH_VALUES:
            preauth = details["preauth_status"]
    return ReferralDecision(
        specialty=request.specialty,
        urgency=request.urgency,
        signoff_required=required,
        signoff_status="not_required" if not required else ("live_referral" if ref else "none"),
        signoff_referral_id=ref["referral_id"] if ref else None,
        referral_status=ref["status"] if ref else None,
        sla_business_days=sla,
        due_on=add_business_days(today, sla).isoformat(),
        preauth=preauth,
    )
