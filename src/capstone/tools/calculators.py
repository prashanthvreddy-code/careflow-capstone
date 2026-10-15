"""Co-pay calculator. M2. Deterministic code, never the LLM (Co-pay Schedule MHP-POL-412 §4.4).

Rules:
- Consultation, telehealth, imaging and lab: the plan's flat co-pay only (plan_copays table).
- Procedures (cost_share_basis = deductible_then_coinsurance):
    unmet deductible (never below 0, never above the price: if it is more than the list price the
    patient pays the list price; a CALCULATOR ASSUMPTION, copay-schedule 4.4 does not cover the case
    and the working says so, M8 fix P2)
    + coinsurance rate x (list price - unmet deductible),
    capped at the plan's out-of-pocket maximum.
Money is computed with Decimal and rounded to cents, so 0.35 x 2600 is exact.
"""
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from pydantic import BaseModel

from capstone.tools.repository import SeedTables

FLAT_SERVICES = ("consultation", "telehealth", "imaging", "lab")
CENT = Decimal("0.01")


class CopayQuote(BaseModel):
    """What the patient owes for one service, with the rule used and the working shown."""

    plan_code: str
    service: str
    basis: Literal["flat_copay", "deductible_then_coinsurance"]
    amount_usd: float
    working: str


def _money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def calculate_copay(seed: SeedTables, plan_code: str, service: str,
                    deductible_met_usd: float = 0) -> CopayQuote:
    """service = a flat category ("consultation", "telehealth", "imaging", "lab")
    or a procedure code (e.g. "ORTH-ARTH"). Unknown plan or service -> ValueError (never a guess)."""
    plan = seed.get("plans", plan_code=plan_code)
    if plan is None:
        raise ValueError(f"unknown plan: {plan_code}")
    if deductible_met_usd < 0:
        raise ValueError("deductible_met_usd cannot be negative")

    category = service
    if service not in FLAT_SERVICES:
        procedure = seed.get("procedures", procedure_code=service)
        if procedure is None:
            raise ValueError(f"unknown service or procedure code: {service}")
        if procedure["cost_share_basis"] == "deductible_then_coinsurance":
            return _procedure_quote(plan, service, Decimal(procedure["list_price_usd"]),
                                    Decimal(str(deductible_met_usd)))
        category = procedure["category"]          # e.g. CARD-ECHO is billed as flat "imaging"

    row = seed.get("plan_copays", plan_code=plan_code, service_category=category)
    if row is None:
        raise ValueError(f"no co-pay for {plan_code} / {category}")
    return CopayQuote(plan_code=plan_code, service=service, basis="flat_copay",
                      amount_usd=float(row["copay_usd"]),
                      working=f"flat {category} co-pay for {plan_code}: ${row['copay_usd']}")


def _procedure_quote(plan, service: str, price: Decimal, met: Decimal) -> CopayQuote:
    deductible = Decimal(plan["annual_deductible_usd"])
    rate = Decimal(str(plan["coinsurance_rate"]))
    oop_max = Decimal(plan["out_of_pocket_max_usd"])

    full_unmet = max(deductible - met, Decimal(0))
    unmet = min(full_unmet, price)
    coinsurance = _money(rate * (price - unmet))
    total = unmet + coinsurance
    capped = min(total, oop_max)

    if full_unmet > price:
        working = (f"unmet deductible ${full_unmet} (${deductible} - ${met} met) is more than the list price "
                   f"${price}, so the patient pays the list price ${price} and no coinsurance -> ${_money(total)} "
                   f"(calculator assumption: copay-schedule 4.4 does not cover this case)")
    else:
        working = (f"unmet deductible ${unmet} (${deductible} - ${met} met) + "
                   f"{rate * 100:.0f}% of (${price} - ${unmet}) = ${coinsurance} -> ${_money(total)}")
    if capped < total:
        working += f"; capped at out-of-pocket max ${oop_max}"
    return CopayQuote(plan_code=plan["plan_code"], service=service,
                      basis="deductible_then_coinsurance", amount_usd=float(_money(capped)),
                      working=working)
