"""Exact cost figures for policy questions, from the co-pay calculator. M8 fix C1.

The policy assistant must never calculate (prompts/rag.md rule 2), so before this fix a question
like "SILVER, $500 of the deductible met, colonoscopy: what will they owe?" ended with "the exact
amount must come from the co-pay calculator" and no amount. Here CODE runs that calculator (M2,
tools/calculators.py: copay-schedule 4.4 on the plan and procedure tables) and hands its result
to the LLM as one extra source, which the LLM may only quote.

The calculator runs only when the question names all of it, otherwise nothing is added:
- exactly one plan (MERIDIAN-GOLD / -SILVER / -BRONZE, CIVIC-BASE);
- a cost intent (owe, pay, co-pay, charged, covers, percent, ...);
- the service: a code (DERM-CONS), a name (Colonoscopy, MRI Knee) or a flat category
  (imaging, lab work, telehealth, specialist visit / consultation). A name that fits several
  services (MRI) is used only if they share one flat category;
- for a procedure (deductible then coinsurance) also the deductible already met ("met $500").
"""
import re
from decimal import Decimal

from capstone.rag.retrieval import Evidence, EvidenceHit, StageRanks
from capstone.tools.calculators import CopayQuote, calculate_copay
from capstone.tools.repository import SeedTables

CALCULATOR_PREFIX = "calculator#"
MAX_QUOTES = 3

INTENT = re.compile(r"\b(owe|owes|owing|pay|pays|paying|co-?pays?|charged?|charges|billed|cost|costs|covers|percent)\b",
                    re.IGNORECASE)
CODE = re.compile(r"\b[A-Z]{3,4}-[A-Z]{3,4}\b")
ALIASES = {"mri": ("ORTH-MRI", "NEUR-MRI")}
CATEGORY_WORDS = [                    # checked only when no code or name matched
    ("telehealth", re.compile(r"\btele-?health\b", re.IGNORECASE)),
    ("imaging", re.compile(r"\bimaging\b", re.IGNORECASE)),
    ("lab", re.compile(r"\blab(oratory)?( work| test| tests)?\b", re.IGNORECASE)),
    ("consultation", re.compile(r"\b(specialist (visit|consultation)|consultation)\b", re.IGNORECASE)),
]
MONEY = r"\$\s?([\d,]+(?:\.\d+)?)"
MET = [re.compile(r"\bmet\s+(?:only\s+)?" + MONEY, re.IGNORECASE),
       re.compile(MONEY + r"\s+(?:of|toward|towards)\s+(?:the\s+|their\s+|her\s+|his\s+)?(?:annual\s+)?deductible",
                  re.IGNORECASE)]


def _plan(question: str, seed: SeedTables) -> str | None:
    found = {p["plan_code"] for p in seed.rows("plans") if re.search(rf"\b{re.escape(p['plan_code'])}\b",
                                                                       question, re.IGNORECASE)}
    return found.pop() if len(found) == 1 else None


def _deductible_met(question: str) -> float | None:
    for pattern in MET:
        m = pattern.search(question)
        if m:
            return float(m.group(1).replace(",", ""))
    return None


def _services(question: str, seed: SeedTables) -> list[str]:
    """Service codes or flat categories named by the question; [] when unclear."""
    procs = {p["procedure_code"]: p for p in seed.rows("procedures")}
    codes = [c for c in dict.fromkeys(CODE.findall(question)) if c in procs]
    if codes:
        return codes
    low = question.lower()
    named = [code for code, p in procs.items() if re.search(rf"\b{re.escape(p['procedure_name'].lower())}\b", low)]
    for alias, alias_codes in ALIASES.items():
        if not named and re.search(rf"\b{alias}\b", low):
            named = list(alias_codes)
            categories = {procs[c]["category"] for c in named}
            flat = all(procs[c]["cost_share_basis"] != "deductible_then_coinsurance" for c in named)
            return [categories.pop()] if len(categories) == 1 and flat else []
    if named:
        return named
    return [cat for cat, pattern in CATEGORY_WORDS if pattern.search(question)][:1]


def cost_quotes(question: str, seed: SeedTables) -> list[CopayQuote]:
    """The calculator's quotes for what the question names, or [] (never a guess)."""
    plan = _plan(question, seed)
    if plan is None or not INTENT.search(question):
        return []
    services = _services(question, seed)
    if not services or len(services) > MAX_QUOTES:
        return []
    procs = {p["procedure_code"]: p for p in seed.rows("procedures")}
    if len({procs[s]["category"] if s in procs else s for s in services}) > 1:
        return []                                  # mixed services: not one clear question
    met = _deductible_met(question)
    quotes = []
    for service in services:
        procedure = service in procs and procs[service]["cost_share_basis"] == "deductible_then_coinsurance"
        if procedure and met is None:
            return []                              # the answer depends on a figure we don't have
        try:
            quotes.append(calculate_copay(seed, plan, service, met or 0))
        except ValueError:
            return []
    return quotes


def _usd(value: float) -> str:
    d = Decimal(str(value))
    return f"${d:,.0f}" if d == d.to_integral_value() else f"${d:,.2f}"


def _working(quote: CopayQuote) -> str:
    """The calculator's working with every amount written like the policy ("$1,000", "$480"),
    plus the coinsurance base (list price minus unmet deductible) it uses but does not print."""
    text = re.sub(r"\$([\d.]+)", lambda m: _usd(float(m.group(1))), quote.working)
    base = re.search(r"of \(\$([\d,.]+) - \$([\d,.]+)\)", text)
    if base:
        remainder = float(base.group(1).replace(",", "")) - float(base.group(2).replace(",", ""))
        text += f"; coinsurance base {_usd(remainder)}"
    return text


def _hit(quote: CopayQuote, rank: int) -> EvidenceHit:
    rule = "flat co-pay from Table 4.1" if quote.basis == "flat_copay" else "unmet deductible plus coinsurance"
    text = (f"Co-pay calculator result for {quote.plan_code}, {quote.service}: the patient owes "
            f"{_usd(quote.amount_usd)} ({quote.amount_usd:,.2f} USD). Rule: {rule} (copay-schedule 4.4). "
            f"Working: {_working(quote)}.")
    return EvidenceHit(chunk_id=f"{CALCULATOR_PREFIX}{quote.plan_code}/{quote.service}",
                       doc_title="CareFlow co-pay calculator", doc_slug="calculator",
                       section="applies copay-schedule 4.4 to the plan and service tables",
                       policy_id=None, effective_date=None, source="tools/calculators.py",
                       text=text, rank=rank, ranks=StageRanks(routed=True))


def with_cost_facts(question: str, evidence: Evidence, seed: SeedTables) -> Evidence:
    """The evidence plus one calculator source per quote (unchanged when nothing applies)."""
    quotes = cost_quotes(question, seed)
    if not quotes:
        return evidence
    hits = list(evidence.hits)
    hits += [_hit(q, len(hits) + i) for i, q in enumerate(quotes, start=1)]
    return evidence.model_copy(update={"hits": hits})
