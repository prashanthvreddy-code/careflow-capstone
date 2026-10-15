"""Seeded fault injection: break a dependency on purpose, the same way every run. M7.

Decided in M7 phase 2:
- Targets: the MCP tools (the EHR / insurance APIs) and the LLM call. A rule names a tool
  ("book_slot"), a dependency group ("ehr", "insurance") or an LLM model ("llm:<model>", or
  "llm" for every model). The most specific matching rule wins.
- Fault kinds: connection error, timeout, malformed reply (the call "works" but a required field
  is missing), or down (every call fails). Rates are probabilities per call.
- Seeded: each target has its own random stream (seed + target name), so the faults one tool
  sees don't depend on how often another tool was called. The same seed replays the same faults,
  which is what makes before/after comparisons provable (A7 Lab B).
- A simulated timeout raises at once and records the time it WOULD have cost (the call's timeout),
  so offline scenarios measure latency without really waiting.
- Off unless a FaultInjector is passed in: production code paths never inject anything.
"""
import random
import zlib
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Fault = Literal["ok", "error", "timeout", "malformed"]


class InjectedConnectionError(ConnectionError):
    """A simulated dropped connection."""


class InjectedTimeout(TimeoutError):
    """A simulated timeout (raised at once; the would-be wait is recorded, not slept)."""


class FaultRule(BaseModel):
    error_rate: float = Field(0.0, ge=0, le=1)
    timeout_rate: float = Field(0.0, ge=0, le=1)
    malformed_rate: float = Field(0.0, ge=0, le=1)
    down: bool = False                                # every call fails with a connection error

    @model_validator(mode="after")
    def _total(self):
        if self.error_rate + self.timeout_rate + self.malformed_rate > 1.0:
            raise ValueError("fault rates must not add up to more than 100%")
        return self


class FaultPlan(BaseModel):
    seed: int = 42
    rules: dict[str, FaultRule] = {}


class FaultEvent(BaseModel):
    target: str
    call_no: int
    fault: Fault


# The four offline M7 scenarios (phase 2 decision 7)
SCENARIOS: dict[str, FaultPlan] = {
    "none": FaultPlan(),
    "flaky_ehr": FaultPlan(rules={"ehr": FaultRule(error_rate=0.2, timeout_rate=0.1, malformed_rate=0.1),
                                  "insurance": FaultRule(error_rate=0.2, timeout_rate=0.1, malformed_rate=0.1)}),
    "ehr_down": FaultPlan(rules={"ehr": FaultRule(down=True), "insurance": FaultRule(down=True)}),
    # the primary model hangs: every call times out (the observed Gemini failure mode)
    "llm_primary_down": FaultPlan(rules={"llm:primary": FaultRule(timeout_rate=1.0)}),
}


class FaultInjector:
    def __init__(self, plan: FaultPlan, *, aliases: dict[str, str] | None = None):
        """aliases maps a real name to a plan name, e.g. {"llm:gemini/...": "llm:primary"}."""
        self.plan, self.aliases = plan, aliases or {}
        self._rngs: dict[str, random.Random] = {}
        self._calls: dict[str, int] = {}
        self.events: list[FaultEvent] = []
        self.simulated_wait_s = 0.0

    def _rule(self, keys: list[str]) -> tuple[str, FaultRule] | None:
        for key in keys:
            key = self.aliases.get(key, key)
            if key in self.plan.rules:
                return key, self.plan.rules[key]
        return None

    def roll(self, *keys: str) -> Fault:
        """Decide the fault for one call. keys = most specific first, e.g. ("book_slot", "ehr")."""
        found = self._rule(list(keys))
        target = keys[0]
        n = self._calls[target] = self._calls.get(target, 0) + 1
        fault: Fault = "ok"
        if found:
            _, rule = found
            rng = self._rngs.setdefault(target, random.Random(self.plan.seed + zlib.crc32(target.encode())))
            r = rng.random()
            if rule.down or r < rule.error_rate:
                fault = "error"
            elif r < rule.error_rate + rule.timeout_rate:
                fault = "timeout"
            elif r < rule.error_rate + rule.timeout_rate + rule.malformed_rate:
                fault = "malformed"
        self.events.append(FaultEvent(target=target, call_no=n, fault=fault))
        return fault

    def before_call(self, *keys: str, timeout_s: float = 0.0) -> bool:
        """Raise the injected error/timeout, if any. Returns True when the reply must be malformed."""
        fault = self.roll(*keys)
        if fault == "error":
            raise InjectedConnectionError(f"injected connection error: {keys[0]}")
        if fault == "timeout":
            self.simulated_wait_s += timeout_s
            raise InjectedTimeout(f"injected timeout after {timeout_s:.0f} s: {keys[0]}")
        return fault == "malformed"

    def malform(self, target: str, value: Any) -> Any:
        """Damage a reply: drop a field chosen by this target's seeded stream (any field, required or not)."""
        rng = self._rngs.setdefault(target, random.Random(self.plan.seed + zlib.crc32(target.encode())))
        return malform(value, pick=lambda keys: keys[rng.randrange(len(keys))])

    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for e in self.events:
            out.setdefault(e.target, {}).setdefault(e.fault, 0)
            out[e.target][e.fault] += 1
        return out


def malform(value: Any, pick=None) -> Any:
    """Damage a reply the way a broken API would: drop one field (the first, or `pick`'s choice) of the (first) record;
    a record that is not a dict becomes None; an empty list gets one record of the wrong shape."""
    if isinstance(value, dict):
        keys = list(value)
        drop = (pick or (lambda ks: ks[0]))(keys) if keys else None
        return {k: v for k, v in value.items() if k != drop}
    if isinstance(value, list):
        return [malform(value[0], pick), *value[1:]] if value else [{"unexpected": True}]
    return None
