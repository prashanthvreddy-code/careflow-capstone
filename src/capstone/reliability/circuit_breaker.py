"""Circuit breaker: stop calling a dependency that is clearly down, and test it again later. M7.

Decided in M7 phase 2 ("balanced"):
- One breaker per dependency group: "insurance" (check_eligibility), "ehr" (referrals,
  appointments, slots, booking check, booking) and one per primary LLM model.
- MCP: opens after 3 failures in a row, stays open 30 s, then lets exactly ONE trial call
  through (half-open). Trial succeeds -> closed again; trial fails -> open for another 30 s.
  LLM primary: same, with a 60 s cool-down (while open, llm/client.py goes straight to the fallback).
- The breaker sits INSIDE the retry loop: every attempt counts. A dead EHR therefore opens the
  breaker during the first request's 3 attempts, and every later request fails at once instead
  of spending its own 3 timeouts (A7 Lab B: "retries make latency worse under sustained outage").
- What counts as a failure: transient errors and malformed replies (the dependency is broken).
  A business answer ("unknown patient", "slot taken") is a working dependency -> it counts as
  a success and resets the count.
- Every state change is reported through on_change, so it lands on the trace.
"""
import threading
import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from capstone.reliability.retries import MalformedReply, is_transient

State = Literal["closed", "open", "half_open"]


class CircuitOpenError(Exception):
    """The breaker is open: the call was refused without touching the dependency."""

    def __init__(self, name: str, retry_in_s: float):
        super().__init__(f"circuit '{name}' is open; next trial in {retry_in_s:.0f} s")
        self.name, self.retry_in_s = name, retry_in_s


class BreakerPolicy(BaseModel):
    failure_threshold: int = Field(3, ge=1)
    reset_timeout_s: float = Field(30.0, ge=0)


def counts_as_failure(error: BaseException) -> bool:
    return isinstance(error, MalformedReply) or is_transient(error)


class CircuitBreaker:
    def __init__(self, name: str, policy: BreakerPolicy | None = None, *,
                 clock: Callable[[], float] = time.monotonic,
                 on_change: Callable[[str, State, State], None] | None = None):
        self.name, self.policy = name, policy or BreakerPolicy()
        self._clock, self._on_change = clock, on_change
        self._lock = threading.Lock()
        self.state: State = "closed"
        self.failures = 0
        self.opened_at: float | None = None
        self._trial_running = False
        self.short_circuited = 0                  # calls refused while open (for the dashboard)

    def _set(self, new: State) -> None:
        old, self.state = self.state, new
        if old != new and self._on_change:
            self._on_change(self.name, old, new)

    def _admit(self) -> None:
        """Decide, before the call, whether it may go through."""
        with self._lock:
            if self.state == "open":
                waited = self._clock() - (self.opened_at or 0.0)
                if waited < self.policy.reset_timeout_s:
                    self.short_circuited += 1
                    raise CircuitOpenError(self.name, self.policy.reset_timeout_s - waited)
                self._set("half_open")
            if self.state == "half_open":
                if self._trial_running:                # exactly one trial call at a time
                    self.short_circuited += 1
                    raise CircuitOpenError(self.name, 0.0)
                self._trial_running = True

    def _record(self, failed: bool) -> None:
        with self._lock:
            trial, self._trial_running = self._trial_running, False
            if not failed:
                self.failures = 0
                self._set("closed")
                return
            self.failures += 1
            if trial or self.failures >= self.policy.failure_threshold:
                self.opened_at = self._clock()
                self._set("open")

    def call(self, fn: Callable[[], Any]) -> Any:
        self._admit()
        try:
            result = fn()
        except BaseException as e:
            self._record(failed=counts_as_failure(e))
            raise
        self._record(failed=False)
        return result


class Breakers:
    """One breaker per dependency name, created on first use and shared by every caller."""

    def __init__(self, policies: dict[str, BreakerPolicy] | None = None, *,
                 clock: Callable[[], float] = time.monotonic,
                 on_change: Callable[[str, State, State], None] | None = None):
        self._policies, self._clock, self._on_change = policies or {}, clock, on_change
        self._by_name: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> CircuitBreaker:
        with self._lock:
            if name not in self._by_name:
                policy = self._policies.get(name) or self._policies.get(name.split(":")[0]) or BreakerPolicy()
                self._by_name[name] = CircuitBreaker(name, policy, clock=self._clock, on_change=self._on_change)
            return self._by_name[name]

    def states(self) -> dict[str, State]:
        return {n: b.state for n, b in self._by_name.items()}
