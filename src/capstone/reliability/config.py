"""The reliability settings and the shared breakers, in one place. M7.

ReliabilityConfig holds the phase-2 "balanced" limits (configs/app.yaml `reliability:` can
override them). Reliability bundles them with the live parts every client shares:
- the breakers (ONE set per process, so a dead EHR seen by one request protects the next),
- an optional fault injector (only scenarios and tests pass one),
- sleep / random source (tests replace them so retries take no real time).
"""
import random
import time
from collections.abc import Callable

from pydantic import BaseModel

from capstone.reliability.circuit_breaker import BreakerPolicy, Breakers
from capstone.reliability.faults import FaultInjector
from capstone.reliability.retries import RetryPolicy


class ReliabilityConfig(BaseModel):
    mcp_timeout_s: float = 5.0
    mcp_retry: RetryPolicy = RetryPolicy(max_attempts=3, base_delay_s=0.5, max_delay_s=2.0)
    mcp_breaker: BreakerPolicy = BreakerPolicy(failure_threshold=3, reset_timeout_s=30)
    llm_timeout_s: float = 30.0
    llm_retry: RetryPolicy = RetryPolicy(max_attempts=2, base_delay_s=1.0, max_delay_s=2.0)   # 1 retry
    llm_breaker: BreakerPolicy = BreakerPolicy(failure_threshold=3, reset_timeout_s=60)


class Reliability:
    def __init__(self, config: ReliabilityConfig | None = None, *, injector: FaultInjector | None = None,
                 sleep: Callable[[float], None] = time.sleep, rng: random.Random | None = None,
                 clock: Callable[[], float] = time.monotonic, on_breaker_change=None):
        self.config = config or ReliabilityConfig()
        self.injector, self.sleep, self.rng = injector, sleep, rng or random.Random()
        c = self.config
        self.breakers = Breakers({"insurance": c.mcp_breaker, "ehr": c.mcp_breaker, "booking": c.mcp_breaker,
                                  "llm": c.llm_breaker},
                                 clock=clock, on_change=on_breaker_change or _trace_breaker_change)


def _trace_breaker_change(name: str, old: str, new: str) -> None:
    from capstone.observability.tracing import event
    event("breaker", breaker=name, old=old, new=new)


_shared: Reliability | None = None


def get_reliability() -> Reliability:
    """The process-wide default (limits from configs/app.yaml, no fault injection)."""
    global _shared
    if _shared is None:
        from capstone.settings import get_app_config
        _shared = Reliability(get_app_config().reliability)
    return _shared


def reset_reliability() -> None:
    global _shared
    _shared = None
