"""Bounded retries for transient failures only. M7.

Decided in M7 phase 2 ("balanced"):
- MCP (EHR / insurance) calls: at most 3 attempts, jittered exponential waits (~0.5 s, ~1 s).
- LLM calls: 1 retry on the primary model, then LiteLLM's fallback model (llm/client.py).
- Only TRANSIENT failures are retried: a dropped connection, a timeout, a provider 429/5xx.
  A malformed reply is not transient (asking again "burns the same bad odds", A7 Lab B), and a
  business answer such as "unknown patient" is not a failure at all -> both are raised at once.
- The caller always learns how many attempts were used (it goes on the trace span), and a
  run of retries that never succeeds ends in RetryExhausted, never in an endless loop.
"""
import random
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

# LiteLLM's provider errors that are worth one more try (names, so this module needs no litellm import)
TRANSIENT_NAMES = {"Timeout", "APIConnectionError", "RateLimitError", "ServiceUnavailableError",
                   "InternalServerError", "BadGatewayError"}


class MalformedReply(Exception):
    """A dependency answered, but not in the shape we need (missing or invalid fields)."""


class RetryExhausted(Exception):
    """Every allowed attempt failed with a transient error."""

    def __init__(self, last: BaseException, attempts: int):
        super().__init__(f"gave up after {attempts} attempt(s): {type(last).__name__}: {last}")
        self.last, self.attempts = last, attempts


class RetryPolicy(BaseModel):
    """How hard to try one call. Defaults = the M7 'balanced' MCP settings."""

    max_attempts: int = Field(3, ge=1, le=10)
    base_delay_s: float = Field(0.5, ge=0)     # wait before the 2nd attempt; doubles each time
    max_delay_s: float = Field(2.0, ge=0)
    jitter: bool = True                        # random 50-100% of the wait, so callers don't retry in step


def is_transient(error: BaseException) -> bool:
    """True for failures that may pass if we simply try again."""
    if isinstance(error, MalformedReply):
        return False
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    return any(cls.__name__ in TRANSIENT_NAMES for cls in type(error).__mro__)


def backoff_delay(attempt: int, policy: RetryPolicy, rng: random.Random) -> float:
    """Wait before attempt number `attempt + 1` (attempt counts from 1)."""
    delay = min(policy.base_delay_s * 2 ** (attempt - 1), policy.max_delay_s)
    return delay * rng.uniform(0.5, 1.0) if policy.jitter else delay


def call_with_retry(fn: Callable[[], Any], policy: RetryPolicy, *, sleep: Callable[[float], None] = time.sleep,
                    rng: random.Random | None = None,
                    on_retry: Callable[[int, BaseException, float], None] | None = None) -> tuple[Any, int]:
    """Run fn until it succeeds or the policy is used up. Returns (result, attempts used).

    Non-transient errors (malformed reply, business errors) are raised immediately, unchanged.
    Transient errors on the last attempt end in RetryExhausted.
    """
    rng = rng or random.Random()
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn(), attempt
        except Exception as e:      # classified below; anything not transient is re-raised
            if not is_transient(e):
                raise
            if attempt == policy.max_attempts:
                raise RetryExhausted(e, attempt) from e
            delay = backoff_delay(attempt, policy, rng)
            if on_retry:
                on_retry(attempt, e, delay)
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover
