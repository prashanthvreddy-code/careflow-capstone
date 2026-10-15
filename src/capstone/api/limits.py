"""API rate and size limits. M8 improvement #6 (ship gate 8, operability).

- Per API key at most `rate_limit_per_minute` requests in any 60-second window (sliding window);
  the next one gets 429 with Retry-After (seconds until a slot frees up). Counted only AFTER the
  key is checked, so wrong-key calls cannot use up a real caller's allowance. /health is never
  limited.
- A request body larger than `max_body_bytes` gets 413 before any parsing, LLM call or workflow
  step (checked from Content-Length; the request models also cap each field's length).
In memory, one process: enough for the single local service; a shared store (e.g. Redis) would be
needed behind several workers.
"""
import math
import threading
import time
from collections import deque
from collections.abc import Callable

from pydantic import BaseModel


class ApiLimits(BaseModel):
    rate_limit_per_minute: int = 60
    max_body_bytes: int = 16384


class RateLimiter:
    def __init__(self, limit: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> int | None:
        """None = allowed (and counted); otherwise the seconds to wait."""
        now = self.clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= now - self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(1, math.ceil(hits[0] + self.window_s - now))
            hits.append(now)
            return None
