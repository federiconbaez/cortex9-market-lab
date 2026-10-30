"""Retry with exponential backoff and jitter, with injectable sleep for testing."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from cortex9_market_lab.observability.logging import get_logger

T = TypeVar("T")
log = get_logger("data.retry")


class RetryExhaustedError(RuntimeError):
    """Raised when all retry attempts failed; ``__cause__`` holds the last error."""


@dataclass
class BackoffPolicy:
    max_attempts: int = 5
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    jitter: float = 0.25
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._rng = random.Random(self.seed)

    def delay(self, attempt: int) -> float:
        """Delay before retry ``attempt`` (1-based): base * 2**(attempt-1), capped, ±jitter."""
        raw = min(self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))
        if self.jitter:
            raw *= 1.0 + self._rng.uniform(-self.jitter, self.jitter)
        return max(0.0, min(raw, self.max_delay_s))


def call_with_retry(
    fn: Callable[[], T],
    *,
    policy: BackoffPolicy,
    retry_on: tuple[type[BaseException], ...],
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, BaseException], None] | None = None,
    op: str = "call",
) -> T:
    """Call ``fn``; on an exception in ``retry_on`` wait and retry up to ``policy.max_attempts``.

    Exceptions not listed in ``retry_on`` propagate immediately (e.g. bad symbol).
    """
    last: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except retry_on as exc:
            last = exc
            if attempt == policy.max_attempts:
                break
            wait = policy.delay(attempt)
            log.warning(
                "retrying",
                extra={"op": op, "attempt": attempt, "wait_s": round(wait, 3), "error": repr(exc)},
            )
            if on_retry is not None:
                on_retry(attempt, exc)
            sleep(wait)
    raise RetryExhaustedError(f"{op} failed after {policy.max_attempts} attempts") from last
