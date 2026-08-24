"""In-process rate limiting for authentication endpoints.

WHY THIS EXISTS
---------------
Atlas AI previously distinguished "no account with that email" from "wrong
password" because it is friendlier. That made `/auth/login` a user-enumeration
oracle: anyone could script it to discover which addresses are registered,
which makes targeted phishing and credential stuffing cheaper.

That distinction is now gone (see `auth_service.authenticate_user`), but
removing it is only half the defence. Without a limit, an attacker can still
mount an online password-guessing attack at whatever rate the server accepts.
Rate limiting is what turns "one guess is cheap" into "a million guesses are
not".

SCOPE, HONESTLY
---------------
This is an in-process counter. It is correct for a single application process
— which is how Atlas AI is deployed, one uvicorn worker behind Compose — and
it does NOT coordinate across processes or machines. Running several workers
would multiply the effective limit by the worker count.

That is a deliberate, documented boundary rather than an oversight. Shared
state means Redis or a database table on the hot path of every login; for the
current deployment shape it would be complexity without benefit. The class
below is the seam where a shared backend would slot in: `check` and
`record_failure` are the only two operations the rest of the code knows about.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class SlidingWindowRateLimiter:
    """Counts failures per key over a moving time window.

    A sliding window rather than a fixed one: fixed windows allow a burst of
    2N attempts across a boundary (N at the end of one window, N at the start
    of the next), which is exactly the pattern an attacker would use.

    Only *failures* are recorded. Someone logging in successfully twenty times
    is a person with several devices, not an attack, and locking them out
    would be a self-inflicted outage.
    """

    def __init__(self, *, max_attempts: int, window_seconds: float) -> None:
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        # Requests are served concurrently and this state is shared, so the
        # read-modify-write below must not interleave.
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque[float]:
        events = self._events[key]
        cutoff = now - self.window_seconds
        while events and events[0] <= cutoff:
            events.popleft()
        return events

    def retry_after(self, key: str) -> float | None:
        """Seconds until `key` may try again, or None if it may try now."""
        now = time.monotonic()
        with self._lock:
            events = self._prune(key, now)
            if len(events) < self.max_attempts:
                return None
            # The window frees up when the oldest recorded failure ages out.
            return max(0.0, events[0] + self.window_seconds - now)

    def record_failure(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._prune(key, now)
            self._events[key].append(now)

    def reset(self, key: str) -> None:
        """Forget a key's failures. Called after a successful login, so one
        forgotten password does not count against the next hour."""
        with self._lock:
            self._events.pop(key, None)

    def clear(self) -> None:
        """Drop all state. For tests; never called by request handling."""
        with self._lock:
            self._events.clear()
