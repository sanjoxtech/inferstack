"""Per-deployment health and admission control for the gateway.

Each deployment has its own circuit breaker, in-flight limit and drain flag, so an
unhealthy or overloaded deployment never blocks other models. The gateway applies
deployment-level admission only; engines keep their own token schedulers.

- closed: requests admitted up to `max_inflight`; beyond that -> overloaded (429).
- open: after `failure_threshold` consecutive upstream failures; requests are
  rejected (503) for `cooldown_seconds`.
- half_open: after cooldown one probe request is admitted; success closes the
  breaker, failure re-opens it.
- draining: no new requests (503); `drain()` waits for in-flight to reach zero or
  a deadline and reports what remained.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


class Rejected(Exception):
    def __init__(self, status, reason, retry_after=None):
        super().__init__(reason)
        self.status, self.reason, self.retry_after = status, reason, retry_after


@dataclass
class DeploymentState:
    max_inflight: int
    failure_threshold: int
    cooldown_seconds: float
    inflight: int = 0
    failures: int = 0
    opened_at: float | None = None
    probing: bool = False
    draining: bool = False
    totals: dict = field(default_factory=lambda: {'admitted': 0, 'succeeded': 0, 'failed': 0, 'rejected': 0})
    idle: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self):
        self.idle.set()


class AdmissionController:
    def __init__(self, *, max_inflight=64, failure_threshold=5, cooldown_seconds=10.0, clock=time.monotonic):
        if max_inflight < 1 or failure_threshold < 1 or cooldown_seconds <= 0:
            raise ValueError('invalid admission policy')
        self.defaults = dict(max_inflight=max_inflight, failure_threshold=failure_threshold,
                             cooldown_seconds=cooldown_seconds)
        self.clock = clock
        self.states: dict[str, DeploymentState] = {}

    def state(self, key) -> DeploymentState:
        if key not in self.states:
            self.states[key] = DeploymentState(**self.defaults)
        return self.states[key]

    def configure(self, key, **policy):
        state = self.state(key)
        for name, value in policy.items():
            setattr(state, name, value)

    def status(self, key):
        s = self.state(key)
        breaker = 'closed' if s.opened_at is None else (
            'half_open' if self.clock() - s.opened_at >= s.cooldown_seconds else 'open')
        return {'breaker': breaker, 'inflight': s.inflight, 'max_inflight': s.max_inflight, 'draining': s.draining,
                'consecutive_failures': s.failures, **s.totals}

    def admit(self, key):
        s = self.state(key)
        if s.draining:
            s.totals['rejected'] += 1
            raise Rejected(503, 'deployment draining', retry_after=5)
        probe = False
        if s.opened_at is not None:
            remaining = s.cooldown_seconds - (self.clock() - s.opened_at)
            if remaining > 0 or s.probing:
                s.totals['rejected'] += 1
                raise Rejected(503, 'deployment unhealthy', retry_after=max(1, int(remaining) + 1))
            s.probing = probe = True
        if s.inflight >= s.max_inflight:
            if probe:
                s.probing = False
            s.totals['rejected'] += 1
            raise Rejected(429, 'deployment at capacity', retry_after=1)
        s.inflight += 1
        s.idle.clear()
        s.totals['admitted'] += 1
        return probe

    def release(self, key, *, ok, probe=False):
        s = self.state(key)
        s.inflight = max(0, s.inflight - 1)
        if s.inflight == 0:
            s.idle.set()
        if probe:
            s.probing = False
        if ok:
            s.totals['succeeded'] += 1
            s.failures, s.opened_at = 0, None
        else:
            s.totals['failed'] += 1
            s.failures += 1
            if probe or s.failures >= s.failure_threshold:
                s.opened_at = self.clock()

    async def drain(self, key, deadline_seconds):
        s = self.state(key)
        s.draining = True
        try:
            await asyncio.wait_for(s.idle.wait(), deadline_seconds)
            return {'drained': True, 'remaining': 0}
        except asyncio.TimeoutError:
            return {'drained': False, 'remaining': s.inflight}

    def undrain(self, key):
        self.state(key).draining = False
