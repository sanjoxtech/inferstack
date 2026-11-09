"""Progressive rollout controller (canary stages with automatic rollback).

Lifecycle: `start` attests the candidate release (when an attestation function
is configured) and routes the first stage's weight to it. Each `step` compares
candidate and stable observations over the current stage window:
- candidate error rate above `max_error_rate` -> rollback;
- candidate median time-to-first-byte regressing more than
  `max_latency_regression` (fraction) over the comparison baseline -> rollback;
- too few candidate requests, unknown latency, or the stage's minimum duration
  not yet elapsed -> hold (never advance on missing evidence);
- otherwise advance to the next stage; after the 100% stage holds cleanly the
  candidate is promoted to stable.

Comparison baseline: stable traffic in the same window while it exists. When the
stage routes all traffic to the candidate, stable gets no requests; the controller
then compares against the stable latency retained from the most recent earlier
stage that had stable traffic, and records that basis. A rollout with no such
baseline (a single 100% stage) holds rather than inventing one.

Ownership and crash safety: the rollout records the exact route body and revision
it last wrote. Before every route mutation the live route must still match; any
other writer (a concurrent rollout, an operator) moves the rollout to `conflict`
without touching the route. Each route change is a two-phase transition: the
intent (target route body, expected revision, follow-up rollout fields) is
persisted on the rollout first, then the route is written, then the rollout
progress. A controller that finds an unfinished intent completes it idempotently
(route already applied -> record progress; route not applied -> apply it).
Every transition is a new revision of the rollout record (audit trail).
`pause`, `resume` and `abort` are operator controls.

Ownership is re-checked after every observation await, immediately before the
transition, and a transition that ends in `conflict` never has physical side
effects. Retiring the losing runtime is a durable obligation: the terminal
transition records `retirement: pending` atomically with the new state, the
runtime is drained and stopped afterwards, and the obligation is marked done only
after it completes. A controller that finds a terminal rollout with a pending
retirement (crash, exception) resumes it on the next `step`.
"""
from __future__ import annotations

import time

from inferstack.releases import gates
from inferstack.releases.manager import ReleaseManager

TERMINAL = {'promoted', 'rolled_back', 'failed', 'conflict'}
NOW = '__now__'  # placeholder in follow-up fields, resolved when the transition completes


class RolloutError(RuntimeError):
    pass


def _route_view(body):
    return {'stable': body.get('stable'), 'candidate': body.get('candidate'), 'weight': body.get('weight', 0),
            'previous': body.get('previous')}


class RolloutController:
    def __init__(self, repository, *, observe, attest=None, clock=time.time, runtime=None):
        """observe(release_id, since) -> {'requests','errors','ttfb_p50_ms'}; attest(release) -> dict with 'status'.

        Library callers may omit `attest` (tests, offline tooling); the control API
        refuses to run rollouts without one (see create_control_app).
        """
        self.repository, self.observe, self.attest, self.clock = repository, observe, attest, clock
        self.releases = ReleaseManager(repository)
        # Optional ReleaseRuntime: launches/warms the candidate before traffic and
        # drains + stops the losing release after promotion or rollback.
        self.runtime = runtime

    def _read(self, rollout_id):
        record = self.repository.read('rollout', rollout_id)
        return record['revision'], dict(record['body'])

    def _write(self, rollout_id, body, revision, **changes):
        body = {**body, **changes, 'updated_at': self.clock()}
        return self.repository.put_record('rollout', rollout_id, body, expected_revision=revision), body

    # --- two-phase route transitions ---------------------------------------------

    def _owned_route(self, body):
        """Live route if it is still exactly what this rollout last wrote, else None."""
        route = self.repository.read('route', body['model'])
        owned = body.get('route_revision') == route['revision'] and \
            _route_view(route['body']) == body.get('route_body')
        return route if owned else None

    def _transition(self, rollout_id, revision, body, target, follow_up):
        """Persist intent, write the route, then record progress. Returns (revision, body)."""
        intent = {'route': _route_view(target), 'expected_route_revision': body['route_revision'],
                  'follow_up': follow_up}
        revision, body = self._write(rollout_id, body, revision, transition=intent)
        return self._complete(rollout_id, revision, body)

    def _complete(self, rollout_id, revision, body):
        intent = body['transition']
        route = self.repository.read('route', body['model'])
        if _route_view(route['body']) == intent['route'] and route['revision'] > intent['expected_route_revision']:
            route_revision = route['revision']  # applied before an interruption
        elif route['revision'] == intent['expected_route_revision']:
            target = intent['route']
            for identifier in filter(None, (target['stable'], target['candidate'])):
                release = self.repository.read('release', identifier)
                if release['body']['deployment']['body']['model'] != body['model']:
                    raise ValueError('release model mismatch')
            route_revision = self.repository.put_record('route', body['model'], target,
                                                        expected_revision=intent['expected_route_revision'])
        else:
            return self._write(rollout_id, body, revision, state='conflict', transition=None,
                               reason='route changed outside this rollout during a transition; '
                                      'operator reconciliation required')
        follow = {k: (self.clock() if v == NOW else v) for k, v in intent['follow_up'].items()}
        return self._write(rollout_id, body, revision, transition=None, route_revision=route_revision,
                           route_body=intent['route'], **follow)

    def _conflict(self, rollout_id, revision, body):
        self._write(rollout_id, body, revision, state='conflict',
                    reason='route no longer matches this rollout (concurrent writer); route left unchanged')
        return 'conflict'

    # --- lifecycle ---------------------------------------------------------------

    async def start(self, rollout_id, model, release_id, stages, *, min_requests=20, evaluation=None):
        """evaluation: {'required': bool, 'suite_sha256'?, 'evaluator_sha256'?}. When required,
        the release's bound held-out evaluation must pass before any stage advances:
        missing/corrupt holds, a recorded quality regression rolls back."""
        route = self.repository.read('route', model)
        if route['body'].get('candidate'):
            raise RolloutError(f'{model} already has a candidate in flight')
        stable = route['body']['stable']
        if stable == release_id:
            raise RolloutError('release is already stable')
        self.repository.read('release', release_id)
        now = self.clock()
        body = {'model': model, 'release_ref': release_id, 'previous_release_ref': stable, 'stable_ref': stable,
                'stages': stages, 'state': 'pending', 'current_stage': 0, 'updated_at': now,
                'min_requests': min_requests, 'observations': {}, 'stage_started_at': None,
                'route_revision': route['revision'], 'route_body': _route_view(route['body']),
                'transition': None, 'stable_baseline': None, 'evaluation_policy': evaluation,
                'policy_sha256': gates.policy_sha256(stages, {'min_requests': min_requests,
                                                              'evaluation': evaluation}),
                'evidence_refs': []}
        revision = self.repository.put_record('rollout', rollout_id, body)
        if self.runtime is not None:
            try:
                await self.runtime.ensure(release_id)
            except Exception as exc:
                self._write(rollout_id, body, revision, state='failed', reason=f'candidate runtime unavailable: {exc}')
                raise RolloutError(f'candidate runtime unavailable: {exc}') from exc
        if self.attest is not None:
            result = await self.attest(self.repository.read('release', release_id))
            if result.get('status') != 'attested':
                self._write(rollout_id, body, revision, state='failed',
                            reason=f"attestation {result.get('status')}: {result.get('checks')}")
                raise RolloutError(f"candidate release not attested: {result.get('status')}")
        target = {'stable': stable, 'candidate': release_id, 'weight': stages[0]['weight'],
                  'previous': route['body'].get('previous')}
        self._transition(rollout_id, revision, body, target,
                         {'state': 'progressing', 'stage_started_at': NOW,
                          'reason': f"stage 0 at {stages[0]['weight']}%"})
        return self.repository.read('rollout', rollout_id)

    async def _rollback(self, rollout_id, revision, body, reason):
        route = self._owned_route(body)
        if route is None:
            return self._conflict(rollout_id, revision, body)
        target = {'stable': body['stable_ref'], 'candidate': None, 'weight': 0,
                  'previous': route['body'].get('previous')}
        revision, body = self._transition(rollout_id, revision, body, target, {
            'state': 'rolled_back', 'reason': reason, **self._retirement(body['release_ref'])})
        if body['state'] == 'rolled_back':
            await self._settle_retirement(rollout_id)
        return body['state']

    def _retirement(self, release_id):
        """Follow-up fields recording the retirement obligation with the terminal transition."""
        if self.runtime is None:
            return {}
        return {'retirement': {'release_id': release_id, 'state': 'pending', 'attempts': 0, 'last_error': None}}

    async def _settle_retirement(self, rollout_id):
        """Run a pending retirement obligation; it is marked done only after the runtime confirms."""
        revision, body = self._read(rollout_id)
        obligation = body.get('retirement') or {}
        if obligation.get('state') != 'pending' or self.runtime is None:
            return obligation or None
        route = self.repository.read('route', body['model'])['body']
        if obligation['release_id'] in (route.get('stable'), route.get('candidate')):
            # Never stop a runtime the live route still serves (operator restored it).
            revision, body = self._write(rollout_id, body, revision, retirement={
                **obligation, 'state': 'skipped', 'last_error': 'release is serving on the live route'})
            return body['retirement']
        attempt = {**obligation, 'attempts': obligation.get('attempts', 0) + 1}
        try:
            outcome = await self.runtime.retire(obligation['release_id'])
        except Exception as exc:
            self._write(rollout_id, body, revision,
                        retirement={**attempt, 'last_error': f'{type(exc).__name__}: {exc}'[:500]})
            raise
        orphaned = [a for a in (outcome or {}).get('allocations', []) if a.get('state') != 'released']
        state = 'done' if not orphaned else 'pending'
        self._write(rollout_id, body, revision, retirement={**attempt, 'state': state, 'outcome': outcome,
                    'last_error': None if not orphaned else 'allocations not confirmed stopped'})
        return {**attempt, 'state': state}

    async def step(self, rollout_id):
        revision, body = self._read(rollout_id)
        if body.get('transition'):
            # Recover an interrupted route transition before evaluating anything else.
            revision, body = self._complete(rollout_id, revision, body)
        if body['state'] in ('promoted', 'rolled_back') and (body.get('retirement') or {}).get('state') == 'pending':
            await self._settle_retirement(rollout_id)  # resume an interrupted cleanup obligation
            return body['state']
        if body['state'] in TERMINAL or body['state'] in ('paused', 'pending'):
            return body['state']
        if self._owned_route(body) is None:
            return self._conflict(rollout_id, revision, body)
        stage = body['stages'][body['current_stage']]
        since = body['stage_started_at']
        until = self.clock()  # a closed window: both sides are measured over [since, until)
        try:
            cand = await self._observe(body['release_ref'], since, until)
            stable = await self._observe(body['stable_ref'], since, until)
        except Exception as exc:  # noqa: BLE001 - telemetry loss is an explicit hold, never a pass
            elapsed = self.clock() - since
            if stage.get('max_duration_s') is not None and elapsed >= stage['max_duration_s']:
                return await self._rollback(rollout_id, revision, body,
                                            f'inconclusive: telemetry unavailable ({type(exc).__name__})')
            return f'holding: telemetry unavailable ({type(exc).__name__})'
        if stable.get('requests'):
            body['stable_baseline'] = {'stage': body['current_stage'], 'at': until,
                                       'ttfb_p50_ms': stable.get('ttfb_p50_ms'),
                                       'metrics': {k: v for k, v in stable.items() if k.endswith('_ms')}}
        baseline, basis = gates.comparable_baseline(stage, stable, body.get('stable_baseline'))
        decision = gates.evaluate(stage, cand, baseline, basis=basis, elapsed=until - since,
                                  min_requests=body.get('min_requests', 20))
        if decision['decision'] == 'advance' and (body.get('evaluation_policy') or {}).get('required'):
            status, why = gates.evaluation_check(self.repository, self.repository.read('release', body['release_ref']),
                                                 body['evaluation_policy'])
            decision['checks']['evaluation'] = {'status': status, 'reason': why}
            if status == 'failed':
                decision = {**decision, 'decision': 'rollback', 'reasons': [why]}
            elif status != 'passed':
                decision = {**decision, 'decision': 'hold', 'reasons': [f'evaluation {status}: {why}']}
        observations = {'candidate': cand, 'stable': stable, 'stage': body['current_stage'], 'at': until,
                        'window': {'since': since, 'until': until}, 'latency_basis': basis,
                        'baseline_ttfb_p50_ms': (baseline or {}).get('ttfb_p50_ms'),
                        'decision': decision['decision'], 'reasons': decision['reasons'],
                        'checks': decision['checks'], 'policy_sha256': body.get('policy_sha256')}
        if decision['decision'] == 'hold':
            return 'holding: ' + '; '.join(decision['reasons'])
        body = {**body, 'evidence_refs': self._evidence(body, observations)}
        if decision['decision'] in ('rollback', 'inconclusive'):
            prefix = 'inconclusive: ' if decision['decision'] == 'inconclusive' else ''
            return await self._rollback(rollout_id, revision, {**body, 'observations': observations},
                                        prefix + '; '.join(decision['reasons']))
        # Observation awaited: another writer may have changed the route meanwhile.
        if self._owned_route(body) is None:
            return self._conflict(rollout_id, revision, body)
        if body['current_stage'] + 1 < len(body['stages']):
            nxt = body['current_stage'] + 1
            target = {'stable': body['stable_ref'], 'candidate': body['release_ref'],
                      'weight': body['stages'][nxt]['weight'], 'previous': body['route_body'].get('previous')}
            _, body = self._transition(rollout_id, revision, body, target,
                                       {'current_stage': nxt, 'stage_started_at': NOW, 'observations': observations,
                                        'stable_baseline': body.get('stable_baseline'),
                                        'evidence_refs': body['evidence_refs'],
                                        'reason': f"stage {nxt} at {body['stages'][nxt]['weight']}%"})
            return 'conflict' if body['state'] == 'conflict' else 'advanced'
        target = {'stable': body['release_ref'], 'candidate': None, 'weight': 0, 'previous': body['stable_ref']}
        _, body = self._transition(rollout_id, revision, body, target,
                                   {'state': 'promoted', 'observations': observations, 'reason': 'all stages passed',
                                    'evidence_refs': body['evidence_refs'],
                                    **self._retirement(body['stable_ref'])})
        if body['state'] != 'promoted':
            return body['state']  # conflict: the route is not ours, so nothing is stopped
        await self._settle_retirement(rollout_id)  # drain in-flight streams, then stop the old stable
        return 'promoted'

    async def _observe(self, release_id, since, until):
        import inspect
        try:
            closed = len(inspect.signature(self.observe).parameters) >= 3
        except (TypeError, ValueError):
            closed = False
        # Legacy observe(release_id, since) callbacks get an open window.
        return await (self.observe(release_id, since, until) if closed else self.observe(release_id, since))

    def _evidence(self, body, observations):
        """Persist the decisive closed-window evidence as an immutable artifact; keep the last 50 refs."""
        artifact = self.repository.artifact(self.repository.encode(
            {'kind': 'rollout_evidence', 'model': body['model'], 'release_ref': body['release_ref'],
             'stable_ref': body['stable_ref'], **observations}).encode())
        ref = {'sha256': artifact['sha256'], 'stage': observations['stage'], 'decision': observations['decision'],
               'window': observations['window']}
        return (list(body.get('evidence_refs') or []) + [ref])[-50:]

    def pause(self, rollout_id):
        revision, body = self._read(rollout_id)
        if body['state'] != 'progressing':
            raise RolloutError(f"cannot pause a {body['state']} rollout")
        self._write(rollout_id, body, revision, state='paused', reason='paused by operator')

    def resume(self, rollout_id):
        revision, body = self._read(rollout_id)
        if body['state'] != 'paused':
            raise RolloutError(f"cannot resume a {body['state']} rollout")
        # The stage window restarts: evidence gathered while paused is not reused.
        self._write(rollout_id, body, revision, state='progressing', stage_started_at=self.clock(),
                    reason='resumed by operator')

    async def settle(self, rollout_id):
        """Resume a pending retirement obligation (recovery loop / operator)."""
        return await self._settle_retirement(rollout_id)

    async def abort(self, rollout_id, reason='aborted by operator'):
        revision, body = self._read(rollout_id)
        if body.get('transition'):
            revision, body = self._complete(rollout_id, revision, body)
        if body['state'] in TERMINAL:
            raise RolloutError(f"rollout already {body['state']}")
        return await self._rollback(rollout_id, revision, body, reason)


def gateway_observer(repository, admission=None):
    """Observation over the durable serving evidence (all gateway workers sharing the store),
    attributed by release ID over a closed window. `admission` is accepted for compatibility."""
    from inferstack.serving.evidence import EvidenceStore
    store = EvidenceStore(repository)

    async def observe(release_id, since, until=None):
        import asyncio
        return await asyncio.to_thread(store.window, release_id=release_id, since=since, until=until)
    return observe


evidence_observer = gateway_observer


def http_observer(repository, gateway_url, token, transport=None):
    """Observation function reading a separate gateway process over its authorized API."""
    import httpx

    async def observe(release_id, since, until=None):
        params = {'release_id': release_id, 'since': since, **({'until': until} if until is not None else {})}
        async with httpx.AsyncClient(base_url=gateway_url, transport=transport, timeout=10,
                                     headers={'Authorization': 'Bearer ' + token}) as client:
            response = await client.get('/gateway/observe', params=params)
            response.raise_for_status()
            return response.json()
    return observe


def http_undrain(gateway_url, token, transport=None):
    """undrain(deployment_id) against a separate gateway process (for ReleaseRuntime.ensure)."""
    import httpx

    async def undrain(deployment_id):
        async with httpx.AsyncClient(base_url=gateway_url, transport=transport, timeout=10,
                                     headers={'Authorization': 'Bearer ' + token}) as client:
            response = await client.post(f'/gateway/undrain/{deployment_id}')
            response.raise_for_status()
            return response.json()
    return undrain


def http_drain(gateway_url, token, transport=None):
    """drain(deployment_id, deadline_s) against a separate gateway process (for ReleaseRuntime.retire)."""
    import httpx

    async def drain(deployment_id, deadline_seconds):
        async with httpx.AsyncClient(base_url=gateway_url, transport=transport, timeout=deadline_seconds + 10,
                                     headers={'Authorization': 'Bearer ' + token}) as client:
            response = await client.post(f'/gateway/drain/{deployment_id}',
                                         params={'deadline_seconds': deadline_seconds})
            response.raise_for_status()
            return response.json()
    return drain
