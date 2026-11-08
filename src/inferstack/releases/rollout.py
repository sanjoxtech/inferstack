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
"""
from __future__ import annotations

import time

from inferstack.releases.manager import ReleaseManager

TERMINAL = {'promoted', 'rolled_back', 'failed', 'conflict'}
NOW = '__now__'  # placeholder in follow-up fields, resolved when the transition completes


class RolloutError(RuntimeError):
    pass


def _route_view(body):
    return {'stable': body.get('stable'), 'candidate': body.get('candidate'), 'weight': body.get('weight', 0),
            'previous': body.get('previous')}


class RolloutController:
    def __init__(self, repository, *, observe, attest=None, clock=time.time):
        """observe(release_id, since) -> {'requests','errors','ttfb_p50_ms'}; attest(release) -> dict with 'status'.

        Library callers may omit `attest` (tests, offline tooling); the control API
        refuses to run rollouts without one (see create_control_app).
        """
        self.repository, self.observe, self.attest, self.clock = repository, observe, attest, clock
        self.releases = ReleaseManager(repository)

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

    async def start(self, rollout_id, model, release_id, stages, *, min_requests=20):
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
                'transition': None, 'stable_baseline': None}
        revision = self.repository.put_record('rollout', rollout_id, body)
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
        revision, body = self._transition(rollout_id, revision, body, target, {'state': 'rolled_back',
                                                                                 'reason': reason})
        return body['state']

    async def step(self, rollout_id):
        revision, body = self._read(rollout_id)
        if body.get('transition'):
            # Recover an interrupted route transition before evaluating anything else.
            revision, body = self._complete(rollout_id, revision, body)
        if body['state'] in TERMINAL or body['state'] in ('paused', 'pending'):
            return body['state']
        if self._owned_route(body) is None:
            return self._conflict(rollout_id, revision, body)
        stage = body['stages'][body['current_stage']]
        since = body['stage_started_at']
        cand = await self.observe(body['release_ref'], since)
        stable = await self.observe(body['stable_ref'], since)
        baseline, basis = stable.get('ttfb_p50_ms'), 'stable_same_window'
        if stable.get('requests') and baseline:
            body['stable_baseline'] = {'ttfb_p50_ms': baseline, 'stage': body['current_stage'], 'at': self.clock()}
        elif stage['weight'] == 100 and body.get('stable_baseline'):
            baseline, basis = body['stable_baseline']['ttfb_p50_ms'], \
                f"retained_stable_stage_{body['stable_baseline']['stage']}"
        else:
            baseline = None
        observations = {'candidate': cand, 'stable': stable, 'stage': body['current_stage'], 'at': self.clock(),
                        'latency_basis': basis if baseline else None, 'baseline_ttfb_p50_ms': baseline}
        if cand['requests']:
            error_rate = cand['errors'] / cand['requests']
            limit = stage.get('max_error_rate')
            if limit is not None and error_rate > limit:
                return await self._rollback(rollout_id, revision, {**body, 'observations': observations},
                                            f'candidate error rate {error_rate:.3f} > {limit}')
        regression_limit = stage.get('max_latency_regression')
        if regression_limit is not None and cand['ttfb_p50_ms'] is not None and baseline:
            regression = (cand['ttfb_p50_ms'] - baseline) / baseline
            if regression > regression_limit:
                return await self._rollback(rollout_id, revision, {**body, 'observations': observations},
                                            f'candidate TTFB regression {regression:.1%} > {regression_limit:.0%}')
        hold = None
        if self.clock() - since < stage['min_duration_s']:
            hold = 'minimum stage duration not reached'
        elif cand['requests'] < body.get('min_requests', 20):
            hold = f"only {cand['requests']} candidate requests (need {body.get('min_requests', 20)})"
        elif regression_limit is not None and (cand['ttfb_p50_ms'] is None or not baseline):
            hold = 'latency evidence unavailable for the stage window'
        if hold:
            return 'holding: ' + hold
        if body['current_stage'] + 1 < len(body['stages']):
            nxt = body['current_stage'] + 1
            target = {'stable': body['stable_ref'], 'candidate': body['release_ref'],
                      'weight': body['stages'][nxt]['weight'], 'previous': body['route_body'].get('previous')}
            self._transition(rollout_id, revision, body, target,
                             {'current_stage': nxt, 'stage_started_at': NOW, 'observations': observations,
                              'stable_baseline': body.get('stable_baseline'),
                              'reason': f"stage {nxt} at {body['stages'][nxt]['weight']}%"})
            return 'advanced'
        target = {'stable': body['release_ref'], 'candidate': None, 'weight': 0, 'previous': body['stable_ref']}
        self._transition(rollout_id, revision, body, target,
                         {'state': 'promoted', 'observations': observations, 'reason': 'all stages passed'})
        return 'promoted'

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

    async def abort(self, rollout_id, reason='aborted by operator'):
        revision, body = self._read(rollout_id)
        if body.get('transition'):
            revision, body = self._complete(rollout_id, revision, body)
        if body['state'] in TERMINAL:
            raise RolloutError(f"rollout already {body['state']}")
        return await self._rollback(rollout_id, revision, body, reason)


def gateway_observer(repository, admission):
    """Observation function backed by the gateway's per-deployment admission records."""
    async def observe(release_id, since):
        deployment_id = repository.read('release', release_id)['body']['deployment']['id']
        return admission.observe(deployment_id, since)
    return observe


def http_observer(repository, gateway_url, token, transport=None):
    """Observation function reading a separate gateway process over its authorized API."""
    import httpx

    async def observe(release_id, since):
        deployment_id = repository.read('release', release_id)['body']['deployment']['id']
        async with httpx.AsyncClient(base_url=gateway_url, transport=transport, timeout=10,
                                     headers={'Authorization': 'Bearer ' + token}) as client:
            response = await client.get('/gateway/observe', params={'deployment_id': deployment_id, 'since': since})
            response.raise_for_status()
            return response.json()
    return observe
