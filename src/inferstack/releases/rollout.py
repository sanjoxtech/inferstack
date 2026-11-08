"""Progressive rollout controller (canary stages with automatic rollback).

Lifecycle: `start` attests the candidate release (when an attestation function
is configured) and routes the first stage's weight to it. Each `step` compares
candidate and stable observations over the current stage window:
- candidate error rate above `max_error_rate` -> rollback;
- candidate median time-to-first-byte regressing more than
  `max_latency_regression` (fraction) over stable -> rollback;
- too few candidate requests, unknown latency, or the stage's minimum duration
  not yet elapsed -> hold (never advance on missing evidence);
- otherwise advance to the next stage; after the 100% stage holds cleanly the
  candidate is promoted to stable.
Every transition is written as a new revision of the rollout record, giving an
audit trail; route changes go through ReleaseManager with revision checks.
`pause`, `resume` and `abort` are operator controls.
"""
from __future__ import annotations

import time

from inferstack.releases.manager import ReleaseManager

TERMINAL = {'promoted', 'rolled_back', 'failed'}


class RolloutError(RuntimeError):
    pass


class RolloutController:
    def __init__(self, repository, *, observe, attest=None, clock=time.time):
        """observe(release_id, since) -> {'requests','errors','ttfb_p50_ms'}; attest(release) -> dict with 'status'."""
        self.repository, self.observe, self.attest, self.clock = repository, observe, attest, clock
        self.releases = ReleaseManager(repository)

    def _read(self, rollout_id):
        record = self.repository.read('rollout', rollout_id)
        return record['revision'], dict(record['body'])

    def _write(self, rollout_id, body, revision, **changes):
        body = {**body, **changes, 'updated_at': self.clock()}
        return self.repository.put_record('rollout', rollout_id, body, expected_revision=revision), body

    async def start(self, rollout_id, model, release_id, stages, *, min_requests=20):
        route = self.repository.read('route', model)
        if route['body'].get('candidate'):
            raise RolloutError(f'{model} already has a candidate in flight')
        stable = route['body']['stable']
        if stable == release_id:
            raise RolloutError('release is already stable')
        now = self.clock()
        body = {'model': model, 'release_ref': release_id, 'previous_release_ref': stable, 'stable_ref': stable,
                'stages': stages, 'state': 'pending', 'current_stage': 0, 'updated_at': now,
                'min_requests': min_requests, 'observations': {}}
        self.repository.put_record('rollout', rollout_id, body)
        revision = 1
        if self.attest is not None:
            result = await self.attest(self.repository.read('release', release_id))
            if result.get('status') != 'attested':
                self._write(rollout_id, body, revision, state='failed',
                            reason=f"attestation {result.get('status')}: {result.get('checks')}")
                raise RolloutError(f"candidate release not attested: {result.get('status')}")
        self.releases.route(model, stable, candidate_release=release_id, weight=stages[0]['weight'],
                            expected_revision=route['revision'])
        self._write(rollout_id, body, revision, state='progressing', stage_started_at=self.clock(),
                    reason=f"stage 0 at {stages[0]['weight']}%")
        return self.repository.read('rollout', rollout_id)

    async def _rollback(self, rollout_id, revision, body, reason):
        route = self.repository.read('route', body['model'])
        if route['body'].get('candidate') == body['release_ref']:
            self.releases.rollback(body['model'], route['revision'])
        self._write(rollout_id, body, revision, state='rolled_back', reason=reason)
        return 'rolled_back'

    async def step(self, rollout_id):
        revision, body = self._read(rollout_id)
        if body['state'] in TERMINAL or body['state'] == 'paused':
            return body['state']
        stage = body['stages'][body['current_stage']]
        since = body['stage_started_at']
        cand = await self.observe(body['release_ref'], since)
        stable = await self.observe(body['stable_ref'], since)
        observations = {'candidate': cand, 'stable': stable, 'stage': body['current_stage'], 'at': self.clock()}
        if cand['requests']:
            error_rate = cand['errors'] / cand['requests']
            limit = stage.get('max_error_rate')
            if limit is not None and error_rate > limit:
                return await self._rollback(rollout_id, revision, {**body, 'observations': observations},
                                            f'candidate error rate {error_rate:.3f} > {limit}')
        regression_limit = stage.get('max_latency_regression')
        if regression_limit is not None and cand['ttfb_p50_ms'] is not None and stable['ttfb_p50_ms']:
            regression = (cand['ttfb_p50_ms'] - stable['ttfb_p50_ms']) / stable['ttfb_p50_ms']
            if regression > regression_limit:
                return await self._rollback(rollout_id, revision, {**body, 'observations': observations},
                                            f'candidate TTFB regression {regression:.1%} > {regression_limit:.0%}')
        hold = None
        if self.clock() - since < stage['min_duration_s']:
            hold = 'minimum stage duration not reached'
        elif cand['requests'] < body.get('min_requests', 20):
            hold = f"only {cand['requests']} candidate requests (need {body.get('min_requests', 20)})"
        elif regression_limit is not None and (cand['ttfb_p50_ms'] is None or not stable['ttfb_p50_ms']):
            hold = 'latency evidence unavailable for the stage window'
        if hold:
            return 'holding: ' + hold
        route = self.repository.read('route', body['model'])
        if body['current_stage'] + 1 < len(body['stages']):
            nxt = body['current_stage'] + 1
            self.releases.route(body['model'], body['stable_ref'], candidate_release=body['release_ref'],
                                weight=body['stages'][nxt]['weight'], expected_revision=route['revision'])
            self._write(rollout_id, body, revision, current_stage=nxt, stage_started_at=self.clock(),
                        observations=observations, reason=f"stage {nxt} at {body['stages'][nxt]['weight']}%")
            return 'advanced'
        self.releases.promote(body['model'], route['revision'])
        self._write(rollout_id, body, revision, state='promoted', observations=observations,
                    reason='all stages passed')
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
