"""Lease-fenced, resumable optimization orchestration.

Each iteration runs two durable steps:
- `iter{i}.plan` (idempotent): provider planning. A succeeded plan is reused on
  resume, so a restart does not re-spend provider budget or change the plan.
- `iter{i}.experiment` (side-effecting): execute -> verify -> cleanup as one unit.
  If a previous attempt was interrupted, its allocations are reconciled (orphan
  processes stopped and released) before the step may run again.

Budgets are checked before every step; exhaustion stops new work and returns the
experiments gathered so far. An absolute deadline cancels work, releases resources
and records the partial history. Cancellation lets the last lease holder attach
partial evidence once. All writes are fenced by the job lease.
"""
from __future__ import annotations

import asyncio

from inferstack.control.errors import (BudgetExhausted, DeadlineExceeded, ReconciliationRequired, classify)
from inferstack.control.store import LeaseLost


class OptimizationWorker:
    def __init__(self, repository, planner, executor, verifier, *, lease_seconds=30, max_iterations=3,
                 worker_id=None, allocations=None, provider=None, faults=None, fetcher=None):
        if lease_seconds < .3 or not 1 <= max_iterations <= 20:
            raise ValueError('invalid worker bounds')
        self.repository = repository
        self.planner, self.executor, self.verifier = planner, executor, verifier
        self.lease_seconds, self.max_iterations = lease_seconds, max_iterations
        self.worker_id = worker_id
        self.allocations, self.provider = allocations, provider
        self.faults = faults or (lambda point: None)
        self.fetcher = fetcher
        self.history = {}

    def record(self, job, stage, body):
        with self.repository.transaction() as db:
            self.repository.owned(db, job['id'], job['lease_token'])
            self.repository.event(db, job['id'], stage, body)

    async def renew(self, job):
        while True:
            await asyncio.sleep(self.lease_seconds / 3)
            self.repository.heartbeat(job['id'], job['lease_token'], self.lease_seconds)
            if self.worker_id:
                self.repository.worker_heartbeat(self.worker_id, self.lease_seconds)

    async def deadline(self, job):
        if job.get('deadline_at') is None:
            await asyncio.Event().wait()
        await asyncio.sleep(max(0.0, job['deadline_at'] - self.repository.clock()))
        raise DeadlineExceeded('job deadline reached')

    # --- steps ------------------------------------------------------------------

    async def reconcile_step(self, job, step_key):
        if self.allocations is None:
            raise ReconciliationRequired(f'{step_key} was interrupted and no allocation manager is configured')
        from inferstack.infrastructure.reconcile import reconcile_allocation
        live = [a for a in self.allocations.list(job_id=job['id'], step_key=step_key)
                if a['state'] in ('intent', 'active', 'releasing', 'orphaned')]
        outcomes = [reconcile_allocation(self.allocations, self.provider, a) for a in live]
        if any(o in ('orphaned', 'foreign_host') for o in outcomes):
            raise ReconciliationRequired(f'{step_key}: allocations could not be proven released: {outcomes}')
        self.repository.step_resolve_ambiguous(job['id'], job['lease_token'], step_key,
                                               {'allocations': len(live), 'outcomes': outcomes})

    async def step(self, job, step_key, fn, *, idempotent):
        exhausted = self.repository.exhausted(job['id'])
        if exhausted:
            raise BudgetExhausted('budget exhausted: ' + ','.join(exhausted))
        begin = self.repository.step_begin(job['id'], job['lease_token'], step_key, idempotent=idempotent,
                                           worker_id=self.worker_id)
        if begin['action'] == 'reuse':
            return begin['result']
        if begin['action'] == 'ambiguous':
            await self.reconcile_step(job, step_key)
            begin = self.repository.step_begin(job['id'], job['lease_token'], step_key, idempotent=idempotent,
                                               worker_id=self.worker_id)
        self.faults(f'step_started:{step_key}')
        try:
            result = await fn({**job, 'step_attempt': begin['attempts']}, step_key)
        except asyncio.CancelledError:
            try:
                self.repository.step_end(job['id'], job['lease_token'], step_key, 'cancelled')
            except LeaseLost:
                pass
            raise
        except Exception as exc:
            try:
                self.repository.step_end(job['id'], job['lease_token'], step_key, 'failed', error=classify(exc))
            except LeaseLost:
                pass
            raise
        self.repository.step_end(job['id'], job['lease_token'], step_key, 'succeeded', result=result)
        self.faults(f'step_done:{step_key}')
        return result

    async def experiment(self, job, step_key, plan, iteration):
        candidate = None
        namespace = f"{job['id']}-{job['attempt']}-{iteration}-{job['step_attempt']}"
        try:
            if getattr(self.executor, 'accepts_context', False):
                candidate = await self.executor.execute(plan, namespace, context={'job': job, 'step_key': step_key})
            else:
                candidate = await self.executor.execute(plan, namespace)
            self.record(job, 'executed', {'iteration': iteration, 'candidate': candidate})
            verdict = await self.verifier.verify(plan, candidate)
            if verdict.get('decision') not in {'keep', 'reject', 'retest', 'inconclusive'}:
                raise ValueError('invalid verifier decision')
            self.record(job, 'verified', {'iteration': iteration, 'verdict': verdict})
            return {'plan': plan, 'candidate': candidate, 'verdict': verdict}
        finally:
            if candidate is not None:
                await self.executor.cleanup(candidate)

    async def research(self, job, step_key):
        """Retrieve supplied URLs as evidence. Unavailable sources are recorded, never fatal."""
        from inferstack.research import EvidenceStore
        store = EvidenceStore(self.repository)
        outcomes = []
        for url in job['payload'].get('technique_urls', []):
            if self.fetcher is None:
                outcomes.append({'evidence_id': store.record_unavailable(
                    url, availability='unavailable', reason='retrieval not configured for this worker',
                    retrieved_at=self.repository.clock()), 'status': 'unavailable'})
            else:
                outcomes.append(await store.ingest_url(url, self.fetcher))
        self.record(job, 'researched', {'sources': outcomes})
        return outcomes

    async def optimize(self, job):
        history = self.history.setdefault(job['id'], [])
        request = dict(job['payload'])
        request['evidence_ids'] = list(request.get('evidence_ids') or [])
        if request.get('technique_urls'):
            sources = await self.step(job, 'research', self.research, idempotent=True)
            request['evidence_ids'] += [s['evidence_id'] for s in sources]
        for iteration in range(self.max_iterations):
            async def plan_fn(j, key, iteration=iteration):
                remaining = self.repository.remaining(job['id'])
                try:
                    plan = await self.planner.plan({**request, 'budget_remaining': remaining}, list(history))
                finally:
                    usage = getattr(self.planner, 'last_usage', None) or {}
                    if usage.get('provider_tokens'):
                        self.repository.charge(job['id'], job['lease_token'],
                                               {'provider_tokens': usage['provider_tokens']})
                self.record(job, 'planned', {'iteration': iteration, 'plan': plan})
                return plan
            try:
                plan = await self.step(job, f'iter{iteration}.plan', plan_fn, idempotent=True)
                outcome = await self.step(
                    job, f'iter{iteration}.experiment',
                    lambda j, key, plan=plan, iteration=iteration: self.experiment(j, key, plan, iteration),
                    idempotent=False)
            except BudgetExhausted as exc:
                return {'decision': history[-1]['verdict']['decision'] if history else 'inconclusive',
                        'experiments': history, 'stop_reason': str(exc)}
            history.append(outcome)
            if outcome['verdict']['decision'] == 'keep':
                # Promotion is separate; a successful experiment is not a live release.
                return {'decision': 'keep', 'experiments': history}
        return {'decision': history[-1]['verdict']['decision'], 'experiments': history,
                'stop_reason': 'iteration budget exhausted'}

    # --- job lifecycle ------------------------------------------------------------

    async def once(self):
        job = self.repository.claim(self.lease_seconds, worker_id=self.worker_id)
        if job is None:
            return None
        self.faults('claimed')
        work = asyncio.create_task(self.optimize(job))
        heartbeat = asyncio.create_task(self.renew(job))
        deadline = asyncio.create_task(self.deadline(job))
        try:
            done, _ = await asyncio.wait({work, heartbeat, deadline}, return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                await heartbeat  # Lease loss propagates; work is cancelled in finally.
            if deadline in done and work not in done:
                await deadline
            result = await work
            self.faults('before_finish')
            self.repository.finish(job['id'], job['lease_token'], result)
        except LeaseLost:
            await self._stop(work, heartbeat, deadline)
            try:
                self.repository.attach_partial(job['id'], job['lease_token'],
                                               {'experiments': self.history.get(job['id'], [])})
            except LeaseLost:
                pass  # Superseded (not cancelled): a newer lease holder owns the job.
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._stop(work, heartbeat, deadline)
            try:
                self.repository.fail(job['id'], job['lease_token'], classify(exc),
                                     partial={'experiments': self.history.get(job['id'], [])})
            except LeaseLost:
                pass
        finally:
            await self._stop(work, heartbeat, deadline)
            self.history.pop(job['id'], None)
        return self.repository.get(job['id'])

    @staticmethod
    async def _stop(*tasks):
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
