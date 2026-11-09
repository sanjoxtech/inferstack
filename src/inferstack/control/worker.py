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
                 worker_id=None, allocations=None, provider=None, faults=None, fetcher=None, implementations=None,
                 source_revision=None):
        if lease_seconds < .3 or not 1 <= max_iterations <= 20:
            raise ValueError('invalid worker bounds')
        self.repository = repository
        self.planner, self.executor, self.verifier = planner, executor, verifier
        self.lease_seconds, self.max_iterations = lease_seconds, max_iterations
        self.worker_id = worker_id
        self.allocations, self.provider = allocations, provider
        self.faults = faults or (lambda point: None)
        self.fetcher = fetcher
        # H3: bounded implementer handoff (ImplementationDesk) and the pinned source revision
        # every handoff is made against. Without them a proposal is only recorded.
        self.implementations, self.source_revision = implementations, source_revision
        self.history = {}

    def record(self, job, stage, body):
        with self.repository.transaction() as db:
            self.repository.owned(db, job['id'], job['lease_token'])
            self.repository.event(db, job['id'], stage, body)

    async def renew(self, job):
        loop = asyncio.get_running_loop()
        last = self._last_renewal = loop.time()
        while True:
            await asyncio.sleep(self.lease_seconds / 3)
            now = loop.time()
            gap, last = now - last, now
            self._last_renewal = now
            try:
                self.repository.heartbeat(job['id'], job['lease_token'], self.lease_seconds)
            except LeaseLost:
                # Operator-visible diagnosis: how long the loop went without renewing.
                import sys
                print(f"lease lost for job {job['id']}: {gap:.2f}s since the previous renewal "
                      f"(lease {self.lease_seconds}s, renewal every {self.lease_seconds / 3:.2f}s)", file=sys.stderr)
                raise
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
        # Reconciliation stops processes and polls; run it off the event loop so lease
        # renewal keeps running while an interrupted step's resources are resolved.
        outcomes = [await asyncio.to_thread(reconcile_allocation, self.allocations, self.provider, a) for a in live]
        if any(o in ('orphaned', 'foreign_host') for o in outcomes):
            raise ReconciliationRequired(f'{step_key}: allocations could not be proven released: {outcomes}')
        self.repository.step_resolve_ambiguous(job['id'], job['lease_token'], step_key,
                                               {'allocations': len(live), 'outcomes': outcomes})

    async def step(self, job, step_key, fn, *, idempotent):
        # Durable results are reused before budget checks: rehydrating history after a
        # restart must not be blocked by a budget that the original work consumed.
        done, result = self.repository.step_succeeded(job['id'], step_key)
        if done:
            return result
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
            outcome = {'plan': plan, 'candidate': candidate, 'verdict': verdict}
            if verdict['decision'] == 'keep' and hasattr(self.executor, 'retain'):
                # The verified runtime stays up under serving ownership for its release.
                serving = await self.executor.retain(candidate, {'job': job})
                candidate = None
                if serving:
                    outcome['serving'] = serving
                    self.record(job, 'retained', {'iteration': iteration, 'serving': serving})
            return outcome
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
                if iteration == 0 and request.get('implementation_ids'):
                    plan = self.implemented_plan(request, history)
                    self.record(job, 'planned', {'iteration': iteration, 'plan': plan,
                                                 'implementation_ids': request['implementation_ids']})
                    return plan
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
                if isinstance(plan, dict) and plan.get('executable') is False:
                    # Nothing to run yet: the proposal waits for an implementation artifact
                    # (bounded implementer handoff, H3). It is recorded, never executed.
                    handoffs = await self.step(job, f'iter{iteration}.handoff',
                                               lambda j, key, plan=plan: self.request_implementation(j, plan),
                                               idempotent=True)
                    self.record(job, 'needs_implementation', {'iteration': iteration,
                                                              'proposals': plan.get('needs_implementation'),
                                                              **handoffs})
                    return {'decision': 'needs_implementation', 'experiments': history,
                            'proposal': plan, 'stop_reason': 'source change has no validated patch artifact',
                            **handoffs}
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

    async def request_implementation(self, job, plan):
        """Open one bounded implementer handoff per proposal (nothing is executed here)."""
        if self.implementations is None or not self.source_revision:
            return {'implementation_ids': [], 'handoff_error': 'no implementation desk configured'}
        ids, errors = [], []
        for proposal in plan.get('needs_implementation') or []:
            try:
                record = self.implementations.request(
                    proposal, plan, baseline_id=plan['baseline_id'], source_revision=self.source_revision,
                    job_id=job['id'], evidence_refs=plan.get('evidence_ids') or ())
                ids.append(record['id'])
            except (KeyError, ValueError) as exc:
                errors.append({'action_id': proposal.get('action_id'), 'error': str(exc)[:500]})
        return {'implementation_ids': ids, 'handoff_errors': errors}

    def implemented_plan(self, request, history):
        """Resume with validated implementer artifacts: the recorded plan with the exact patches."""
        from inferstack.implementation import ValidationFailed
        from inferstack.memory.experience import experience
        from inferstack.optimization.dag import PlanRejected, validate_plan
        if self.implementations is None:
            raise PlanRejected('implementation_ids given but no implementation desk is configured')
        try:
            raw = self.implementations.executable_plan(request['implementation_ids'])
        except (KeyError, ValueError, ValidationFailed) as exc:
            raise PlanRejected(f'cannot resume from implementations: {exc}') from None
        baseline_id = request['deployment_id']
        failed = {e['plan'].get('treatment_signature') for e in history
                  if e['verdict'].get('decision') != 'retest' and e['plan'].get('treatment_signature')}
        failed |= set(experience(self.repository, baseline_id, limit=10)['failed_here_signatures'])
        _, plan = validate_plan(raw, repository=self.repository, baseline_id=baseline_id,
                                candidate_ids=tuple(getattr(self.planner, 'candidate_ids', ()) or ()),
                                allowed_evidence=tuple(raw.get('evidence_ids') or ()), history_signatures=failed,
                                production_ids=tuple(getattr(self.planner, 'production_ids', ()) or ()))
        return plan

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
            import sys
            since = asyncio.get_running_loop().time() - getattr(self, '_last_renewal', 0.0)
            print(f"lease lost for job {job['id']}: detected by a fenced write {since:.2f}s after the last "
                  f"renewal attempt (lease {self.lease_seconds}s)", file=sys.stderr)
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
