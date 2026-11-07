"""Lease-fenced optimization orchestration; components supply typed operations."""
from __future__ import annotations
import asyncio
from inferstack.control.store import LeaseLost


class OptimizationWorker:
    def __init__(self, repository, planner, executor, verifier, *, lease_seconds=30, max_iterations=3):
        if lease_seconds < .3 or not 1 <= max_iterations <= 20:
            raise ValueError('invalid worker bounds')
        self.repository = repository
        self.planner, self.executor, self.verifier = planner, executor, verifier
        self.lease_seconds, self.max_iterations = lease_seconds, max_iterations

    def record(self, job, stage, body):
        with self.repository.transaction() as db:
            self.repository.owned(db, job['id'], job['lease_token'])
            self.repository.event(db, job['id'], stage, body)

    async def renew(self, job):
        while True:
            await asyncio.sleep(self.lease_seconds / 3)
            self.repository.heartbeat(job['id'], job['lease_token'], self.lease_seconds)

    async def optimize(self, job):
        history = []
        for iteration in range(self.max_iterations):
            plan = await self.planner.plan(job['payload'], history)
            self.record(job, 'planned', {'iteration': iteration, 'plan': plan})
            candidate = None
            try:
                # Each attempt receives a distinct namespace; retries never mutate serving.
                candidate = await self.executor.execute(plan, f"{job['id']}-{job['attempt']}-{iteration}")
                self.record(job, 'executed', {'iteration': iteration, 'candidate': candidate})
                verdict = await self.verifier.verify(plan, candidate)
                if verdict.get('decision') not in {'keep','reject','retest','inconclusive'}:
                    raise ValueError('invalid verifier decision')
                self.record(job, 'verified', {'iteration': iteration, 'verdict': verdict})
                history.append({'plan': plan, 'candidate': candidate, 'verdict': verdict})
                if verdict['decision'] == 'keep':
                    # Promotion is separate; a successful experiment is not a live release.
                    return {'decision': 'keep', 'experiments': history}
            finally:
                if candidate is not None:
                    await self.executor.cleanup(candidate)
        return {'decision': history[-1]['verdict']['decision'], 'experiments': history,
                'stop_reason': 'iteration budget exhausted'}

    async def once(self):
        job = self.repository.claim(self.lease_seconds)
        if job is None:
            return None
        work = asyncio.create_task(self.optimize(job))
        heartbeat = asyncio.create_task(self.renew(job))
        try:
            done, _ = await asyncio.wait({work, heartbeat}, return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                await heartbeat  # Lease loss propagates and cancels execution below.
            result = await work
            self.repository.finish(job['id'], job['lease_token'], result)
        except LeaseLost:
            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            try:
                self.repository.finish(job['id'], job['lease_token'],
                                       {'error_type': type(exc).__name__}, state='failed')
            except LeaseLost:
                pass
        finally:
            work.cancel()
            heartbeat.cancel()
            await asyncio.gather(work, heartbeat, return_exceptions=True)
        return self.repository.get(job['id'])
