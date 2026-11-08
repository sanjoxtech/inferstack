"""Release runtimes: keep, replace, warm and retire the processes that serve releases.

A kept candidate's verified process is retained under serving ownership
(`serving:<deployment>` allocation owner). `ensure` returns a healthy runtime for
a release: the retained one when it is alive and ready, otherwise a replacement
launched from the release manifest (frozen deployment command, runnable
workspace with its tree digest re-checked) under a new allocation, warmed until
ready and optionally attested before it is reported usable. `retire` drains
in-flight requests with a deadline (when a drain function is supplied, e.g. the
gateway's admission controller), then stops the process and confirms release.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from inferstack.engines.managed import DeploymentSpec, ManagedEngine
from inferstack.infrastructure.allocations import SERVING_PREFIX, AllocationManager


class RuntimeUnavailable(RuntimeError):
    pass


class ReleaseRuntime:
    def __init__(self, repository, provider, *, log_root, engine_factory=ManagedEngine, attest=None,
                 drain=None, ready_timeout=900):
        """drain(deployment_id, deadline_s) -> {'drained': bool, 'remaining': int} (gateway admission)."""
        self.repository, self.provider = repository, provider
        self.log_root, self.engine_factory = Path(log_root), engine_factory
        self.attest, self.drain, self.ready_timeout = attest, drain, ready_timeout

    def _manifest(self, release_id):
        release = self.repository.read('release', release_id)
        return release, release['body']['deployment']['body']

    def serving_allocations(self, release_id):
        _, deployment = self._manifest(release_id)
        manager = AllocationManager(self.repository, SERVING_PREFIX + deployment['id'])
        return [a for a in manager.list(state='active') if a['worker_id'] == SERVING_PREFIX + deployment['id']]

    async def _ready(self, deployment):
        engine = self.engine_factory(DeploymentSpec.from_record({**deployment, 'command': []}))
        try:
            return await engine.ready()
        finally:
            await engine.close()

    async def ensure(self, release_id):
        """A ready runtime for the release: the retained one, else a warmed, attested replacement."""
        release, deployment = self._manifest(release_id)
        for alloc in self.serving_allocations(release_id):
            if self.provider.alive(alloc['external']) and await self._ready(deployment):
                return {'allocation_id': alloc['id'], 'replaced': False}
        if not deployment.get('command'):
            if await self._ready(deployment):
                return {'allocation_id': None, 'replaced': False, 'attached': True}
            raise RuntimeUnavailable(f'attached endpoint for release {release_id} is not ready')
        runnable = release['body'].get('runnable')
        if runnable:
            from inferstack.execution.executor import IsolatedExecutor
            path = Path(runnable['path'])
            if not path.is_dir() or IsolatedExecutor.tree_sha256(path) != runnable['tree_sha256']:
                raise RuntimeUnavailable(f'runnable artifact for {release_id} is missing or modified')
        return await self._launch(release_id, release, deployment)

    async def _launch(self, release_id, release, deployment):
        owner = SERVING_PREFIX + deployment['id']
        manager = AllocationManager(self.repository, owner)
        alloc_id = 'alloc-' + uuid.uuid4().hex[:20]
        argv = list(deployment['command'])
        with self.repository.transaction() as db:  # intent before launch, no job lease (serving owner)
            now = self.repository.clock()
            db.execute('INSERT INTO allocations(id,idempotency_key,job_id,step_key,worker_id,provider,resource_type,'
                       'spec,state,gpu_count,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                       (alloc_id, f'serving:{release_id}:{alloc_id}', None, None, owner, self.provider.name,
                        self.provider.resource_type, self.repository.encode({'release_id': release_id, 'argv': argv,
                                                                              'kind': 'serving'}),
                        'intent', 0, now, now))
        self.log_root.mkdir(parents=True, exist_ok=True)
        log = (self.log_root / f'{alloc_id}.log').open('ab')
        try:
            process = await self.provider.spawn(alloc_id, argv, log)
            external = {**self.provider.external_for(process.pid), 'allocation_id': alloc_id}
            manager.record_external(alloc_id, external)
            with self.repository.transaction() as db:
                now = self.repository.clock()
                db.execute("UPDATE allocations SET state='active',active_at=?,updated_at=? WHERE id=?",
                           (now, now, alloc_id))
            deadline = time.monotonic() + self.ready_timeout
            while not await self._ready(deployment):
                if time.monotonic() > deadline or not self.provider.alive(external):
                    raise RuntimeUnavailable(f'replacement for {release_id} did not become ready')
                await asyncio.sleep(0.25)
            if self.attest is not None:
                result = await self.attest(release)
                if result.get('status') != 'attested':
                    raise RuntimeUnavailable(f"replacement for {release_id} not attested: {result.get('status')}")
            return {'allocation_id': alloc_id, 'replaced': True}
        except BaseException:
            ext = manager.get(alloc_id)['external']
            if ext is None or self.provider.stop(ext):
                manager.release(alloc_id, 'replacement_failed')
            else:
                manager.mark_orphaned(alloc_id, 'replacement_stop_failed')
            raise
        finally:
            log.close()

    async def retire(self, release_id, *, drain_seconds=30.0):
        """Drain in-flight requests (deadline), then stop and release every serving runtime of the release."""
        _, deployment = self._manifest(release_id)
        drained = await self.drain(deployment['id'], drain_seconds) if self.drain else {'drained': None}
        outcomes = []
        for alloc in self.serving_allocations(release_id):
            manager = AllocationManager(self.repository, alloc['worker_id'])
            if self.provider.stop(alloc['external']):
                manager.release(alloc['id'], 'retired')
                outcomes.append({'allocation_id': alloc['id'], 'state': 'released'})
            else:
                manager.mark_orphaned(alloc['id'], 'retire_stop_failed')
                outcomes.append({'allocation_id': alloc['id'], 'state': 'orphaned'})
        return {'drain': drained, 'allocations': outcomes}
