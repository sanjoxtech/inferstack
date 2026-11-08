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
                 drain=None, undrain=None, ready_timeout=900, providers=None, gpu_devices=None):
        """drain(deployment_id, deadline_s) -> {'drained': bool, 'remaining': int} (gateway admission);
        undrain(deployment_id) restores admission for a runtime brought back into service.

        `provider` launches process releases; `providers` maps provider name -> provider
        for every placement this runtime may manage (container releases launch through
        the provider named in their profile; retirement always dispatches to the
        provider recorded on the allocation)."""
        self.repository, self.provider = repository, provider
        self.providers = {provider.name: provider, **(providers or {})}
        self.log_root, self.engine_factory = Path(log_root), engine_factory
        self.attest, self.drain, self.undrain, self.ready_timeout = attest, drain, undrain, ready_timeout
        # Identified host devices: GPU-engine replacements reserve specific free ones
        # through the same exclusive placement as experiments (fail closed without them).
        self.gpu_devices = list(gpu_devices or [])

    def _manifest(self, release_id):
        release = self.repository.read('release', release_id)
        return release, release['body']['deployment']['body']

    def serving_allocations(self, release_id):
        _, deployment = self._manifest(release_id)
        manager = AllocationManager(self.repository, SERVING_PREFIX + deployment['id'])
        return [a for a in manager.list(state='active') if a['worker_id'] == SERVING_PREFIX + deployment['id']]

    def _provider_for(self, name):
        """The provider that owns a placement; Docker needs no configuration, others must be supplied."""
        if name not in self.providers and name == 'docker':
            from inferstack.infrastructure.remote import DockerProvider
            self.providers['docker'] = DockerProvider()
        return self.providers.get(name)

    async def _ready(self, deployment):
        engine = self.engine_factory(DeploymentSpec.from_record({**deployment, 'command': []}))
        try:
            return await engine.ready()
        finally:
            await engine.close()

    async def _restore(self, deployment):
        """A runtime brought (back) into service admits requests again."""
        if self.undrain is not None:
            await self.undrain(deployment['id'])

    async def ensure(self, release_id):
        """A ready runtime for the release: the retained one, else a warmed, attested replacement."""
        release, deployment = self._manifest(release_id)
        for alloc in self.serving_allocations(release_id):
            provider = self._provider_for(alloc['provider'])
            if provider is not None and provider.alive(alloc['external']) and await self._ready(deployment):
                await self._restore(deployment)
                return {'allocation_id': alloc['id'], 'replaced': False, 'provider': alloc['provider']}
        container = (deployment.get('profile') or {}).get('container')
        if container:
            result = await self._launch(release_id, release, deployment, container=dict(container))
            await self._restore(deployment)
            return result
        if not deployment.get('command'):
            if await self._ready(deployment):
                await self._restore(deployment)
                return {'allocation_id': None, 'replaced': False, 'attached': True}
            raise RuntimeUnavailable(f'attached endpoint for release {release_id} is not ready')
        runnable = release['body'].get('runnable')
        if runnable:
            from inferstack.execution.executor import IsolatedExecutor
            path = Path(runnable['path'])
            if not path.is_dir() or IsolatedExecutor.tree_sha256(path) != runnable['tree_sha256']:
                raise RuntimeUnavailable(f'runnable artifact for {release_id} is missing or modified')
        result = await self._launch(release_id, release, deployment)
        await self._restore(deployment)
        return result

    def _placement(self, container):
        name = container.get('provider')
        if name is None:
            remote = [n for n, p in self.providers.items() if getattr(p, 'launches_remotely', False)]
            name = remote[0] if len(remote) == 1 else 'docker'
        provider = self._provider_for(name)
        if provider is None or not getattr(provider, 'launches_remotely', False):
            raise RuntimeUnavailable(f'container release needs a configured {name!r} provider')
        return provider

    async def _launch(self, release_id, release, deployment, container=None):
        owner = SERVING_PREFIX + deployment['id']
        manager = AllocationManager(self.repository, owner)
        alloc_id = 'alloc-' + uuid.uuid4().hex[:20]
        provider = self._placement(container) if container is not None else self.provider
        spec = {'release_id': release_id, 'kind': 'serving'}
        from inferstack.infrastructure.allocations import ResourceConflict, place_devices
        from inferstack.optimization.experiment import GPU_ENGINES
        resources = deployment.get('resources') or {}
        if container is not None:
            spec['container'] = container  # persisted placement: recovery can relaunch or stop it
            listed = container.get('gpus') if isinstance(container.get('gpus'), list) else []
            gpus = len(listed) if listed else int(container.get('gpus') or 0)
            if listed:
                spec['gpu_devices'] = [str(g) for g in listed]
        else:
            spec['argv'] = list(deployment['command'])
            gpus = int(resources.get('gpu_count') or len(resources.get('gpu_devices') or []) or
                       (1 if deployment.get('engine') in GPU_ENGINES else 0))
            if resources.get('gpu_devices'):
                spec['gpu_devices'] = list(resources['gpu_devices'])
            elif gpus:
                spec['gpu_pool'] = self.gpu_devices
        try:
            with self.repository.transaction() as db:  # intent before launch, no job lease (serving owner)
                spec = place_devices(db, provider.name, spec, gpus)
                now = self.repository.clock()
                db.execute('INSERT INTO allocations(id,idempotency_key,job_id,step_key,worker_id,provider,'
                           'resource_type,spec,state,gpu_count,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                           (alloc_id, f'serving:{release_id}:{alloc_id}', None, None, owner, provider.name,
                            provider.resource_type, self.repository.encode(spec), 'intent', gpus, now, now))
        except ResourceConflict as exc:
            raise RuntimeUnavailable(f'no exclusive GPU placement for {release_id}: {exc}') from exc
        self.log_root.mkdir(parents=True, exist_ok=True)
        log = (self.log_root / f'{alloc_id}.log').open('ab')
        try:
            if container is not None:
                external = {**provider.launch(alloc_id, container), 'allocation_id': alloc_id}
            else:
                env = {'CUDA_VISIBLE_DEVICES': ','.join(spec['gpu_devices'])} if spec.get('gpu_devices') else None
                process = await provider.spawn(alloc_id, spec['argv'], log, env=env)
                external = {**provider.external_for(process.pid), 'allocation_id': alloc_id}
            manager.record_external(alloc_id, external)
            with self.repository.transaction() as db:
                now = self.repository.clock()
                db.execute("UPDATE allocations SET state='active',active_at=?,updated_at=? WHERE id=?",
                           (now, now, alloc_id))
            deadline = time.monotonic() + self.ready_timeout
            while not await self._ready(deployment):
                if time.monotonic() > deadline or not provider.alive(external):
                    raise RuntimeUnavailable(f'replacement for {release_id} did not become ready')
                await asyncio.sleep(0.25)
            if self.attest is not None:
                result = await self.attest(release)
                if result.get('status') != 'attested':
                    raise RuntimeUnavailable(f"replacement for {release_id} not attested: {result.get('status')}")
            return {'allocation_id': alloc_id, 'replaced': True, 'provider': provider.name}
        except BaseException:
            ext = manager.get(alloc_id)['external']
            if ext is None or provider.stop(ext):
                manager.release(alloc_id, 'replacement_failed')
            else:
                manager.mark_orphaned(alloc_id, 'replacement_stop_failed')
            raise
        finally:
            log.close()

    async def retire(self, release_id, *, drain_seconds=30.0):
        """Drain in-flight requests (deadline), then stop and release every serving runtime of the release.

        Each allocation is stopped by the provider recorded on it. An allocation whose
        provider is not configured is marked orphaned (operator-visible), never left active."""
        _, deployment = self._manifest(release_id)
        drained = await self.drain(deployment['id'], drain_seconds) if self.drain else {'drained': None}
        outcomes = []
        for alloc in self.serving_allocations(release_id):
            manager = AllocationManager(self.repository, alloc['worker_id'])
            provider = self._provider_for(alloc['provider'])
            if provider is None:
                manager.mark_orphaned(alloc['id'], f"retire_no_provider_{alloc['provider']}")
                outcomes.append({'allocation_id': alloc['id'], 'state': 'orphaned',
                                 'reason': f"provider {alloc['provider']!r} not configured"})
            elif provider.stop(alloc['external']):
                manager.release(alloc['id'], 'retired')
                outcomes.append({'allocation_id': alloc['id'], 'state': 'released'})
            else:
                manager.mark_orphaned(alloc['id'], 'retire_stop_failed')
                outcomes.append({'allocation_id': alloc['id'], 'state': 'orphaned'})
        return {'drain': drained, 'allocations': outcomes}
