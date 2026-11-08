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

    def _owned(self, release_id):
        _, deployment = self._manifest(release_id)
        owner = SERVING_PREFIX + deployment['id']
        return [a for a in AllocationManager(self.repository, owner).list() if a['worker_id'] == owner]

    def serving_allocations(self, release_id):
        return [a for a in self._owned(release_id) if a['state'] == 'active']

    def unresolved_allocations(self, release_id):
        """Every serving resource of the release not yet proven released (intent, active,
        releasing or orphaned): retirement is complete only when this is empty."""
        from inferstack.infrastructure.allocations import LIVE_STATES
        return [a for a in self._owned(release_id) if a['state'] in (*LIVE_STATES, 'orphaned')]

    def _history(self, release_id, key):
        """Most recent value of a spec field recorded on any serving allocation of the release's
        deployment (e.g. the verified launch environment or the container placement)."""
        for alloc in reversed(self._owned(release_id)):
            if (alloc['spec'] or {}).get(key) is not None:
                return alloc
        return None

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

    def _placement(self, container, release_id=None, release=None):
        """The provider a container release must be replaced on: the placement frozen in the
        release, else the provider stated in its profile, else the provider of its recorded
        serving allocation. Never guessed: with several remote providers configured and no
        recorded placement the replacement is refused."""
        name = ((release or {}).get('body') or {}).get('placement', {}).get('provider') if release else None
        name = name or container.get('provider')
        if name is None and release_id is not None:
            recorded = self._history(release_id, 'container')
            name = recorded['provider'] if recorded else None
        if name is None:
            remote = [n for n, p in self.providers.items() if getattr(p, 'launches_remotely', False)]
            if len(remote) > 1:
                raise RuntimeUnavailable(f'container release has no recorded placement and several remote '
                                         f'providers are configured ({sorted(remote)}); refusing to guess')
            name = remote[0] if remote else 'docker'
        provider = self._provider_for(name)
        if provider is None or not getattr(provider, 'launches_remotely', False):
            raise RuntimeUnavailable(f'container release needs a configured {name!r} provider')
        return provider

    async def _launch(self, release_id, release, deployment, container=None):
        owner = SERVING_PREFIX + deployment['id']
        manager = AllocationManager(self.repository, owner)
        alloc_id = 'alloc-' + uuid.uuid4().hex[:20]
        provider = self._placement(container, release_id, release) if container is not None else self.provider
        spec = {'release_id': release_id, 'kind': 'serving'}
        from inferstack.infrastructure.allocations import ResourceConflict, place_devices
        from inferstack.optimization.experiment import GPU_ENGINES
        resources = deployment.get('resources') or {}
        if container is not None:
            spec['container'] = container  # persisted placement: recovery can relaunch or stop it
            listed = container.get('gpus') if isinstance(container.get('gpus'), list) else []
            gpus = len(listed) if listed else int(container.get('gpus') or 0)
            scheduled = getattr(provider, 'schedules_devices', provider.name == 'kubernetes')
            if scheduled:
                pass  # the cluster scheduler places count-based GPU requests; gpu_count is accounting only
            elif listed:
                spec['gpu_devices'] = [str(g) for g in listed]
            elif gpus:
                spec['gpu_pool'] = self.gpu_devices  # a local container runtime needs identified host devices
        else:
            spec['argv'] = list(deployment['command'])
            spec['launch_env'] = self._launch_env(release_id, release, alloc_id)
            scheduled = False
            gpus = int(resources.get('gpu_count') or len(resources.get('gpu_devices') or []) or
                       (1 if deployment.get('engine') in GPU_ENGINES else 0))
            if resources.get('gpu_devices'):
                spec['gpu_devices'] = list(resources['gpu_devices'])
            elif gpus:
                spec['gpu_pool'] = self.gpu_devices
        try:
            with self.repository.transaction() as db:  # intent before launch, no job lease (serving owner)
                if not scheduled:  # host devices; a cluster scheduler does its own placement
                    spec = place_devices(db, provider.name, spec, gpus)
                now = self.repository.clock()
                db.execute('INSERT INTO allocations(id,idempotency_key,job_id,step_key,worker_id,provider,'
                           'resource_type,spec,state,gpu_count,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                           (alloc_id, f'serving:{release_id}:{alloc_id}', None, None, owner, provider.name,
                            provider.resource_type, self.repository.encode(spec), 'intent', gpus, now, now))
        except ResourceConflict as exc:
            raise RuntimeUnavailable(f'no exclusive GPU placement for {release_id}: {exc}') from exc
        if container is not None and spec.get('gpu_devices') and not isinstance(container.get('gpus'), list):
            container = {**container, 'gpus': spec['gpu_devices']}  # docker --gpus device=<reserved>
        self.log_root.mkdir(parents=True, exist_ok=True)
        log = (self.log_root / f'{alloc_id}.log').open('ab')
        try:
            if container is not None:
                external = {**provider.launch(alloc_id, container), 'allocation_id': alloc_id}
            else:
                # Exactly the frozen, credential-free launch environment: never the control
                # process's own environment. Only reserved devices are added deliberately.
                env = dict(spec['launch_env'])
                if spec.get('gpu_devices'):
                    env['CUDA_VISIBLE_DEVICES'] = ','.join(spec['gpu_devices'])
                process = await provider.spawn(alloc_id, spec['argv'], log, env=env, inherit_env=False)
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

    def _launch_env(self, release_id, release, alloc_id):
        """Credential-free environment for a replacement process.

        The release's frozen `execution.env` (copied at release creation from the verified
        launch) wins; else the environment recorded on the release's earlier serving
        allocation; else a minimal allowlisted environment. Every source is re-checked
        through `scrubbed_env`, which refuses credential-like variables."""
        from inferstack.execution.sandbox import scrubbed_env
        frozen = ((release.get('body') or {}).get('execution') or {}).get('env')
        if frozen is None:
            recorded = self._history(release_id, 'launch_env')
            frozen = recorded['spec']['launch_env'] if recorded else None
        runnable = (release.get('body') or {}).get('runnable') or {}
        if frozen is not None and frozen.get('HOME'):
            home = Path(frozen['HOME'])
        elif runnable.get('path'):
            home = Path(runnable['path'])
        else:
            home = self.log_root / f'{alloc_id}.home'
        home.mkdir(parents=True, exist_ok=True)
        extra = {k: v for k, v in (frozen or {}).items() if k not in ('HOME', 'TMPDIR')}
        return scrubbed_env(extra, home=home)

    async def retire(self, release_id, *, drain_seconds=30.0):
        """Drain in-flight requests (deadline), then stop and release every serving runtime of the release.

        Each allocation is stopped by the provider recorded on it. An allocation whose
        provider is not configured is marked orphaned (operator-visible), never left active."""
        _, deployment = self._manifest(release_id)
        drained = await self.drain(deployment['id'], drain_seconds) if self.drain else {'drained': None}
        outcomes = []
        for alloc in self.unresolved_allocations(release_id):
            manager = AllocationManager(self.repository, alloc['worker_id'])
            provider = self._provider_for(alloc['provider'])
            if alloc['external'] is None and provider is not None:
                tagged = provider.find_tagged(alloc['id']) if hasattr(provider, 'find_tagged') else None
                if tagged == []:
                    manager.release(alloc['id'], 'retired_never_launched')
                    outcomes.append({'allocation_id': alloc['id'], 'state': 'released'})
                else:
                    manager.mark_orphaned(alloc['id'], 'retire_unidentified_launch')
                    outcomes.append({'allocation_id': alloc['id'], 'state': 'orphaned',
                                     'reason': 'launch identity not recorded'})
                continue
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
