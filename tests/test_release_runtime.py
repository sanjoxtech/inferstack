"""V08: a kept candidate is served from its release; replacements launch and retire cleanly."""
import os
import socket
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inferstack.infrastructure import AllocationManager, LocalProcessProvider
from inferstack.releases.manager import ReleaseManager
from inferstack.releases.rollout import RolloutController
from inferstack.releases.runtime import ReleaseRuntime, RuntimeUnavailable
from inferstack.serving.gateway import create_gateway
from inferstack.storage.repository import Repository

FIXTURE = str(Path(__file__).parent / 'fixtures/fake_openai_server.py')


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def release_for(repo, name, port):
    repo.put_record('deployment', name, {'id': name, 'engine': 'generic_openai', 'model': 'm',
                                         'base_url': f'http://127.0.0.1:{port}',
                                         'command': [sys.executable, FIXTURE, '--port', str(port)]})
    repo.put('release', 'rel-' + name, {'deployment': repo.read('deployment', name),
                                        'evidence': {'sha256': 'f' * 64, 'bytes': 0}, 'job_id': 'fixture',
                                        'verification_scope': 'fixture'})
    return 'rel-' + name


def completion(repo):
    client = TestClient(create_gateway(repo, 'tok'))
    return client.post('/v1/completions', headers={'Authorization': 'Bearer tok'},
                       json={'model': 'm', 'prompt': '2+2'}).status_code


@pytest.mark.asyncio
async def test_replacement_launch_serve_and_retire(tmp_path):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    provider = LocalProcessProvider()
    runtime = ReleaseRuntime(repo, provider, log_root=tmp_path / 'logs', ready_timeout=20)
    release = release_for(repo, 'cand', free_port())
    try:
        first = await runtime.ensure(release)
        assert first['replaced'] is True
        assert (await runtime.ensure(release))['replaced'] is False  # healthy runtime is reused, not duplicated
        ReleaseManager(repo).route('m', release)
        assert completion(repo) == 200
        [alloc] = runtime.serving_allocations(release)
        provider.stop(alloc['external'], grace_seconds=1)  # the runtime dies
        assert (await runtime.ensure(release))['replaced'] is True
        assert completion(repo) == 200
    finally:
        outcome = await runtime.retire(release)
    assert outcome['allocations'] and all(a['state'] == 'released' for a in outcome['allocations'])
    assert all(a['state'] == 'released' for a in AllocationManager(repo, 'x').list())
    assert completion(repo) == 502


@pytest.mark.asyncio
async def test_modified_runnable_artifact_is_refused(tmp_path):
    from inferstack.execution.executor import IsolatedExecutor
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    ws = tmp_path / 'ws'
    ws.mkdir()
    (ws / 'model.py').write_text('x = 1')
    release = release_for(repo, 'cand', free_port())
    body = dict(repo.read('release', release)['body'])
    repo.put('release', 'rel-built', {**body, 'runnable': {'path': str(ws), 'tree_sha256': IsolatedExecutor.tree_sha256(ws)}})
    (ws / 'model.py').write_text('x = 2')
    runtime = ReleaseRuntime(repo, LocalProcessProvider(), log_root=tmp_path / 'logs')
    with pytest.raises(RuntimeUnavailable, match='modified'):
        await runtime.ensure('rel-built')


@pytest.mark.asyncio
async def test_promotion_retires_previous_stable(tmp_path):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    runtime = ReleaseRuntime(repo, LocalProcessProvider(), log_root=tmp_path / 'logs', ready_timeout=20)
    stable, cand = release_for(repo, 'stable', free_port()), release_for(repo, 'cand', free_port())
    try:
        await runtime.ensure(stable)
        ReleaseManager(repo).route('m', stable)

        async def observe(release_id, since):
            return {'requests': 100, 'errors': 0, 'ttfb_p50_ms': 100.0}
        controller = RolloutController(repo, observe=observe, runtime=runtime)
        await controller.start('r1', 'm', cand, [{'weight': 100, 'min_duration_s': 0}])
        assert runtime.serving_allocations(cand)
        assert await controller.step('r1') == 'promoted'
        assert runtime.serving_allocations(stable) == [] and runtime.serving_allocations(cand)
        assert completion(repo) == 200
    finally:
        for release in (stable, cand):
            await runtime.retire(release)
