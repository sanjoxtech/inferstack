"""Independent lifecycle and launch-isolation probes for f50aab1.

Run explicitly: python -m pytest docs/verification/v2_third_pass_probes.py -v
No GPU, cloud resources or real secrets are used. Temporary local servers are
stopped in finally blocks; provider observations are controlled test doubles.
"""
import asyncio
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from inferstack.execution.executor import IsolatedExecutor
from inferstack.execution.sandbox import Sandbox, scrubbed_env
from inferstack.infrastructure import AllocationManager, LocalProcessProvider
from inferstack.optimization.experiment import EndpointExecutor
from inferstack.releases.manager import ReleaseManager
from inferstack.releases.rollout import RolloutController
from inferstack.releases.runtime import ReleaseRuntime
from inferstack.storage.repository import Repository


@pytest.fixture
def repo(tmp_path):
    return Repository(tmp_path / 'db', tmp_path / 'artifacts')


def release(repo, name, *, profile=None, command=None, base_url=None):
    repo.put_record('deployment', name, {'id': name, 'engine': 'generic_openai', 'model': 'm',
        'base_url': base_url or 'http://' + name, 'command': command or [], 'profile': profile or {}})
    identifier = 'rel-' + name
    repo.put('release', identifier, {'deployment': repo.read('deployment', name),
        'evidence': {'sha256': 'f' * 64, 'bytes': 0}, 'job_id': 'fixture', 'verification_scope': 'fixture'})
    return identifier


def serving_allocation(repo, deployment, provider, external, spec=None):
    job = repo.submit('owner-' + deployment, {})
    claim = repo.claim(100)
    manager = AllocationManager(repo, 'worker')
    row = manager.intent(claim, 'experiment', purpose=deployment, provider=provider,
        resource_type='container' if provider != 'local' else 'process', spec=spec or {})
    manager.activate(row['id'], claim, external)
    manager.transfer_to_serving(row['id'], deployment, claim)
    return manager, row['id']


class ReadyEngine:
    def __init__(self, spec):
        self.spec = spec

    async def ready(self):
        return True

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_orphaned_retirement_remains_pending_until_resource_is_stopped(repo, tmp_path):
    stable, candidate = release(repo, 'stable'), release(repo, 'candidate')
    ReleaseManager(repo).route('m', stable)

    class Provider:
        name = 'local'
        stop_calls = 0

        def stop(self, external):
            self.stop_calls += 1
            return False  # resource remains running; cleanup could not be confirmed

    provider = Provider()
    manager, allocation = serving_allocation(repo, 'stable', 'local', {'pid': 1})
    runtime = ReleaseRuntime(repo, provider, engine_factory=ReadyEngine, log_root=tmp_path / 'logs')

    async def observe(release_id, since):
        return {'requests': 100, 'errors': 0, 'ttfb_p50_ms': 100.0}

    controller = RolloutController(repo, observe=observe, runtime=runtime)
    await controller.start('r', 'm', candidate, [{'weight': 100, 'min_duration_s': 0}])
    await controller.step('r')
    assert manager.get(allocation)['state'] == 'orphaned'
    assert repo.read('rollout', 'r')['body']['retirement']['state'] == 'pending'
    await controller.step('r')
    obligation = repo.read('rollout', 'r')['body']['retirement']
    assert obligation['state'] == 'pending', (obligation, provider.stop_calls, manager.get(allocation)['state'])


class ContainerProvider:
    launches_remotely = True
    resource_type = 'container'

    def __init__(self, name):
        self.name, self.launches = name, []

    def alive(self, external):
        return not external.get('dead', False)

    def launch(self, allocation_id, spec):
        self.launches.append(spec)
        return {'provider': self.name, 'id': allocation_id}

    def stop(self, external):
        return True


@pytest.mark.asyncio
async def test_replacement_preserves_retained_kubernetes_placement(repo, tmp_path):
    docker, kubernetes = ContainerProvider('docker'), ContainerProvider('kubernetes')
    container = {'image': 'python@sha256:' + 'a' * 64, 'container_port': 8000, 'host_port': 8100}
    identifier = release(repo, 'remote', profile={'container': container})
    serving_allocation(repo, 'remote', 'kubernetes', {'provider': 'kubernetes', 'dead': True},
                       spec={'container': container})
    runtime = ReleaseRuntime(repo, LocalProcessProvider(), providers={'docker': docker, 'kubernetes': kubernetes},
        engine_factory=ReadyEngine, log_root=tmp_path / 'logs')
    result = await runtime.ensure(identifier)
    assert result['provider'] == 'kubernetes', (result, len(docker.launches), len(kubernetes.launches))


@pytest.mark.asyncio
async def test_gpu_kubernetes_replacement_uses_cluster_scheduler(repo, tmp_path):
    kubernetes = ContainerProvider('kubernetes')
    container = {'provider': 'kubernetes', 'image': 'python@sha256:' + 'a' * 64,
                 'container_port': 8000, 'gpus': 1}
    identifier = release(repo, 'remote', profile={'container': container})
    runtime = ReleaseRuntime(repo, LocalProcessProvider(), providers={'kubernetes': kubernetes},
        engine_factory=ReadyEngine, log_root=tmp_path / 'logs')
    result = await runtime.ensure(identifier)
    assert result['provider'] == 'kubernetes' and len(kubernetes.launches) == 1


# Amended 9 Nov 2026 (platform semantics): on Linux/bwrap a path outside the boundary is
# invisible (FileNotFoundError) or on a read-only root (EROFS) rather than EPERM as on macOS.
# Any OS-level denial is accepted; the asserted property (no secret read, no outside write) is unchanged.
def test_candidate_launcher_does_not_grant_read_access_to_script_sibling_secrets(tmp_path):
    host = tmp_path / 'host-project'
    host.mkdir()
    (host / '.env').write_text('AUDIT_FAKE_SECRET=not-a-real-credential\n')
    script = host / 'server.py'
    script.write_text('from pathlib import Path\n'
        'try:\n Path(__file__).with_name(".env").read_text(); print("SIBLING_SECRET_READABLE")\n'
        'except OSError:\n print("denied")\n')
    workspace = tmp_path / 'candidate'
    workspace.mkdir()
    sandbox = Sandbox(require_filesystem_isolation=True)
    argv = sandbox.launcher([sys.executable, str(script)], workspace=workspace)
    result = subprocess.run(argv, cwd=workspace, env=scrubbed_env(home=workspace),
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'SIBLING_SECRET_READABLE' not in result.stdout, sandbox.describe()


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


SERVER = '''import argparse,json
from http.server import BaseHTTPRequestHandler,HTTPServer
p=argparse.ArgumentParser();p.add_argument('--port',type=int);a=p.parse_args()
class H(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_GET(self):
  b=json.dumps({'data':[{'id':'m'}]}).encode()
  self.send_response(200);self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
HTTPServer(('127.0.0.1',a.port),H).serve_forever()
'''


@pytest.mark.asyncio
async def test_retained_runnable_replacement_preserves_verified_pythonpath(repo, tmp_path):
    workspace = tmp_path / 'runnable'
    workspace.mkdir()
    (workspace / 'audit_candidate_server.py').write_text(SERVER)
    sandbox = Sandbox(require_filesystem_isolation=True)
    port = free_port()
    command = sandbox.launcher([sys.executable, '-m', 'audit_candidate_server', '--port', str(port)],
                               workspace=workspace)
    repo.put_record('deployment', 'base', {'id': 'base', 'engine': 'generic_openai', 'model': 'm',
        'base_url': 'http://127.0.0.1:9'})
    repo.put_record('deployment', 'candidate', {'id': 'candidate', 'engine': 'generic_openai', 'model': 'm',
        'base_url': f'http://127.0.0.1:{port}', 'command': command})
    job = repo.submit('runnable', {})
    claim = repo.claim(100)
    manager, provider = AllocationManager(repo, 'worker'), LocalProcessProvider()
    executor = EndpointExecutor(repo, tmp_path / 'logs', allocations=manager, provider=provider)
    runtime = ReleaseRuntime(repo, provider, log_root=tmp_path / 'replacements', ready_timeout=3)
    candidate = None
    try:
        record = repo.read('deployment', 'candidate')
        # This is the exact launch path used by IsolatedExecutor after a source patch:
        # patched modules come from its workspace via a scrubbed PYTHONPATH.
        candidate = await executor.launch(record, {'baseline_id': 'base'}, 'runnable-test',
            context={'job': claim, 'step_key': 'experiment'},
            env=scrubbed_env({'PYTHONPATH': str(workspace)}, home=workspace))
        retained = await executor.retain(candidate, {'job': claim})
        repo.put('release', 'rel-runnable', {'deployment': record,
            'evidence': {'sha256': 'f' * 64, 'bytes': 0}, 'job_id': 'fixture', 'verification_scope': 'fixture',
            'runnable': {'path': str(workspace), 'tree_sha256': IsolatedExecutor.tree_sha256(workspace)}})
        assert await runtime._ready(record['body'])  # the initial patched runtime works
        assert provider.stop(retained['external'], grace_seconds=1)
        manager.release(retained['allocation_id'], 'injected runtime death')
        result = await runtime.ensure('rel-runnable')
        assert result['replaced'] and await runtime._ready(record['body'])
    finally:
        if repo.list_records('release'):
            await runtime.retire('rel-runnable')
        if candidate is not None:
            await executor.cleanup(candidate)


@pytest.mark.asyncio
async def test_sandboxed_replacement_does_not_inherit_control_process_secrets(repo, tmp_path, monkeypatch):
    # Read only this invented sentinel; no real host secret is inspected or printed.
    monkeypatch.setenv('AUDIT_ONLY_FAKE_TOKEN', 'audit-sentinel-not-a-real-credential')
    workspace = tmp_path / 'replacement'
    workspace.mkdir()
    marker = workspace / 'inherited.txt'
    code = ('import os,time; '
            f'open({str(marker)!r},"w").write("visible" if os.environ.get("AUDIT_ONLY_FAKE_TOKEN") else "hidden"); '
            'time.sleep(30)')
    command = Sandbox(require_filesystem_isolation=True).launcher([sys.executable, '-c', code], workspace=workspace)
    identifier = release(repo, 'replacement', command=command)
    runtime = ReleaseRuntime(repo, LocalProcessProvider(), engine_factory=ReadyEngine,
                             log_root=tmp_path / 'logs')
    try:
        await runtime.ensure(identifier)
        for _ in range(100):
            if marker.exists():
                break
            await asyncio.sleep(0.01)
        assert marker.read_text() == 'hidden'
    finally:
        await runtime.retire(identifier)
