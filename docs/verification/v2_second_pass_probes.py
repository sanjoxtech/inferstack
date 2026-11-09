"""Independent regression/acceptance probes for the faa5cda remediation.

Run explicitly; outside default testpaths. Fake secrets are temporary sentinels,
all provider/controller observations are injected, and no real credentials are read.
Failing assertions represent unfulfilled contracts, not claimed live GPU evidence.
"""
import asyncio
import importlib.util
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

from inferstack.control.store import Store
from inferstack.engines.managed import DeploymentSpec
from inferstack.execution.executor import IsolatedExecutor
from inferstack.execution.sandbox import Sandbox
from inferstack.infrastructure import AllocationManager, LocalProcessProvider
from inferstack.integrations.outbox import Outbox
from inferstack.integrations.triggers import TriggerService
from inferstack.optimization.experiment import EndpointExecutor, EndpointVerifier
from inferstack.releases.attestation import _expected
from inferstack.releases.manager import ReleaseManager
from inferstack.releases.rollout import RolloutController
from inferstack.releases.runtime import ReleaseRuntime
from inferstack.storage.repository import Repository
from fastapi.testclient import TestClient
from inferstack.control.api import create_control_app
from inferstack.serving.gateway import create_gateway

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def repo(tmp_path):
    return Repository(tmp_path / 'db', tmp_path / 'artifacts')


def fixtures(repo):
    repo.put_record('hardware', 'hw', {'provider': 'attached', 'observed_at': 1.0, 'gpus': []})
    repo.put_record('model', 'mdl', {'source': 'huggingface', 'repo_id': 'org/m', 'revision': 'b' * 40})
    for name in ('base', 'cand'):
        repo.put_record('deployment', name, {'id': name, 'engine': 'generic_openai', 'model': 'm',
            'base_url': 'http://' + name, 'hardware_ref': 'hw', 'model_ref': 'mdl'})
        repo.put('release', 'rel-' + name, {'deployment': repo.read('deployment', name),
            'evidence': {'sha256': 'f' * 64, 'bytes': 0}, 'job_id': 'fixture', 'verification_scope': 'fixture'})


class MemoryEngine:
    def __init__(self, spec):
        self.spec = spec

    async def start(self, *args, **kwargs):
        pass

    async def ready(self):
        return True

    async def close(self):
        pass


class DeterministicVerifier(EndpointVerifier):
    async def trial(self, engine):
        return {'e2e_ms': 100.0 if engine.spec.id == 'base' else 90.0}, True, ['four']


def verifier(repo, tmp_path):
    executor = EndpointExecutor(repo, tmp_path / 'logs', engine_factory=MemoryEngine)
    return DeterministicVerifier(repo, executor, 'base',
        [{'prompt': '2+2', 'expected': 'four', 'max_tokens': 4}],
        [{'metric': 'e2e_ms', 'direction': 'min'}])


# Amended 9 Nov 2026 (platform semantics): on Linux/bwrap a path outside the boundary is
# invisible (FileNotFoundError) or on a read-only root (EROFS) rather than EPERM as on macOS.
# Any OS-level denial is accepted; the asserted property (no secret read, no outside write) is unchanged.
def test_candidate_cannot_read_host_project_dotenv(tmp_path):
    sentinel = tmp_path / 'host-project/.env'
    sentinel.parent.mkdir()
    sentinel.write_text('FAKE_API_KEY=audit-only-not-a-real-key\n')
    workspace = tmp_path / 'candidate'
    workspace.mkdir()
    sandbox = Sandbox(require_filesystem_isolation=True)
    code, output = sandbox.run([sys.executable, '-c',
        f'from pathlib import Path\ntry:\n Path({str(sentinel)!r}).read_text(); print("HOST_SECRET_READABLE")\n'
        'except OSError:\n print("denied")'], cwd=workspace, home=workspace)
    assert code == 0, output
    assert 'HOST_SECRET_READABLE' not in output, sandbox.describe()


@pytest.mark.asyncio
async def test_missing_declared_measurement_cannot_keep(repo, tmp_path):
    fixtures(repo)
    v = verifier(repo, tmp_path)
    plan = {'baseline_id': 'base', 'baseline_revision': 1, 'candidate_id': 'cand', 'candidate_revision': 1,
            'expected_measurements': ['e2e_ms', 'output_tokens_per_s']}
    candidate = await v.executor.execute(plan, 'expected-metrics')
    result = await v.verify(plan, candidate)
    assert result['decision'] != 'keep', result


@pytest.mark.asyncio
async def test_ttft_ignores_role_and_usage_metadata(repo, tmp_path, monkeypatch):
    v = verifier(repo, tmp_path)

    class Stream:
        async def stream(self, payload):
            yield b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"four"}}]}\n\n'
            yield b'data: [DONE]\n\n'

    ticks = iter([0.0, 0.001, 0.100])
    monkeypatch.setattr('inferstack.optimization.experiment.time.perf_counter', lambda: next(ticks))
    result = await v.phases(Stream())
    assert result['ttft_client_ms'] == pytest.approx(100.0), result


def test_logprob_gate_cannot_accept_a_truncated_matching_prefix(repo, tmp_path):
    v = verifier(repo, tmp_path)
    passed, divergence, reason = v.logit_agreement(
        [(['the', 'cat', 'sat'], [-0.1, -0.2, -0.3])], [(['the'], [-0.1])])
    assert passed is not True, (passed, divergence, reason)


@pytest.mark.asyncio
async def test_release_model_identity_does_not_follow_later_record_updates(repo, tmp_path):
    fixtures(repo)
    v = verifier(repo, tmp_path)
    plan = {'baseline_id': 'base', 'baseline_revision': 1, 'candidate_id': 'cand', 'candidate_revision': 1}
    candidate = await v.executor.execute(plan, 'release-identity')
    verdict = await v.verify(plan, candidate)
    assert verdict['decision'] == 'keep'
    job = repo.submit('verified-release', {'deployment_id': 'base'})
    claim = repo.claim(100)
    repo.finish(job, claim['lease_token'], {'decision': 'keep', 'experiments': [
        {'plan': plan, 'candidate': candidate, 'verdict': verdict}]})
    release = repo.read('release', ReleaseManager(repo).create(job))
    frozen = _expected(repo, release['body']['deployment']['body'], release)['model_revision']
    body = repo.read('model', 'mdl')['body']
    repo.put_record('model', 'mdl', {**body, 'revision': 'c' * 40}, expected_revision=1)
    observed = _expected(repo, release['body']['deployment']['body'], release)['model_revision']
    assert observed == frozen, (frozen, observed)


@pytest.mark.asyncio
async def test_rollout_route_conflict_does_not_retire_live_stable(repo):
    fixtures(repo)
    manager = ReleaseManager(repo)
    manager.route('m', 'rel-base')
    retired = []

    class Runtime:
        async def ensure(self, release_id):
            return {}

        async def retire(self, release_id):
            retired.append(release_id)

    async def observe(release_id, since):
        # Another authorized writer restores the old stable while this step is observing.
        # The controller must see the ownership conflict and leave both runtimes alone.
        route = repo.read('route', 'm')
        if route['body'].get('candidate'):
            manager.route('m', 'rel-base', expected_revision=route['revision'])
        return {'requests': 100, 'errors': 0, 'ttfb_p50_ms': 100.0}

    controller = RolloutController(repo, observe=observe, runtime=Runtime())
    await controller.start('r', 'm', 'rel-cand', [{'weight': 100, 'min_duration_s': 0}])
    outcome = await controller.step('r')
    assert retired == [], (outcome, retired, repo.read('rollout', 'r')['body']['state'])


@pytest.mark.asyncio
async def test_terminal_rollout_retries_interrupted_runtime_retirement(repo):
    fixtures(repo)
    ReleaseManager(repo).route('m', 'rel-base')
    calls = []

    class Runtime:
        async def ensure(self, release_id):
            return {}

        async def retire(self, release_id):
            calls.append(release_id)
            if len(calls) == 1:
                raise RuntimeError('injected interruption before resource retirement')

    async def observe(release_id, since):
        return {'requests': 100, 'errors': 0, 'ttfb_p50_ms': 100.0}

    runtime = Runtime()
    controller = RolloutController(repo, observe=observe, runtime=runtime)
    await controller.start('r', 'm', 'rel-cand', [{'weight': 100, 'min_duration_s': 0}])
    with pytest.raises(RuntimeError, match='injected interruption'):
        await controller.step('r')
    await RolloutController(repo, observe=observe, runtime=runtime).step('r')
    assert len(calls) == 2, calls


@pytest.mark.asyncio
async def test_post_promotion_rollback_restores_a_live_previous_runtime(repo, tmp_path):
    def port():
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            return sock.getsockname()[1]

    for name in ('base', 'cand'):
        endpoint = port()
        repo.put_record('deployment', name, {'id': name, 'engine': 'generic_openai', 'model': 'm',
            'base_url': f'http://127.0.0.1:{endpoint}',
            'command': [sys.executable, str(ROOT / 'tests/fixtures/fake_openai_server.py'), '--port', str(endpoint)]})
        repo.put('release', 'rel-' + name, {'deployment': repo.read('deployment', name),
            'evidence': {'sha256': 'f' * 64, 'bytes': 0}, 'job_id': 'fixture', 'verification_scope': 'fixture'})
    manager = ReleaseManager(repo)
    runtime = ReleaseRuntime(repo, LocalProcessProvider(), log_root=tmp_path / 'logs', ready_timeout=10)

    async def observe(release_id, since):
        return {'requests': 100, 'errors': 0, 'ttfb_p50_ms': 100.0}

    try:
        await runtime.ensure('rel-base')
        manager.route('m', 'rel-base')
        controller = RolloutController(repo, observe=observe, runtime=runtime)
        await controller.start('r', 'm', 'rel-cand', [{'weight': 100, 'min_duration_s': 0}])
        assert await controller.step('r') == 'promoted'
        client = TestClient(create_control_app(repo, 'test', release_runtime=runtime))
        response = client.post('/control/routes/m/rollback', headers={'Authorization': 'Bearer test'},
            params={'expected_revision': repo.read('route', 'm')['revision']})
        assert response.status_code == 200, response.text
        gateway = TestClient(create_gateway(repo, 'test'))
        response = gateway.post('/v1/completions', headers={'Authorization': 'Bearer test'},
            json={'model': 'm', 'prompt': '2+2'})
        assert response.status_code == 200, response.json()
    finally:
        await runtime.retire('rel-base')
        await runtime.retire('rel-cand')


@pytest.mark.asyncio
async def test_control_runtime_dispatches_container_retirement_to_its_provider(repo, tmp_path):
    fixtures(repo)
    job = repo.submit('container', {})
    claim = repo.claim(100)
    manager = AllocationManager(repo, 'worker')
    allocation = manager.intent(claim, 'experiment', purpose='candidate', provider='docker',
        resource_type='container', spec={'deployment_id': 'cand'})
    manager.activate(allocation['id'], claim, {'provider': 'docker', 'container_id': 'fake-container'})
    manager.transfer_to_serving(allocation['id'], 'cand', claim)
    # This is the exact provider selection used by control.__main__.serve.
    runtime = ReleaseRuntime(repo, LocalProcessProvider(), log_root=tmp_path / 'logs')
    await runtime.retire('rel-cand')
    assert manager.get(allocation['id'])['state'] != 'active'


@pytest.mark.asyncio
async def test_build_process_carries_its_persisted_allocation_identity(repo, tmp_path):
    repo.put_record('deployment', 'base', {'id': 'base', 'engine': 'vllm', 'model': 'm',
        'base_url': 'http://127.0.0.1:9000', 'command': [sys.executable, '-m', 'vllm', '--port', '9000']})
    repo.put_record('technique', 'fixture', {'name': 'fixture', 'summary': 'fixture', 'engines': [{'engine': 'vllm'}]})
    job = repo.submit('build', {})
    claim = repo.claim(100)
    manager = AllocationManager(repo, 'worker')
    executor = IsolatedExecutor(repo, tmp_path / 'logs', work_root=tmp_path / 'work', source_root=ROOT,
        source_revision='a' * 40, allocations=manager, provider=LocalProcessProvider())
    observed = []

    def build(ws, action, artifacts, log):
        code, out = executor.sandbox.run([sys.executable, '-c',
            'import os; print(os.environ.get("INFERSTACK_ALLOCATION_ID", "MISSING"))'], cwd=ws, home=ws)
        assert code == 0, out
        observed.append(out.strip())

    async def launch(record, plan, namespace, context, **kwargs):
        return {'id': namespace}

    executor.build_plugin, executor.launch = build, launch
    try:
        await executor.execute({'baseline_id': 'base', 'baseline_revision': 1,
            'actions': [{'id': 'plugin', 'type': 'plugin_install', 'params': {'technique_id': 'fixture'}}],
            'action_order': ['plugin']}, 'build-probe', context={'job': claim, 'step_key': 'experiment'})
        [allocation] = manager.list(job_id=job)
        assert observed == [allocation['id']], (observed, allocation['state'], allocation['external'])
    finally:
        executor._discard('build-probe')


def test_stale_process_group_identity_never_signals_reused_group(monkeypatch):
    provider = LocalProcessProvider()
    monkeypatch.setattr(provider, 'alive', lambda ext: False)
    monkeypatch.setattr(provider, 'group_members', lambda pgid: [456789])
    monkeypatch.setattr(provider, 'find_tagged', lambda alloc: [])
    signalled = []
    monkeypatch.setattr(os, 'killpg', lambda pgid, sig: signalled.append((pgid, sig)))
    monkeypatch.setattr(os, 'kill', lambda pid, sig: None)
    # Record predates a PID/PGID reuse; no member has our allocation tag.
    provider.stop({'host': provider.host, 'pid': 456789, 'pgid': 456789,
                   'start': 'old-process-start-time', 'allocation_id': 'old-allocation'}, grace_seconds=0)
    assert signalled == [], signalled


@pytest.mark.asyncio
async def test_two_implicit_gpu_reservations_do_not_launch_on_unidentified_devices(repo, tmp_path):
    spawned = []

    class Process:
        pid = 123

    class Provider:
        name, resource_type = 'local', 'process'

        async def spawn(self, alloc_id, argv, log, **kwargs):
            spawned.append(kwargs)
            return Process()

        def external_for(self, pid):
            return {'host': 'test', 'pid': pid}

    manager = AllocationManager(repo, 'worker')
    executor = EndpointExecutor(repo, tmp_path / 'logs', allocations=manager, provider=Provider())
    from inferstack.infrastructure.allocations import ResourceConflict
    for name in ('a', 'b'):
        repo.submit(name, {})
        claim = repo.claim(100)
        spawn = executor._spawner({'job': claim, 'step_key': 'experiment', 'namespace': name},
            DeploymentSpec(name, 'vllm', 'm', 'http://' + name, command=('vllm', 'serve')))
        try:
            await spawn(['vllm', 'serve'], None)
        except (ResourceConflict, ValueError):
            break  # refusing unidentified/conflicting GPU ownership is safe
    assert len(spawned) < 2 or all('CUDA_VISIBLE_DEVICES' in (s.get('env') or {}) for s in spawned), spawned


def test_trigger_global_admission_is_atomic_across_tickers(repo, monkeypatch):
    clock = threading.local()
    repo.clock = lambda: getattr(clock, 'now', 119.9)
    specs = {'deployment_id': 'base', 'goal': 'tune', 'cooldown_s': 60, 'interval_s': 60,
             'budget': {'experiments': 1}, 'max_jobs_per_day': 2}
    service = TriggerService(repo, max_active=1)
    service.create('a', 'periodic', specs)
    barrier = threading.Barrier(2)
    original = repo.submit

    def submit(*args, **kwargs):
        barrier.wait(timeout=5)  # both tickers have passed their separate admission transaction
        return original(*args, **kwargs)

    monkeypatch.setattr(repo, 'submit', submit)
    def tick(now):
        clock.now = now
        return asyncio.run(TriggerService(repo, max_active=1).tick())
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as pool:
        # Identical ticker invocations straddle a normal periodic window boundary.
        futures = [pool.submit(tick, now) for now in (119.9, 120.1)]
        outcomes = [future.result(timeout=10) for future in futures]
    with repo.connection() as db:
        count = db.execute("SELECT COUNT(*) FROM jobs WHERE state='pending'").fetchone()[0]
    assert count <= 1, (count, outcomes)


def test_cancelled_factoryos_job_enqueues_terminal_callback(repo):
    job = repo.submit('callback', {'outcome_callback': {'issue_id': 'issue', 'idempotency_key': 'callback'}})
    assert repo.cancel(job)
    assert len(Outbox(repo).list()) == 1


def test_optional_container_skips_do_not_fail_gpu_suite_gate():
    spec = importlib.util.spec_from_file_location('q_review', ROOT / 'scripts/v2_qualify.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.suite_gate(0, 'SKIPPED [2] tests/test_docker_live.py: needs Docker daemon\n'
                                'SKIPPED [1] tests/test_kubernetes_live.py: INFERSTACK_TEST_K8S_CONTEXT required\n'
                                '246 passed, 3 skipped', has_gpu=True)
    assert result['status'] == 'passed', result
