"""Independent acceptance probes for ff777007 (intentionally failing contracts).

Run explicitly: python -m pytest docs/verification/v2_contract_probes.py -v
These are outside the default testpaths so they do not alter the baseline suite.
Only temporary files, local processes and injected transports are used. No keys,
GPU, Docker daemon, Kubernetes account or external requests are needed.
"""
import asyncio
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inferstack.control.api import create_control_app
from inferstack.control.worker import OptimizationWorker
from inferstack.evaluation.numerical import check
from inferstack.execution.sandbox import Sandbox
from inferstack.execution.executor import IsolatedExecutor
from inferstack.infrastructure import AllocationManager
from inferstack.infrastructure.remote import KubernetesProvider
from inferstack.optimization.experiment import EndpointExecutor, EndpointVerifier
from inferstack.releases.manager import ReleaseManager
from inferstack.releases.rollout import RolloutController
from inferstack.storage.repository import Repository
from inferstack.serving.gateway import create_gateway

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def repo(tmp_path):
    return Repository(tmp_path / 'db', tmp_path / 'artifacts')


def deployments(repo):
    for name in ('base', 'cand'):
        repo.put_record('deployment', name, {
            'id': name, 'engine': 'generic_openai', 'model': 'm',
            'base_url': 'http://' + name,
        })


def releases(repo):
    deployments(repo)
    # Focused rollout fixtures, equivalent to the existing test_rollout fixtures.
    # These are not evidence of a real, verified model release.
    for name in ('base', 'cand'):
        repo.put('release', 'rel-' + name, {
            'deployment': repo.read('deployment', name),
            'evidence': {'sha256': 'f' * 64, 'bytes': 0},
            'job_id': 'fixture', 'verification_scope': 'fixture',
        })
    ReleaseManager(repo).route('m', 'rel-base')


async def healthy_observer(release_id, since):
    return {'requests': 100, 'errors': 0, 'ttfb_p50_ms': 100.0}


def test_numerical_gate_rejects_nonfinite_candidate_and_reference():
    result = check('nonfinite', [float('nan')], [0.0], dtype='bfloat16',
                   reference=[float('nan')])
    assert result['pass'] is False, result


class MemoryEngine:
    def __init__(self, spec):
        self.spec = spec

    async def ready(self):
        return True

    async def start(self, *args, **kwargs):
        pass

    async def close(self):
        pass


class DeterministicVerifier(EndpointVerifier):
    async def trial(self, engine):
        # Controlled timing inputs test gate dispatch, not physical performance.
        return {'e2e_ms': 100.0 if engine.spec.id == 'base' else 90.0}, True, ['four']


async def endpoint_verdict(repo, tmp_path):
    deployments(repo)
    executor = EndpointExecutor(repo, tmp_path / 'logs', engine_factory=MemoryEngine)
    plan = {'baseline_id': 'base', 'baseline_revision': 1,
            'candidate_id': 'cand', 'candidate_revision': 1,
            'gates': ['correctness', 'quality', 'health', 'comparable', 'complete',
                      'numerical_oracle', 'cancellation', 'overload'],
            'actions': [{'id': 'measure', 'type': 'measure', 'depends_on': [],
                         'params': {'repeats': 100}}]}
    candidate = await executor.execute(plan, 'probe')
    verifier = DeterministicVerifier(repo, executor, 'base',
        [{'prompt': '2+2', 'expected': 'four', 'max_tokens': 4}],
        [{'metric': 'e2e_ms', 'direction': 'min'}], repeats=3)
    return await verifier.verify(plan, candidate)


@pytest.mark.asyncio
async def test_endpoint_cannot_keep_without_requested_numerical_and_serving_gates(repo, tmp_path):
    verdict = await endpoint_verdict(repo, tmp_path)
    assert verdict['decision'] != 'keep', verdict


@pytest.mark.asyncio
async def test_unknown_hardware_cannot_satisfy_comparability(repo, tmp_path):
    verdict = await endpoint_verdict(repo, tmp_path)
    assert verdict['gates']['comparable'] is not True, verdict


def test_rollout_api_refuses_traffic_when_attestation_is_not_configured(repo):
    releases(repo)
    client = TestClient(create_control_app(repo, 'test-token', rollout_observer=healthy_observer))
    result = client.post('/control/rollouts', headers={'Authorization': 'Bearer test-token'}, json={
        'rollout_id': 'r1', 'model': 'm', 'release_id': 'rel-cand',
        'stages': [{'weight': 100, 'min_duration_s': 0}],
    })
    assert result.status_code != 200, result.json()


@pytest.mark.asyncio
async def test_rollout_recovers_route_write_before_progress_write(repo, monkeypatch):
    releases(repo)
    controller = RolloutController(repo, observe=healthy_observer)
    original = repo.put_record

    def interrupted(kind, identifier, body, *args, **kwargs):
        if kind == 'rollout' and body.get('state') == 'progressing':
            raise RuntimeError('injected crash after route update')
        return original(kind, identifier, body, *args, **kwargs)

    monkeypatch.setattr(repo, 'put_record', interrupted)
    with pytest.raises(RuntimeError, match='injected crash'):
        await controller.start('r1', 'm', 'rel-cand', [{'weight': 100, 'min_duration_s': 0}])
    monkeypatch.setattr(repo, 'put_record', original)
    # A new controller represents a restarted control process.
    restarted = RolloutController(repo, observe=healthy_observer)
    outcome = await restarted.step('r1')
    assert outcome in ('promoted', 'rolled_back', 'progressing'), outcome


@pytest.mark.asyncio
async def test_rollout_cannot_promote_a_different_concurrent_candidate(repo):
    releases(repo)
    controller = RolloutController(repo, observe=healthy_observer)
    await controller.start('r1', 'm', 'rel-cand', [{'weight': 100, 'min_duration_s': 0}])
    other = dict(repo.read('release', 'rel-base')['body'])
    repo.put('release', 'rel-other', other)
    manager = ReleaseManager(repo)
    route = repo.read('route', 'm')
    manager.route('m', 'rel-base', candidate_release='rel-other', weight=100,
                  expected_revision=route['revision'])
    await controller.step('r1')
    assert repo.read('route', 'm')['body']['stable'] != 'rel-other'


@pytest.mark.asyncio
async def test_latency_gated_rollout_can_finish_100_percent_stage(repo):
    releases(repo)
    now = [1000.0]

    async def observe(release_id, since):
        if release_id == 'rel-base' and repo.read('route', 'm')['body']['weight'] == 100:
            return {'requests': 0, 'errors': 0, 'ttfb_p50_ms': None}
        return await healthy_observer(release_id, since)

    controller = RolloutController(repo, observe=observe, clock=lambda: now[0])
    stages = [{'weight': w, 'min_duration_s': 1, 'max_latency_regression': 0.2} for w in (50, 100)]
    await controller.start('r1', 'm', 'rel-cand', stages)
    now[0] += 2
    assert await controller.step('r1') == 'advanced'
    now[0] += 2
    assert await controller.step('r1') == 'promoted'


def test_sandbox_cannot_rewrite_files_outside_candidate_workspace(tmp_path):
    protected = tmp_path / 'trusted-evaluator-sentinel'
    protected.write_text('trusted')
    workspace = tmp_path / 'candidate'
    workspace.mkdir()
    sandbox = Sandbox()
    # Amended after review: a deny-based boundary fails the write closed (the original
    # `code == 0` precondition presumed a copy-on-write view). The candidate must still
    # be able to write inside its workspace, and the outside write must be refused.
    code, output = sandbox.run([sys.executable, '-c',
        'from pathlib import Path; Path("inside").write_text("ok")\n'
        f'try:\n    Path({str(protected)!r}).write_text("tampered")\n'
        'except PermissionError:\n    print("denied")'],
        cwd=workspace, home=workspace)
    assert code == 0, output
    assert (workspace / 'inside').read_text() == 'ok', output
    assert 'denied' in output, (output, sandbox.describe())
    assert protected.read_text() == 'trusted', sandbox.describe()


def test_kubernetes_stop_waits_for_pods_not_just_deployment_object():
    pod = {'running': True}
    deployment = {'exists': True}

    def runner(argv, input_text=None, timeout=120):
        args = argv[argv.index('-n') + 2:]
        if args[:2] == ['get', 'deployment']:
            return (0, json.dumps({'metadata': {'uid': 'original'}}), '') if deployment['exists'] else (1, '', 'NotFound')
        if args[:2] == ['delete', 'deployment']:
            deployment['exists'] = False  # realistic asynchronous garbage collection
            return 0, '', ''
        if args[0] == 'delete':
            return 0, '', ''
        if args[:2] == ['get', 'pods']:
            return 0, json.dumps({'items': [{'status': {'phase': 'Running'}}] if pod['running'] else []}), ''
        raise AssertionError(argv)

    provider = KubernetesProvider(namespace='exp', runner=runner)
    stopped = provider.stop({'provider': 'kubernetes', 'namespace': 'exp', 'name': 'candidate', 'uid': 'original'},
                            grace_seconds=0.01)
    assert not (stopped and pod['running']), 'stop returned True while the GPU pod was still running'


@pytest.mark.asyncio
async def test_local_stop_cleans_children_after_process_leader_has_exited(tmp_path):
    from inferstack.infrastructure.local import LocalProcessProvider
    child_path = tmp_path / 'child.pid'
    leader_code = (
        'import subprocess,sys,time; from pathlib import Path; '
        'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
        f'Path({str(child_path)!r}).write_text(str(p.pid)); time.sleep(0.3)'
    )
    provider = LocalProcessProvider()
    with open(os.devnull, 'wb') as log:
        leader = await provider.spawn('audit-child-cleanup', [sys.executable, '-c', leader_code], log)
        external = provider.external_for(leader.pid)
        try:
            await leader.wait()
            child = int(child_path.read_text())
            stopped = provider.stop(external, grace_seconds=0.01)
            try:
                os.kill(child, 0)
                child_alive = True
            except ProcessLookupError:
                child_alive = False
            assert not (stopped and child_alive), 'stop returned True with a surviving process-group member'
        finally:
            # Always remove the harmless test process, including on assertion failure.
            try:
                os.killpg(external['pgid'], signal.SIGKILL)
            except ProcessLookupError:
                pass


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@pytest.mark.asyncio
async def test_kept_owned_candidate_can_be_served_from_its_release(repo, tmp_path):
    from inferstack.infrastructure.local import LocalProcessProvider
    fixture = str(ROOT / 'tests/fixtures/fake_openai_server.py')
    base_port, cand_port = free_port(), free_port()
    baseline = subprocess.Popen([sys.executable, fixture, '--port', str(base_port)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    executor = None
    try:
        for _ in range(100):
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1', base_port)) == 0:
                    break
            await asyncio.sleep(0.02)
        # Amended after review: V04 requires pinned hardware and model identity for
        # comparability, so this fixture declares them (otherwise the verdict is inconclusive).
        repo.put_record('hardware', 'hw', {'provider': 'attached', 'observed_at': 1.0, 'gpus': []})
        repo.put_record('model', 'mdl', {'source': 'huggingface', 'repo_id': 'org/m', 'revision': 'b' * 40})
        for name, port in (('base', base_port), ('cand', cand_port)):
            repo.put_record('deployment', name, {
                'id': name, 'engine': 'generic_openai', 'model': 'm', 'hardware_ref': 'hw', 'model_ref': 'mdl',
                'base_url': f'http://127.0.0.1:{port}',
                'command': [sys.executable, fixture, '--port', str(port)] if name == 'cand' else [],
            })
        job_id = repo.submit('owned-release', {'deployment_id': 'base'})
        worker_id = repo.register_worker(30)
        allocations = AllocationManager(repo, worker_id)
        provider = LocalProcessProvider()
        executor = EndpointExecutor(repo, tmp_path / 'logs', allocations=allocations, provider=provider)

        class Planner:
            async def plan(self, request, history):
                return {'baseline_id': 'base', 'baseline_revision': 1,
                        'candidate_id': 'cand', 'candidate_revision': 1}

        verifier = DeterministicVerifier(repo, executor, 'base',
            [{'prompt': '2+2', 'expected': 'four', 'max_tokens': 4}],
            [{'metric': 'e2e_ms', 'direction': 'min'}])
        result = await OptimizationWorker(repo, Planner(), executor, verifier,
            worker_id=worker_id, allocations=allocations, provider=provider).once()
        assert result['result']['decision'] == 'keep', result
        manager = ReleaseManager(repo)
        release = manager.create(job_id)
        manager.route('m', release)
        client = TestClient(create_gateway(repo, 'test-token'))
        response = client.post('/v1/completions', headers={'Authorization': 'Bearer test-token'},
                               json={'model': 'm', 'prompt': '2+2'})
        assert response.status_code == 200, response.json()
    finally:
        # Amended after review: kept runtimes are serving-owned; retire them explicitly.
        from inferstack.releases.runtime import ReleaseRuntime
        if executor is not None:
            for record in repo.list_records('release'):
                await ReleaseRuntime(repo, LocalProcessProvider(), log_root=tmp_path / 'rt').retire(record['id'])
            for namespace in list(executor.active):
                await executor.cleanup({'id': namespace})
        baseline.kill()
        baseline.wait()


@pytest.mark.asyncio
async def test_isolated_build_yields_to_worker_heartbeats(repo, tmp_path):
    from inferstack.infrastructure.local import LocalProcessProvider
    repo.put_record('technique', 'fixture', {'name': 'fixture', 'summary': 'test fixture',
        'engines': [{'engine': 'vllm'}]})
    repo.put_record('deployment', 'base', {'id': 'base', 'engine': 'vllm', 'model': 'm',
        'base_url': 'http://127.0.0.1:9000', 'command': [sys.executable, '-m', 'vllm', '--port', '9000']})
    executor = IsolatedExecutor(repo, tmp_path / 'logs', work_root=tmp_path / 'work',
        source_root=ROOT, source_revision='a' * 40,
        allocations=AllocationManager(repo, 'worker'), provider=LocalProcessProvider())

    def build_plugin(ws, action, artifacts, log):
        # Exercise the production blocking build primitive with a harmless slow command.
        code, _ = executor.sandbox.run([sys.executable, '-c', 'import time; time.sleep(0.4)'],
                                       cwd=ws, home=ws)
        assert code == 0
        return None

    async def launch(record, plan, namespace, context, **kwargs):
        return {'id': namespace}

    executor.build_plugin = build_plugin
    executor.launch = launch
    ticks = []

    async def heartbeat():
        for _ in range(12):
            ticks.append(time.monotonic())
            await asyncio.sleep(0.01)

    task = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.02)
    try:
        await executor.execute({'baseline_id': 'base', 'baseline_revision': 1,
            'actions': [{'id': 'plugin', 'type': 'plugin_install', 'params': {'technique_id': 'fixture'}}],
            'action_order': ['plugin']}, 'blocking-build', context={})
        await task
        worst_gap = max(b - a for a, b in zip(ticks, ticks[1:]))
        assert worst_gap < 0.2, f'heartbeat blocked for {worst_gap:.3f}s by a 0.4s build'
    finally:
        executor._discard('blocking-build')


@pytest.mark.asyncio
async def test_restart_at_exhausted_budget_preserves_prior_experiment_results(repo):
    job_id = repo.submit('resume', {'deployment_id': 'base'})
    claim = repo.claim(100)
    plan = {'hypothesis': 'previous failed treatment', 'treatment_signature': 'already-tried'}
    repo.step_begin(job_id, claim['lease_token'], 'iter0.plan', idempotent=True)
    repo.step_end(job_id, claim['lease_token'], 'iter0.plan', 'succeeded', result=plan)
    repo.step_begin(job_id, claim['lease_token'], 'iter0.experiment', idempotent=False)
    repo.step_end(job_id, claim['lease_token'], 'iter0.experiment', 'succeeded',
                  result={'plan': plan, 'candidate': {'id': 'old'}, 'verdict': {'decision': 'reject'}})
    # Use the exhausted-experiments case: succeeded durable steps must be read
    # before budget checks short-circuit their reuse after restart.
    with repo.transaction() as db:
        db.execute('UPDATE jobs SET budget=? WHERE id=?', (repo.encode({'experiments': 1}), job_id))
    worker = OptimizationWorker(repo, None, None, None)
    result = await worker.optimize(claim)
    assert len(result['experiments']) == 1, result


@pytest.mark.asyncio
async def test_native_local_gpu_launch_does_not_default_gpu_usage_to_zero(repo, tmp_path):
    from inferstack.engines.managed import DeploymentSpec

    class Process:
        pid = 123

    class Provider:
        name, resource_type = 'local', 'process'

        async def spawn(self, *args, **kwargs):
            return Process()

        def external_for(self, pid):
            return {'host': 'test', 'pid': pid}

    job_id = repo.submit('gpu', {}, budget={'gpu_seconds': 1})
    claim = repo.claim(100)
    allocations = AllocationManager(repo, 'worker')
    executor = EndpointExecutor(repo, tmp_path / 'logs', allocations=allocations, provider=Provider())
    spawn = executor._spawner({'job': claim, 'step_key': 'iter0.experiment', 'namespace': 'probe'},
        DeploymentSpec('candidate', 'inferstack_native', 'm', 'http://candidate', command=('inferstack', 'serve')))
    await spawn(['inferstack', 'serve'], None)
    [allocation] = allocations.list(job_id=job_id)
    assert allocation['gpu_count'] > 0, allocation


def test_qualification_does_not_accept_nonzero_kernel_process_exit(repo, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('qualification_under_review', ROOT / 'scripts/v2_qualify.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from inferstack.infrastructure import inventory
    monkeypatch.setattr(inventory, 'collect', lambda **kw: ({'gpus': [{'name': 'test-GPU'}], 'libraries': {}}, []))

    def run(argv, log, timeout, env=None):
        if 'scripts/v160_kernel_equivalence.py' in argv:
            return 1, json.dumps({'pass': True, 'failed': [], 'method': 'fixture', 'checks': []}), 0
        return 0, '1 passed, 99 skipped', 0

    monkeypatch.setattr(module, 'run', run)
    # The matrix path must be inside ROOT because the runner calls relative_to(ROOT).
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    manifest_path = tmp_path / 'examples/model_foundry/factorylm-2b/candidate/manifest.json'
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({'source': {'revision': 'a' * 40}, 'tokenizer': {'revision': 'a' * 40}}))
    monkeypatch.setattr(sys, 'argv', ['v2_qualify.py', '--out', 'out'])
    with pytest.raises(SystemExit):
        module.main()
    result = json.loads((tmp_path / 'out/qualification.json').read_text())
    assert result['gates']['kernel_equivalence']['status'] == 'failed', result
