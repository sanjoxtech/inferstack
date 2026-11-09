"""Package B acceptance with real processes: crash at every transition, restart,
stale workers, cancellation, deadlines, budgets and orphan recovery."""
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inferstack.control.api import create_control_app
from inferstack.control.worker import OptimizationWorker
from inferstack.infrastructure import AllocationManager, LocalProcessProvider, reconcile
from inferstack.optimization.experiment import EndpointExecutor, EndpointVerifier
from inferstack.storage.repository import Repository

FIXTURES = Path(__file__).parent / 'fixtures'
SRC = Path(__file__).parents[1] / 'src'
LEASE = 2.0  # was 1.0; a 1.6 s CI event-loop stall lost the lease before the planned crash point


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def port_open(port):
    with socket.socket() as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def server_argv(port, answer='four', delay=0.0):
    return [sys.executable, str(FIXTURES / 'fake_openai_server.py'), '--port', str(port), '--answer', answer,
            '--delay', str(delay)]


@pytest.fixture
def env(tmp_path):
    base_port, cand_port = free_port(), free_port()
    baseline = subprocess.Popen(server_argv(base_port))
    deadline = time.monotonic() + 10
    while not port_open(base_port):
        assert time.monotonic() < deadline, 'baseline server did not start'
        time.sleep(0.05)
    repo = Repository(tmp_path / 'control.db', tmp_path / 'artifacts')
    repo.put_record('deployment', 'baseline', {'id': 'baseline', 'engine': 'generic_openai', 'model': 'm',
                                               'base_url': f'http://127.0.0.1:{base_port}'})

    def candidate(answer='four', delay=0.0):
        repo.put_record('deployment', 'candidate', {'id': 'candidate', 'engine': 'generic_openai', 'model': 'm',
                                                    'base_url': f'http://127.0.0.1:{cand_port}',
                                                    'command': server_argv(cand_port, answer, delay)})
    try:
        yield {'repo': repo, 'tmp': tmp_path, 'cand_port': cand_port, 'candidate': candidate}
    finally:
        baseline.kill()
        baseline.wait()
        provider = LocalProcessProvider()
        for alloc in AllocationManager(repo, 't').list():  # never leak a process from a failed test
            for pid in provider.find_tagged(alloc['id']):
                provider.stop(provider.external_for(pid), grace_seconds=1)


def run_worker(env, *extra, wait=True):
    argv = [sys.executable, str(FIXTURES / 'crash_worker.py'), '--db', str(env['tmp'] / 'control.db'),
            '--state', str(env['tmp']), '--lease', str(LEASE), *extra]
    environ = {**os.environ, 'PYTHONPATH': f"{SRC}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    if not wait:
        return subprocess.Popen(argv, env=environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(argv, env=environ, capture_output=True, text=True, timeout=90)


def tagged_pids(repo):
    provider = LocalProcessProvider()
    return [pid for a in AllocationManager(repo, 't').list() for pid in provider.find_tagged(a['id'])]


SPAWNED_POINTS = ('process_started', 'process_spawned', 'allocation_active')
CRASH_POINTS = ('claimed', 'step_started:iter0.plan', 'step_done:iter0.plan', 'step_started:iter0.experiment',
                'allocation_intent', *SPAWNED_POINTS, 'step_done:iter0.experiment', 'before_finish')
RECONCILE_NOTE = {'allocation_intent': 'never_launched', 'process_started': 'unrecorded_launch_stopped',
                  'process_spawned': 'orphan_stopped', 'allocation_active': 'orphan_stopped'}


@pytest.mark.parametrize('point', CRASH_POINTS)
def test_crash_at_every_transition_then_restart_recovers(env, point):
    env['candidate']()
    repo = env['repo']
    # On a loaded CI runner the worker can lose its (short) lease before reaching the
    # injected crash point; that is the fencing contract working, not this scenario. Then
    # the job is finished by a normal worker and the crash is retried on a fresh job (at
    # most three times); every such retry must be explained by a reported lease loss.
    for attempt in range(3):
        job = repo.submit(f'crash-{point}-{attempt}', {'deployment_id': 'baseline', 'goal': 'reduce latency'})
        crashed = run_worker(env, '--crash-at', point)
        if crashed.returncode == 137:
            break
        assert crashed.returncode == 0 and 'lease lost' in crashed.stderr, crashed.stderr
        for _ in range(3):
            time.sleep(LEASE * 1.3)
            if repo.get(job)['state'] != 'running':
                break
            assert run_worker(env).returncode == 0
    assert crashed.returncode == 137, crashed.stderr
    if point in SPAWNED_POINTS:
        assert tagged_pids(repo), 'expected an orphaned candidate process after the crash'

    time.sleep(LEASE * 1.3)  # job and worker leases of the crashed worker expire
    # Durability contract: a worker that loses its lease stops (fenced); the next worker
    # resumes. On a loaded runner a 1 s test lease can be lost, so up to three restarts
    # are allowed; each must exit cleanly and the job must finish exactly once.
    runs = []
    for _ in range(3):
        resumed = run_worker(env)
        runs.append(resumed)
        assert resumed.returncode == 0, resumed.stderr
        if repo.get(job)['state'] != 'running':
            break
        time.sleep(LEASE * 1.3)
    final = repo.get(job)
    assert final['state'] == 'completed', (final['state'], final['result'],
                                           [(r.stdout[-300:], r.stderr[-500:]) for r in runs])
    if len(runs) > 1:  # every extra restart must be explained by a reported lease loss
        assert all('lease lost' in r.stderr for r in runs[:-1]), [r.stderr[-500:] for r in runs]
    assert final['result']['decision'] in {'keep', 'reject', 'retest', 'inconclusive'}
    steps = {s['step_key']: s for s in final['steps']}
    allocations = AllocationManager(repo, 't').list(job_id=job)
    assert allocations and all(a['state'] == 'released' for a in allocations)
    assert len({a['idempotency_key'] for a in allocations}) == len(allocations)
    if len(runs) == 1:  # exact attempt accounting when only the injected crash interrupted the job
        assert steps['iter0.plan']['attempts'] == (2 if point == 'step_started:iter0.plan' else 1)
        crashed_in_experiment = point in ('step_started:iter0.experiment', 'allocation_intent', *SPAWNED_POINTS)
        assert steps['iter0.experiment']['attempts'] == (2 if crashed_in_experiment else 1)
        assert len(allocations) == (2 if point in RECONCILE_NOTE else 1)
        if point in RECONCILE_NOTE:
            assert allocations[0]['note'] == RECONCILE_NOTE[point]
    assert not tagged_pids(repo) and not port_open(env['cand_port'])
    assert sum(e['kind'] == 'completed' for e in final['events']) == 1


def test_wedged_worker_is_superseded_and_cannot_publish(env):
    env['candidate']()
    repo = env['repo']
    job = repo.submit('stale', {'deployment_id': 'baseline', 'goal': 'x'})
    wedged = run_worker(env, '--hang-at', 'step_started:iter0.experiment', '--hang-seconds', '6', wait=False)
    deadline = time.monotonic() + 20
    while not (env['tmp'] / 'hanging').exists():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    time.sleep(LEASE * 1.3)
    runs = []
    for _ in range(3):  # see the crash test: a 1 s lease may be lost on a loaded runner
        takeover = run_worker(env)
        runs.append(takeover)
        assert takeover.returncode == 0, takeover.stderr
        if repo.get(job)['state'] != 'running':
            break
        time.sleep(LEASE * 1.3)
    out, err = wedged.communicate(timeout=60)
    assert wedged.returncode == 0, err
    assert (env['tmp'] / 'woke').exists()
    final = repo.get(job)
    diagnostics = [(r.stdout[-300:], r.stderr[-500:]) for r in runs] + [('wedged', err[-500:])]
    assert final['state'] == 'completed', (final['state'], diagnostics)
    if len(runs) > 1:
        assert all('lease lost' in r.stderr for r in runs[:-1]), diagnostics
    assert final['attempt'] == 1 + len(runs), (final['attempt'], diagnostics)
    assert sum(e['kind'] == 'completed' for e in final['events']) == 1
    claims = [e['body']['worker_id'] for e in final['events'] if e['kind'] == 'claimed']
    # The woken worker created no resources after losing its lease.
    owners = {a['worker_id'] for a in AllocationManager(repo, 't').list(job_id=job)}
    assert claims[0] not in owners and owners <= set(claims[1:]), (claims, owners)
    assert not tagged_pids(repo)


def in_process_worker(env, **kw):
    repo = env['repo']
    worker_id = repo.register_worker(0.6)
    provider = LocalProcessProvider()
    allocations = AllocationManager(repo, worker_id)
    executor = EndpointExecutor(repo, env['tmp'] / 'logs', allocations=allocations, provider=provider)
    verifier = EndpointVerifier(repo, executor, 'baseline', [{'prompt': '2+2', 'expected': 'four', 'max_tokens': 4}],
                                [{'metric': 'e2e_ms', 'direction': 'min'}], repeats=3)
    sys.path.insert(0, str(FIXTURES))
    from crash_worker import FixedPlanner
    return OptimizationWorker(repo, FixedPlanner(repo), executor, verifier, lease_seconds=0.6, worker_id=worker_id,
                              allocations=allocations, provider=provider, **kw)


async def wait_active(repo, job):
    for _ in range(400):
        if any(a['state'] == 'active' for a in AllocationManager(repo, 't').list(job_id=job)):
            return
        await asyncio.sleep(0.025)
    raise AssertionError('allocation never became active')


@pytest.mark.asyncio
async def test_cancel_stops_process_releases_and_keeps_partial_evidence(env):
    env['candidate'](delay=0.5)
    repo = env['repo']
    job = repo.submit('cancel', {'deployment_id': 'baseline', 'goal': 'x'})
    task = asyncio.create_task(in_process_worker(env).once())
    await wait_active(repo, job)
    assert repo.cancel(job)
    final = await asyncio.wait_for(task, 30)
    assert final['state'] == 'cancelled' and final['result'] is None
    assert [e['kind'] for e in final['events']].count('partial_evidence') == 1
    assert all(a['state'] == 'released' for a in AllocationManager(repo, 't').list(job_id=job))
    assert not tagged_pids(repo) and not port_open(env['cand_port'])


@pytest.mark.asyncio
async def test_deadline_stops_work_and_cleans_up(env):
    env['candidate'](delay=0.5)
    repo = env['repo']
    job = repo.submit('deadline', {'deployment_id': 'baseline', 'goal': 'x'}, deadline_at=time.time() + 3)
    final = await asyncio.wait_for(in_process_worker(env).once(), 30)
    assert final['state'] == 'failed' and final['last_error']['type'] == 'DeadlineExceeded'
    assert final['last_error']['retryable'] is False and 'partial' in final['result']
    assert all(a['state'] == 'released' for a in AllocationManager(repo, 't').list(job_id=job))
    assert not tagged_pids(repo)


@pytest.mark.asyncio
async def test_budget_exhaustion_stops_new_steps_and_retains_results(env):
    env['candidate'](answer='wrong')  # every experiment is rejected by the exact-match gate
    repo = env['repo']
    job = repo.submit('budget', {'deployment_id': 'baseline', 'goal': 'x'}, budget={'experiments': 1})
    final = await asyncio.wait_for(in_process_worker(env, max_iterations=3).once(), 60)
    assert final['state'] == 'completed'
    assert final['result']['stop_reason'] == 'budget exhausted: experiments'
    assert len(final['result']['experiments']) == 1
    assert final['result']['experiments'][0]['verdict']['decision'] == 'reject'
    assert [s['step_key'] for s in final['steps']] == ['iter0.plan', 'iter0.experiment']
    assert all(a['state'] == 'released' for a in AllocationManager(repo, 't').list(job_id=job))


def test_unstoppable_orphan_is_operator_visible_and_recoverable(env):
    env['candidate']()
    repo = env['repo']
    job = repo.submit('orphan', {'deployment_id': 'baseline', 'goal': 'x'})
    assert run_worker(env, '--crash-at', 'allocation_active').returncode == 137
    time.sleep(LEASE * 1.3)

    class StuckProvider(LocalProcessProvider):
        def stop(self, external, grace_seconds=10.0):
            return False
    assert [a['outcome'] for a in reconcile(repo, StuckProvider())] == ['orphaned']
    client = TestClient(create_control_app(repo, 'tok'))
    h = {'Authorization': 'Bearer tok'}
    orphans = client.get('/control/allocations?state=orphaned', headers=h).json()['allocations']
    assert len(orphans) == 1 and orphans[0]['job_id'] == job
    assert tagged_pids(repo)  # still running: the operator can see it
    recovered = client.post(f"/control/allocations/{orphans[0]['id']}/recover", headers=h, json={}).json()
    assert recovered['state'] == 'released' and recovered['outcome'] == 'orphan_stopped'
    assert not tagged_pids(repo)
    assert client.post(f"/control/allocations/{orphans[0]['id']}/recover", headers=h, json={}).status_code == 409
    workers = client.get('/control/workers', headers=h).json()['workers']
    assert workers and all('id' in w for w in workers)
