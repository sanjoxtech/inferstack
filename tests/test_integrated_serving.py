"""P5: integrated serving lifecycle on real local processes (CPU).

shadow -> bounded canary -> induced regression -> rollback -> controller restart ->
second canary -> promotion -> retirement, then serving cost attribution over the
same interval. GPU behaviour, a real data plane and provider billing are out of scope
here (see docs/CLOUD_SERVING_RESEARCH_PLAN.md P5).
"""
import socket
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inferstack.infrastructure import LocalProcessProvider
from inferstack.infrastructure.pricing import PriceTable
from inferstack.infrastructure.serving_cost import compare_capacity, reconcile_bill, serving_cost
from inferstack.releases.manager import ReleaseManager
from inferstack.releases.rollout import RolloutController, evidence_observer
from inferstack.releases.runtime import ReleaseRuntime
from inferstack.serving.evidence import EvidenceStore
from inferstack.serving.gateway import create_gateway
from inferstack.storage.repository import Repository

ROOT = Path(__file__).resolve().parents[1]
H = {'Authorization': 'Bearer t'}


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def release(repo, name, delay):
    p = port()
    repo.put_record('deployment', name, {
        'id': name, 'engine': 'generic_openai', 'model': 'm', 'base_url': f'http://127.0.0.1:{p}',
        'resources': {'gpu_count': 1},
        'command': [sys.executable, str(ROOT / 'tests/fixtures/fake_openai_server.py'), '--port', str(p),
                    '--delay', str(delay)]})
    repo.put('release', 'rel-' + name, {'deployment': repo.read('deployment', name),
             'evidence': {'sha256': 'f' * 64, 'bytes': 0}, 'job_id': 'fixture', 'verification_scope': 'fixture'})
    return 'rel-' + name


def traffic(client, n):
    for i in range(n):
        r = client.post('/v1/completions', headers={**H, 'X-Request-Id': f'req-{time.time_ns()}-{i}'},
                        json={'model': 'm', 'prompt': '2+2'})
        assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_integrated_lifecycle_and_cost(tmp_path):
    repo = Repository(tmp_path / 'db', tmp_path / 'artifacts')
    prices = PriceTable({'local/test-gpu': {'usd_per_gpu_hour': 3.6, 'source': 'test fixture', 'as_of': '2026-11-09'}})
    runtime = ReleaseRuntime(repo, LocalProcessProvider(), log_root=tmp_path / 'logs', ready_timeout=15,
                             gpu_devices=['GPU-a', 'GPU-b', 'GPU-c'], pricing=prices, gpu_name='test-gpu')
    # Base and good share a stable 50 ms service time so relative latency gates compare real
    # differences, not scheduler noise on near-zero values; slow adds a clear 250 ms regression.
    base, slow, good = release(repo, 'base', 0.05), release(repo, 'slow', 0.25), release(repo, 'good', 0.05)
    started = time.time()
    try:
        await runtime.ensure(base)
        ReleaseManager(repo).route('m', base)
        portal = TestClient(create_gateway(repo, 't'))
        client = portal.__enter__()  # one long-lived event loop, so shadow tasks are not cut off

        # 1. Shadow: sampled, budgeted copies to the candidate; never client-visible.
        await runtime.ensure(slow)
        now = time.time()
        repo.put_record('shadow', 'm', {'release': slow, 'sample_rate': 1.0, 'budget_requests': 4,
                                        'started_at': now, 'expires_at': now + 600})
        traffic(client, 8)
        deadline = time.time() + 10
        store = EvidenceStore(repo)
        while len(store.rows(release_id=slow, since=now, shadow=True)) < 4 and time.time() < deadline:
            time.sleep(0.05)
        assert len(store.rows(release_id=slow, since=now, shadow=True)) == 4  # budget respected
        assert store.window(release_id=slow, since=now)['requests'] == 0       # no client traffic yet
        repo.put_record('shadow', 'm', {**repo.read('shadow', 'm')['body'], 'enabled': False}, expected_revision=1)

        # 2. Canary with an induced latency regression -> rollback, across a controller restart.
        stages = [{'weight': 50, 'min_duration_s': 0, 'max_latency_regression': 0.5, 'max_error_rate': 0.05},
                  {'weight': 100, 'min_duration_s': 0}]
        await RolloutController(repo, observe=evidence_observer(repo), runtime=runtime).start(
            'r1', 'm', slow, stages, min_requests=8)
        traffic(client, 40)
        restarted = RolloutController(repo, observe=evidence_observer(repo), runtime=runtime)  # fresh process
        assert await restarted.step('r1') == 'rolled_back'
        body = repo.read('rollout', 'r1')['body']
        assert 'TTFB regression' in body['reason'] and body['retirement']['state'] == 'done'
        assert repo.read('route', 'm')['body'] == {'stable': base, 'candidate': None, 'weight': 0, 'previous': None}
        assert runtime.serving_allocations(slow) == []  # the regressed runtime was stopped

        # 3. A healthy candidate advances through every stage and is promoted; the old stable retires.
        await RolloutController(repo, observe=evidence_observer(repo), runtime=runtime).start(
            'r2', 'm', good, [{**stages[0], 'max_latency_regression': 1.0}, stages[1]], min_requests=8)
        controller = RolloutController(repo, observe=evidence_observer(repo), runtime=runtime)
        traffic(client, 30)
        assert await controller.step('r2') == 'advanced'
        traffic(client, 12)
        assert await controller.step('r2') == 'promoted'
        assert repo.read('route', 'm')['body']['stable'] == good
        assert runtime.serving_allocations(base) == [] and len(runtime.serving_allocations(good)) == 1
        evidence = repo.read('rollout', 'r2')['body']['evidence_refs']
        assert [e['decision'] for e in evidence] == ['advance', 'advance']
    finally:
        if 'portal' in locals():
            portal.__exit__(None, None, None)
        for rel in ('rel-base', 'rel-slow', 'rel-good'):
            await runtime.retire(rel)

    # 4. Cost attribution over the whole interval: idle, shadow, canary and rollback capacity included.
    ended = time.time()
    report = serving_cost(repo, since=started, until=ended, slo={'e2e_ms': 200},
                          roles={'base': 'stable', 'slow': 'candidate', 'good': 'candidate'})
    assert report['gross_usd'] is not None and report['gross_usd'] > 0
    gpu_seconds = sum(a['gpu_seconds'] for a in report['allocations'])
    assert report['gross_usd'] == pytest.approx(gpu_seconds / 3600 * 3.6)
    assert report['price_provenance'] == {'local:test fixture': '2026-11-09'}
    assert report['requests']['shadow'] == 4 and report['requests']['client'] == 90
    assert report['tokens']['output'] == 90                       # fake engine reports 1 token per request
    assert report['tokens']['goodput_output'] < report['tokens']['output']  # the slow canary missed the SLO
    assert report['usd_per_million_goodput_tokens'] > report['usd_per_million_output_tokens']
    assert reconcile_bill(report, report['gross_usd'] * 1.01)['reconciled'] is True
    assert reconcile_bill(report, report['gross_usd'] * 2)['reconciled'] is False
    assert compare_capacity(report, report)['gross_usd']['delta'] == 0


def test_cost_is_unavailable_rather_than_guessed(tmp_path):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    with repo.transaction() as db:
        db.execute("INSERT INTO allocations(id,idempotency_key,job_id,step_key,worker_id,provider,resource_type,"
                   "spec,state,gpu_count,created_at,updated_at,active_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   ('a1', 'k1', None, None, 'serving:d', 'local', 'process', '{"price": null}', 'active', 1,
                    1.0, 1.0, 100.0))
    EvidenceStore(repo).record(__import__('inferstack.serving.evidence', fromlist=['RequestRecord']).RequestRecord(
        'rel', 'd', 'ok', False, at=150.0, e2e_ms=10.0, output_tokens=None))
    report = serving_cost(repo, since=100.0, until=200.0)
    assert report['gross_usd'] is None and report['usd_per_million_output_tokens'] is None
    assert len(report['unavailable_reasons']) == 2
    assert reconcile_bill(report, 1.0)['reconciled'] is False
    with pytest.raises(ValueError, match='source'):
        serving_cost(repo, since=100.0, until=200.0, extra_costs=[{'item': 'storage', 'usd': 1.0}])
