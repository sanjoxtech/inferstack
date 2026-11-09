"""Package I: progressive rollout with automatic rollback, and FactoryOS feedback."""
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from inferstack.control.api import create_control_app
from inferstack.releases.manager import ReleaseManager
from inferstack.releases.rollout import RolloutController, RolloutError, http_observer
from inferstack.serving.gateway import create_gateway
from inferstack.serving.health import AdmissionController
from inferstack.storage.repository import Repository
from test_engines import GatewayServer, servers  # noqa: F401  (fixture reuse)

STAGES = [{'weight': 10, 'min_duration_s': 60, 'max_error_rate': 0.02, 'max_latency_regression': 0.2},
          {'weight': 50, 'min_duration_s': 60, 'max_error_rate': 0.02, 'max_latency_regression': 0.2},
          {'weight': 100, 'min_duration_s': 60, 'max_error_rate': 0.02, 'max_latency_regression': 0.2}]


def setup_release(repo, model='m', base_urls=('http://stable', 'http://cand')):
    for name, url in zip(('stable', 'cand'), base_urls):
        repo.put_record('deployment', name, {'id': name, 'engine': 'generic_openai', 'model': model, 'base_url': url})
        repo.put('release', 'rel-' + name, {'deployment': repo.read('deployment', name),
                                            'evidence': {'sha256': 'f' * 64, 'bytes': 0}, 'job_id': 'j',
                                            'verification_scope': 'test'})
    ReleaseManager(repo).route(model, 'rel-stable')


class Observations:
    def __init__(self):
        self.data = {'rel-stable': {'requests': 500, 'errors': 0, 'ttfb_p50_ms': 100.0},
                     'rel-cand': {'requests': 100, 'errors': 0, 'ttfb_p50_ms': 105.0}}

    async def __call__(self, release_id, since):
        return dict(self.data[release_id])


@pytest.fixture
def env(tmp_path):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    now = [1_000_000.0]
    setup_release(repo)
    obs = Observations()
    controller = RolloutController(repo, observe=obs, clock=lambda: now[0])
    return repo, controller, obs, now


@pytest.mark.asyncio
async def test_stages_advance_only_on_evidence_then_promote(env):
    repo, c, obs, now = env
    await c.start('r1', 'm', 'rel-cand', STAGES)
    assert repo.read('route', 'm')['body'] == {'stable': 'rel-stable', 'candidate': 'rel-cand', 'weight': 10,
                                               'previous': None}
    assert (await c.step('r1')).startswith('holding: minimum stage duration')
    now[0] += 61
    obs.data['rel-cand']['requests'] = 5
    assert 'only 5 candidate requests' in await c.step('r1')
    obs.data['rel-cand']['requests'] = 100
    assert await c.step('r1') == 'advanced' and repo.read('route', 'm')['body']['weight'] == 50
    now[0] += 61
    assert await c.step('r1') == 'advanced' and repo.read('route', 'm')['body']['weight'] == 100
    now[0] += 61
    assert await c.step('r1') == 'promoted'
    assert repo.read('route', 'm')['body'] == {'stable': 'rel-cand', 'candidate': None, 'weight': 0,
                                               'previous': 'rel-stable'}
    history = repo.history('rollout', 'r1')
    # Each route change is two-phase: an intent revision precedes the committed one.
    intents = [h for h in history if h['body'].get('transition')]
    assert len(intents) == 4 and all(h['body']['transition']['route'] for h in intents)
    states = [(h['body']['state'], h['body']['current_stage']) for h in history if not h['body'].get('transition')]
    assert states == [('pending', 0), ('progressing', 0), ('progressing', 1), ('progressing', 2), ('promoted', 2)]
    assert await c.step('r1') == 'promoted'  # terminal is stable


@pytest.mark.asyncio
@pytest.mark.parametrize('change,reason', [({'errors': 5}, 'error rate'), ({'ttfb_p50_ms': 150.0}, 'TTFB regression')])
async def test_breach_rolls_back_immediately(env, change, reason):
    repo, c, obs, now = env
    await c.start('r1', 'm', 'rel-cand', STAGES)
    obs.data['rel-cand'].update(change)
    assert await c.step('r1') == 'rolled_back'  # even before the minimum duration
    assert repo.read('route', 'm')['body']['candidate'] is None
    assert repo.read('route', 'm')['body']['stable'] == 'rel-stable'
    assert reason in repo.read('rollout', 'r1')['body']['reason']


@pytest.mark.asyncio
async def test_missing_latency_holds_and_operator_controls(env):
    repo, c, obs, now = env
    await c.start('r1', 'm', 'rel-cand', STAGES)
    with pytest.raises(RolloutError, match='already has a candidate'):
        await c.start('r2', 'm', 'rel-cand', STAGES)
    now[0] += 61
    obs.data['rel-cand']['ttfb_p50_ms'] = None
    assert 'latency evidence unavailable' in await c.step('r1')
    c.pause('r1')
    assert await c.step('r1') == 'paused'
    c.resume('r1')
    assert (await c.step('r1')).startswith('holding: minimum stage duration')  # window restarted
    assert await c.abort('r1') == 'rolled_back' and repo.read('route', 'm')['body']['candidate'] is None
    with pytest.raises(RolloutError):
        await c.abort('r1')


@pytest.mark.asyncio
async def test_unattested_candidate_never_receives_traffic(env):
    repo, _, obs, now = env

    async def attest(release):
        return {'status': 'mismatch', 'checks': {'engine_version': False}}
    c = RolloutController(repo, observe=obs, attest=attest, clock=lambda: now[0])
    with pytest.raises(RolloutError, match='not attested'):
        await c.start('r1', 'm', 'rel-cand', STAGES)
    assert repo.read('route', 'm')['body']['candidate'] is None
    assert repo.read('rollout', 'r1')['body']['state'] == 'failed'


def test_real_gateway_failures_trigger_rollback(tmp_path, servers):  # noqa: F811
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    stable_url, _ = servers()
    cand_url, _ = servers(fail_status=503)
    setup_release(repo, base_urls=(stable_url, cand_url))
    admission = AdmissionController(failure_threshold=1000)
    with GatewayServer(create_gateway(repo, 'gw', admission=admission)) as gw:
        async def attested(release):  # attestation itself is covered in test_attestation.py
            return {'status': 'attested', 'checks': {}}
        control = TestClient(create_control_app(repo, 'ctl', rollout_observer=http_observer(repo, gw, 'gw'),
                                                rollout_attest=attested))
        h = {'Authorization': 'Bearer ctl'}
        started = control.post('/control/rollouts', headers=h, json={
            'rollout_id': 'r1', 'model': 'm', 'release_id': 'rel-cand', 'min_requests': 5,
            'stages': [{'weight': 50, 'min_duration_s': 0, 'max_error_rate': 0.05}, {'weight': 100, 'min_duration_s': 0}]})
        assert started.status_code == 200, started.text
        codes = [httpx.post(gw + '/v1/completions', headers={'Authorization': 'Bearer gw', 'X-Request-Id': f'r{i}'},
                            json={'model': 'm', 'prompt': 'x'}).status_code for i in range(40)]
        assert 502 in codes and 200 in codes
        step = control.post('/control/rollouts/r1/step', headers=h).json()
        assert step['outcome'] == 'rolled_back' and 'error rate' in step['rollout']['body']['reason']
        assert {httpx.post(gw + '/v1/completions', headers={'Authorization': 'Bearer gw'},
                           json={'model': 'm', 'prompt': 'x'}).status_code for _ in range(10)} == {200}
        assert control.post('/control/rollouts/r1/explode', headers=h).status_code == 404


def test_rollout_requires_configured_observer(tmp_path):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    client = TestClient(create_control_app(repo, 'ctl'))
    assert client.post('/control/rollouts/x/step', headers={'Authorization': 'Bearer ctl'}).status_code == 503


# --- FactoryOS feedback --------------------------------------------------------------------

def issue(**over):
    body = {'issue_id': 'fos-1', 'idempotency_key': 'fos-1', 'source': 'factoryos', 'trace_ids': ['abc'],
            'deployment_ref': 'stable@1', 'category': 'latency', 'observations': {'p95_client_ttft_ms': 1840},
            'reported_at': time.time()}
    body.update(over)
    return body


def test_factoryos_feedback_contract(tmp_path):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    setup_release(repo)
    client = TestClient(create_control_app(repo, 'ctl'))
    h = {'Authorization': 'Bearer ctl'}
    body = issue()
    first = client.post('/control/feedback', headers=h, json=body).json()
    assert first['status'] == 'queued' and first['contract_version'] == 1 and first['job_id']
    assert client.post('/control/feedback', headers=h, json=body).json()['job_id'] == first['job_id']
    assert client.post('/control/feedback', headers=h,
                       json={**body, 'observations': {'p95_client_ttft_ms': 1}}).status_code == 409
    job = repo.get(first['job_id'])
    assert job['payload']['deployment_id'] == 'stable' and job['payload']['feedback_ref'] == 'fos-1@1'
    assert 'latency issue fos-1' in job['payload']['goal']
    status = client.get('/control/feedback/fos-1', headers=h).json()
    assert status['status'] == 'pending' and 'explicit progressive rollout' in status['note']
    quality = client.post('/control/feedback', headers=h, json=issue(issue_id='fos-2', idempotency_key='fos-2',
                                                                     category='quality')).json()
    assert {k: quality[k] for k in ('issue_id', 'contract_version', 'job_id', 'status')} == \
        {'issue_id': 'fos-2', 'contract_version': 1, 'job_id': None, 'status': 'recorded_not_actionable'}
    assert quality['problem']['state'] == 'needs_measurement' and quality['problem']['measurement_requests']
    assert client.post('/control/feedback', headers=h, json=issue(issue_id='fos-3', category='mystery')).status_code == 422
    assert client.post('/control/feedback', headers=h, json={**issue(issue_id='fos-4'), 'approve': True}).status_code == 422
    assert client.post('/control/feedback', headers=h,
                       json=issue(issue_id='fos-5', idempotency_key='fos-5', deployment_ref='ghost@1')).status_code == 409
    assert client.get('/control/feedback/nope', headers=h).status_code == 404
