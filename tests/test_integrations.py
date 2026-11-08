"""FactoryOS outcome outbox and bounded triggers (Package I scope from the verification report)."""
import httpx
import pytest
from fastapi.testclient import TestClient

from inferstack.control.api import create_control_app
from inferstack.integrations.outbox import Outbox, http_sender
from inferstack.integrations.triggers import TriggerService
from inferstack.storage.repository import Repository

ISSUE = {'issue_id': 'iss-1', 'idempotency_key': 'idem-1', 'source': 'factoryos', 'trace_ids': ['t1'],
         'deployment_ref': 'base', 'category': 'latency', 'observations': {'p95_ms': 900}, 'reported_at': 1.0}
H = {'Authorization': 'Bearer tok'}


@pytest.fixture
def repo(tmp_path):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    repo.put_record('deployment', 'base', {'id': 'base', 'engine': 'generic_openai', 'model': 'm',
                                           'base_url': 'http://base'})
    return repo


def finish_job(repo, job_id, decision='reject'):
    lease = repo.claim(60)
    assert lease['id'] == job_id
    repo.finish(job_id, lease['lease_token'], {'decision': decision, 'experiments': []})


class Receiver:
    """Receiver that deduplicates on Idempotency-Key, like FactoryOS must."""
    def __init__(self, fail_first=0):
        self.effects, self.seen, self.calls, self.fail_first = [], set(), 0, fail_first

    def __call__(self, request):
        self.calls += 1
        if self.calls <= self.fail_first:
            return httpx.Response(503, text='busy')
        key = request.headers['Idempotency-Key']
        assert request.headers['Authorization'] == 'Bearer secret-token'
        if key not in self.seen:
            self.seen.add(key)
            self.effects.append(request.read())
        return httpx.Response(200, text='ok')


@pytest.mark.asyncio
async def test_outcome_is_written_with_the_terminal_state_and_delivered_once(repo, monkeypatch):
    monkeypatch.setenv('CB_TOKEN', 'secret-token')
    client = TestClient(create_control_app(repo, 'tok'))
    first = client.post('/control/feedback', headers=H, json=ISSUE).json()
    again = client.post('/control/feedback', headers=H, json=ISSUE).json()
    assert first['job_id'] == again['job_id']  # duplicate feedback -> one workflow
    outbox = Outbox(repo, base_backoff=0.0)
    assert outbox.list() == []  # nothing before the job is terminal
    finish_job(repo, first['job_id'])
    [row] = outbox.list()
    assert row['payload']['issue_id'] == 'iss-1' and row['payload']['decision'] == 'reject'
    receiver = Receiver(fail_first=2)
    send = http_sender('http://127.0.0.1/outcome', token_env='CB_TOKEN', transport=httpx.MockTransport(receiver))
    assert await outbox.deliver_due(send) == ['retry']
    assert await outbox.deliver_due(send) == ['retry']
    assert await outbox.deliver_due(send) == ['delivered']
    assert await outbox.deliver_due(send) == []  # delivered rows are never resent
    assert len(receiver.effects) == 1


@pytest.mark.asyncio
async def test_lost_ack_redelivery_has_no_duplicate_effect_and_rejections_go_dead(repo, monkeypatch):
    monkeypatch.setenv('CB_TOKEN', 'secret-token')
    client = TestClient(create_control_app(repo, 'tok'))
    job = client.post('/control/feedback', headers=H, json=ISSUE).json()['job_id']
    finish_job(repo, job, 'keep')
    outbox = Outbox(repo, base_backoff=0.0)
    receiver = Receiver()
    transport = httpx.MockTransport(receiver)

    async def lost_ack(row):  # delivered, but the response never arrives
        await http_sender('http://127.0.0.1/o', token_env='CB_TOKEN', transport=transport)(row)
        raise httpx.ReadTimeout('ack lost')
    assert await outbox.deliver_due(lost_ack) == ['retry']
    assert await outbox.deliver_due(http_sender('http://127.0.0.1/o', token_env='CB_TOKEN',
                                                transport=transport)) == ['delivered']
    assert len(receiver.effects) == 1 and receiver.calls == 2
    # a non-retryable rejection is dead-lettered and visible, then re-armable
    other = {**ISSUE, 'issue_id': 'iss-2', 'idempotency_key': 'idem-2'}
    finish_job(repo, client.post('/control/feedback', headers=H, json=other).json()['job_id'])
    reject = http_sender('http://127.0.0.1/o', token_env='CB_TOKEN',
                         transport=httpx.MockTransport(lambda r: httpx.Response(400, text='bad')))
    assert await outbox.deliver_due(reject) == ['dead']
    [dead] = client.get('/control/outbox?state=dead', headers=H).json()['items']
    assert client.post(f"/control/outbox/{dead['id']}/retry", headers=H).status_code == 200


def test_callbacks_require_https():
    with pytest.raises(ValueError, match='https'):
        http_sender('http://factoryos.example/outcome', token_env='X')


@pytest.mark.asyncio
async def test_periodic_trigger_is_bounded(repo):
    now = [10_000.0]
    repo.clock = lambda: now[0]
    service = TriggerService(repo, max_active=1)
    spec = {'deployment_id': 'base', 'goal': 'nightly tune', 'interval_s': 3600, 'cooldown_s': 600,
            'budget': {'experiments': 2}, 'max_jobs_per_day': 2}
    service.create('nightly', 'periodic', spec)
    [fired] = await service.tick()
    assert fired['action'] == 'fired'
    assert (await service.tick())[0]['action'] == 'cooldown'  # repeated ticks create no new jobs
    now[0] += 3600
    assert (await service.tick())[0]['action'] == 'job_in_flight'  # one unfinished job per trigger
    finish_job(repo, fired['job_id'])
    assert (await service.tick())[0]['action'] == 'fired'
    finish_job(repo, _pending(repo))
    now[0] += 3600
    assert (await service.tick())[0]['action'] == 'daily_limit'
    assert repo.get(fired['job_id'])['budget'] == {'experiments': 2}  # triggered jobs are always budgeted


def _pending(repo):
    with repo.connection() as db:
        return db.execute("SELECT id FROM jobs WHERE state='pending'").fetchone()['id']


@pytest.mark.asyncio
async def test_slo_trigger_fires_only_on_observed_breach(repo):
    values = {'v': None}

    async def observe(deployment_id, metric, window_s):
        return values['v']
    service = TriggerService(repo, observe=observe)
    service.create('slo', 'slo', {'deployment_id': 'base', 'goal': 'error spike', 'metric': 'error_rate',
                                  'threshold': 0.05, 'direction': 'above', 'window_s': 300, 'cooldown_s': 900,
                                  'budget': {'experiments': 1}, 'max_jobs_per_day': 4})
    assert (await service.tick())[0]['action'] == 'not_due'  # missing telemetry never fires
    values['v'] = 0.01
    assert (await service.tick())[0]['action'] == 'not_due'
    values['v'] = 0.2
    assert (await service.tick())[0]['action'] == 'fired'


def test_trigger_specs_must_be_bounded(repo):
    service = TriggerService(repo)
    with pytest.raises(ValueError, match='budget'):
        service.create('t', 'periodic', {'deployment_id': 'base', 'goal': 'g', 'interval_s': 60, 'cooldown_s': 60,
                                         'max_jobs_per_day': 2})
    with pytest.raises(ValueError, match='max_jobs_per_day'):
        service.create('t', 'periodic', {'deployment_id': 'base', 'goal': 'g', 'interval_s': 60, 'cooldown_s': 60,
                                         'budget': {'experiments': 1}, 'max_jobs_per_day': 1000})
