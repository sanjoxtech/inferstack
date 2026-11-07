"""Package B store semantics, run on SQLite and PostgreSQL."""
from concurrent.futures import ThreadPoolExecutor

import pytest

from inferstack.control.store import LeaseLost, Store
from inferstack.infrastructure import AllocationManager
from inferstack.storage.migrations import LATEST
from inferstack.storage.repository import Repository


def clocked(url, start=1000.0):
    now = [start]
    return Store(url, clock=lambda: now[0]), now


def test_both_backends_migrate_to_latest(database_url, tmp_path):
    repo = Repository(database_url, tmp_path / 'artifacts')
    assert repo.schema_version == LATEST
    assert repo.put_record('deployment', 'd', {'id': 'd', 'engine': 'vllm', 'model': 'm',
                                                'base_url': 'http://d'}) == 1
    assert Repository(database_url, tmp_path / 'artifacts').history('deployment', 'd')[0]['origin'] == 'typed'


def test_concurrent_submit_is_idempotent(database_url):
    s = Store(database_url)
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda _: s.submit('same-key', {'goal': 'x'}, budget={'experiments': 2}), range(8)))
    assert len(set(ids)) == 1
    with pytest.raises(ValueError, match='different payload'):
        s.submit('same-key', {'goal': 'x'}, budget={'experiments': 3})
    with pytest.raises(ValueError, match='positive'):
        s.submit('other', {}, budget={'gpu_seconds': 0})
    with pytest.raises(ValueError, match='subset'):
        s.submit('other', {}, budget={'coffee': 1})


def test_steps_resume_and_stale_worker_cannot_publish(database_url):
    s, now = clocked(database_url)
    job = s.submit('j', {})
    a = s.claim(10, worker_id='w1')
    assert s.step_begin(job, a['lease_token'], 'iter0.plan', idempotent=True)['action'] == 'run'
    s.step_end(job, a['lease_token'], 'iter0.plan', 'succeeded', result={'plan': 1})
    assert s.step_begin(job, a['lease_token'], 'iter0.experiment', idempotent=False)['action'] == 'run'
    now[0] += 11  # worker a stalls; lease expires
    b = s.claim(10, worker_id='w2')
    assert b['attempt'] == 2
    for stale in (lambda: s.step_end(job, a['lease_token'], 'iter0.experiment', 'succeeded', result={}),
                  lambda: s.step_begin(job, a['lease_token'], 'iter1.plan', idempotent=True),
                  lambda: s.charge(job, a['lease_token'], {'provider_tokens': 5}),
                  lambda: s.finish(job, a['lease_token'], {'decision': 'keep'})):
        with pytest.raises(LeaseLost):
            stale()
    assert s.step_begin(job, b['lease_token'], 'iter0.plan', idempotent=True) == \
        {'action': 'reuse', 'result': {'plan': 1}, 'attempts': 1}
    assert s.step_begin(job, b['lease_token'], 'iter0.experiment', idempotent=False)['action'] == 'ambiguous'
    s.step_resolve_ambiguous(job, b['lease_token'], 'iter0.experiment', {'allocations': 0})
    assert s.step_begin(job, b['lease_token'], 'iter0.experiment', idempotent=False) == {'action': 'run', 'attempts': 2}
    s.finish(job, b['lease_token'], {'decision': 'reject'})
    final = s.get(job)
    assert final['result'] == {'decision': 'reject'} and final['worker_id'] == 'w2'
    assert {st['step_key']: st['attempts'] for st in final['steps']} == {'iter0.plan': 1, 'iter0.experiment': 2}


def test_budgets_derive_and_exhaust(database_url):
    s, now = clocked(database_url)
    job = s.submit('j', {}, budget={'experiments': 1, 'provider_tokens': 100, 'wall_seconds': 50, 'gpu_seconds': 4})
    c = s.claim(10)
    assert s.exhausted(job) == []
    assert s.charge(job, c['lease_token'], {'provider_tokens': 60}) == {'provider_tokens': 60}
    with pytest.raises(ValueError):
        s.charge(job, c['lease_token'], {'experiments': 1})  # derived, not chargeable
    s.charge(job, c['lease_token'], {'provider_tokens': 40})
    assert s.exhausted(job) == ['provider_tokens']
    s.step_begin(job, c['lease_token'], 'iter0.experiment', idempotent=False)
    s.step_end(job, c['lease_token'], 'iter0.experiment', 'succeeded', result={})
    alloc = AllocationManager(s, 'w').intent({**c, 'id': job}, 'iter0.experiment', purpose='p', provider='local',
                                             resource_type='process', spec={}, gpu_count=2)
    AllocationManager(s, 'w').activate(alloc['id'], {**c, 'id': job}, {'host': 'h', 'pid': 1})
    now[0] += 3  # 3 s x 2 GPUs while active
    assert s.exhausted(job) == ['experiments', 'gpu_seconds', 'provider_tokens']
    now[0] += 60
    assert 'wall_seconds' in s.exhausted(job)


def test_retryable_failure_reschedules_with_backoff(database_url):
    s, now = clocked(database_url)
    job = s.submit('j', {}, max_attempts=2)
    c = s.claim(10)
    assert s.fail(job, c['lease_token'], {'type': 'ProviderUnavailable', 'retryable': True}) == 'pending'
    assert s.claim(10) is None  # backoff not elapsed
    now[0] += 6
    c2 = s.claim(10)
    assert c2['attempt'] == 2
    assert s.fail(job, c2['lease_token'], {'type': 'ProviderUnavailable', 'retryable': True},
                  partial={'experiments': [1]}) == 'failed'
    final = s.get(job)
    assert final['state'] == 'failed' and final['result']['partial'] == {'experiments': [1]}
    job2 = s.submit('k', {}, max_attempts=3)
    c3 = s.claim(10)
    assert s.fail(job2, c3['lease_token'], {'type': 'ValueError', 'retryable': False}) == 'failed'


def test_deadline_passed_before_claim_fails_job(database_url):
    s, now = clocked(database_url)
    job = s.submit('j', {}, deadline_at=1005.0)
    now[0] = 1006
    assert s.claim(10) is None
    final = s.get(job)
    assert final['state'] == 'failed' and final['last_error']['type'] == 'DeadlineExceeded'


def test_cancelled_worker_attaches_partial_evidence_once(database_url):
    s = Store(database_url)
    job = s.submit('j', {})
    c = s.claim(10)
    assert s.cancel(job)
    s.attach_partial(job, c['lease_token'], {'experiments': ['e1']})
    with pytest.raises(LeaseLost):
        s.attach_partial(job, c['lease_token'], {'experiments': []})
    final = s.get(job)
    assert final['state'] == 'cancelled' and final['result'] is None
    assert [e['body'] for e in final['events'] if e['kind'] == 'partial_evidence'] == [{'experiments': ['e1']}]


def test_allocation_intent_is_idempotent_and_fenced(database_url):
    s, now = clocked(database_url)
    job = s.submit('j', {})
    c = s.claim(10)
    ctx = {**c, 'id': job, 'step_attempt': 1}
    m = AllocationManager(s, 'w1')
    a1 = m.intent(ctx, 'iter0.experiment', purpose='candidate', provider='local', resource_type='process', spec={})
    a2 = m.intent(ctx, 'iter0.experiment', purpose='candidate', provider='local', resource_type='process', spec={})
    assert a1['id'] == a2['id'] and len(m.list(job_id=job)) == 1
    a3 = m.intent({**ctx, 'step_attempt': 2}, 'iter0.experiment', purpose='candidate', provider='local',
                  resource_type='process', spec={})
    assert a3['id'] != a1['id']
    now[0] += 11
    with pytest.raises(LeaseLost):
        m.intent({**ctx, 'step_attempt': 3}, 'iter0.experiment', purpose='candidate', provider='local',
                 resource_type='process', spec={})
    m.release(a1['id'], 'test')  # cleanup is never fenced
    assert m.get(a1['id'])['state'] == 'released'


def test_worker_registry(database_url):
    s, now = clocked(database_url)
    w = s.register_worker(10)
    s.worker_heartbeat(w, 10)
    with s.connection() as db:
        assert s.worker_alive(db, w)
    now[0] += 11
    with s.connection() as db:
        assert not s.worker_alive(db, w)
    s.retire_worker(w)
    with pytest.raises(LeaseLost):
        s.worker_heartbeat(w)
