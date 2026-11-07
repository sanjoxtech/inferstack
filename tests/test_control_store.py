from concurrent.futures import ThreadPoolExecutor
import pytest
from inferstack.control.store import Store, LeaseLost


def test_restart_and_idempotency(database_url):
    path = database_url
    s = Store(path)
    job = s.submit('one', {'target': 'deployment-a'})
    assert Store(path).submit('one', {'target': 'deployment-a'}) == job
    with pytest.raises(ValueError):
        s.submit('one', {'target': 'deployment-b'})
    attempt = Store(path).claim()
    s.finish(job, attempt['lease_token'], {'decision': 'reject'})
    assert Store(path).get(job)['result']['decision'] == 'reject'


def test_expired_worker_cannot_finish_or_extend(database_url):
    now = [100.0]
    s = Store(database_url, clock=lambda: now[0])
    job = s.submit('one', {})
    old = s.claim(10)
    now[0] = 110
    with pytest.raises(LeaseLost):
        s.heartbeat(job, old['lease_token'])
    new = s.claim(10)
    with pytest.raises(LeaseLost):
        s.finish(job, old['lease_token'], {})
    s.finish(job, new['lease_token'], {'ok': True})
    assert s.get(job)['attempt'] == 2


def test_competing_workers_claim_once(database_url):
    s = Store(database_url)
    s.submit('one', {})
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: s.claim(), range(8)))
    assert sum(x is not None for x in claims) == 1


def test_cancel_invalidates_worker(database_url):
    s = Store(database_url)
    job = s.submit('one', {})
    claim = s.claim()
    assert s.cancel(job)
    assert not s.cancel(job)
    with pytest.raises(LeaseLost):
        s.finish(job, claim['lease_token'], {})
    assert s.claim() is None
    assert 'lease_token' not in s.get(job)
