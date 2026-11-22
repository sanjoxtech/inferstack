"""G6: the SLO watchdog opens a settings campaign on a sustained SLO breach, inside a spend cap, and closes it."""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from inferstack.evaluation.goodput import GoodputCase
from inferstack.optimization.watchdog import (
    ServingSLO,
    Watchdog,
    fix_from,
    judge_window,
)
from inferstack.storage.repository import Repository

ROOT = Path(__file__).resolve().parents[1]
FAKE = str(ROOT / 'tests' / 'fixtures' / 'fake_native_server.py')
BASE = str(ROOT / 'tests' / 'fixtures' / 'profiles' / 'fake-native.toml')
SLO = ServingSLO(rate_rps=5.0, cases=(GoodputCase('chat', 8, 4, ttft_ms=500.0, tpot_ms=50.0),))


def _repo(tmp_path):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    repo.put_record('deployment', 'base', {'id': 'base', 'engine': 'inferstack_native', 'model': 'm',
                                           'base_url': 'http://127.0.0.1:9'})
    return repo


def _windows(*fracs):
    it = iter(fracs)

    async def observe(_):
        f = next(it)
        return {'source': 'stub', 'requests': 0 if f is None else 40, 'good_frac': f}
    return observe


def _keep(carried, patch=None):
    return {'decision': 'keep', 'cost_usd': 0.4, 'tune': {'best': {'patch': patch or {'static_decode_slots': 48}}},
            'verdict': {'metrics': {'goodput99_rps': {'candidate': {'median': carried}}}}}


def test_window_rules():
    assert judge_window({'requests': 40, 'good_frac': 0.995}, SLO)[0] == 'ok'
    assert judge_window({'requests': 40, 'good_frac': 0.9}, SLO)[0] == 'breach'
    assert judge_window({'requests': 3, 'good_frac': 0.0}, SLO)[0] == 'unmeasured'       # too few to judge
    assert judge_window({'requests': 40, 'ttft_p99_ms': 900.0}, SLO)[0] == 'breach'      # telemetry p99
    assert judge_window({'requests': 40}, SLO)[0] == 'unmeasured'
    assert fix_from(_keep(6.0), SLO)['patch'] == {'static_decode_slots': 48}
    assert fix_from(_keep(4.0), SLO) is None                                             # kept, but short of 5 req/s


def test_noise_does_not_open_and_a_sustained_breach_opens_once(tmp_path):
    repo, calls = _repo(tmp_path), []
    w = Watchdog(repo, _windows(0.9, 1.0, 0.9, None, 0.9, 0.9, 0.8, 0.8), lambda d, inc: calls.append(inc) or _keep(6.0),
                 SLO, consecutive=3)
    states = [asyncio.run(w.tick('base'))['status'] for _ in range(6)]
    assert states == ['suspect', 'healthy', 'suspect', 'suspect', 'suspect', 'fix_pending']  # unmeasured: no reset
    assert len(calls) == 1 and calls[0]['trigger']['kind'] == 'slo_breach'
    assert [asyncio.run(w.tick('base'))['status'] for _ in range(2)] == ['fix_pending'] * 2  # no re-open
    assert len(calls) == 1
    inc = w.incident('base--inc0001')
    assert inc['fix']['patch'] == {'static_decode_slots': 48} and inc['outcome'] is None   # open until verified


def test_promotion_is_verified_before_the_incident_closes(tmp_path):
    repo = _repo(tmp_path)
    w = Watchdog(repo, _windows(0.5, 0.5, 1.0, 1.0), lambda d, inc: _keep(6.0), SLO, consecutive=2)
    asyncio.run(w.tick('base')), asyncio.run(w.tick('base'))
    assert w.state('base')['status'] == 'fix_pending'
    repo.put_record('deployment', 'base', {'id': 'base', 'engine': 'inferstack_native', 'model': 'm',
                                           'base_url': 'http://127.0.0.1:9', 'profile': {'static_decode_slots': 48}},
                    expected_revision=1)                                            # a promotion: new revision
    assert asyncio.run(w.tick('base'))['status'] == 'verifying'
    assert asyncio.run(w.tick('base'))['status'] == 'healthy'
    assert w.incident('base--inc0001')['outcome'] == 'resolved'


def test_no_fix_escalates_and_infra_failure_cools_down(tmp_path):
    notes = []
    repo = _repo(tmp_path)
    w = Watchdog(repo, _windows(0.5, 0.5, 0.5, 0.5), lambda d, inc: _keep(4.0), SLO, consecutive=2,
                 notify=notes.append)
    asyncio.run(w.tick('base')), asyncio.run(w.tick('base'))
    assert w.state('base')['status'] == 'escalated' and w.incident('base--inc0001')['outcome'] == 'unresolved'
    assert notes[0]['kind'] == 'watchdog_unresolved'
    asyncio.run(w.tick('base')), asyncio.run(w.tick('base'))
    assert w.state('base')['incidents'] == 1                                        # no loop on the same revision

    now = [1000.0]
    repo2 = _repo(tmp_path / 'b')
    w2 = Watchdog(repo2, _windows(*[0.5] * 6), lambda d, inc: {'decision': 'infra_failed', 'cost_usd': 0.0}, SLO,
                  consecutive=2, cooldown_s=60, clock=lambda: now[0])
    asyncio.run(w2.tick('base')), asyncio.run(w2.tick('base'))
    assert w2.state('base')['status'] == 'cooldown' and w2.incident('base--inc0001')['outcome'] == 'infra_failed'
    asyncio.run(w2.tick('base'))
    assert w2.state('base')['incidents'] == 1
    now[0] += 61
    asyncio.run(w2.tick('base')), asyncio.run(w2.tick('base'))
    assert w2.state('base')['incidents'] == 2                                       # retried after the cooldown


def test_spend_cap_blocks_the_campaign(tmp_path):
    calls = []
    w = Watchdog(_repo(tmp_path), _windows(0.5, 0.5), lambda d, inc: calls.append(1), SLO, consecutive=2,
                 daily_cap_usd=10.0, campaign_estimate_usd=12.0)
    asyncio.run(w.tick('base')), asyncio.run(w.tick('base'))
    assert not calls and w.incident('base--inc0001')['outcome'] == 'blocked_budget'


def test_planted_regression_opens_and_closes_a_campaign_unattended(tmp_path):
    """Exit test (G6): the live engine runs a planted bad setting (16 slots: first token 0.7 s against a 0.5 s
    SLO). The watchdog sees two breached windows, runs the settings tune + A/B on its own, applies the kept
    patch, re-measures the live endpoint and closes the incident as resolved. No human step, $0."""
    sys.path.insert(0, str(ROOT / 'scripts'))
    import engine_scorecard as es

    from inferstack.engines.profiles import native_overlay
    from inferstack.execution.gpu_backend import LocalBackend
    from inferstack.optimization.remote_verify import RemoteABVerifier
    from inferstack.optimization.tune import TuneAction
    from inferstack.optimization.watchdog import probe_observer

    repo = _repo(tmp_path)
    argv = [sys.executable, FAKE, '--profile', BASE, '--port', '{port}']
    profiles = tmp_path / 'profiles.json'
    profiles.write_text(json.dumps({'profiles': [{'name': 'nat', 'engine': 'inferstack_native', 'model': 'm',
                                                  'argv': argv}]}))
    workload = tmp_path / 'workload.json'
    workload.write_text(json.dumps({'cases': [{'name': 'chat', 'prompt_words': 8, 'max_tokens': 4,
                                               'ttft_ms': 500.0, 'tpot_ms': 50.0}]}))
    live = {}

    def serve(server_argv):
        port = es.free_port()
        proc = subprocess.Popen([a.replace('{port}', str(port)) for a in server_argv],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base = f'http://127.0.0.1:{port}'
        ready, detail = asyncio.run(es.wait_ready(base, 'm', proc, 30))
        assert ready, detail
        if live.get('proc'):
            live['proc'].terminate()
            live['proc'].wait(10)
        live['proc'] = proc
        return base

    def apply(deployment_id, patch):
        patched, _ = native_overlay(argv, patch, tmp_path / 'live', source_root=ROOT)
        base = serve(patched)
        cur = repo.read('deployment', deployment_id)
        repo.put_record('deployment', deployment_id, {**cur['body'], 'base_url': base, 'profile': patch},
                        expected_revision=cur['revision'])

    (tmp_path / 'live').mkdir()
    try:
        cur = repo.read('deployment', 'base')
        repo.put_record('deployment', 'base', {**cur['body'], 'base_url': serve(argv)}, expected_revision=1)
        v = RemoteABVerifier(repo, LocalBackend(price_per_hour=0.0), tmp_path / 'runs', profiles=str(profiles),
                             workload=str(workload), slo='', repeats=3, rates=(5, 10), seconds_per_rate=0.4,
                             min_requests=4, hours_per_trial=0.01)
        tune = TuneAction(v, trials=8, seed=1, hours_per_trial=0.01, knobs=['static_decode_slots', 'gemm_blas_library'])
        w = Watchdog(repo, probe_observer(SLO, seconds=0.8, min_requests=4, warmup=0),
                     lambda d, inc: tune.run(d, 'nat', name=inc['id'].replace('--', '-')), SLO, apply=apply,
                     consecutive=2, min_requests=4)
        statuses = [asyncio.run(w.tick('base'))['status'] for _ in range(4)]
    finally:
        if live.get('proc'):
            live['proc'].terminate()
            live['proc'].wait(10)
    assert statuses == ['suspect', 'verifying', 'verifying', 'healthy'], statuses
    inc = w.incident('base--inc0001')
    assert inc['outcome'] == 'resolved' and inc['campaign']['decision'] == 'keep', inc
    assert 32 <= inc['fix']['patch']['static_decode_slots'] < 64 and inc['campaign']['cost_usd'] == 0.0
    assert repo.read('deployment', 'base')['body']['profile'] == inc['fix']['patch']
