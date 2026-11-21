"""G2: GPU-host A/B runs judged by the harness, spend-gated, own sessions only, stored in memory."""
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from inferstack.execution.gpu_backend import ColabBackend, LocalBackend, RemoteJob, SpendRefused, check_spend, run_job

ROOT = Path(__file__).resolve().parents[1]
FAKE = str(ROOT / 'tests' / 'fixtures' / 'fake_openai_server.py')


def fake_profile(name, *extra):
    return {'name': name, 'engine': 'generic_openai', 'model': 'm',
            'argv': [sys.executable, FAKE, '--port', '{port}', '--usage', '--stream-tokens', '4',
                     '--chunk-delay', '0.002', *extra]}


@pytest.fixture
def setup(tmp_path):
    from inferstack.storage.repository import Repository
    profiles = tmp_path / 'profiles.json'
    profiles.write_text(json.dumps({'profiles': [fake_profile('fast'), fake_profile('fast2'),
                                                 fake_profile('slow', '--first-delay', '0.7')]}))
    workload = tmp_path / 'workload.json'
    workload.write_text(json.dumps({'cases': [{'name': 'chat', 'prompt_words': 8, 'max_tokens': 4,
                                               'ttft_ms': 500.0, 'tpot_ms': 50.0}]}))
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    repo.put_record('hardware', 'l4', {'provider': 'colab', 'observed_at': 1.0, 'gpus': []})
    repo.put_record('model', 'qwen', {'source': 'huggingface', 'repo_id': 'Qwen/x', 'revision': 'b' * 40})
    repo.put_record('deployment', 'base', {'id': 'base', 'engine': 'generic_openai', 'model': 'm',
                                           'base_url': 'http://127.0.0.1:9', 'hardware_ref': 'l4', 'model_ref': 'qwen'})
    return repo, profiles, workload, tmp_path


def verifier(repo, profiles, workload, tmp_path, **kw):
    from inferstack.optimization.remote_verify import RemoteABVerifier
    return RemoteABVerifier(repo, LocalBackend(price_per_hour=0.0), tmp_path / 'runs', profiles=str(profiles),
                            workload=str(workload), slo='', repeats=3, rates=(5, 10), seconds_per_rate=0.4,
                            min_requests=4, hours_per_trial=0.01, **kw)


def test_identical_sides_claim_nothing_and_outcome_reaches_memory(setup):
    from inferstack.memory import experience
    repo, profiles, workload, tmp_path = setup
    plan = {'id': 'same', 'hypothesis': 'identical engines', 'baseline_deployment': 'base',
            'baseline_profile': 'fast', 'candidate_profile': 'fast2'}
    verdict, run = verifier(repo, profiles, workload, tmp_path).verify(plan)
    assert run.ok and run.cost_usd == 0.0
    assert all(verdict['gates'].values()), verdict
    assert verdict['decision'] in {'inconclusive', 'retest'} and verdict['metrics']['goodput99_rps']['gain'] == 0
    ab = json.loads((run.evidence_dir / 'ab.json').read_text())
    assert [t['side'] for t in ab['trials']] == ['baseline', 'candidate', 'candidate', 'baseline',
                                                 'baseline', 'candidate']           # AB, BA, AB
    out = experience(repo, 'base', limit=5)
    assert len(out['items']) == 1 and out['items'][0]['decision'] == verdict['decision']


def test_slower_candidate_is_rejected(setup):
    repo, profiles, workload, tmp_path = setup
    plan = {'id': 'slow', 'hypothesis': 'slower first token', 'baseline_deployment': 'base',
            'baseline_profile': 'fast', 'candidate_profile': 'slow'}
    verdict, run = verifier(repo, profiles, workload, tmp_path).verify(plan)
    assert verdict['decision'] == 'reject'
    assert verdict['metrics']['goodput99_rps']['candidate']['median'] == 0
    knee = verdict['knee']['candidate']                          # G4: why it missed, not only that it did
    assert knee['bottleneck'] == 'uncertain' and knee['cases']['chat']['signal']['dominant_miss'] == 'ttft'
    from inferstack.evaluation.bottleneck import latest_measurements
    assert latest_measurements(repo, 'base')['knee']['summary']['bottleneck'] == 'none'   # baseline met every rate


def test_spend_gate():
    job = RemoteJob('x', ('scripts/remote_ab.py',), expected_hours=15)
    with pytest.raises(SpendRefused):
        check_spend(job, 0.80)                                   # $12 > $10 cap
    assert check_spend(job, 0.80, approved_usd=12.0) == 12.0     # explicit approval covers it
    with pytest.raises(SpendRefused):
        check_spend(job, None)                                   # unknown price is never guessed
    with pytest.raises(SpendRefused):
        run_job(ColabBackend(runner=lambda *a, **k: pytest.fail('launched despite refusal')), job, '/nonexistent')


def test_colab_owns_only_its_sessions_and_retries_a_reclaimed_vm(tmp_path):
    calls = []

    def runner(argv, **kw):
        session = argv[argv.index('--session') + 1]
        calls.append(session)
        if len(calls) == 1:
            return subprocess.CompletedProcess(argv, 3, 'VM reclaimed', '')
        (tmp_path / f'{session}-abc1234' / 'out-x' / 'job').mkdir(parents=True)
        return subprocess.CompletedProcess(argv, 0, 'ok', '')

    backend = ColabBackend(runner=runner)
    job = RemoteJob('p14-try', ('scripts/remote_ab.py', '--out', '{OUT}'), expected_hours=1)
    res = run_job(backend, job, tmp_path)
    assert calls == ['ifs-p14-try', 'ifs-p14-try-1'] and all(not c.startswith('mf-') for c in calls)
    assert res.ok and res.evidence_dir.name == 'job' and res.estimate_usd == 0.8
    argv = backend.driver_argv(job, 'ifs-p14-try', tmp_path)
    assert shlex.split(argv[argv.index('--step') + 1]) == ['scripts/remote_ab.py', '--out', '{OUT}/job']
    patched = RemoteJob('p', ('scripts/remote_ab.py', '--patch', '{"k": 3}'), expected_hours=1)
    step = backend.driver_argv(patched, 'ifs-p', tmp_path)[-1]
    assert json.loads(shlex.split(step)[2]) == {'k': 3}            # JSON survives the driver's shlex.split
    with pytest.raises(ValueError):
        RemoteJob('MF_session', ('x.py',), expected_hours=1)


def test_attainment_sees_a_regression_between_grid_points():
    """P10b replay on L4 (ifs-g2-proof-k3b): goodput@99 tied at 8, the 12 req/s step collapsed."""
    from inferstack.optimization.remote_verify import judge

    def trial(side, repeat, top):
        steps = [{'rate': 4.0, 'good_frac': 1.0}, {'rate': 8.0, 'good_frac': 1.0}, {'rate': 12.0, 'good_frac': top}]
        return {'side': side, 'repeat': repeat, 'profile': side, 'ready': True, 'check': ['a', 'b'],
                'goodput': {'goodput99_rps': 8.0, 'goodput90_rps': 8.0, 'token_counts_missing': 0,
                            'cases': {'interactive': {'steps': steps}}}}
    base = [0.713, 0.642, 0.858]
    cand = [0.167, 0.208, 0.2]
    ab = {'commit': 'c', 'repeats': 3, 'sweep': {'rates': [4, 8, 12, 16]},
          'trials': [trial('baseline', i, b) for i, b in enumerate(base)] +
                    [trial('candidate', i, c) for i, c in enumerate(cand)]}
    verdict, gates, _, _ = judge(ab)
    assert all(gates.values()) and verdict['decision'] == 'reject'
    assert verdict['metrics']['goodput99_rps']['gain'] == 0 and verdict['metrics']['slo_attainment']['gain'] < 0


def test_colab_cost_counts_only_while_the_vm_exists(tmp_path):
    """A100 2026-11-20: five failed `colab new` attempts were billed $1.44 of wall time. Waiting for a VM
    that never came is free; only the driver's BILLING start..stop window costs money."""
    from inferstack.execution.gpu_backend import billed_seconds
    assert billed_seconds('retry 5/5 after failure\nfailed after 5 tries', 1000.0) == 0.0
    assert billed_seconds('BILLING start 100.0\nx\nBILLING stop 460.0', 9999.0) == 360.0
    assert billed_seconds('BILLING start 100.0\nreclaimed', 400.0) == 300.0      # no stop: billed to the end

    def failed_create(argv, **kw):
        kw['stdout'].write('+ colab new\nretry 5/5 after failure\n')
        return subprocess.CompletedProcess(argv, 1, '', '')
    job = RemoteJob('a100-try', ('scripts/remote_ab.py', '--out', '{OUT}'), expected_hours=1)
    res = run_job(ColabBackend(price_per_hour=1.80, runner=failed_create), job, tmp_path)
    assert not res.ok and res.cost_usd == 0.0

    def one_hour(argv, **kw):
        session = argv[argv.index('--session') + 1]
        kw['stdout'].write('BILLING start 0.0\nBILLING stop 3600.0\n')
        (tmp_path / f'{session}-abc1234' / 'out-x' / 'job').mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(argv, 0, '', '')
    res = run_job(ColabBackend(price_per_hour=1.80, runner=one_hour), job, tmp_path)
    assert res.ok and res.cost_usd == 1.8
