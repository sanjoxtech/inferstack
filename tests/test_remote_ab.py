"""G2: GPU-host A/B runs judged by the harness, spend-gated, own sessions only, stored in memory."""
import json
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
    assert argv[argv.index('--step') + 1] == 'scripts/remote_ab.py --out {OUT}/job'
    with pytest.raises(ValueError):
        RemoteJob('MF_session', ('x.py',), expected_hours=1)
