"""Package G: ULP numerical oracle, judge noise floor, evidence-computed gates,
typed trial records, and analyzer null semantics."""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from inferstack.evaluation.judge import Objective, evaluate
from inferstack.evaluation.numerical import check, check_relative, round_to
from inferstack.optimization.experiment import VERIFIER_CODE_SHA256, EndpointExecutor, EndpointVerifier
from inferstack.storage.repository import Repository

FIXTURES = Path(__file__).parent / 'fixtures'
ROOT = Path(__file__).parents[1]
GATES = {'correctness': True, 'quality': True, 'health': True, 'comparable': True, 'complete': True}


# --- numerical oracle -----------------------------------------------------------------

def silu(x):
    return x / (1 + np.exp(-x))


def test_correctly_rounded_output_passes_where_absolute_tolerance_failed():
    rng = np.random.default_rng(123)
    a = round_to(rng.standard_normal(98304), 'bfloat16')
    b = round_to(rng.standard_normal(98304), 'bfloat16')
    oracle = silu(a) * b                              # FP64 oracle on the BF16 inputs
    fused = round_to(oracle, 'bfloat16')              # single correct rounding
    double = round_to(round_to(silu(a), 'bfloat16') * b, 'bfloat16')  # how the BF16 reference rounds
    # The v1.6 criterion rejected exactly this: a 1-ULP gap at |y| >= 8 is 0.0625 > 3e-2.
    assert np.abs(fused - double).max() > 3e-2
    result = check('silu_mul', fused, oracle, dtype='bfloat16', reference=double)
    assert result['pass'] and result['basis'] == 'ulp_bound'
    # round_to goes FP64->FP32->BF16, so allow the tiny FP32 intermediate rounding above 0.5 ULP.
    assert result['max_ulp'] <= 0.5 + 1e-4 and 0.2 < result['mean_ulp'] < 0.3
    assert result['reference_max_ulp'] > 1.0  # the reference is the less accurate side


def test_real_errors_fail_and_reference_fallback_is_bounded():
    rng = np.random.default_rng(7)
    oracle = rng.standard_normal(4096) * 6
    good = round_to(oracle, 'bfloat16')
    ulp = np.exp2(np.floor(np.log2(np.abs(oracle))) - 7)
    off_by_two = good + 2 * ulp
    assert not check('x', off_by_two, oracle, dtype='bfloat16')['pass']
    assert check('x', off_by_two, oracle, dtype='bfloat16', reference=off_by_two + ulp)['basis'] == \
        'no_worse_than_reference'
    assert not check('x', off_by_two, oracle, dtype='bfloat16', reference=good)['pass']
    broken = good.copy()
    broken[17] = np.nan
    assert not check('x', broken, oracle, dtype='bfloat16', reference=good)['pass']
    with pytest.raises(ValueError, match='shape'):
        check('x', good[:10], oracle, dtype='bfloat16')
    state = rng.standard_normal(1000)
    assert check_relative('state', state * (1 + 5e-5), state)['pass']
    assert not check_relative('state', state * (1 + 5e-3), state)['pass']


# --- judge --------------------------------------------------------------------------------

def trials(values, metric='e2e_ms'):
    return [{metric: v} for v in values]


def test_improvement_within_noise_is_not_a_keep():
    noisy_base, noisy_cand = [97, 100, 103, 97, 103], [93, 96, 99, 93, 99]  # 4% gain, MAD 3 on each side
    verdict = evaluate(trials(noisy_base), trials(noisy_cand), [Objective('e2e_ms', 'min', epsilon=0.02)],
                       gates=GATES, min_trials=5)
    assert verdict['decision'] == 'inconclusive' and verdict['metrics']['e2e_ms']['noise_floor'] == 6
    tight = evaluate(trials([100, 100.2, 99.8, 100, 100]), trials([96, 96.1, 95.9, 96, 96]),
                     [Objective('e2e_ms', 'min', epsilon=0.02)], gates=GATES, min_trials=5)
    assert tight['decision'] == 'keep'
    assert evaluate(trials([100] * 3), trials([90] * 3), [Objective('e2e_ms', 'min')],
                    gates={**GATES, 'complete': False})['decision'] == 'reject'
    assert evaluate(trials([100] * 3), trials([90, None, 90]), [Objective('e2e_ms', 'min')],
                    gates=GATES)['decision'] == 'inconclusive'


# --- verifier with real servers ----------------------------------------------------------------

def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture
def servers():
    procs = []

    def start(answer='four'):
        port = free_port()
        procs.append(subprocess.Popen([sys.executable, str(FIXTURES / 'fake_openai_server.py'), '--port', str(port),
                                       '--answer', answer]))
        for _ in range(200):
            with socket.socket() as s:
                if s.connect_ex(('127.0.0.1', port)) == 0:
                    return f'http://127.0.0.1:{port}'
            time.sleep(0.05)
        raise AssertionError('server did not start')
    yield start
    for p in procs:
        p.kill()
        p.wait()


async def run_verifier(tmp_path, servers, *, answer='four', candidate_hw=None, baseline_hw=None, model_ref='qwen'):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    for hw in {candidate_hw, baseline_hw} - {None}:
        repo.put_record('hardware', hw, {'provider': 'colab', 'observed_at': 1.0, 'gpus': []})
    if model_ref:
        repo.put_record('model', model_ref, {'source': 'huggingface', 'repo_id': 'Qwen/x', 'revision': 'b' * 40})
    repo.put_record('deployment', 'base', {'id': 'base', 'engine': 'generic_openai', 'model': 'm',
                                           'base_url': servers(), 'hardware_ref': baseline_hw, 'model_ref': model_ref})
    repo.put_record('deployment', 'cand', {'id': 'cand', 'engine': 'generic_openai', 'model': 'm',
                                           'base_url': servers(answer), 'hardware_ref': candidate_hw,
                                           'model_ref': model_ref})
    executor = EndpointExecutor(repo, tmp_path / 'logs')
    cases = [{'prompt': '2+2', 'expected': 'four', 'max_tokens': 4}]
    verifier = EndpointVerifier(repo, executor, 'base', cases, [{'metric': 'e2e_ms', 'direction': 'min'}], repeats=3)
    plan = {'candidate_id': 'cand', 'candidate_revision': 1, 'baseline_id': 'base', 'baseline_revision': 1}
    candidate = await executor.execute(plan, 'exp')
    try:
        return repo, await verifier.verify(plan, candidate)
    finally:
        await executor.cleanup(candidate)


@pytest.mark.asyncio
async def test_gates_are_computed_and_trials_recorded(tmp_path, servers):
    repo, verdict = await run_verifier(tmp_path, servers, baseline_hw='a100', candidate_hw='a100')
    assert verdict['gates'] == GATES and verdict['decision'] in {'keep', 'reject', 'retest', 'inconclusive'}
    assert len(verdict['trial_refs']) == 6
    trial = repo.read_typed('trial', 'exp-candidate-0')
    assert trial.deployment_ref == 'cand@1' and trial.raw_artifact_sha256 == verdict['evidence']['sha256']
    assert repo.read_typed('trial', 'exp-baseline-2').deployment_ref == 'base@1'
    evidence = json.loads(repo.read_artifact(verdict['evidence']['sha256']))
    assert evidence['verifier']['code_sha256'] == VERIFIER_CODE_SHA256 and len(evidence['verifier']['fixture_sha256']) == 64
    assert evidence['comparability']['hardware_ref'] == 'same' and evidence['identity']['hardware_ref'] == ['a100@1'] * 2
    assert evidence['policy']['required'] == sorted(GATES) and len(evidence['policy']['sha256']) == 64


@pytest.mark.asyncio
async def test_missing_identity_is_not_comparable(tmp_path, servers):
    # V04: absent hardware or model identity is unknown, never proven comparability.
    _, no_hw = await run_verifier(tmp_path / 'a', servers)
    assert no_hw['gates']['comparable'] is None and no_hw['decision'] == 'inconclusive'
    _, no_model = await run_verifier(tmp_path / 'b', servers, baseline_hw='a100', candidate_hw='a100', model_ref=None)
    assert no_model['gates']['comparable'] is None and no_model['decision'] == 'inconclusive'


@pytest.mark.asyncio
async def test_wrong_answers_and_different_hardware_are_rejected(tmp_path, servers):
    _, wrong = await run_verifier(tmp_path / 'a', servers, answer='five')
    assert wrong['decision'] == 'reject' and wrong['gates']['correctness'] is False
    _, moved = await run_verifier(tmp_path / 'b', servers, baseline_hw='a100', candidate_hw='t4')
    assert moved['decision'] == 'reject' and moved['gates']['comparable'] is False
    _, same = await run_verifier(tmp_path / 'c', servers, baseline_hw='a100', candidate_hw='a100')
    assert same['gates']['comparable'] is True


# --- analyzer -----------------------------------------------------------------------------------

def run_analyzer(results):
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    out = subprocess.run([sys.executable, str(ROOT / 'scripts/v160_analyze_results.py'), '--results', str(results)],
                         capture_output=True, text=True, env=env, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    return json.loads((results / 'analysis.json').read_text())


def write_runs(path, runs):
    path.write_text(''.join(json.dumps({'concurrency': 8, 'summary': s}) + '\n' for s in runs))


def test_analyzer_keeps_missing_metrics_null_and_requires_repeats(tmp_path):
    results = tmp_path / 'results'
    results.mkdir()
    base = {'output_token_throughput_tps': 100.0, 'tpot_ms': {'p99': 20.0}, 'ttft_ms': {'p99': 50.0}}
    write_runs(results / 'baseline-sweep.jsonl', [base])
    write_runs(results / 'norm-sweep.jsonl', [{'output_token_throughput_tps': 130.0}])  # no TPOT/TTFT reported
    analysis = run_analyzer(results)
    norm = analysis['ablations_c8'][0]
    assert norm['c8_tpot_p99_ms'] is None and norm['tpot_reduction_pct'] is None and norm['c8_ttft_p99_ms'] is None
    assert set(norm['missing_metrics']) == {'tpot'} and norm['verdict']['decision'] == 'inconclusive_single_run'
    assert analysis['ablations_c8'][1]['verdict']['decision'] == 'missing'  # mlp lane never ran
    write_runs(results / 'baseline-sweep.jsonl', [base, base, base])
    write_runs(results / 'full-sweep.jsonl', [{'output_token_throughput_tps': 130.0, 'tpot_ms': {'p99': 15.0},
                                               'ttft_ms': {'p99': 50.0}}] * 3)
    full = run_analyzer(results)['ablations_c8'][-1]
    assert full['verdict']['decision'] == 'keep' and full['tps_speedup_vs_baseline'] == pytest.approx(1.3)
