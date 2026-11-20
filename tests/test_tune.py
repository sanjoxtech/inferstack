"""G5: settings tuning as a campaign action: derived space, crash-region learning, goodput objective, A/B confirm."""
import json
import sys
from pathlib import Path

import pytest

from inferstack.execution.gpu_backend import LocalBackend
from inferstack.optimization.tune import CrashRegion, TuneAction, best_from_tune, space_for

ROOT = Path(__file__).resolve().parents[1]
FAKE = str(ROOT / 'tests' / 'fixtures' / 'fake_native_server.py')
BASE = str(ROOT / 'tests' / 'fixtures' / 'profiles' / 'fake-native.toml')


def test_space_follows_hardware_and_the_measured_bottleneck():
    l4, why = space_for({'gpu': 'NVIDIA L4', 'memory_gb': 22.0}, target=['capacity', 'gemm', 'gdn'])
    names = {p['name']: p['choices'] for p in l4}
    assert max(names['static_decode_slots']) == 64 and 'gemm_blas_library' in names     # 22 GB x 4 rows/GB = 88
    assert 'static_decode_admit_pace_min_rows' not in names                              # admission: not the bottleneck
    assert any(w.startswith('static_decode_admit_pace_min_rows: dropped') for w in why)
    a100, _ = space_for({'gpu': 'A100', 'memory_gb': 79.2}, target=['capacity'])
    assert max(next(p['choices'] for p in a100 if p['name'] == 'static_decode_slots')) == 256
    assert {p['name'] for p in space_for({'memory_gb': 22.0}, target=['queue_admission'])[0]} >= {
        'static_decode_admit_every_steps', 'static_decode_admit_pace_min_rows'}
    assert len(space_for({'memory_gb': 22.0})[0]) == 9                                   # no measurement: every knob


def test_crash_region_predicts_larger_memory_points():
    r = CrashRegion()
    r.add({'static_decode_slots': 64, 'gemm_blas_library': ''})
    assert r.covers({'static_decode_slots': 96, 'gemm_blas_library': ''})
    assert not r.covers({'static_decode_slots': 48, 'gemm_blas_library': ''})
    assert not r.covers({'static_decode_slots': 96, 'gemm_blas_library': 'cublas'})       # other knobs must match


def test_best_must_beat_the_baseline():
    s = lambda g, a: {'goodput99_rps': g, 'slo_attainment': a}  # noqa: E731
    tune = {'trials': [{'trial': 0, 'baseline': True, 'patch': {}, 'score': s(8, 0.7), 'feasible': True},
                       {'trial': 1, 'patch': {'x': 1}, 'score': s(12, 0.8), 'feasible': False},
                       {'trial': 2, 'patch': {'x': 2}, 'score': s(8, 0.75), 'feasible': True}]}
    assert best_from_tune(tune)['patch'] == {'x': 2}                       # infeasible winner never picked
    tune['trials'][2]['score'] = s(8, 0.6)
    assert best_from_tune(tune) is None


@pytest.fixture
def setup(tmp_path):
    from inferstack.storage.repository import Repository
    profiles = tmp_path / 'profiles.json'
    profiles.write_text(json.dumps({'profiles': [{'name': 'nat', 'engine': 'inferstack_native', 'model': 'm',
                                                  'argv': [sys.executable, FAKE, '--profile', BASE,
                                                           '--port', '{port}']}]}))
    workload = tmp_path / 'workload.json'
    workload.write_text(json.dumps({'cases': [{'name': 'chat', 'prompt_words': 8, 'max_tokens': 4,
                                               'ttft_ms': 500.0, 'tpot_ms': 50.0}]}))
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    repo.put_record('deployment', 'base', {'id': 'base', 'engine': 'inferstack_native', 'model': 'm',
                                           'base_url': 'http://127.0.0.1:9'})
    return repo, profiles, workload, tmp_path


def test_tune_finds_a_setting_learns_the_crash_region_and_the_ab_confirms_it(setup):
    """Exit test on a fake engine: baseline (16 slots) queues; >= 64 slots crashes; cuBLASLt changes greedy
    text. The tuner must pick a 32-48 slot patch without cuBLASLt, never rerun a predicted crash, store the
    result, and the interleaved A/B must keep it."""
    from inferstack.evaluation.bottleneck import latest_measurements
    from inferstack.optimization.remote_verify import RemoteABVerifier
    repo, profiles, workload, tmp_path = setup
    v = RemoteABVerifier(repo, LocalBackend(price_per_hour=0.0), tmp_path / 'runs', profiles=str(profiles),
                         workload=str(workload), slo='', repeats=3, rates=(5, 10), seconds_per_rate=0.4,
                         min_requests=4, hours_per_trial=0.01)
    action = TuneAction(v, trials=8, seed=1, hours_per_trial=0.01,
                        knobs=['static_decode_slots', 'gemm_blas_library'])
    out = action.run('base', 'nat', name='tune-fake')
    tune = latest_measurements(repo, 'base')['tune']['summary']
    best = tune['best']['patch']
    assert 32 <= best['static_decode_slots'] < 64 and best['gemm_blas_library'] != 'cublaslt'
    assert tune['crashed'] >= 1 and all(c['static_decode_slots'] >= 64 for c in tune['crash_region'])
    raw = json.loads(next((tmp_path / 'runs').rglob('tune.json')).read_text())
    for t in raw['trials']:
        if t.get('pruned'):
            assert t['patch']['static_decode_slots'] >= t['predicted_by']['static_decode_slots']
        if t['patch'].get('gemm_blas_library') == 'cublaslt' and t.get('ready'):
            assert not t['feasible']                                      # quality gate inside the tuner
    assert out['decision'] == 'keep', out.get('verdict', {}).get('reasons')


def test_crash_region_skips_predicted_crashes_on_the_host(setup):
    """On a big GPU (60 GB -> slots up to 192) every slot count >= 64 crashes; after the first crash the
    larger ones are skipped without launching a server."""
    import subprocess
    repo, profiles, workload, tmp_path = setup
    out = tmp_path / 'tune'
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'remote_tune.py'), '--profiles', str(profiles),
                    '--baseline', 'nat', '--workload', str(workload), '--slo', '', '--knobs', 'static_decode_slots',
                    '--hardware', '{"gpu": "big", "memory_gb": 60}', '--trials', '8', '--seed', '3',
                    '--rates', '5', '10', '--seconds-per-rate', '0.3', '--min-requests', '3', '--warmup', '0',
                    '--out', str(out)], check=True, cwd=ROOT, capture_output=True)
    trials = json.loads((out / 'tune.json').read_text())['trials'][1:]
    pruned = [t for t in trials if t.get('pruned')]
    crashed = [t for t in trials if t.get('crashed')]
    assert pruned and crashed and not any(t.get('ready') for t in trials if t['patch']['static_decode_slots'] >= 64)
    first = min(c['trial'] for c in crashed)
    assert all(p['trial'] > first for p in pruned)
