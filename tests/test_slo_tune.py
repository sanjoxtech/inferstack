"""Phase 2 SLO tuner: profile writing, scoring against an SLO, search-space sampling (no GPU)."""
import importlib.util
import random
import tomllib
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('slo_tune', Path(__file__).parent.parent / 'scripts' / 'slo_tune.py')
slo_tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(slo_tune)


def test_profile_roundtrip_with_overrides_loads_as_a_profile(tmp_path):
    from inferstack.config import load_profile
    base_path = Path('configs/profiles/production-v1.20.0.toml')
    base = tomllib.loads(base_path.read_text())
    out = slo_tune.apply_overrides(base, {'native.max_requests_per_batch': 4, 'native.batch_wait_ms': 0.0,
                                          'native.static_decode_compact_from_tail': False}, 'tune-000')
    text = slo_tune.dump_toml(out)
    assert tomllib.loads(text) == out                          # writer is lossless for profiles
    p = tmp_path / 'tune-000.toml'
    p.write_text(text)
    prof = load_profile(p)
    assert prof.native.max_requests_per_batch == 4 and prof.native.static_decode_compact_from_tail is False
    assert base['native']['max_requests_per_batch'] == 16      # base not mutated
    untouched = {k: v for k, v in base['native'].items() if k not in ('max_requests_per_batch', 'batch_wait_ms',
                                                                      'static_decode_compact_from_tail')}
    assert all(out['native'][k] == v for k, v in untouched.items())
    with pytest.raises(KeyError):
        slo_tune.apply_overrides(base, {'nosuch.x': 1}, 'bad')


def _card(profile, cells):
    return {'cells': [{'profile': profile, 'case': c, 'concurrency': n,
                       'metrics': {k: {'median': v} for k, v in m.items()}} for (c, n), m in cells.items()]}


SLO = {'objective': 'output_tps', 'cells': [['a', 1], ['b', 16]], 'min_success_rate': 1.0,
       'limits': {'ttft_p95_ms': 100}, 'cell_limits': {'b@16': {'ttft_p95_ms': 300}}}


def test_score_geometric_mean_and_per_cell_limits():
    card = _card('t', {('a', 1): {'output_tps': 100, 'success_rate': 1.0, 'ttft_p95_ms': 90},
                       ('b', 16): {'output_tps': 400, 'success_rate': 1.0, 'ttft_p95_ms': 250}})
    obj, viol, detail = slo_tune.score(card, 't', SLO)
    assert obj == pytest.approx(200.0)                          # sqrt(100 * 400)
    assert viol == [pytest.approx(-0.1), pytest.approx(-50 / 300)]   # relative; b@16 judged by its own limit
    assert slo_tune.feasible(viol) and detail['b@16']['ttft_p95_ms'] == 250


def test_score_marks_crash_missing_and_lost_requests_infeasible():
    over = _card('t', {('a', 1): {'output_tps': 100, 'success_rate': 1.0, 'ttft_p95_ms': 120},
                       ('b', 16): {'output_tps': 400, 'success_rate': 1.0, 'ttft_p95_ms': 250}})
    assert not slo_tune.feasible(slo_tune.score(over, 't', SLO)[1])          # a@1 over its limit
    lost = _card('t', {('a', 1): {'output_tps': 100, 'success_rate': 0.9, 'ttft_p95_ms': 50},
                       ('b', 16): {'output_tps': 400, 'success_rate': 1.0, 'ttft_p95_ms': 250}})
    obj, viol, _ = slo_tune.score(lost, 't', SLO)
    assert obj == 0.0 and max(viol) >= slo_tune.INFEASIBLE
    obj, viol, detail = slo_tune.score({'cells': []}, 't', SLO)             # server never came up
    assert obj == 0.0 and not slo_tune.feasible(viol) and detail['a@1'] == 'missing'


def test_suggest_random_is_seeded_and_in_space():
    space = [{'name': 'native.x', 'choices': [1, 2, 4]}, {'name': 'native.y', 'low': 0.0, 'high': 1.0},
             {'name': 'native.z', 'low': 1, 'high': 8, 'int': True}]
    a = [slo_tune.suggest(space, rng=random.Random(3)) for _ in range(1)]
    b = [slo_tune.suggest(space, rng=random.Random(3)) for _ in range(1)]
    assert a == b
    p = a[0]
    assert p['native.x'] in (1, 2, 4) and 0.0 <= p['native.y'] <= 1.0 and 1 <= p['native.z'] <= 8


def test_shipped_space_and_slo_fit_the_base_profile():
    import json
    base = tomllib.loads(Path('configs/profiles/production-v1.20.0.toml').read_text())
    space = json.loads(Path('configs/tune/space-l4.json').read_text())
    for p in space:
        sec, _, key = p['name'].partition('.')
        assert key in base[sec], p['name']                      # every knob exists in the profile
    slo = json.loads(Path('configs/tune/slo-default.json').read_text())
    wl = json.loads(Path('configs/scorecard/workload-fast.json').read_text())
    cases = {c['name'] for c in wl['cases']}
    assert all(c in cases and n in wl['concurrency'] for c, n in slo['cells'])


def test_floors_require_beating_a_rival_throughput():
    slo = {**SLO, 'cell_floors': {'b@16': {'output_tps': 450}}}
    card = _card('t', {('a', 1): {'output_tps': 100, 'success_rate': 1.0, 'ttft_p95_ms': 90},
                       ('b', 16): {'output_tps': 400, 'success_rate': 1.0, 'ttft_p95_ms': 250}})
    obj, viol, detail = slo_tune.score(card, 't', slo)
    assert viol[-1] == pytest.approx(50 / 450) and not slo_tune.feasible(viol)   # 11% short of the floor


def test_cells_met_counts_limits_and_floors():
    slo = {**SLO, 'cell_floors': {'b@16': {'output_tps': 450}}}
    detail = {'a@1': {'output_tps': 100, 'success_rate': 1.0, 'ttft_p95_ms': 90},
              'b@16': {'output_tps': 400, 'success_rate': 1.0, 'ttft_p95_ms': 250}, 'c@1': 'missing'}
    assert slo_tune._cells_met(detail, slo) == 1          # a@1 meets; b@16 misses its floor; c missing
