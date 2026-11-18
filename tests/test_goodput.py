"""P9 goodput helpers: per-request good rule and goodput at 99% / 90% (no GPU)."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('goodput', Path(__file__).parent.parent / 'scripts' / 'goodput.py')
gp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gp)

LIM = {'ttft_ms': 500.0, 'tpot_ms': 30.0}


def test_good_needs_success_ttft_and_tpot():
    assert gp.is_good({'success': True, 'ttft_ms': 120, 'client_decode_ms': 20}, LIM)
    assert not gp.is_good({'success': True, 'ttft_ms': 600, 'client_decode_ms': 20}, LIM)
    assert not gp.is_good({'success': True, 'ttft_ms': 120, 'client_decode_ms': 35}, LIM)
    assert not gp.is_good({'success': False, 'ttft_ms': 120, 'client_decode_ms': 20}, LIM)
    assert gp.is_good({'success': True, 'ttft_ms': 120, 'client_decode_ms': None}, LIM)   # 1-token output


def test_goodput_is_highest_rate_meeting_each_attainment():
    steps = [{'rate': 1, 'good_frac': 1.0}, {'rate': 2, 'good_frac': 0.995}, {'rate': 4, 'good_frac': 0.95},
             {'rate': 8, 'good_frac': 0.6}]
    assert gp.summarize(steps) == {'goodput99': 2, 'goodput90': 4}
    assert gp.summarize([{'rate': 1, 'good_frac': 0.8}]) == {'goodput99': 0.0, 'goodput90': 0.0}


def test_limits_come_from_the_industry_layer():
    import json
    slo = json.loads(Path('configs/tune/slo-v2-l4.json').read_text())
    assert gp.limits_for(slo, 'prefill_heavy') == {'ttft_ms': 1000.0, 'tpot_ms': 40.0}
    assert gp.limits_for(slo, 'interactive') == {'ttft_ms': 500.0, 'tpot_ms': 30.0}
