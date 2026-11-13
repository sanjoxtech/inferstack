import importlib.util
from pathlib import Path

from inferstack.runtime.qwen_continuous import QwenHybridContinuousEngine
from test_qwen_scheduler import Model, Tok, cfg

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('n2_prefill_diagnosis', ROOT / 'scripts' / 'n2_prefill_diagnosis.py')
n2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n2)


def engine(**native):
    c = cfg()
    for k, v in native.items():
        setattr(c.native, k, v)
    return QwenHybridContinuousEngine(Model(), Tok(), c, 'cpu'), c


def test_prompts_match_the_scorecard_prefill_heavy_case():
    texts, case = n2.scorecard_prompts(n=2)
    assert case['name'] == 'prefill_heavy' and case['prompt_words'] == 1500
    assert len(texts) == 2 and texts[0] != texts[1]
    assert texts == n2.scorecard_prompts(n=2)[0]  # deterministic


def test_chunk_size_sets_number_of_forward_calls():
    eng, _ = engine(prefix_state_cache=False)
    ids = [list(range(1, 11))]  # 10 tokens
    r4 = n2.measure_prefill(eng, ids, chunk=4, prefix_cache=False)
    r10 = n2.measure_prefill(eng, ids, chunk=10, prefix_cache=False)
    assert r4['model_calls'] == 3 and r10['model_calls'] == 1
    assert r4['wall_ms'] >= r4['model_ms'] + r4['snapshot_ms'] - 1e-6
    assert r4['snapshots'] == 0 and r4['prompt_tokens'] == [10]


def test_prefix_cache_snapshots_counted_only_when_enabled_and_single_request():
    eng, c = engine(prefix_state_cache=True, prefix_cache_min_tokens=4, prefix_cache_block_tokens=4)
    ids = [list(range(1, 13))]  # 12 tokens -> snapshots at 4, 8, 12
    on = n2.measure_prefill(eng, ids, chunk=4, prefix_cache=True)
    off = n2.measure_prefill(eng, ids, chunk=4, prefix_cache=False)
    two = n2.measure_prefill(eng, ids * 2, chunk=4, prefix_cache=True)
    assert on['snapshots'] == 3 and off['snapshots'] == 0 and two['snapshots'] == 0
    assert c.native.prefill_chunk_tokens == 4


def test_measurement_restores_engine_methods_and_uses_a_fresh_cache():
    eng, _ = engine(prefix_state_cache=True, prefix_cache_min_tokens=4, prefix_cache_block_tokens=4)
    call, store = eng._model_call, eng._maybe_store_prefix
    ids = [list(range(1, 9))]
    first = n2.measure_prefill(eng, ids, chunk=4, prefix_cache=True)
    second = n2.measure_prefill(eng, ids, chunk=4, prefix_cache=True)
    assert eng._model_call == call and eng._maybe_store_prefix == store
    assert first['model_calls'] == second['model_calls'] == 2  # no prefix hit carried over


def test_phases_records_errors_and_continues(monkeypatch):
    eng, _ = engine(prefix_state_cache=False)
    real = n2.measure_prefill

    def flaky(e, ids, *, chunk, prefix_cache, max_new_tokens=32):
        if chunk == 8:
            raise RuntimeError('CUDA out of memory')
        return real(e, [x[:16] for x in ids], chunk=chunk, prefix_cache=prefix_cache, max_new_tokens=2)

    monkeypatch.setattr(n2, 'measure_prefill', flaky)
    rows = n2.run_phases(eng, Tok(), batches=[1], chunks=[4, 8], repeats=2)
    assert len(rows) == 4
    assert all('error' in r for r in rows if r['chunk'] == 8)
    ok = [r for r in rows if r['chunk'] == 4]
    assert all(r['repeats'] == 2 and r['model_calls'] == 4 for r in ok)
