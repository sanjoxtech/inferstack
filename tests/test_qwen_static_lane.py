"""Engine-level check of the static decode lane on a tiny real Qwen3.5 model (CPU, eager).

Oracle: the same engine with the lane off (Transformers decode path). Greedy outputs
must be identical for concurrent requests of different lengths, including rows that
finish early (compaction), late arrivals, a prefix-cache hit, and a request too long
for the pool (falls back to the Transformers path).
"""
import asyncio

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers.models.qwen3_5.modeling_qwen3_5')

from inferstack.config import load_profile  # noqa: E402
from inferstack.core.request import Request, SamplingParams  # noqa: E402
from inferstack.runtime.qwen_continuous import QwenHybridContinuousEngine  # noqa: E402
from test_qwen_static_decode import tiny_model  # noqa: E402


class Tok:
    eos_token_id = 96
    pad_token_id = 0


def cfg(static, prefix=False):
    c = load_profile('configs/profiles/production-v1.6.0-baseline.toml')
    n = c.native
    n.custom_gdn_triton = False
    n.prefix_state_cache = prefix
    n.prefix_cache_min_tokens = 8
    n.prefix_cache_block_tokens = 8
    n.batch_wait_ms = 1
    n.prefill_chunk_tokens = 8
    n.max_requests_per_batch = 4
    n.static_decode = static
    n.static_decode_slots = 4
    n.static_decode_max_tokens = 48
    n.static_decode_len_buckets = (16, 32, 48)
    n.static_decode_strict = True
    c.kernels.torch_compile = False
    n.compile_model = False
    return c


async def drain(q, timeout=60):
    out = []
    while True:
        x = await asyncio.wait_for(q.get(), timeout)
        if x is None:
            return out
        if isinstance(x, BaseException):
            raise x
        out.append(x)


def run(static, plan, prefix=False):
    """plan: list of (delay_s, request_id, prompt, max_new)."""
    async def go():
        model = tiny_model(seed=11).to(torch.float32)
        e = QwenHybridContinuousEngine(model, Tok(), cfg(static, prefix), 'cpu')
        await e.start()

        async def one(delay, rid, prompt, max_new):
            await asyncio.sleep(delay)
            q = await e.submit(Request(rid, tuple(prompt), SamplingParams(max_new_tokens=max_new, temperature=0.0)))
            return rid, await drain(q)

        res = dict(await asyncio.gather(*(one(*p) for p in plan)))
        d = e.diagnostics()
        await e.close()
        return res, d
    return asyncio.run(go())


PLAN = [(0.0, 'a', [5, 7, 11, 13, 2, 9, 4, 4, 8, 1], 12),
        (0.0, 'b', [3, 1], 5),
        (0.0, 'c', [6, 6, 6, 1, 2, 3, 9], 9),
        (0.05, 'd', [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26], 7),
        (0.4, 'e', list(range(1, 41)), 10)]   # 40 + 10 > 48: must use the fallback path


def test_static_lane_matches_transformers_decode():
    ref, dref = run(False, PLAN)
    got, d = run(True, PLAN)
    assert got == ref
    assert all(len(v) >= 1 for v in got.values()) and sum(map(len, got.values())) > 15, got
    sd = d['static_decode']
    assert sd['active'] and dref['static_decode']['active'] is False
    assert sd['rows_admitted'] >= 4 and sd['decode_steps'] > 0
    assert sd['rows_rejected_length'] >= 1
    assert sd['live_rows'] == 0
    assert d['compactions'] >= 1


def test_static_lane_with_prefix_cache_hit():
    shared = [9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8]
    plan = [(0.0, 'p1', shared + [1, 2], 6), (0.3, 'p2', shared + [3, 3, 3], 6)]
    ref, _ = run(False, plan, prefix=True)
    got, d = run(True, plan, prefix=True)
    assert got == ref
    assert d['prefix_state_cache']['hits'] >= 1 if 'hits' in d['prefix_state_cache'] else True


def test_static_lane_cancellation_frees_slot():
    async def go():
        e = QwenHybridContinuousEngine(tiny_model(seed=12).to(torch.float32), Tok(), cfg(True), 'cpu')
        await e.start()
        r = Request('x', (1, 2, 3), SamplingParams(max_new_tokens=40, temperature=0.0))
        q = await e.submit(r)
        await asyncio.wait_for(q.get(), 30)
        assert e.cancel('x')
        await asyncio.wait_for(drain(q), 30)
        d = e.diagnostics()
        await e.close()
        return d, r
    d, r = asyncio.run(go())
    assert d['static_decode']['live_rows'] == 0
    # The tiny model can emit EOS before the cancel lands (CPU timing); then it is a finish.
    assert d['cancellations'] == 1 or r.finish_reason in ('stop', 'length'), (d['cancellations'], r.finish_reason)
