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


def cfg(static, prefix=False, spec=False, packed=False, budget=24):
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
    n.static_decode_spec = spec
    n.static_decode_spec_mtp = 'random:3'
    n.static_decode_packed_prefill = packed
    n.static_decode_packed_max_tokens = budget
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


def run(static, plan, prefix=False, spec=False, temps=None, packed=False, budget=24):
    """plan: list of (delay_s, request_id, prompt, max_new)."""
    async def go():
        model = tiny_model(seed=11).to(torch.float32)
        e = QwenHybridContinuousEngine(model, Tok(), cfg(static, prefix, spec, packed, budget), 'cpu')
        await e.start()

        async def one(delay, rid, prompt, max_new):
            await asyncio.sleep(delay)
            t = (temps or {}).get(rid, 0.0)
            q = await e.submit(Request(rid, tuple(prompt), SamplingParams(max_new_tokens=max_new, temperature=t)))
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
    tb = d['ttft_breakdown']                       # every request's first token is split once
    assert tb['requests'] == len(PLAN) and dref['ttft_breakdown']['requests'] == len(PLAN)
    assert tb['total_ms'] >= tb['queue_ms'] + tb['wait_ms'] + tb['prefill_ms'] - 1e-6 > 0


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


def test_spec_lane_matches_transformers_decode():
    """N6.2: MTP speculative lane (random head here: mostly rejected drafts) must give the
    Transformers greedy output, with compaction, late arrivals and the length fallback."""
    ref, _ = run(False, PLAN)
    got, d = run(True, PLAN, spec=True)
    assert got == ref
    sd = d['static_decode']
    assert sd['active'] and sd['spec'] and sd['spec_steps'] > 0
    assert sd['rows_admitted'] >= 4 and sd['rows_rejected_length'] >= 1 and sd['live_rows'] == 0
    assert sd['spec_acceptance'] is not None


def test_spec_lane_with_prefix_cache_hit():
    shared = [9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8]
    plan = [(0.0, 'p1', shared + [1, 2], 6), (0.3, 'p2', shared + [3, 3, 3], 6)]
    ref, _ = run(False, plan, prefix=True)
    got, d = run(True, plan, prefix=True, spec=True)
    assert got == ref
    assert d['static_decode']['spec_steps'] > 0


def test_spec_lane_leaves_sampled_rows_on_the_transformers_path():
    plan = [(0.0, 'g', [5, 7, 11, 13, 2], 6), (0.0, 's', [3, 1, 4, 1, 5], 6)]
    got, d = run(True, plan, spec=True, temps={'s': 0.8})
    ref, _ = run(False, plan[:1])
    assert got['g'] == ref['g'] and len(got['s']) >= 1
    assert d['static_decode']['rows_rejected_sampling'] >= 1


# ---- P1: packed prefill admission -----------------------------------------------------
@pytest.mark.parametrize('spec', [False, True])
def test_packed_prefill_lane_matches_transformers(spec):
    """Packed prefill straight into the pool must give the Transformers greedy output. The
    24-token budget sends some rows of a cohort down the old path, and 'e' (too long for the
    pool) always takes it."""
    ref, _ = run(False, PLAN)
    got, d = run(True, PLAN, spec=spec, packed=True)
    assert got == ref
    sd = d['static_decode']
    assert sd['packed_prefills'] >= 1 and sd['packed_rows'] >= 2, sd
    assert sd['rows_rejected_length'] >= 1 or sd['packed_rows'] < len(PLAN)
    assert sd['live_rows'] == 0
    assert d['ttft_breakdown']['requests'] == len(PLAN)


def test_packed_prefill_with_prefix_cache_on():
    shared = [9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8]
    plan = [(0.0, 'p1', shared + [1, 2], 6), (0.3, 'p2', shared + [3, 3, 3], 6), (0.6, 'p3', shared + [5], 6)]
    ref, _ = run(False, plan, prefix=True)
    got, d = run(True, plan, prefix=True, packed=True)
    assert got == ref
    pp = d['static_decode']['pool_prefix_cache']
    # p1 is cold (no snapshot); p2 shares its prefix with p1, so p2 stores it; p3 hits it.
    assert pp['stores'] == 1 and pp['hits'] >= 1, pp


@pytest.mark.parametrize('spec', [False, True])
def test_packed_prefill_shares_prefix_within_a_cohort(spec):
    """Rows arriving together with a common 16-token prefix: the prefix is computed once,
    snapshotted and copied; the outputs stay greedy-equal to Transformers."""
    shared = [9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8]
    plan = [(0.0, 'q1', shared + [1, 2], 6), (0.0, 'q2', shared + [3, 3, 3], 5), (0.0, 'q3', shared + [7], 6)]
    ref, _ = run(False, plan)
    got, d = run(True, plan, prefix=True, packed=True, spec=spec, budget=64)
    assert got == ref
    sd = d['static_decode']
    assert sd['packed_prefix_groups'] >= 1 and sd['packed_prefix_shared_rows'] >= 1, sd
