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


def cfg(static, prefix=False, spec=False, packed=False, budget=24, first_sight=False, single=False, emit=False, tail=False, overlap=0, pipe=False, lazy=False):
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
    n.prefix_cache_store_first_sight = first_sight
    n.prefix_cache_single_pass_snapshot = single
    n.emit_batched = emit
    n.static_decode_compact_from_tail = tail
    n.static_decode_overlap_prefill = bool(overlap)
    n.static_decode_overlap_slots = overlap
    n.static_decode_overlap_min_tokens = 8
    n.static_decode_spec_pipeline = pipe
    n.static_decode_spec_lazy_bank = lazy
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


def run(static, plan, prefix=False, spec=False, temps=None, packed=False, budget=24, first_sight=False,
        single=False, emit=False, tail=False, overlap=0, pipe=False, lazy=False):
    """plan: list of (delay_s, request_id, prompt, max_new)."""
    async def go():
        model = tiny_model(seed=11).to(torch.float32)
        e = QwenHybridContinuousEngine(model, Tok(), cfg(static, prefix, spec, packed, budget, first_sight, single, emit, tail, overlap, pipe, lazy), 'cpu')
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


def test_packed_prefill_first_sight_snapshot_serves_exact_repeat():
    """P5: with store_first_sight a cold prompt is snapshotted on its first visit, so its exact
    repeat restores from the cache (default policy: the repeat would only store it). Greedy
    outputs stay equal to the Transformers path."""
    long = [9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1]
    plan = [(0.0, 'a1', long, 6), (0.6, 'a2', list(long), 6)]
    ref, _ = run(False, plan, prefix=True)
    got, d = run(True, plan, prefix=True, packed=True, first_sight=True)
    assert got == ref
    pp = d['static_decode']['pool_prefix_cache']
    assert pp['stores'] == 1 and pp['hits'] >= 1, pp
    _, d0 = run(True, plan, prefix=True, packed=True)
    assert d0['static_decode']['pool_prefix_cache']['hits'] == 0, d0['static_decode']['pool_prefix_cache']


@pytest.mark.parametrize('spec', [False, True])
def test_packed_prefill_single_pass_snapshot_serves_exact_repeat(spec):
    """P5: with single_pass_snapshot the cold prompt's snapshot is recorded inside its one
    prefill (no prefix pass), its exact repeat restores from it, and greedy outputs stay equal
    to the Transformers path."""
    long = [9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1]
    plan = [(0.0, 'a1', long, 6), (0.6, 'a2', list(long), 6)]
    ref, _ = run(False, plan, prefix=True)
    got, d = run(True, plan, prefix=True, packed=True, first_sight=True, single=True, spec=spec)
    assert got == ref
    sd = d['static_decode']
    pp = sd['pool_prefix_cache']
    assert pp['stores'] == 1 and pp['hits'] >= 1, pp
    assert sd['packed_prefix_single_pass'] == 1 and sd['packed_prefix_groups'] == 0, sd


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


@pytest.mark.parametrize('spec', [False, True])
def test_batched_emission_keeps_streams_identical(spec):
    """P6: emit_batched delivers a step's tokens in one event-loop wakeup; every stream must be
    token-for-token what the per-token path gives (order, stop, length)."""
    shared = [9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8]
    plan = [(0.0, 'q1', shared + [1, 2], 9), (0.0, 'q2', [3, 3, 3], 5), (0.05, 'q3', shared + [7], 12),
            (0.1, 'q4', [5, 1], 3)]
    ref, _ = run(True, plan, packed=True, spec=spec, budget=64)
    got, _ = run(True, plan, packed=True, spec=spec, budget=64, emit=True)
    assert got == ref


@pytest.mark.parametrize('spec', [False, True])
def test_tail_fill_compaction_keeps_outputs(spec):
    """P6: with compact_from_tail, finished rows' slots are refilled from the tail (rows reorder).
    Every request's greedy output must still equal the Transformers path, with rows finishing
    at different steps so holes open in the middle of the batch."""
    shared = [9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8]
    plan = [(0.0, 'r1', shared + [1, 2], 2), (0.0, 'r2', [3, 3, 3], 9), (0.0, 'r3', shared + [7], 3),
            (0.0, 'r4', [5, 1, 4], 11), (0.15, 'r5', [2, 6, 2, 6], 4), (0.2, 'r6', shared[:9], 7)]
    ref, _ = run(False, plan)
    got, d = run(True, plan, packed=True, spec=spec, budget=64, tail=True)
    assert got == ref
    assert d['static_decode']['decoder'].get('row_moves', 0) >= 1, d['static_decode']['decoder']


@pytest.mark.parametrize('spec', [False, True])
@pytest.mark.parametrize('prefix', [False, True])
def test_overlap_prefill_keeps_outputs(spec, prefix):
    """P7: long prompts arriving while rows decode are prefilled on a second thread into the
    spare top slots, then moved down; outputs must stay greedy-equal to Transformers, including
    rows that finish while an overlap prefill is in flight and requests deferred past the spare
    slots."""
    shared = [9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8]
    plan = [(0.0, 'd1', [3, 3, 3], 14), (0.0, 'd2', [5, 1], 4),
            (0.05, 'l1', shared + [1, 2], 6), (0.05, 'l2', shared + [7, 7, 7], 5), (0.05, 'l3', shared[:12], 5),
            (0.3, 'l4', shared + [4], 3)]
    ref, _ = run(False, plan)
    got, d = run(True, plan, packed=True, spec=spec, budget=64, overlap=2, prefix=prefix)
    got3, _ = run(True, plan + [(0.06, 's1', [2, 2], 4), (0.07, 'l5', shared + [5, 5], 4)], packed=True, spec=spec,
                  budget=64, overlap=3, prefix=prefix)   # v2: a short prompt and another long one during a job
    ref3, _ = run(False, plan + [(0.06, 's1', [2, 2], 4), (0.07, 'l5', shared + [5, 5], 4)])
    assert got3 == ref3
    assert got == ref
    sd = d['static_decode']
    assert sd.get('overlap_jobs', 0) >= 1 and sd['slots'] == 6, sd


def test_packed_budget_overflow_is_requeued_not_sent_to_transformers():
    """P9: with static_decode_budget_new_tokens, rows that fit the lane but exceed one prefill's
    token budget are re-queued for the next packed prefill (not the Transformers path); outputs
    stay greedy-equal."""
    shared = [9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8]
    plan = [(0.0, f'b{i}', shared + [i + 1, 2], 4) for i in range(4)]
    ref, _ = run(False, plan)

    async def go():
        model = tiny_model(seed=11).to(torch.float32)
        c = cfg(True, prefix=True, packed=True, budget=40)
        c.native.static_decode_budget_new_tokens = True
        c.native.max_batch_tokens = 1 << 16
        e = QwenHybridContinuousEngine(model, Tok(), c, 'cpu')
        await e.start()

        async def one(delay, rid, prompt, max_new):
            await asyncio.sleep(delay)
            q = await e.submit(Request(rid, tuple(prompt), SamplingParams(max_new_tokens=max_new, temperature=0.0)))
            return rid, await drain(q)
        res = dict(await asyncio.gather(*(one(*p) for p in plan)))
        d = e.diagnostics()
        await e.close()
        return res, d
    got, d = asyncio.run(go())
    assert got == ref
    sd = d['static_decode']
    assert sd['packed_rows'] == 4 and sd['packed_prefills'] >= 2, sd   # split across prefills, all in the lane


def test_full_lane_queues_rows_instead_of_transformers_path():
    """P9b: with static_decode_wait_for_slot, requests beyond the slot pool wait for a free slot
    and all run in the lane (no Transformers side cohort); outputs stay greedy-equal."""
    plan = [(0.0, f'w{i}', [3 + i, 5, 7, 11, 2 + i], 6) for i in range(7)]   # 7 rows, 4 slots
    ref, _ = run(False, plan)

    async def go():
        model = tiny_model(seed=11).to(torch.float32)
        c = cfg(True, packed=True, budget=64)
        c.native.static_decode_wait_for_slot = True
        c.native.max_batch_tokens = 1 << 16
        e = QwenHybridContinuousEngine(model, Tok(), c, 'cpu')
        await e.start()

        async def one(delay, rid, prompt, max_new):
            await asyncio.sleep(delay)
            q = await e.submit(Request(rid, tuple(prompt), SamplingParams(max_new_tokens=max_new, temperature=0.0)))
            return rid, await drain(q)
        res = dict(await asyncio.gather(*(one(*p) for p in plan)))
        d = e.diagnostics()
        await e.close()
        return res, d
    got, d = asyncio.run(go())
    assert got == ref
    assert d['static_decode']['packed_rows'] == 7, d['static_decode']


def test_admission_pacing_keeps_outputs_and_batches_arrivals():
    """P9d: with static_decode_admit_every_steps, rows arriving while the lane decodes wait up to K
    decode steps and share a packed prefill; outputs stay greedy-equal."""
    plan = [(0.0, 'p0', [3, 5, 7, 11, 2], 12)] + [(0.01 * i, f'p{i}', [3 + i, 5, 7, 2 + i], 6) for i in range(1, 4)]
    ref, _ = run(False, plan)

    async def go():
        model = tiny_model(seed=11).to(torch.float32)
        c = cfg(True, packed=True, budget=64)
        c.native.static_decode_wait_for_slot = True
        c.native.static_decode_admit_every_steps = 4
        c.native.static_decode_admit_pace_min_rows = 1
        c.native.max_batch_tokens = 1 << 16
        e = QwenHybridContinuousEngine(model, Tok(), c, 'cpu')
        await e.start()

        async def one(delay, rid, prompt, max_new):
            await asyncio.sleep(delay)
            q = await e.submit(Request(rid, tuple(prompt), SamplingParams(max_new_tokens=max_new, temperature=0.0)))
            return rid, await drain(q)
        res = dict(await asyncio.gather(*(one(*p) for p in plan)))
        d = e.diagnostics()
        await e.close()
        return res, d
    got, d = asyncio.run(go())
    assert got == ref
    assert d['static_decode']['packed_rows'] == 4, d['static_decode']


@pytest.mark.parametrize('tail', [False, True])
def test_spec_pipeline_keeps_outputs(tail):
    """P9f: with static_decode_spec_pipeline, spec step N+1 is launched before step N's tokens are
    accepted, and compaction happens while it runs. Rows finish at different steps (holes, moves),
    others arrive mid-decode; every greedy output must equal the Transformers path."""
    shared = [9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8]
    plan = [(0.0, 'r1', shared + [1, 2], 2), (0.0, 'r2', [3, 3, 3], 9), (0.0, 'r3', shared + [7], 3),
            (0.0, 'r4', [5, 1, 4], 11), (0.15, 'r5', [2, 6, 2, 6], 4), (0.2, 'r6', shared[:9], 7),
            (0.25, 'r7', [4, 4], 1)]
    ref, _ = run(False, plan)
    got, d = run(True, plan, packed=True, spec=True, budget=64, tail=tail, pipe=True)
    assert got == ref
    sd = d['static_decode']
    assert sd['spec_pipelined_steps'] > 0 and sd['decoder'].get('row_moves', 0) >= 1, sd


@pytest.mark.parametrize('pipe', [False, True])
def test_spec_lazy_bank_keeps_outputs(pipe):
    """P10: with static_decode_spec_lazy_bank the GDN verify writes only token 2's state and
    rejected rows rebuild token 1's (the random MTP head rejects most drafts, so the fix-up path
    runs on most rows); greedy outputs must equal the Transformers path, with row moves."""
    shared = [9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8]
    plan = [(0.0, 'r1', shared + [1, 2], 2), (0.0, 'r2', [3, 3, 3], 9), (0.0, 'r3', shared + [7], 3),
            (0.0, 'r4', [5, 1, 4], 11), (0.15, 'r5', [2, 6, 2, 6], 4), (0.2, 'r6', shared[:9], 7)]
    ref, _ = run(False, plan)
    got, d = run(True, plan, packed=True, spec=True, budget=64, tail=True, pipe=pipe, lazy=True)
    assert got == ref
    assert d['static_decode']['spec_steps'] > 0
