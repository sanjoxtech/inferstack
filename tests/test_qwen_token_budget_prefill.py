"""N3.1 token-budget prefill on a tiny real Qwen3.5 model (CPU, eager).

Oracle: the same engine with the fixed one-chunk-per-turn policy. Greedy outputs must be
identical; the budget policy must need fewer prefill forwards, keep prefix-cache block
snapshots usable, and switch to the smaller budget while requests are decoding.
"""
import asyncio

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers.models.qwen3_5.modeling_qwen3_5')

from inferstack.core.request import Request, SamplingParams  # noqa: E402
from inferstack.runtime.qwen_continuous import PrefillJob, QwenHybridContinuousEngine  # noqa: E402
from test_qwen_static_decode import tiny_model  # noqa: E402
from test_qwen_static_lane import PLAN, Tok, cfg, drain  # noqa: E402


def budget_cfg(idle, busy, static=False, prefix=False):
    c = cfg(static, prefix)
    c.native.prefill_token_budget_idle = idle
    c.native.prefill_token_budget_busy = busy
    return c


def run(c, plan):
    async def go():
        e = QwenHybridContinuousEngine(tiny_model(seed=11).to(torch.float32), Tok(), c, 'cpu')
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


def job(rows, width, prefix_len=0, cursor=0):
    ids = torch.zeros((rows, width), dtype=torch.long)
    return PrefillJob([object()] * rows, [None] * rows, ids, None, None, [width] * rows, width, 4, prefix_len, cursor)


def engine(c):
    return QwenHybridContinuousEngine(tiny_model(seed=11).to(torch.float32), Tok(), c, 'cpu')


def test_prefill_end_fixed_policy_is_one_chunk():
    e = engine(budget_cfg(0, 0))
    assert e._prefill_budget() == 0
    assert e._prefill_end(job(1, 37), 0) == 8 and e._prefill_end(job(1, 37, cursor=32), 0) == 37


def test_prefill_end_budget_divides_by_rows_in_whole_chunks():
    e = engine(budget_cfg(64, 16))
    assert e._prefill_end(job(1, 100), 64) == 64        # 64 tokens / 1 row
    assert e._prefill_end(job(2, 100), 64) == 32        # 64 tokens / 2 rows
    assert e._prefill_end(job(3, 100), 64) == 16        # 21 rounded down to whole 8-token chunks
    assert e._prefill_end(job(16, 100), 64) == 8        # never below one chunk
    assert e._prefill_end(job(2, 20), 64) == 20         # clipped to the prompt


def test_prefill_end_keeps_prefix_cache_block_boundaries():
    e = engine(budget_cfg(1000, 1000, prefix=True))     # block = 8 tokens
    assert e._prefill_end(job(1, 37), 1000) == 32       # stop at the last full block
    assert e._prefill_end(job(1, 37, cursor=32), 1000) == 37
    assert e._prefill_end(job(1, 37), 20) == 16         # partial span ends on a block
    # A prefix-cache hit shifts logical positions: 5 cached tokens + 30 new = 35 → block 32.
    assert e._prefill_end(job(1, 30, prefix_len=5), 1000) == 27
    assert e._prefill_end(job(2, 37), 1000) == 37       # multi-row jobs store no snapshots


def test_budget_switches_to_busy_while_decoding():
    e = engine(budget_cfg(512, 32))
    assert e._prefill_budget() == 512
    e._active.append(object())
    assert e._prefill_budget() == 32
    e = engine(budget_cfg(512, 0))
    e._active.append(object())
    assert e._prefill_budget() == 512                   # busy 0 falls back to idle


@pytest.mark.parametrize('static', [False, True])
def test_budget_prefill_matches_fixed_chunks(static):
    ref, dref = run(budget_cfg(0, 0, static), PLAN)
    got, d = run(budget_cfg(64, 16, static), PLAN)
    assert got == ref
    assert d['prefill_forward_calls'] < dref['prefill_forward_calls'], (d['prefill_forward_calls'], dref['prefill_forward_calls'])


def test_budget_prefill_keeps_prefix_cache_hits():
    shared = [9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 1]
    plan = [(0.0, 'p1', shared + [1, 2, 3], 6), (0.3, 'p2', shared + [3, 3], 6)]
    ref, dref = run(budget_cfg(0, 0, prefix=True), plan)
    got, d = run(budget_cfg(256, 64, prefix=True), plan)
    assert got == ref
    assert d['prefix_state_cache']['hits'] >= 1
    assert d['prefix_state_cache']['hits'] == dref['prefix_state_cache']['hits']


def test_backlog_raises_busy_budget_up_to_idle():
    c = budget_cfg(512, 32)
    c.native.prefill_backlog_drain_steps = 4
    e = engine(c)
    e._active.append(object())
    assert e._prefill_budget() == 32                    # empty backlog: plain busy budget
    e._prefill_jobs.append(job(2, 100))                 # 200 queued tokens / 4 steps = 50
    assert e._prefill_budget() == 50
    e._prefill_jobs.append(job(4, 200, cursor=40))      # + 4 x 160 = 840 tokens / 4 = 210
    assert e._prefill_budget() == 210
    e._prefill_jobs.append(job(16, 200))                # 4040 / 4 = 1010, capped at idle
    assert e._prefill_budget() == 512
    e._active.clear()
    assert e._prefill_budget() == 512                   # idle GPU: idle budget as before


def test_backlog_off_keeps_fixed_busy_budget():
    e = engine(budget_cfg(512, 32))
    e._active.append(object())
    e._prefill_jobs.append(job(16, 200))
    assert e._prefill_budget() == 32


@pytest.mark.parametrize('static', [False, True])
def test_backlog_budget_matches_fixed_chunks(static):
    c = budget_cfg(64, 8, static)
    c.native.prefill_backlog_drain_steps = 2
    ref, _ = run(budget_cfg(0, 0, static), PLAN)
    got, d = run(c, PLAN)
    assert got == ref
