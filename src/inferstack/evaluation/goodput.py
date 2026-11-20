"""Open-loop goodput (G1): the harness's serving objective, engine-neutral.

For each workload case, requests arrive as a Poisson process at increasing rates. A request is good when it
succeeds and meets every SLO at once (vLLM `bench serve --goodput` rule): client TTFT <= the case's TTFT limit
and per-token time (e2e - TTFT) / (tokens - 1) <= the case's per-token limit. goodput@99 is the highest
swept rate with >= 99% good requests (MLPerf's percentile), goodput@90 the same at 90%. A case's sweep
stops after the first rate below `stop_below` good.

Per-token time needs the engine's usage count (requested with `stream_options.include_usage`); a request
without it cannot be shown to meet the per-token limit and counts as not good, and the sweep reports
`token_counts_missing` so an engine that never reports usage reads as unmeasured, not as slow.

Works on anything with `stream(payload)` yielding SSE bytes (ManagedEngine): InferStack, vLLM, SGLang.
Prompts are deterministic per (seed, case, rate step) and unique per request, except `shared_prefix`
cases, whose long prefix is shared by design. Ideas from scripts/goodput.py (P9); no rival code.
"""
from __future__ import annotations

import asyncio
import hashlib
import random
import time
from dataclasses import asdict, dataclass

from inferstack.evaluation.metrics import StreamMeasurement

DEFAULT_RATES = (0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24)
_WORDS = ('system latency token cache kernel memory batch request model engine queue throughput window state '
          'vector matrix decode prefill attention layer graph stream budget policy signal sample bound limit').split()


@dataclass(frozen=True)
class GoodputCase:
    name: str
    prompt_words: int
    max_tokens: int
    ttft_ms: float            # TTFT limit (the SLO's p99 class limit)
    tpot_ms: float            # per-token limit
    shared_prefix: bool = False


def cases_from_slo(workload_cases, slo):
    """GoodputCase list from scorecard-style workload cases and an SLO file's `industry_limits`."""
    out = []
    for c in workload_cases:
        lim = slo['industry_limits'][c['name']]
        out.append(GoodputCase(c['name'], int(c['prompt_words']), int(c['max_tokens']),
                               float(lim.get('ttft_p99_ms', lim['ttft_p95_ms'])), float(lim['client_decode_p95_ms']),
                               bool(c.get('shared_prefix'))))
    return out


def prompts(case: GoodputCase, n: int, seed: int, step: int):
    def text(k, rng):
        return ' '.join(rng.choice(_WORDS) for _ in range(k))
    if case.shared_prefix:
        prefix = text(case.prompt_words, random.Random(f'{seed}:{case.name}'))
        return [f'{prefix}\nQuestion {step}.{i}: summarize the above in one sentence.' for i in range(n)]
    return [f'Request {step}.{i}. ' + text(case.prompt_words, random.Random(f'{seed}:{case.name}:{step}:{i}'))
            for i in range(n)]


def miss(row, case: GoodputCase):
    """Why a request is not good: 'error', 'ttft', 'tpot', 'no_tokens', or None when it is good (G4: the
    first failing SLO is what a planner needs to know about the knee)."""
    if not row.get('success') or row.get('ttft_ms') is None:
        return 'error'
    if row['ttft_ms'] > case.ttft_ms:
        return 'ttft'
    tokens = row.get('output_tokens')
    if tokens is None:
        return 'no_tokens'
    if tokens > 1 and (row['e2e_ms'] - row['ttft_ms']) / (tokens - 1) > case.tpot_ms:
        return 'tpot'
    return None


def is_good(row, case: GoodputCase) -> bool:
    return miss(row, case) is None


def summarize(steps):
    def best(th):
        ok = [s['rate'] for s in steps if s['good_frac'] >= th]
        return max(ok) if ok else 0.0
    return {'goodput99': best(0.99), 'goodput90': best(0.90)}


async def one(stream, payload, timeout):
    m = StreamMeasurement(time.perf_counter())
    error, buffer = None, b''
    gen = stream(payload)
    try:
        async with asyncio.timeout(timeout):
            async for chunk in gen:
                buffer += chunk.replace(b'\r\n', b'\n')
                *events, buffer = buffer.split(b'\n\n')
                for event in events:
                    data = [line[5:].strip() for line in event.split(b'\n') if line.startswith(b'data:')]
                    if data:
                        m.feed(b'\n'.join(data).decode(), time.perf_counter())
                if m.done:
                    break
    except Exception as exc:  # noqa: BLE001 - recorded per request
        error = f'{type(exc).__name__}: {exc}'[:200]
    finally:
        await gen.aclose()
    r = m.result(time.perf_counter(), error)
    return {k: r[k] for k in ('success', 'error', 'ttft_ms', 'e2e_ms', 'output_tokens')}


async def run_rate(stream, texts, max_tokens, rate, seed, timeout):
    rng = random.Random(seed)
    t, delays = 0.0, []
    for _ in texts:
        t += rng.expovariate(rate)
        delays.append(t)

    async def fire(text, delay):
        await asyncio.sleep(delay)
        return await one(stream, {'prompt': text, 'max_tokens': max_tokens, 'temperature': 0,
                                  'stream_options': {'include_usage': True}}, timeout)
    start = time.perf_counter()
    rows = await asyncio.gather(*(fire(x, d) for x, d in zip(texts, delays)))
    return rows, time.perf_counter() - start


def _pct(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


async def _read_probe(probe):
    if probe is None:
        return None
    try:
        return await probe()
    except Exception:  # noqa: BLE001 - counters are diagnostic; a failed read never fails the sweep
        return None


async def sweep(stream, cases, *, rates=DEFAULT_RATES, seconds_per_rate=20.0, min_requests=20, seed=20261119,
                timeout=300.0, stop_below=0.5, warmup=2, probe=None):
    """Per case: rate steps and goodput@99/@90. Returns {'cases': {...}, 'goodput99_rps': sum over cases}.

    Each step records why requests missed (`misses`: ttft/tpot/error/no_tokens counts). `probe` (G4) is an
    optional async () -> {counter: cumulative number} read before and after every step (engine counters such
    as the native /v1/native/status TTFT split); the step stores the difference as `counters`."""
    out = {}
    for case in cases:
        if warmup:
            await run_rate(stream, prompts(case, warmup, seed, -1), case.max_tokens, 1.0, 0, timeout)
        steps = []
        for k, rate in enumerate(rates):
            n = max(min_requests, int(rate * seconds_per_rate))
            before = await _read_probe(probe)
            rows, wall = await run_rate(stream, prompts(case, n, seed, k), case.max_tokens, rate,
                                        1000 * k + sum(map(ord, case.name)), timeout)
            after = await _read_probe(probe)
            reasons = [miss(r, case) for r in rows]
            good = reasons.count(None)
            step = {'rate': rate, 'requests': n, 'good': good, 'good_frac': good / n, 'wall_s': wall,
                    'success': sum(bool(r['success']) for r in rows) / n,
                    'token_counts_missing': sum(r['success'] and r['output_tokens'] is None for r in rows),
                    'misses': {m: reasons.count(m) for m in ('ttft', 'tpot', 'error', 'no_tokens') if reasons.count(m)},
                    'ttft_ms_p50': _pct([r['ttft_ms'] for r in rows if r.get('ttft_ms') is not None], 0.5)}
            if before is not None and after is not None:
                step['counters'] = {k: after[k] - before.get(k, 0) for k in after
                                    if isinstance(after[k], (int, float)) and not isinstance(after[k], bool)}
            steps.append(step)
            if good / n < stop_below:
                break
        out[case.name] = {'case': asdict(case), 'steps': steps, **summarize(steps)}
    missing = sum(s['token_counts_missing'] for c in out.values() for s in c['steps'])
    return {'cases': out, 'goodput99_rps': sum(c['goodput99'] for c in out.values()),
            'goodput90_rps': sum(c['goodput90'] for c in out.values()), 'token_counts_missing': missing,
            'fixture_sha256': hashlib.sha256(repr((seed, [asdict(c) for c in cases], tuple(rates),
                                                   seconds_per_rate, min_requests)).encode()).hexdigest()}
