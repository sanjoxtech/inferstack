#!/usr/bin/env python
"""P9: open-loop goodput per engine and workload class (Poisson arrivals), and cost at that goodput.

A request is "good" when its TTFT and its per-token time both meet the class limits (vLLM
`bench serve --goodput` rule: every SLO at once). For each engine and case, requests arrive as a
Poisson process at increasing rates; goodput@99 is the highest rate with >= 99% good requests
(MLPerf's percentile), goodput@90 the same at 90% (DistServe / Mooncake convention).
Cost at goodput = GPU $/h / 3600 / goodput * 1000  ($ per 1,000 requests).

Prompts are unique per rate step (fresh seeds), except the shared_prefix case, whose long prefix is
shared by design, so caches only help where the workload says they should. The sweep for a case stops
after the first rate below 50% good.

  python scripts/goodput.py --profiles configs/scorecard/a100-qwen3.5-2b.json --only native_bb4k vllm sglang \\
      --slo configs/tune/slo-v2-l4.json --out OUT/goodput
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import engine_scorecard as es  # noqa: E402

DEFAULT_RATES = [0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24]


def limits_for(slo, case):
    lim = slo['industry_limits'][case]
    return {'ttft_ms': float(lim.get('ttft_p99_ms', lim['ttft_p95_ms'])),
            'tpot_ms': float(lim['client_decode_p95_ms'])}


def is_good(row, lim):
    return bool(row.get('success') and row.get('ttft_ms') is not None and row['ttft_ms'] <= lim['ttft_ms']
                and (row.get('client_decode_ms') is None or row['client_decode_ms'] <= lim['tpot_ms']))


def summarize(steps):
    """steps: [{'rate', 'good_frac', ...}] in rate order -> goodput at 99% and 90%."""
    def best(th):
        ok = [s['rate'] for s in steps if s['good_frac'] >= th]
        return max(ok) if ok else 0.0
    return {'goodput99': best(0.99), 'goodput90': best(0.90)}


async def run_rate(base, model, texts, max_tokens, rate, timeout, seed):
    rng = random.Random(seed)
    async with es.httpx.AsyncClient(limits=es.httpx.Limits(max_connections=len(texts) + 4)) as client:
        async def fire(i, text, delay):
            await asyncio.sleep(delay)
            return {'i': i, **await es.one(client, base, model, text, max_tokens, 0.0, timeout)}
        t, delays = 0.0, []
        for _ in texts:
            t += rng.expovariate(rate)
            delays.append(t)
        start = time.perf_counter()
        rows = await asyncio.gather(*(fire(i, x, d) for i, (x, d) in enumerate(zip(texts, delays))))
        wall = time.perf_counter() - start
    return rows, wall


async def run_profile(profile, workload, slo, rates, per_rate, min_requests, out_dir, ready_timeout, timeout):
    out_dir.mkdir(parents=True, exist_ok=True)
    port = es.free_port()
    base = f'http://127.0.0.1:{port}'
    proc, log, argv = es.launch(profile, port, out_dir / 'server.log')
    res = {'profile': profile['name'], 'argv': argv, 'cases': {}}
    try:
        ready, detail = await es.wait_ready(base, profile['model'], proc, ready_timeout)
        res['ready'] = ready
        if not ready:
            res['error'] = str(detail)[:500]
            return res
        with (out_dir / 'requests.jsonl').open('w') as fout:
            for case in workload['cases']:
                lim = limits_for(slo, case['name'])
                warm = es.prompts({**case, 'name': case['name'] + ':warmup'}, 2, workload['seed'])
                await run_rate(base, profile['model'], warm, case['max_tokens'], 1.0, timeout, 0)
                steps = []
                for k, rate in enumerate(rates):
                    n = max(min_requests, int(rate * per_rate))
                    texts = es.prompts({**case, 'name': f"{case['name']}:r{k}"}, n, workload['seed'])
                    before = await es.ttft_breakdown(base)
                    rows, wall = await run_rate(base, profile['model'], texts, case['max_tokens'], rate, timeout,
                                                seed=1000 * k + sum(map(ord, case['name'])))
                    good = sum(is_good(r, lim) for r in rows)
                    step = {'rate': rate, 'requests': n, 'good': good, 'good_frac': good / n,
                            'success': sum(bool(r.get('success')) for r in rows) / n, 'wall_s': wall,
                            'ttft_p50_ms': es.pct([r['ttft_ms'] for r in rows if r.get('ttft_ms') is not None], .5),
                            'ttft_p99_ms': es.pct([r['ttft_ms'] for r in rows if r.get('ttft_ms') is not None], .99)}
                    lane = es.breakdown_delta(before, await es.ttft_breakdown(base))   # native only, else None
                    if lane:
                        step['engine'] = lane
                    steps.append(step)
                    for r in rows:
                        fout.write(json.dumps({'profile': profile['name'], 'case': case['name'], 'rate': rate, **r}) + '\n')
                    print(json.dumps({'profile': profile['name'], 'case': case['name'], **step}), flush=True)
                    if step['good_frac'] < 0.5:
                        break
                res['cases'][case['name']] = {'limits': lim, 'steps': steps, **summarize(steps)}
    finally:
        res['exit_code'] = es.stop(proc)
        log.close()
    return res


def report(results, price):
    lines = [f'# Open-loop goodput and cost (Poisson arrivals), GPU ${price:.2f}/h', '',
             '| case | profile | goodput@99 req/s | $ / 1k req @99 | goodput@90 req/s | $ / 1k req @90 |',
             '|---|---|---|---|---|---|']
    cost = lambda g: f'{price / 3600 / g * 1000:.4f}' if g else 'n/a'   # noqa: E731
    for r in results:
        for case, c in r.get('cases', {}).items():
            lines.append(f"| {case} | {r['profile']} | {c['goodput99']:g} | {cost(c['goodput99'])} | "
                         f"{c['goodput90']:g} | {cost(c['goodput90'])} |")
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profiles', default='configs/scorecard/a100-qwen3.5-2b.json')
    ap.add_argument('--only', nargs='+', required=True)
    ap.add_argument('--workload', help='workload JSON (default: the scorecard default workload)')
    ap.add_argument('--slo', default='configs/tune/slo-v2-l4.json')
    ap.add_argument('--rates', type=float, nargs='*', default=DEFAULT_RATES)
    ap.add_argument('--seconds-per-rate', type=float, default=20.0, help='requests per step = rate x this')
    ap.add_argument('--min-requests', type=int, default=20)
    ap.add_argument('--gpu-price-per-hour', type=float, default=0.80)
    ap.add_argument('--ready-timeout', type=float, default=1200)
    ap.add_argument('--request-timeout', type=float, default=300)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.profiles).read_text())
    workload = json.loads(Path(a.workload).read_text()) if a.workload else es.DEFAULT_WORKLOAD
    slo = json.loads(Path(a.slo).read_text())
    out = Path(a.out)
    results = []
    for name in a.only:
        prof = next(p for p in cfg['profiles'] if p['name'] == name)
        results.append(asyncio.run(run_profile(prof, workload, slo, a.rates, a.seconds_per_rate, a.min_requests,
                                               out / name, a.ready_timeout, a.request_timeout)))
        (out / 'goodput.json').write_text(json.dumps(results, indent=2, default=str))
    text = report(results, a.gpu_price_per_hour)
    (out / 'goodput.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
