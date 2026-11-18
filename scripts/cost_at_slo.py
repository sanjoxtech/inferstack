#!/usr/bin/env python
"""P9 (closed-loop version): $ per 1,000 requests at the SLO, per engine and workload case.

For each profile and case, take the highest-concurrency cell whose medians meet the SLO's
latency limits (TTFT p95/p99, per-token p95; the industry layer only, without the "beat the
rival" floors, so every engine is judged by the same bar) and 100% success. Its throughput
gives the cost:

    requests/s      = output tok/s / mean output tokens per request
    $ / 1k requests = GPU $/h / 3600 / (requests/s) * 1000
    $ / 1M tokens   = GPU $/h / 3600 / (output tok/s) * 1e6

This is a closed-loop approximation (fixed concurrency 1/4/8/16), not open-loop goodput
(Poisson arrivals); a case with no cell meeting the SLO gets no price. The GPU price is an input
(default: a public on-demand L4 list price, see --gpu-price-per-hour), so the table shows relative
cost, not a bill.

  python scripts/cost_at_slo.py SCORECARD.json --slo configs/tune/slo-v2-l4.json --gpu-price-per-hour 0.80
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def meets(metrics: dict, limits: dict, min_success: float = 1.0) -> bool:
    med = lambda k: (metrics.get(k) or {}).get('median')   # noqa: E731
    if (med('success_rate') or 0) < min_success:
        return False
    return all(med(m) is not None and med(m) <= float(v) for m, v in limits.items())


def industry_limits(slo: dict, case: str, conc: int) -> dict:
    """Per-cell latency limits from the SLO's industry layer. In slo-v2 the per-cell ttft_p95
    may be tightened to the best rival; rivals are then judged against themselves, so use the
    class limit stored under 'industry_limits' when present, else the cell limits."""
    if 'industry_limits' in slo:
        return dict(slo['industry_limits'][case])
    return {**(slo.get('limits') or {}), **((slo.get('cell_limits') or {}).get(f'{case}@{conc}') or {})}


def table(card: dict, slo: dict, price: float):
    rows = []
    cells = {}
    for c in card.get('cells', []):
        cells.setdefault((c['profile'], c['case']), []).append(c)
    for (prof, case), cs in sorted(cells.items()):
        ok = [c for c in cs if meets(c['metrics'], industry_limits(slo, case, c['concurrency']),
                                     slo.get('min_success_rate', 1.0))]
        if not ok:
            rows.append({'profile': prof, 'case': case, 'concurrency': None})
            continue
        best = max(ok, key=lambda c: c['concurrency'])
        tps = best['metrics']['output_tps']['median']
        mot = best['metrics']['mean_output_tokens']['median']
        rps = tps / mot
        rows.append({'profile': prof, 'case': case, 'concurrency': best['concurrency'], 'output_tps': tps,
                     'requests_per_s': rps, 'usd_per_1k_requests': price / 3600 / rps * 1000,
                     'usd_per_1m_tokens': price / 3600 / tps * 1e6})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('scorecard')
    ap.add_argument('--slo', default='configs/tune/slo-v2-l4.json')
    ap.add_argument('--gpu-price-per-hour', type=float, default=0.80,
                    help='GPU $/h used for the cost (default 0.80, roughly a public on-demand L4 price)')
    ap.add_argument('--out')
    a = ap.parse_args()
    card = json.loads(Path(a.scorecard).read_text())
    slo = json.loads(Path(a.slo).read_text())
    rows = table(card, slo, a.gpu_price_per_hour)
    lines = [f'# Cost at SLO (closed loop), GPU ${a.gpu_price_per_hour:.2f}/h', '',
             '| case | profile | highest conc. meeting SLO | out tok/s | req/s | $ / 1k requests | $ / 1M tokens |',
             '|---|---|---|---|---|---|---|']
    for r in rows:
        if r['concurrency'] is None:
            lines.append(f"| {r['case']} | {r['profile']} | none | | | | |")
        else:
            lines.append(f"| {r['case']} | {r['profile']} | {r['concurrency']} | {r['output_tps']:.0f} | "
                         f"{r['requests_per_s']:.2f} | {r['usd_per_1k_requests']:.4f} | {r['usd_per_1m_tokens']:.3f} |")
    text = '\n'.join(lines) + '\n'
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == '__main__':
    main()
