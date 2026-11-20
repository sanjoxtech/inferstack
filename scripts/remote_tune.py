#!/usr/bin/env python
"""G5: settings tuning on the GPU host (slo_tune v2). Writes tune.json; the controller judges and confirms.

Steps, all on the host where latency is measured:
  1. probe the GPU (memory, SMs) and derive the knob space from it and the measured bottleneck (--target);
  2. run the baseline profile once (fresh server, greedy check, open-loop goodput sweep) as the reference;
  3. for each trial: the sampler proposes a patch; if the crash region predicts a failure it is skipped
     (recorded as pruned, no GPU time); otherwise the patched profile runs like the baseline. A server
     that fails to start grows the crash region; greedy outputs that disagree with the baseline make the
     trial infeasible.

The pick is only a proposal: optimization/tune.py:TuneAction confirms it with an interleaved A/B.

  python scripts/remote_tune.py --profiles configs/scorecard/a100-qwen3.5-2b.json --baseline native_lazy \\
      --target capacity gemm gdn --trials 8 --cases interactive --rates 4 8 12 16 --out OUT/tune
"""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / 'src'))
import remote_ab  # noqa: E402

from inferstack.evaluation.goodput import GoodputCase, cases_from_slo  # noqa: E402
from inferstack.optimization.tune import (CrashRegion, Sampler, agreement, objective, probe_hardware,  # noqa: E402
                                          score, space_for)

MIN_AGREEMENT = 0.75


async def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    profiles = json.loads(Path(a.profiles).read_text())['profiles']
    workload = json.loads(Path(a.workload).read_text())
    cases = [c for c in workload['cases'] if not a.cases or c['name'] in a.cases]
    cases = cases_from_slo(cases, json.loads(Path(a.slo).read_text())) if a.slo else [GoodputCase(**c) for c in cases]
    sweep_args = {'cases': cases, 'rates': tuple(a.rates), 'seconds_per_rate': a.seconds_per_rate,
                  'min_requests': a.min_requests, 'warmup': a.warmup}
    hardware = json.loads(a.hardware) if a.hardware else probe_hardware()
    space, why = space_for(hardware, target=a.target or (), knobs=a.knobs)
    sampler, region = Sampler(space, a.seed), CrashRegion()
    result = {'version': 1, 'host': platform.node(), 'hardware': hardware, 'target': a.target or [],
              'space': space, 'why': why, 'method': sampler.method, 'baseline_profile': a.baseline,
              'sweep': {**sweep_args, 'cases': [c.__dict__ for c in cases]}, 'trials': []}
    print(json.dumps({'hardware': hardware, 'space': space}), flush=True)

    def save():
        result['crash_region'] = region.crashes
        (out / 'tune.json').write_text(json.dumps(result, indent=2, default=str))

    async def measure(i, patch):
        workdir = out / f't{i:03d}'
        workdir.mkdir(exist_ok=True)
        prof = remote_ab.side_profile(profiles, a.baseline, patch or None, workdir)
        return await remote_ab.trial(prof, f't{i:03d}', 0, sweep_args, workdir, a.ready_timeout)

    base_row = await measure(0, {})
    reference = base_row.get('check')
    base = {'trial': 0, 'baseline': True, 'patch': {}, 'ready': base_row.get('ready'),
            'score': score(base_row.get('goodput'), a.rates), 'check': reference, 'feasible': True,
            'error': base_row.get('error')}
    result['trials'].append(base)
    save()
    if not base_row.get('ready') or base['score'] is None:
        result['error'] = 'baseline did not start or was unmeasured; nothing to tune against'
        save()
        return
    seen, asks = set(), 0
    i = 1
    while i <= a.trials and asks < 4 * a.trials and space:
        asks += 1
        patch = sampler.ask()
        key = json.dumps(patch, sort_keys=True)
        if key in seen:
            sampler.tell(objective(next(t['score'] for t in result['trials'] if
                                        json.dumps(t['patch'], sort_keys=True) == key) or None))
            continue
        seen.add(key)
        hit = region.covers(patch)
        if hit is not None:
            sampler.tell(-1.0)
            result['trials'].append({'trial': i, 'patch': patch, 'pruned': True, 'feasible': False,
                                     'predicted_by': hit, 'score': None})
            save()
            continue
        row = await measure(i, patch)
        s = score(row.get('goodput'), a.rates)
        crashed = not row.get('ready')
        agree = agreement(reference, row.get('check'))
        feasible = not crashed and s is not None and agree >= MIN_AGREEMENT
        if crashed:
            region.add(patch)
        sampler.tell(objective(s) if feasible else -1.0)
        result['trials'].append({'trial': i, 'patch': patch, 'ready': row.get('ready'), 'crashed': crashed,
                                 'error': row.get('error'), 'score': s, 'agreement': round(agree, 2),
                                 'feasible': feasible, 'knee': (row.get('goodput') or {}).get('goodput99_rps')})
        print(json.dumps({'trial': i, 'patch': patch, 'score': s, 'feasible': feasible}), flush=True)
        save()
        i += 1
    result['commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                                      cwd=ROOT).stdout.strip() or None
    result['gpu'] = remote_ab.gpu_name()
    save()
    print('tune.json written:', out / 'tune.json', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profiles', required=True)
    ap.add_argument('--baseline', required=True, help='native profile name to tune from')
    ap.add_argument('--workload', default=str(ROOT / 'configs/scorecard/workload-fast.json'))
    ap.add_argument('--cases', nargs='*')
    ap.add_argument('--slo', default=str(ROOT / 'configs/tune/slo-v2-l4.json'))
    ap.add_argument('--target', nargs='*', help='measured bottleneck labels / kernel families (G4)')
    ap.add_argument('--knobs', nargs='*', help='restrict to these knobs')
    ap.add_argument('--hardware', help='JSON hardware facts instead of probing (tests)')
    ap.add_argument('--trials', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rates', type=float, nargs='+', default=[1, 2, 4, 6, 8, 12, 16])
    ap.add_argument('--seconds-per-rate', type=float, default=20.0)
    ap.add_argument('--min-requests', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--ready-timeout', type=float, default=1200)
    ap.add_argument('--out', required=True)
    asyncio.run(run(ap.parse_args()))


if __name__ == '__main__':
    main()
