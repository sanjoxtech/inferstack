#!/usr/bin/env python
"""Phase 2: SLO auto-tuner. Search engine settings for the best throughput that meets an SLO.

Each trial writes a profile (base profile + knob overrides), serves it through the engine
scorecard for one round of the chosen cells, and scores it:

    objective   : geometric mean of output tok/s over the SLO's cells (maximised)
    constraints : for every cell and limit, metric - limit (feasible when all <= 0)

The sampler is Optuna TPE with constraints when optuna is installed (seeded, reproducible),
otherwise seeded random search. A trial whose server fails to start or whose requests fail is
recorded as infeasible with a large violation instead of being dropped: on a new GPU or model
the crash region is information (SLO-Guard, arXiv 2604.17627, ideas only).

Outputs in --out: trials.jsonl (one row per trial), best.toml (best feasible profile),
summary.md. --confirm N reruns base vs best for N rounds with the normal scorecard so the pick
is judged by the usual non-overlapping-range rule, not by one noisy round.

Example (on the GPU host):
  python scripts/slo_tune.py --base configs/profiles/production-v1.20.0.toml \\
      --space configs/tune/space-l4.json --slo configs/tune/slo-default.json \\
      --workload configs/scorecard/workload-fast.json --trials 12 --out results/tune-l4
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import tomllib
from pathlib import Path

INFEASIBLE = 1e6          # violation recorded for a trial that crashed or lost requests
NATIVE_ARGV = ['inferstack', 'serve', '--profile', '{profile}', '--foundry-root', 'examples/model_foundry',
               '--model', 'foundry://factorylm-2b@candidate', '--device', 'cuda:0', '--host', '127.0.0.1',
               '--port', '{port}']


# ---- profile writing ------------------------------------------------------------------------
def _fmt(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return '[' + ', '.join(_fmt(x) for x in v) + ']'
    raise TypeError(f'cannot write {type(v).__name__} to TOML')


def dump_toml(data: dict) -> str:
    """Minimal TOML writer for profiles: top-level scalars, then one table per section."""
    lines = [f'{k} = {_fmt(v)}' for k, v in data.items() if not isinstance(v, dict)]
    for sec, body in data.items():
        if isinstance(body, dict):
            lines += ['', f'[{sec}]'] + [f'{k} = {_fmt(v)}' for k, v in body.items()]
    return '\n'.join(lines) + '\n'


def apply_overrides(base: dict, overrides: dict, name: str) -> dict:
    """overrides: {'native.max_requests_per_batch': 4, ...} (section.key); unknown sections fail."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    out['name'] = name
    for dotted, value in overrides.items():
        sec, _, key = dotted.partition('.')
        if not key or not isinstance(out.get(sec), dict):
            raise KeyError(f'override {dotted!r}: no [{sec}] section in the base profile')
        out[sec][key] = value
    return out


# ---- search space ---------------------------------------------------------------------------
def suggest(space, trial=None, rng=None):
    """space: [{'name': 'native.x', 'choices': [...]} | {'name', 'low', 'high', 'int': bool, 'log': bool}]."""
    pick = {}
    for p in space:
        n = p['name']
        if 'choices' in p:
            pick[n] = trial.suggest_categorical(n, p['choices']) if trial else rng.choice(p['choices'])
        elif p.get('int'):
            pick[n] = trial.suggest_int(n, p['low'], p['high'], log=p.get('log', False)) if trial else \
                rng.randint(p['low'], p['high'])
        else:
            pick[n] = trial.suggest_float(n, p['low'], p['high'], log=p.get('log', False)) if trial else \
                rng.uniform(p['low'], p['high'])
    return pick


# ---- scoring --------------------------------------------------------------------------------
def score(card: dict, profile: str, slo: dict):
    """(objective, violations, detail) for one profile in an aggregated scorecard.
    slo: {'cells': [[case, conc], ...], 'objective': 'output_tps',
          'limits': {'ttft_p95_ms': 400, ...},              # every cell
          'cell_limits': {'prefill_heavy@16': {'ttft_p95_ms': 600}},   # per cell, overrides limits
          'min_success_rate': 1.0}"""
    want = {(c, int(n)) for c, n in slo['cells']}
    got = {(c['case'], c['concurrency']): c['metrics'] for c in card.get('cells', []) if c['profile'] == profile}
    detail, objs, viol = {}, [], []
    for key in sorted(want):
        m = got.get(key)
        if m is None:
            viol.append(INFEASIBLE)
            detail[f'{key[0]}@{key[1]}'] = 'missing'
            continue
        val = lambda k: (m.get(k) or {}).get('median')   # noqa: E731
        sr = val('success_rate')
        tps = val(slo.get('objective', 'output_tps'))
        d = {'output_tps': tps, 'success_rate': sr}
        if sr is None or sr < slo.get('min_success_rate', 1.0) or not tps:
            viol.append(INFEASIBLE)
        else:
            objs.append(tps)
        lim = {**(slo.get('limits') or {}), **((slo.get('cell_limits') or {}).get(f'{key[0]}@{key[1]}') or {})}
        for metric, limit in lim.items():
            v = val(metric)
            d[metric] = v
            viol.append(INFEASIBLE if v is None else v - float(limit))
        detail[f'{key[0]}@{key[1]}'] = d
    obj = math.exp(sum(map(math.log, objs)) / len(objs)) if objs and len(objs) == len(want) else 0.0
    return obj, viol, detail


def feasible(viol):
    return all(v <= 0 for v in viol)


# ---- running a trial ------------------------------------------------------------------------
def run_profiles(profiles: dict, workload, out: Path, rounds: int, baseline=None):
    """Serve each {name: profile_path} through engine_scorecard; returns the aggregated card."""
    cfg = {'version': 1, 'description': 'slo_tune trial', 'baseline': baseline or next(iter(profiles)),
           'profiles': [{'name': n, 'engine': 'inferstack_native', 'model': 'factorylm-2b',
                         'argv': [a.replace('{profile}', str(p)) for a in NATIVE_ARGV]} for n, p in profiles.items()]}
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / 'profiles.json'
    cfg_path.write_text(json.dumps(cfg, indent=2))
    cmd = [sys.executable, 'scripts/engine_scorecard.py', 'run', '--profiles', str(cfg_path), '--out', str(out),
           '--rounds', str(rounds), '--run-id', out.name]
    if workload:
        cmd += ['--workload', str(workload)]
    with open(out / 'scorecard.log', 'w') as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    card_path = next(out.rglob('scorecard.json'), None)
    card = json.loads(card_path.read_text()) if card_path else {'cells': []}
    card['_exit'] = code
    return card


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', required=True, help='base profile TOML')
    ap.add_argument('--space', required=True, help='JSON list of knobs')
    ap.add_argument('--slo', required=True, help='JSON SLO (cells, objective, limits)')
    ap.add_argument('--workload', help='scorecard workload JSON (default: scorecard default)')
    ap.add_argument('--trials', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--confirm', type=int, default=0, help='rounds for a base-vs-best confirmation run (0 = skip)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = tomllib.loads(Path(a.base).read_text())
    space = json.loads(Path(a.space).read_text())
    slo = json.loads(Path(a.slo).read_text())
    log = out / 'trials.jsonl'

    def evaluate(i, pick):
        name = f'tune-{i:03d}'
        prof = out / 'profiles' / f'{name}.toml'
        prof.parent.mkdir(parents=True, exist_ok=True)
        prof.write_text(dump_toml(apply_overrides(base, pick, name)))
        card = run_profiles({name: prof}, a.workload, out / 'runs' / name, 1)
        obj, viol, detail = score(card, name, slo)
        row = {'trial': i, 'name': name, 'params': pick, 'objective': obj, 'violations': viol,
               'feasible': feasible(viol), 'exit': card.get('_exit'), 'cells': detail}
        with open(log, 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps({k: row[k] for k in ('trial', 'params', 'objective', 'feasible')}), flush=True)
        return row

    rows = []
    try:
        import optuna
        sampler = optuna.samplers.TPESampler(seed=a.seed, constraints_func=lambda t: t.user_attrs['viol'])
        study = optuna.create_study(direction='maximize', sampler=sampler)
        for i in range(a.trials):
            t = study.ask()
            row = evaluate(i, suggest(space, trial=t))
            t.set_user_attr('viol', row['violations'])
            study.tell(t, row['objective'])
            rows.append(row)
        method = f'optuna {optuna.__version__} TPE (constraints)'
    except ImportError:
        rng = random.Random(a.seed)
        rows = [evaluate(i, suggest(space, rng=rng)) for i in range(a.trials)]
        method = 'seeded random search (optuna not installed)'

    ok = [r for r in rows if r['feasible']]
    best = max(ok, key=lambda r: r['objective']) if ok else None
    lines = ['# SLO tune', '', f'Method: {method} · trials: {len(rows)} · feasible: {len(ok)}',
             f'SLO: `{json.dumps(slo)}`', '', '| trial | feasible | objective tok/s | params |', '|---|---|---|---|']
    lines += [f"| {r['trial']} | {r['feasible']} | {r['objective']:.1f} | `{json.dumps(r['params'])}` |" for r in rows]
    if best:
        (out / 'best.toml').write_text(dump_toml(apply_overrides(base, best['params'], base.get('name', 'base') + '-tuned')))
        lines += ['', f"Best: trial {best['trial']} ({best['objective']:.1f} tok/s) → best.toml"]
        if a.confirm:
            card = run_profiles({'base': Path(a.base), 'tuned': out / 'best.toml'}, a.workload, out / 'confirm',
                                a.confirm, baseline='base')
            md = next((out / 'confirm').rglob('scorecard.md'), None)
            lines += ['', f'Confirmation ({a.confirm} rounds, exit {card.get("_exit")}): see '
                      f'`{md.relative_to(out) if md else "confirm/ (no scorecard)"}`']
    else:
        lines += ['', 'No feasible trial: the SLO is not met anywhere in this space.']
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
