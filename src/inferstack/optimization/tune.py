"""G5: settings tuning as a campaign action (slo_tune v2). Deterministic, no model, settings only.

The tuner searches allowlisted native profile knobs (engines/profiles.py) for the best open-loop goodput
under the declared SLO, on the GPU host, and proposes one patch. It never ships anything: the proposal is
then confirmed by a fresh interleaved A/B (RemoteABVerifier), whose verdict alone decides keep/reject.

What is new against scripts/slo_tune.py (v1, closed-loop tok/s per scorecard cell):

  objective     goodput@99 (req/s at >= 99% good) with SLO attainment (mean good fraction over the rate
                grid) as tie-break, the same metrics the A/B verifier judges.
  space         derived, not hand-written: each knob carries what it acts on and whether larger values need
                more GPU memory; the hardware probe (GPU memory) caps the memory knobs, and the measured
                bottleneck (G4 knee label + dominant kernel families) keeps only knobs that act on it.
  crash region  a trial whose server fails to start (OOM, bad graph capture) marks a region: any later point
                that is at least as large on every memory knob and equal on the others is predicted to crash
                and is skipped without spending GPU time (SCOOT/SLO-Guard idea; ideas only, no code copied).
  quality       a trial whose greedy outputs disagree with the baseline's is infeasible, like the A/B gate.

`remote_tune.py` runs the loop on the host and writes tune.json; this module holds the pure logic both
sides share, and `TuneAction` drives it from the controller (spend gate, store, confirm).
"""
from __future__ import annotations

import json
import random

from inferstack.engines.profiles import PROFILES, check

# acts_on: knee labels (G4) and kernel families this knob can move; memory: larger value needs more memory
KNOBS = {
    'static_decode_slots': {'acts_on': {'capacity', 'decode', 'queue_admission'}, 'memory': True,
                            'rows': True},
    'max_requests_per_batch': {'acts_on': {'capacity', 'queue_admission', 'decode'}, 'memory': True,
                               'rows': True},
    'static_decode_admit_every_steps': {'acts_on': {'queue_admission', 'prefill'}, 'memory': False,
                                        'choices': (0, 2, 3, 4, 6, 8)},
    'static_decode_admit_pace_min_rows': {'acts_on': {'queue_admission'}, 'memory': False,
                                          'choices': (0, 4, 8, 16)},
    'max_prefill_chunk_tokens': {'acts_on': {'prefill', 'queue_admission'}, 'memory': True,
                                 'choices': (64, 128, 256, 512)},
    'static_decode_spec_drafts': {'acts_on': {'decode', 'capacity', 'gdn'}, 'memory': True, 'choices': (1, 2, 3)},
    'static_decode_spec_pipeline': {'acts_on': {'decode', 'capacity'}, 'memory': False, 'choices': (True, False)},
    'static_decode_spec_lazy_bank': {'acts_on': {'decode', 'capacity', 'gdn'}, 'memory': False,
                                     'choices': (True, False)},
    'gemm_blas_library': {'acts_on': {'capacity', 'decode', 'gemm'}, 'memory': False,
                          'choices': ('', 'cublas', 'cublaslt')},
}
ROW_CHOICES = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256)
ROWS_PER_GB = 4            # prior only: rows the decode lane may hold per GB of GPU memory (L4 24 GB -> 96)
KNOWN_LABELS = {'queue_admission', 'prefill', 'decode', 'capacity'}


def probe_hardware():
    """GPU facts the space is derived from (on the host). CPU hosts return {'gpu': None}."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {'gpu': None}
        p = torch.cuda.get_device_properties(0)
        return {'gpu': p.name, 'memory_gb': round(p.total_memory / 2**30, 1), 'sm_count': p.multi_processor_count,
                'capability': f'{p.major}.{p.minor}'}
    except Exception:  # noqa: BLE001 - no torch: CPU host
        return {'gpu': None}


def targets(measurements):
    """Bottleneck targets from G4 measurements: knee labels and kernel families with share >= 0.15."""
    from inferstack.evaluation.bottleneck import bottleneck_terms
    return sorted(bottleneck_terms(measurements or {}))


def space_for(hardware, *, target=(), engine='inferstack_native', knobs=None):
    """[{'name', 'choices'}] for the tuner plus a `why` list naming every knob kept or dropped and why."""
    schema = PROFILES[engine]
    memory_gb = (hardware or {}).get('memory_gb')
    max_rows = int(memory_gb * ROWS_PER_GB) if memory_gb else 64
    want = set(target) & (KNOWN_LABELS | {'gdn', 'gemm', 'attention'})
    space, why = [], []
    for name, meta in KNOBS.items():
        if knobs is not None and name not in knobs:
            continue
        if name not in schema:
            why.append(f'{name}: not an allowlisted {engine} key')
            continue
        if want and not meta['acts_on'] & want:
            why.append(f"{name}: dropped, acts on {sorted(meta['acts_on'])} not on {sorted(want)}")
            continue
        opt = schema[name]
        choices = [r for r in ROW_CHOICES if r <= max_rows] if meta.get('rows') else list(meta['choices'])
        choices = [c for c in choices if (opt.minimum is None or c >= opt.minimum)
                   and (opt.maximum is None or c <= opt.maximum) and (opt.values is None or c in opt.values)]
        if len(choices) < 2:
            why.append(f'{name}: dropped, fewer than 2 legal values')
            continue
        space.append({'name': name, 'choices': choices})
        cap = f' (rows <= {max_rows} from {memory_gb} GB)' if meta.get('rows') else ''
        why.append(f"{name}: kept, acts on {sorted(meta['acts_on'] & want) or 'all (no measured target)'}{cap}")
    return space, why


class CrashRegion:
    """Points that failed to start; predicts that a point at least as large on every memory knob (and equal on
    the others) fails too."""

    def __init__(self, crashes=()):
        self.crashes = [dict(c) for c in crashes]

    def add(self, point):
        self.crashes.append(dict(point))

    def covers(self, point):
        for c in self.crashes:
            if set(c) != set(point):
                continue
            if all((point[k] >= c[k]) if KNOBS.get(k, {}).get('memory') and not isinstance(c[k], bool)
                   else point[k] == c[k] for k in c):
                return c
        return None


def score(goodput, rates):
    """(goodput99_rps, slo_attainment) of one sweep; None when unmeasured."""
    from inferstack.optimization.remote_verify import attainment
    if not goodput or goodput.get('token_counts_missing'):
        return None
    return {'goodput99_rps': goodput.get('goodput99_rps') or 0.0,
            'slo_attainment': round(attainment(goodput, [float(r) for r in rates]) or 0.0, 4)}


def objective(s):
    """Single number for the sampler: goodput@99 with attainment (< 1) as tie-break between grid points."""
    return -1.0 if s is None else s['goodput99_rps'] + s['slo_attainment']


def agreement(reference, check):
    if not reference or not check:
        return 0.0
    return sum(a == b for a, b in zip(reference, check)) / len(reference)


def baseline_point(argv, space, *, source_root='.'):
    """The value each knob in the space really runs with in the baseline profile: its TOML value, else the
    engine default (config.py dataclasses). Knobs with neither are left out."""
    import dataclasses
    from pathlib import Path
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib
    from inferstack import config
    argv = list(argv)
    data = {}
    if '--profile' in argv:
        path = Path(argv[argv.index('--profile') + 1])
        path = path if path.is_absolute() else Path(source_root) / path
        data = tomllib.loads(path.read_text()) if path.is_file() else {}
    sections = {'scheduler': config.SchedulerConfig, 'kernels': config.KernelConfig, 'native': config.NativeConfig}
    schema = PROFILES['inferstack_native']
    point = {}
    for p in space:
        name = p['name']
        section = schema[name].section
        if name in data.get(section, {}):
            point[name] = data[section][name]
        elif section in sections:
            field = {f.name: f for f in dataclasses.fields(sections[section])}.get(name)
            if field is not None and field.default is not dataclasses.MISSING:
                point[name] = field.default
    return point


def with_baseline(space, start):
    """Make sure each knob's baseline value is one of its choices (numbers stay sorted)."""
    out = []
    for p in space:
        choices, v = list(p['choices']), start.get(p['name'])
        if v is not None and v not in choices:
            choices.append(v)
            if all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in choices):
                choices.sort()
        out.append({**p, 'choices': choices})
    return out


def neighbours(space, start):
    """One knob at a time, one step either side of the baseline value (bools flip)."""
    out, seen = [], set()
    for p in space:
        v = start.get(p['name'])
        if v is None or v not in p['choices']:
            continue
        i = p['choices'].index(v)
        for j in (i - 1, i + 1):
            if 0 <= j < len(p['choices']) and p['choices'][j] != v:
                point = {**start, p['name']: p['choices'][j]}
                key = json.dumps(point, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    out.append(point)
    return out


class Sampler:
    """Three phases, ask/tell throughout:
      1. neighbours: from a baseline point, change one knob one step (a hand-tuned profile is usually near
         the best; random jumps far from it mostly land worse, as the L4 G5 run showed);
      2. combine: if two or more single moves beat the baseline, try them all together once;
      3. explore: Optuna TPE when installed (seeded, warm-started with phases 1-2), else seeded random.
    Without a baseline point it is phase 3 only."""

    def __init__(self, space, seed=0, start=None):
        self.space, self.rng, self.start = space, random.Random(seed), dict(start or {})
        self.queue = neighbours(space, self.start) if self.start else []
        self.base_value, self.singles, self.combined = None, [], False
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.optuna = optuna
            self.study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
            method = f'optuna {optuna.__version__} TPE'
        except ImportError:
            self.optuna, self.study, method = None, None, 'seeded random'
        self.method = (f'baseline neighbours ({len(self.queue)}) then {method}' if self.queue else method)
        self._open = self._pending = None

    def ask(self):
        if self.queue:
            self._pending = ('single', self.queue.pop(0))
            return dict(self._pending[1])
        if not self.combined and self.base_value is not None:
            best = {}                       # knob -> (score, value): the best winning move per knob
            for p, v in self.singles:
                if v > self.base_value:
                    for k, x in p.items():
                        if self.start.get(k) != x and (k not in best or v > best[k][0]):
                            best[k] = (v, x)
            if len(best) >= 2:
                self.combined = True
                point = {**self.start, **{k: x for k, (_, x) in best.items()}}
                self._pending = ('combined', point)
                return dict(point)
        self._pending = None
        if self.study is not None:
            self._open = self.study.ask()
            return {p['name']: self._open.suggest_categorical(p['name'], p['choices']) for p in self.space}
        return {p['name']: self.rng.choice(p['choices']) for p in self.space}

    def tell(self, value):
        if self._pending is not None:
            kind, point = self._pending
            self._pending = None
            if kind == 'single':
                self.singles.append((point, value))
            self._warm(point, value)
            return
        if self.study is not None and self._open is not None:
            self.study.tell(self._open, value)
            self._open = None

    def _warm(self, point, value):
        """Feed a phase-1/2 result to TPE so phase 3 starts from what was learned."""
        if self.study is None or set(point) != {p['name'] for p in self.space}:
            return
        dists = {p['name']: self.optuna.distributions.CategoricalDistribution(p['choices']) for p in self.space}
        self.study.add_trial(self.optuna.trial.create_trial(params=point, distributions=dists, value=value))


def best_from_tune(tune, *, min_agreement=0.75):
    """The tuned patch to confirm: the best feasible trial that beats the baseline trial, else None."""
    trials = tune.get('trials') or []
    base = next((t for t in trials if t.get('baseline')), None)
    if base is None or base.get('score') is None:
        return None
    ok = [t for t in trials if not t.get('baseline') and t.get('feasible') and t.get('score')]
    if not ok:
        return None
    best = max(ok, key=lambda t: objective(t['score']))
    if objective(best['score']) <= objective(base['score']):
        return None
    return {'patch': best['patch'], 'trial': best['trial'], 'score': best['score'], 'baseline_score': base['score']}


def summarize(tune):
    trials = tune.get('trials') or []
    return {'gpu': (tune.get('hardware') or {}).get('gpu'), 'hardware': tune.get('hardware'),
            'target': tune.get('target'), 'space': tune.get('space'), 'why': tune.get('why'),
            'method': tune.get('method'), 'trials': len(trials),
            'ran': sum(1 for t in trials if not t.get('pruned')),
            'pruned': sum(1 for t in trials if t.get('pruned')),
            'crashed': sum(1 for t in trials if t.get('crashed')),
            'feasible': sum(1 for t in trials if t.get('feasible')),
            'crash_region': tune.get('crash_region'), 'best': best_from_tune(tune)}


class TuneAction:
    """Controller side: run remote_tune.py on a backend, store the result, confirm the pick with an A/B.

    The tune result is stored as the deployment's `tune` measurement (space, crash region, best patch); the
    confirmation is an ordinary RemoteABVerifier experiment, so memory and the planner see it like any other.
    """

    def __init__(self, verifier, *, trials=8, seed=0, hours_per_trial=0.1, knobs=None, confirm=True,
                 install_optuna=False):
        self.verifier, self.trials, self.seed, self.install_optuna = verifier, trials, seed, install_optuna
        self.hours_per_trial, self.knobs, self.confirm = hours_per_trial, knobs, confirm

    def job(self, name, baseline_profile, target):
        from inferstack.execution.gpu_backend import RemoteJob
        v = self.verifier
        argv = ['scripts/remote_tune.py', '--profiles', v.profiles, '--baseline', baseline_profile,
                '--workload', v.workload, '--slo', v.slo, '--trials', str(self.trials), '--seed', str(self.seed),
                '--rates', *map(str, v.rates), '--seconds-per-rate', str(v.seconds_per_rate),
                '--min-requests', str(v.min_requests), '--out', '{OUT}']
        if self.install_optuna:
            argv.append('--install-optuna')
        if target:
            argv += ['--target', *target]
        if self.knobs:
            argv += ['--knobs', *self.knobs]
        if v.cases:
            argv += ['--cases', *v.cases]
        return RemoteJob(name, tuple(argv), expected_hours=(self.trials + 1) * self.hours_per_trial,
                         gpu=v.gpu, setup=v.setup)

    def run(self, deployment_id, baseline_profile, *, name):
        from inferstack.evaluation.bottleneck import latest_measurements, record_measurement
        from inferstack.execution.gpu_backend import run_job
        v = self.verifier
        target = targets({k: m for k, m in latest_measurements(v.repository, deployment_id).items()
                          if not m.get('stale')})
        run = run_job(v.backend, self.job(name, baseline_profile, target), v.out_root, cap_usd=v.cap_usd,
                      approved_usd=v.approved_usd)
        path = run.evidence_dir / 'tune.json' if run.evidence_dir else None
        if path is None or not path.is_file():
            return {'decision': 'infra_failed', 'reason': f'tune produced no tune.json (exit {run.exit_code})',
                    'cost_usd': run.cost_usd, 'log_tail': run.log_tail[-300:]}
        tune = json.loads(path.read_text())
        summary = summarize(tune)
        record_measurement(v.repository, deployment_id, 'tune', summary, tune,
                           source={'job': name, 'run': run.record()})
        out = {'tune': summary, 'cost_usd': run.cost_usd}
        best = summary['best']
        if best is None:
            return {**out, 'decision': 'no_candidate',
                    'reason': 'no feasible trial beat the baseline inside the tuned space'}
        if not self.confirm:
            return {**out, 'decision': 'proposed'}
        check('inferstack_native', best['patch'])
        plan = {'id': f'{name}-confirm'[:41], 'baseline_deployment': deployment_id,
                'baseline_profile': baseline_profile, 'patch': best['patch'],
                'hypothesis': f"settings tune targeting {', '.join(target) or 'no measured bottleneck'}: "
                              f"patch {json.dumps(best['patch'], sort_keys=True)} raised goodput@99 from "
                              f"{best['baseline_score']['goodput99_rps']} to {best['score']['goodput99_rps']} "
                              'req/s in one tuning trial; confirm with interleaved A/B'}
        verdict, confirm_run = v.verify(plan)
        return {**out, 'decision': verdict['decision'], 'verdict': verdict,
                'cost_usd': round((run.cost_usd or 0) + (confirm_run.cost_usd or 0), 4)}


__all__ = ['KNOBS', 'CrashRegion', 'Sampler', 'TuneAction', 'best_from_tune', 'objective', 'probe_hardware',
           'score', 'space_for', 'summarize', 'targets']
