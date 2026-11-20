"""G4: bounded measurement actions the controller runs before planning, on the same GPU backends as G2.

  kernels  scripts/decode_step_profile.py on the host (torch.profiler over one decode step at a few batch
           sizes) -> kernel-family shares, stored as the deployment's `kernels` measurement.
  knee     a goodput sweep of the deployment alone is what every remote A/B already does for its baseline
           (remote_verify stores it); `KneeMeasurer` exists for a deployment with no A/B yet.

Both go through `run_job`, so the spend gate ($10 cap, unknown price refused), own-session rule and
measured cost apply. A measurement only describes; it never changes a deployment or decides anything.
"""
from __future__ import annotations

import json
from pathlib import Path

from inferstack.evaluation.bottleneck import diagnose_trials, record_measurement, summarize_kernels
from inferstack.execution.gpu_backend import RemoteJob, run_job


class MeasurementFailed(RuntimeError):
    pass


class KernelProfileMeasurer:
    def __init__(self, repository, backend, out_root, *, iters=20, positions=(200, 1600), prof_rows=(8, 32, 48),
                 expected_hours=0.25, gpu='L4', setup=(), cap_usd=10.0, approved_usd=None):
        self.repository, self.backend, self.out_root = repository, backend, Path(out_root)
        self.iters, self.positions, self.prof_rows = iters, tuple(positions), tuple(prof_rows)
        self.expected_hours, self.gpu, self.setup = expected_hours, gpu, tuple(setup)
        self.cap_usd, self.approved_usd = cap_usd, approved_usd

    def job(self, name, profile_path):
        argv = ('scripts/decode_step_profile.py', '--profile', str(profile_path), '--iters', str(self.iters),
                '--positions', *map(str, self.positions), '--prof-rows', *map(str, self.prof_rows), '--out', '{OUT}')
        return RemoteJob(name, argv, expected_hours=self.expected_hours, gpu=self.gpu, setup=self.setup)

    def measure(self, deployment_id, profile_path, *, name=None):
        job = self.job(name or f'kp-{deployment_id}'[:41].lower().replace('_', '-').replace('.', '-'), profile_path)
        run = run_job(self.backend, job, self.out_root, cap_usd=self.cap_usd, approved_usd=self.approved_usd)
        path = run.evidence_dir / 'decode_step.json' if run.evidence_dir else None
        if path is None or not path.is_file():
            raise MeasurementFailed(f'kernel profile produced no decode_step.json (exit {run.exit_code}): '
                                    f'{run.log_tail[-300:]}')
        raw = json.loads(path.read_text())
        return record_measurement(self.repository, deployment_id, 'kernels', summarize_kernels(raw), raw,
                                  source={'job': job.name, 'run': run.record()}), run


class KneeMeasurer:
    """Goodput knee of a deployment that has no A/B yet: scripts/remote_ab.py with the baseline profile on
    both sides (A/A; the second side doubles as a run-to-run check). Stored only as a measurement, never as
    an experiment, so memory does not gain a fake treatment."""

    def __init__(self, verifier):
        self.verifier = verifier      # a RemoteABVerifier (its sweep settings, backend and spend gate)

    def measure(self, deployment_id, profile_name, *, name):
        from inferstack.optimization.remote_verify import gates_from_ab, knees_from_ab
        v = self.verifier
        plan = {'id': name, 'baseline_profile': profile_name, 'candidate_profile': profile_name}
        run = run_job(v.backend, v.job(plan), v.out_root, cap_usd=v.cap_usd, approved_usd=v.approved_usd)
        path = run.evidence_dir / 'ab.json' if run.evidence_dir else None
        if path is None or not path.is_file():
            raise MeasurementFailed(f'knee run produced no ab.json (exit {run.exit_code}): {run.log_tail[-300:]}')
        ab = json.loads(path.read_text())
        gates, _ = gates_from_ab(ab)
        knees = knees_from_ab(ab)
        if not (gates['health'] and gates['complete']) or not knees['baseline']:
            raise MeasurementFailed(f'knee run unhealthy or incomplete: {gates}')
        merged = diagnose_trials([t['goodput'] for t in ab['trials'] if t.get('goodput')])
        return record_measurement(v.repository, deployment_id, 'knee', merged, ab,
                                  source={'job': name, 'run': run.record()}), run


__all__ = ['KernelProfileMeasurer', 'KneeMeasurer', 'MeasurementFailed', 'diagnose_trials']
