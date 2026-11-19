"""G2: run a harness job on a GPU host, unattended, with spend and ownership rules.

A `RemoteJob` is a repository script plus arguments (`{OUT}` = the evidence directory on the host). A
backend runs it where the GPU is and brings the evidence back:

  - `LocalBackend`: a subprocess on this machine (tests; or a rented GPU box the harness itself runs on).
  - `ColabBackend`: drives scripts/colab_fast_check.py (the proven driver: git-bundle shipping, detached job,
    poll through network drops, reclaimed-VM detection, download before stop) as a `--step` job.

Paid hosts plug in as another backend with the same `run()`.

Rules every backend obeys:
  - Spend gate: before launch, expected cost = expected_hours x price. Over the cap ($10 by default) the
    run is refused unless the caller passes an approved amount at least that large. An unknown price
    is refused the same way; spend is never guessed.
  - Ownership: Colab sessions this harness creates are named `ifs-<job>`; the backend only ever starts,
    polls and stops that one session. Any other session (e.g. the principal's `mf-*`) is never touched.
  - Cost is recorded from measured wall time x price, next to the evidence.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAP_USD = 10.0
_NAME = re.compile(r'^[a-z0-9][a-z0-9-]{0,40}$')


class SpendRefused(RuntimeError):
    """Expected cost over the cap without approval, or price unknown."""


@dataclass(frozen=True)
class RemoteJob:
    name: str                       # lowercase id; the Colab session is 'ifs-' + name
    argv: tuple[str, ...]           # repository script + args, run from the repo root; '{OUT}' = evidence dir
    expected_hours: float
    gpu: str = 'L4'
    setup: tuple[str, ...] = ()     # backend setup flags, e.g. ('--install-fla',) for Colab

    def __post_init__(self):
        if not _NAME.match(self.name):
            raise ValueError(f'job name must match {_NAME.pattern}: {self.name!r}')
        if not self.argv or not str(self.argv[0]).endswith('.py'):
            raise ValueError('argv must start with a repository script (.py)')
        if not self.expected_hours > 0:
            raise ValueError('expected_hours must be positive')


@dataclass
class RunResult:
    backend: str
    session: str
    exit_code: int
    evidence_dir: Path | None
    wall_s: float
    price_per_hour: float | None
    cost_usd: float | None
    estimate_usd: float | None
    log_tail: str = ''
    attempts: list = field(default_factory=list)

    @property
    def ok(self):
        return self.exit_code == 0 and self.evidence_dir is not None

    def record(self):
        return {'backend': self.backend, 'session': self.session, 'exit_code': self.exit_code,
                'evidence_dir': str(self.evidence_dir) if self.evidence_dir else None, 'wall_s': round(self.wall_s, 1),
                'price_per_hour': self.price_per_hour, 'cost_usd': self.cost_usd, 'estimate_usd': self.estimate_usd,
                'attempts': self.attempts}


def check_spend(job, price_per_hour, *, cap_usd=DEFAULT_CAP_USD, approved_usd=None):
    """Expected cost, or SpendRefused. Under the cap runs freely; over it needs an approval that covers it."""
    if price_per_hour is None:
        if approved_usd is None:
            raise SpendRefused(f'{job.name}: price unknown; refusing to guess spend')
        return None
    estimate = round(job.expected_hours * price_per_hour, 4)
    if estimate > cap_usd and (approved_usd is None or estimate > approved_usd):
        raise SpendRefused(f'{job.name}: expected ${estimate:.2f} is over the ${cap_usd:.2f} cap; '
                           'needs explicit approval of at least that amount')
    return estimate


class LocalBackend:
    """Runs the job as a subprocess here. `price_per_hour` is what this host costs (0 for a laptop)."""
    name = 'local'

    def __init__(self, *, price_per_hour=0.0, python=sys.executable, root=ROOT):
        self.price_per_hour, self.python, self.root = price_per_hour, python, Path(root)

    def run(self, job, out_root):
        out = Path(out_root) / f'local-{job.name}'
        evidence = out / 'evidence'
        evidence.mkdir(parents=True, exist_ok=True)
        argv = [self.python, '-u', *[str(a).replace('{OUT}', str(evidence)) for a in job.argv]]
        with open(out / 'job.log', 'w') as log:
            try:
                code = subprocess.run(argv, cwd=self.root, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=max(600, job.expected_hours * 3600 * 3)).returncode
            except subprocess.TimeoutExpired:
                code = 124
        tail = (out / 'job.log').read_text()[-1500:]
        return code, (evidence if code == 0 else None), tail, f'local-{job.name}'


class ColabBackend:
    """Google Colab through the `colab` CLI, via scripts/colab_fast_check.py. Own sessions only."""
    name = 'colab'
    PREFIX = 'ifs-'

    def __init__(self, *, price_per_hour=0.80, driver=ROOT / 'scripts' / 'colab_fast_check.py', python=sys.executable,
                 poll_s=60, retries_on_reclaim=1, runner=subprocess.run):
        self.price_per_hour, self.driver, self.python = price_per_hour, Path(driver), python
        self.poll_s, self.retries_on_reclaim, self.runner = poll_s, retries_on_reclaim, runner

    def session_for(self, job, attempt=0):
        name = self.PREFIX + job.name + (f'-{attempt}' if attempt else '')
        if not name.startswith(self.PREFIX) or name.startswith('mf-'):
            raise ValueError(f'refusing session name {name!r}: not owned by the harness')
        return name

    def driver_argv(self, job, session, out_root):
        step = ' '.join(str(a).replace('{OUT}', '{OUT}/job') for a in job.argv)
        return [self.python, str(self.driver), '--session', session, '--gpu', job.gpu, '--out', str(out_root),
                '--poll-s', str(self.poll_s), '--step-timeout', str(int(job.expected_hours * 3600 * 2)), *job.setup,
                '--step', step]

    def run(self, job, out_root):
        """Driver exit codes: 0 evidence fetched (VM stopped), 2 not fetched (VM left running, says how to
        fetch), 3 VM reclaimed (nothing to fetch). Only a reclaim is retried, under a new session name."""
        attempt, code, tail, session = 0, 1, '', ''
        while True:
            session = self.session_for(job, attempt)
            r = self.runner(self.driver_argv(job, session, out_root), cwd=ROOT, capture_output=True, text=True,
                            timeout=max(3600, job.expected_hours * 3600 * 3))
            code, tail = r.returncode, (r.stdout or '')[-1500:]
            if code == 3 and attempt < self.retries_on_reclaim:
                attempt += 1
                continue
            break
        runs = sorted(Path(out_root).glob(f'{session}-*'))
        evidence = None
        if code == 0 and runs:
            found = sorted(runs[-1].glob('*/job'))
            evidence = found[-1] if found else None
        return code, evidence, tail, session


def run_job(backend, job, out_root, *, cap_usd=DEFAULT_CAP_USD, approved_usd=None):
    """Spend-gated run. Returns RunResult with measured cost; raises SpendRefused before any launch."""
    estimate = check_spend(job, backend.price_per_hour, cap_usd=cap_usd, approved_usd=approved_usd)
    Path(out_root).mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    code, evidence, tail, session = backend.run(job, out_root)
    wall = time.monotonic() - t0
    price = backend.price_per_hour
    cost = None if price is None else round(wall / 3600 * price, 4)
    return RunResult(backend.name, session, code, evidence, wall, price, cost, estimate, tail)

