"""G7: a code idea becomes a pull request only on evidence; the agent never ships anything.

  handoff (controller-owned files and tests) -> coding agent attempts (each validated by the desk:
  confinement, compile, independent tests in the sandbox) -> GPU A/B of the validated diff against the
  pinned baseline (greedy-equality quality gate, repeats, open-loop goodput + SLO attainment)
  -> too-good breaker -> PR bundle (diff, evidence, description) for human review.

The evaluator is fixed: its files' hash is taken when the run starts (optionally pinned by the operator)
and checked again before a bundle is written; a candidate diff can never touch it (patch confinement), and
the host restores tests and the evaluator from the pinned revision before the candidate runs.

The breaker: one patch that multiplies goodput is far more often a broken measurement or a benchmark
shortcut than a real win. A keep whose goodput@99 grows more than `max_gain` times (or from zero) is
`suspicious`: no bundle, a fresh-seed rerun and a human look first.

Publishing (push a branch, open a draft PR) is a separate call that refuses without `allow_publish=True`:
the principal decides when anything leaves this machine.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from inferstack.implementation import AttemptLimit

ROOT = Path(__file__).resolve().parents[3]
EVALUATOR_FILES = ('src/inferstack/evaluation/judge.py', 'src/inferstack/evaluation/goodput.py',
                   'src/inferstack/evaluation/metrics.py', 'src/inferstack/optimization/remote_verify.py',
                   'src/inferstack/optimization/code_change.py', 'src/inferstack/execution/patches.py',
                   'src/inferstack/execution/gpu_backend.py', 'scripts/remote_ab.py')


def evaluator_fingerprint(root=ROOT):
    h = hashlib.sha256()
    for rel in EVALUATOR_FILES:
        h.update(rel.encode() + b'\0' + (Path(root) / rel).read_bytes() + b'\0')
    return h.hexdigest()


class EvaluatorChanged(RuntimeError):
    pass


def breaker(verdict, max_gain):
    """Reason string when a kept result is too good to trust, else None."""
    metric = ((verdict or {}).get('metrics') or {}).get('goodput99_rps') or {}
    base, cand = (metric.get('baseline') or {}).get('median'), (metric.get('candidate') or {}).get('median')
    if base is None or cand is None:
        return None
    if base <= 0 < cand:
        return f'goodput@99 rose from 0 to {cand}: nothing to compare a plausible gain against'
    if base > 0 and cand / base > max_gain:
        return f'goodput@99 x{cand / base:.2f} ({base} -> {cand} req/s) exceeds the x{max_gain} breaker'
    return None


def pr_text(record, verdict, summary, fingerprint):
    handoff, cand = record['handoff'], record['candidate']
    rows = []
    for name, m in sorted((verdict.get('metrics') or {}).items()):
        b, c = (m.get('baseline') or {}).get('median'), (m.get('candidate') or {}).get('median')
        rows.append(f'| {name} | {b} | {c} | {m.get("gain")} |')
    gates = ', '.join(f'{k}: {"pass" if v else "FAIL"}' for k, v in sorted((verdict.get('gates') or {}).items()))
    title = f"perf: {(summary or handoff['description'] or handoff['hypothesis'])[:60].rstrip()}"
    body = '\n'.join([
        '## Hypothesis', handoff['hypothesis'], '', '## Change', summary or handoff['description'] or '-', '',
        f"Files: {', '.join(cand['touched_files'])}. Pinned source `{cand['source_revision'][:12]}`, candidate "
        f"`{cand['candidate_id']}`, patch sha256 `{cand['patch_sha256'][:16]}`.", '',
        '## Evidence (interleaved A/B on the GPU host, medians over repeats)', '',
        '| metric | baseline | candidate | gain |', '|---|---|---|---|', *rows, '',
        f"Decision: **{verdict.get('decision')}**. Gates: {gates}. Reasons: {'; '.join(verdict.get('reasons') or [])}.",
        f"Independent tests: {', '.join(cand['independent_tests'])}. Evidence sha256 "
        f"`{(verdict.get('evidence') or {}).get('sha256', '')[:16]}`. Evaluator fingerprint `{fingerprint[:16]}`.",
        f"GPU cost: ${verdict.get('cost_usd') or 0}.", '',
        'Opened by the InferStack code-change agent (G7). Review before merging; nothing was merged or deployed.'])
    return title, body


class CodeChange:
    """One proposal -> at most one PR bundle. `implementer.run(desk, impl_id, feedback)` makes one attempt."""

    def __init__(self, desk, implementer, verifier, *, out_root, max_gain=1.5, expected_fingerprint=None,
                 evaluator_root=ROOT):
        self.desk, self.implementer, self.verifier = desk, implementer, verifier
        self.out_root, self.max_gain, self.evaluator_root = Path(out_root), max_gain, evaluator_root
        self.expected_fingerprint = expected_fingerprint

    def _fingerprint(self, pinned):
        now = evaluator_fingerprint(self.evaluator_root)
        if pinned is not None and now != pinned:
            raise EvaluatorChanged(f'evaluator files changed (expected {pinned[:16]}, found {now[:16]}); refusing '
                                   'to judge a candidate with a different evaluator')
        return now

    async def run(self, proposal, plan, *, baseline_id, baseline_profile, source_revision, name, tests,
                  job_id=None):
        fingerprint = self._fingerprint(self.expected_fingerprint)
        impl = self.desk.request(proposal, plan, baseline_id=baseline_id, source_revision=source_revision,
                                 job_id=job_id, independent_tests=tests)
        impl_id, record = impl['id'], impl['body']
        out = {'implementation_id': impl_id, 'evaluator_fingerprint': fingerprint}
        feedback = None
        while record['state'] != 'validated':   # a rerun of a validated change goes straight to the GPU
            try:
                record = (await self.implementer.run(self.desk, impl_id, feedback))['body']
            except AttemptLimit:
                record = self.desk.read(impl_id)['body']
            if record['state'] == 'validated':
                break
            if record['state'] in ('exhausted', 'cancelled'):
                last = record['attempts'][-1] if record['attempts'] else {}
                return {**out, 'decision': 'implementation_failed', 'attempts': len(record['attempts']),
                        'reason': last.get('reason') or last.get('outcome')}
            last = record['attempts'][-1]
            feedback = last.get('reason') or f"attempt {last['outcome']}"
        out['attempts'] = len(record['attempts'])
        attempt = next(a for a in record['attempts'] if a.get('candidate_id') == record['candidate']['candidate_id'])
        diff = self.desk.repository.read_artifact(record['candidate']['patch_sha256']).decode()
        summary = None
        if attempt.get('logs_sha256'):
            summary = self.desk.repository.read_artifact(attempt['logs_sha256']).decode(errors='replace')[:2000]
        ab_plan = {'id': name, 'baseline_deployment': baseline_id, 'baseline_profile': baseline_profile,
                   'source_diff': diff, 'hypothesis': record['handoff']['hypothesis'],
                   'source_tests': record['handoff']['acceptance']['independent_tests']}   # again, on the GPU
        verdict, _ = self.verifier.verify(ab_plan)
        self._fingerprint(fingerprint)                      # nothing changed the judge while the A/B ran
        out.update(candidate_id=record['candidate']['candidate_id'], verdict=verdict,
                   cost_usd=verdict.get('cost_usd'))
        if verdict.get('decision') != 'keep':
            return {**out, 'decision': verdict.get('decision') or 'inconclusive'}
        tripped = breaker(verdict, self.max_gain)
        if tripped:
            return {**out, 'decision': 'suspicious', 'reason': tripped,
                    'next': 'rerun the A/B with a fresh seed and have a human read the diff before any PR'}
        title, body = pr_text(record, verdict, summary, fingerprint)
        bundle = self.out_root / impl_id
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / 'patch.diff').write_text(diff)
        (bundle / 'PR.md').write_text(f'# {title}\n\n{body}\n')
        (bundle / 'pr.json').write_text(json.dumps({
            'title': title, 'implementation_id': impl_id, 'candidate': record['candidate'],
            'source_revision': source_revision, 'evaluator_fingerprint': fingerprint,
            'evidence_sha256': (verdict.get('evidence') or {}).get('sha256'), 'decision': verdict['decision'],
            'metrics': verdict.get('metrics'), 'gates': verdict.get('gates')}, indent=2, sort_keys=True, default=str))
        return {**out, 'decision': 'pr_ready', 'bundle': str(bundle), 'title': title}


def publish_pr(bundle, *, source_root, base='main', allow_publish=False, remote='origin'):
    """Push the bundle as a branch and open a draft PR. Refuses unless the principal explicitly allowed it."""
    if not allow_publish:
        raise PermissionError('publishing a PR needs the principal to say so (allow_publish=True)')
    bundle, source_root = Path(bundle), Path(source_root)
    meta = json.loads((bundle / 'pr.json').read_text())
    branch = f"agent/{meta['implementation_id']}"
    tree = Path(tempfile.mkdtemp(prefix='pr-'))
    git = ['git', '-C', str(source_root)]
    try:
        subprocess.run([*git, 'worktree', 'add', '-q', '-b', branch, str(tree), meta['source_revision']], check=True)
        subprocess.run(['git', '-C', str(tree), 'apply', str(bundle / 'patch.diff')], check=True)
        subprocess.run(['git', '-C', str(tree), 'add', '-A'], check=True)
        subprocess.run(['git', '-C', str(tree), 'commit', '-qm', meta['title']], check=True)
        subprocess.run(['git', '-C', str(tree), 'push', '-u', remote, branch], check=True)
        body = (bundle / 'PR.md').read_text().split('\n', 2)[-1]
        done = subprocess.run(['gh', 'pr', 'create', '--draft', '--base', base, '--head', branch, '--title',
                               meta['title'], '--body', body], cwd=tree, capture_output=True, text=True, check=True)
        return done.stdout.strip()
    finally:
        subprocess.run([*git, 'worktree', 'remove', '--force', str(tree)], capture_output=True)
        shutil.rmtree(tree, ignore_errors=True)


__all__ = ['CodeChange', 'EvaluatorChanged', 'breaker', 'evaluator_fingerprint', 'pr_text', 'publish_pr']
