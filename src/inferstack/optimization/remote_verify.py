"""G2: judge an A/B run made on a GPU host (scripts/remote_ab.py) with the harness's own rules.

The host only measures; the decision is made here, by `evaluation.judge.evaluate` with the same gates and
noise rules as `EndpointVerifier`. The outcome is stored like every other experiment (a content-addressed
evidence artifact + a terminal job whose result lists {plan, candidate, verdict}), so `memory.experience`
and the planner's no-identical-retry rule see it without a human step.

Gates are computed from ab.json, never asserted:
  - health: every trial's server became ready and the goodput sweep ran;
  - complete: `repeats` trials per side, every objective metric observed (an engine that hides token
    counts is unmeasured, not slow);
  - correctness: every greedy check request returned text on both sides;
  - quality: greedy check outputs agree with the baseline's on at least `min_agreement` of prompts
    (a spec-decode or kernel change that alters greedy text fails here);
  - comparable: one host, one GPU, one commit for both sides.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from inferstack.evaluation.bottleneck import diagnose_trials, record_measurement
from inferstack.evaluation.judge import Objective, evaluate
from inferstack.execution.gpu_backend import RemoteJob, run_job

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
# goodput@99 alone is blind between grid points: on the L4 proof run (P10b replay) both sides scored 8 req/s
# while the candidate's 12 req/s good fraction fell from 64-86% to 17-21%. SLO attainment (mean good fraction
# over the whole rate grid) sees that, so a change must not regress it.
DEFAULT_OBJECTIVES = (Objective('goodput99_rps', 'max'), Objective('slo_attainment', 'max'))


def attainment(goodput, rates):
    """Mean good fraction over every case and every rate in the grid; rates skipped after the sweep stopped
    count as 0 (the sweep only stops below 50% good, so the curve is already falling there)."""
    fracs = []
    for case in goodput['cases'].values():
        by_rate = {s['rate']: s['good_frac'] for s in case['steps']}
        fracs += [by_rate.get(r, 0.0) for r in rates]
    return sum(fracs) / len(fracs) if fracs else None


def trials_from_ab(ab):
    """Per side: [{'goodput99_rps', 'goodput90_rps', 'slo_attainment'}] in repeat order; None where unmeasured."""
    sides = {'baseline': [], 'candidate': []}
    rates = [float(r) for r in ab['sweep']['rates']]
    for t in sorted(ab['trials'], key=lambda t: t['repeat']):
        g = t.get('goodput') or {}
        measured = bool(g) and not g.get('token_counts_missing')
        sides[t['side']].append({'goodput99_rps': g.get('goodput99_rps') if measured else None,
                                 'goodput90_rps': g.get('goodput90_rps') if measured else None,
                                 'slo_attainment': attainment(g, rates) if measured else None})
    return sides['baseline'], sides['candidate']


def gates_from_ab(ab, *, min_agreement=0.75):
    trials, repeats = ab['trials'], ab['repeats']
    by = {s: [t for t in trials if t['side'] == s] for s in ('baseline', 'candidate')}
    notes = {}
    health = all(t.get('ready') and t.get('goodput') for t in trials)
    complete = all(len(v) == repeats for v in by.values()) and all(
        t.get('goodput') and not t['goodput'].get('token_counts_missing') for t in trials)
    checks = [t.get('check') or [] for t in trials]
    correctness = bool(checks) and all(c and all(isinstance(x, str) and x.strip() for x in c) for c in checks)
    reference = next((t['check'] for t in by['baseline'] if t.get('check')), None)
    agree = []
    for t in by['candidate']:
        if reference and t.get('check'):
            agree.append(sum(a == b for a, b in zip(reference, t['check'])) / len(reference))
    quality = bool(agree) and min(agree) >= min_agreement
    notes['quality'] = f'greedy agreement with baseline: min {min(agree) if agree else 0:.2f} (need {min_agreement})'
    comparable = bool(ab.get('commit')) and all(t.get('profile') for t in trials)
    notes['comparable'] = f"host {ab.get('host')}, gpu {ab.get('gpu')}, commit {ab.get('commit')}"
    if not health:
        notes['health'] = '; '.join(f"{t['side']} r{t['repeat']}: {t.get('error', 'no sweep')}"
                                    for t in trials if not (t.get('ready') and t.get('goodput')))[:500]
    return {'health': health, 'complete': complete, 'correctness': correctness, 'quality': quality,
            'comparable': comparable}, notes


def knees_from_ab(ab):
    """G4: majority knee diagnosis per side (which SLO broke first, and the engine counters there)."""
    return {side: diagnose_trials([t.get('goodput') for t in sorted(ab['trials'], key=lambda t: t['repeat'])
                                   if t['side'] == side and t.get('goodput')])
            for side in ('baseline', 'candidate')}


def judge(ab, objectives=DEFAULT_OBJECTIVES, *, min_agreement=0.75):
    base, cand = trials_from_ab(ab)
    gates, notes = gates_from_ab(ab, min_agreement=min_agreement)
    verdict = evaluate(base, cand, list(objectives), gates=gates)
    verdict['knee'] = knees_from_ab(ab)
    return verdict, gates, notes, (base, cand)


def treatment_signature(actions):
    body = sorted(json.dumps({'type': a['type'], 'params': a['params']}, sort_keys=True)
                  for a in actions if a['type'] != 'measure')
    return hashlib.sha256('\n'.join(body).encode()).hexdigest()


class RemoteABVerifier:
    """Plan -> RemoteJob on a backend -> ab.json -> verdict + evidence + stored outcome.

    plan: {'id', 'hypothesis', 'baseline_deployment' (record id), 'baseline_profile' (scorecard name),
           and one of 'candidate_profile', 'patch' (native profile patch) or 'source_diff' (G7: unified diff
           to the patchable roots; the candidate is the baseline profile on patched source)}.
    """

    def __init__(self, repository, backend, out_root, *, profiles='configs/scorecard/a100-qwen3.5-2b.json',
                 workload='configs/scorecard/workload-fast.json', slo='configs/tune/slo-v2-l4.json', cases=(),
                 repeats=3, rates=(1, 2, 4, 6, 8, 12, 16), seconds_per_rate=20.0, min_requests=20,
                 objectives=DEFAULT_OBJECTIVES, min_agreement=0.75, hours_per_trial=0.2, gpu='L4', setup=(),
                 cap_usd=10.0, approved_usd=None, source_root=None, source_revision=None):
        self.repository, self.backend, self.out_root = repository, backend, Path(out_root)
        self.profiles, self.workload, self.slo, self.cases = profiles, workload, slo, tuple(cases)
        self.repeats, self.rates, self.seconds_per_rate, self.min_requests = repeats, rates, seconds_per_rate, min_requests
        self.objectives, self.min_agreement, self.hours_per_trial = tuple(objectives), min_agreement, hours_per_trial
        self.gpu, self.setup, self.cap_usd, self.approved_usd = gpu, tuple(setup), cap_usd, approved_usd
        self.source_root, self.source_revision = source_root, source_revision

    def job(self, plan):
        if sum(bool(plan.get(k)) for k in ('candidate_profile', 'patch', 'source_diff')) != 1:
            raise ValueError('plan needs exactly one of candidate_profile, patch or source_diff')
        argv = ['scripts/remote_ab.py', '--profiles', self.profiles, '--baseline', plan['baseline_profile'],
                '--workload', self.workload, '--slo', self.slo, '--repeats', str(self.repeats),
                '--rates', *map(str, self.rates), '--seconds-per-rate', str(self.seconds_per_rate),
                '--min-requests', str(self.min_requests), '--out', '{OUT}']
        if plan.get('source_diff'):
            import base64
            argv += ['--candidate-diff-b64', base64.b64encode(plan['source_diff'].encode()).decode()]
            if self.source_root:
                argv += ['--source-root', str(self.source_root)]
            if self.source_revision:
                argv += ['--source-revision', self.source_revision]
            if plan.get('source_tests'):
                argv += ['--source-tests', *plan['source_tests']]
        elif plan.get('candidate_profile'):
            argv += ['--candidate', plan['candidate_profile']]
        else:
            argv += ['--patch', json.dumps(plan['patch'], sort_keys=True, separators=(',', ':'))]
        if self.cases:
            argv += ['--cases', *self.cases]
        return RemoteJob(plan['id'], tuple(argv), expected_hours=2 * self.repeats * self.hours_per_trial,
                         gpu=self.gpu, setup=self.setup)

    def actions(self, plan):
        if plan.get('source_diff'):
            change = {'id': 'change', 'type': 'source_patch',
                      'params': {'patch_sha256': hashlib.sha256(plan['source_diff'].encode()).hexdigest()}}
        else:
            change = {'id': 'change', 'type': 'config_change',
                      'params': {'profile_patch': plan['patch']} if plan.get('patch')
                      else {'candidate_profile': plan['candidate_profile']}}
        actions = [change,
                   {'id': 'measure', 'type': 'measure', 'depends_on': ['change'], 'params': {}}]
        return actions

    def verify(self, plan):
        job = self.job(plan)
        run = run_job(self.backend, job, self.out_root, cap_usd=self.cap_usd, approved_usd=self.approved_usd)
        return self.conclude(plan, run.evidence_dir / 'ab.json' if run.evidence_dir else None, run), run

    def conclude(self, plan, ab_path, run):
        """Judge an ab.json (or its absence) and store the outcome. Also used to re-judge a past run."""
        baseline = self.repository.read('deployment', plan['baseline_deployment'])
        if ab_path is None or not ab_path.is_file():
            # nothing was measured (VM refused, reclaimed, driver died): an infrastructure failure, never
            # 'inconclusive', which would read as 'measured, no effect'
            verdict = {'decision': 'infra_failed', 'reasons': [f'remote run produced no ab.json (exit {run.exit_code})']}
            ab, gates, notes, trials = None, {}, {'run': run.log_tail[-500:]}, ([], [])
        else:
            ab = json.loads(ab_path.read_text())
            tests = ab.get('source_tests')
            if tests is not None and not tests.get('passed'):
                # G7: the controller's tests failed on the GPU against the patched source; no A/B was run
                verdict = {'decision': 'reject', 'reasons': [
                    f"independent tests failed on the GPU host against the patched source (exit {tests.get('exit_code')})"]}
                gates, notes, trials = {'source_tests': False}, {'source_tests': tests}, ([], [])
            else:
                verdict, gates, notes, trials = judge(ab, self.objectives, min_agreement=self.min_agreement)
                if tests is not None:
                    gates, notes = {**gates, 'source_tests': True}, {**notes, 'source_tests': tests}
        identity = {'model_ref': [baseline['body'].get('model_ref')] * 2,
                    'hardware_ref': [baseline['body'].get('hardware_ref')] * 2}
        evidence = {'baseline': baseline, 'candidate': {'profile': plan.get('candidate_profile'), 'patch': plan.get('patch'),
                                  'source_diff_sha256': hashlib.sha256(plan['source_diff'].encode()).hexdigest()
                                  if plan.get('source_diff') else None},
                    'identity': identity, 'trials': list(trials), 'gates': gates, 'gate_notes': notes,
                    'verdict': verdict, 'ab': ab, 'run': run.record(),
                    'verifier': {'kind': 'remote_open_loop_goodput', 'code_sha256': CODE_SHA256,
                                 'fixture_sha256': (ab or {}).get('sweep') and hashlib.sha256(
                                     json.dumps(ab['sweep'], sort_keys=True).encode()).hexdigest(),
                                 'repeats': self.repeats, 'order': 'alternating AB/BA, fresh server per trial'}}
        artifact = self.repository.artifact(self.repository.encode(evidence).encode())
        digest = artifact if isinstance(artifact, str) else artifact['sha256']
        verdict = {**verdict, 'evidence': {'sha256': digest}, 'gates': gates, 'cost_usd': run.cost_usd}
        knee = (verdict.get('knee') or {}).get('baseline')
        if knee and gates.get('health') and gates.get('complete'):
            # every A/B run also measures the baseline: keep its knee as the deployment's latest measurement
            record_measurement(self.repository, plan['baseline_deployment'], 'knee', knee,
                               source={'remote_ab': plan['id'], 'evidence_sha256': digest})
        self.record(plan, verdict)
        return verdict

    def record(self, plan, verdict):
        actions = self.actions(plan)
        baseline = self.repository.read('deployment', plan['baseline_deployment'])
        experiment = {'plan': {'hypothesis': plan.get('hypothesis', ''), 'actions': actions,
                               'baseline_ref': f"{baseline['id']}@{baseline['revision']}",
                               'treatment_signature': treatment_signature(actions)},
                      'candidate': None, 'verdict': verdict}
        key = f"remote-ab:{plan['id']}:{verdict['evidence']['sha256'][:16]}"
        repo = self.repository
        with repo.transaction() as db:
            job_id = repo.submit_in(db, key, {'kind': 'remote_ab', 'plan_id': plan['id']})
            if db.execute('SELECT state FROM jobs WHERE id=?', (job_id,)).fetchone()['state'] == 'pending':
                db.execute("UPDATE jobs SET state='completed',result=?,updated=? WHERE id=?",
                           (repo.encode({'experiments': [experiment]}), repo.clock(), job_id))
                repo.event(db, job_id, 'completed', {'remote_ab': plan['id'], 'decision': verdict['decision']})
        return job_id
