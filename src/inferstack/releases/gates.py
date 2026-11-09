"""Rollout stage gates over durable serving evidence windows (pure functions).

A stage policy is frozen in the rollout before the canary starts (its SHA-256 is
recorded). Each step evaluates the candidate window against:
- absolute SLOs (`ttft_p95_ms`, `ttft_p99_ms`, `e2e_p95_ms`, `itl_p95_ms`,
  `stall_p95_ms`) and relative regressions against a comparable baseline
  (`*_regression`, a fraction; `max_latency_regression` keeps its legacy meaning
  on median time-to-first-byte);
- error, cancellation and empty-stream rates with separate denominators.

Outcomes: `rollback` (a sufficiently evidenced violation), `hold` (evidence not yet
sufficient or not yet comparable), `inconclusive` (the stage exceeded
`max_duration_s` without sufficient evidence; ends the rollout safely) or `advance`.
Missing data never passes and is never treated as zero. A percentile is only
available once its minimum sample count is reached (p95: 200, p99: 1000); the
policy cannot lower it below that.
"""
from __future__ import annotations

import hashlib
import json

PERCENTILE_GATES = {  # policy key -> (summary field, sample series)
    'ttft_p95_ms': ('ttft_p95_ms', 'ttft'), 'ttft_p99_ms': ('ttft_p99_ms', 'ttft'),
    'e2e_p95_ms': ('e2e_p95_ms', 'e2e'), 'itl_p95_ms': ('itl_p95_ms', 'itl'),
    'stall_p95_ms': ('stall_p95_ms', 'stall'),
}
REGRESSION_GATES = {'ttft_p95_regression': 'ttft_p95_ms', 'e2e_p95_regression': 'e2e_p95_ms',
                    'itl_p95_regression': 'itl_p95_ms', 'max_latency_regression': 'ttfb_p50_ms'}


def policy_sha256(stages, extra=None):
    body = json.dumps({'stages': stages, **(extra or {})}, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(body.encode()).hexdigest()


def comparable_baseline(stage, stable, retained):
    """Baseline metrics for relative gates: stable traffic in the same window, else the
    retained earlier-stage stable evidence (100% stage), else none."""
    if stable and stable.get('requests'):
        return stable, 'stable_same_window'
    if stage['weight'] == 100 and retained:
        return retained['metrics'], f"retained_stable_stage_{retained['stage']}"
    return None, None


def evaluate(stage, candidate, baseline, *, basis=None, elapsed, min_requests):
    checks, violations, waiting = {}, [], []
    requests = candidate.get('requests') or 0

    def rate(name, numerator, limit):
        if limit is None:
            return
        value = numerator / requests if requests else None
        ok = None if value is None else value <= limit
        checks[name] = {'value': value, 'limit': limit, 'pass': ok}
        if ok is False:
            violations.append(f"candidate {name.replace('_', ' ')} {value:.3f} > {limit}")

    rate('error_rate', candidate.get('errors') or 0, stage.get('max_error_rate'))
    rate('cancel_rate', candidate.get('cancelled') or 0, stage.get('max_cancel_rate'))
    rate('empty_stream_rate', candidate.get('streams_without_content') or 0, stage.get('max_empty_stream_rate'))
    for key, (field, series) in PERCENTILE_GATES.items():
        limit = stage.get(key)
        if limit is None:
            continue
        value = candidate.get(field)
        samples = (candidate.get('samples') or {}).get(series, 0)
        ok = None if value is None else value <= limit
        checks[key] = {'value': value, 'limit': limit, 'samples': samples, 'pass': ok}
        if ok is False:
            violations.append(f'{key} {value:.1f}ms > {limit}ms over {samples} samples')
        elif ok is None:
            waiting.append(f'{key}: insufficient samples ({samples})')
    for key, field in REGRESSION_GATES.items():
        limit = stage.get(key)
        if limit is None:
            continue
        value = candidate.get(field)
        base = (baseline or {}).get(field)
        if value is None or not base:
            checks[key] = {'value': value, 'baseline': base, 'basis': basis, 'limit': limit, 'pass': None}
            waiting.append('latency evidence unavailable for the stage window' if key == 'max_latency_regression'
                           else f'{key}: ' + ('candidate evidence insufficient' if value is None
                                             else 'no comparable baseline'))
            continue
        regression = (value - base) / base
        ok = regression <= limit
        checks[key] = {'value': value, 'baseline': base, 'basis': basis, 'regression': regression,
                       'limit': limit, 'pass': ok}
        if not ok:
            label = 'candidate TTFB regression' if key == 'max_latency_regression' else key
            violations.append(f'{label} {regression:.1%} > {limit:.0%} ({field} {value:.1f} vs {base:.1f}, {basis})')
    needed = max(stage.get('min_samples') or 0, min_requests)
    # Percentile violations are already sample-gated (a percentile exists only past its
    # minimum count); rate and median violations act on any observed traffic.
    if violations and requests:
        return {'decision': 'rollback', 'reasons': violations, 'checks': checks}
    if requests < needed:
        waiting.insert(0, f'only {requests} candidate requests (need {needed})')
    if elapsed < stage['min_duration_s']:
        waiting.insert(0, 'minimum stage duration not reached')
    if waiting:
        max_duration = stage.get('max_duration_s')
        if max_duration is not None and elapsed >= max_duration:
            return {'decision': 'inconclusive', 'reasons': waiting, 'checks': checks}
        return {'decision': 'hold', 'reasons': waiting, 'checks': checks}
    return {'decision': 'advance', 'reasons': [], 'checks': checks}


def evaluation_check(repository, release, policy):
    """Release-bound held-out evaluation. Returns (status, reason): passed | failed | missing | corrupt.

    The evaluation is the verification evidence the release was created from (suite =
    workload fixture digest, evaluator = verifier code digest). Candidate code cannot
    change either; a policy may pin both digests."""
    evaluation = (release.get('body') or {}).get('evaluation')
    if not evaluation:
        return 'missing', 'release has no bound evaluation'
    for key in ('suite_sha256', 'evaluator_sha256'):
        if policy.get(key) and policy[key] != evaluation.get(key):
            return 'corrupt', f'{key} {evaluation.get(key)} is not the pinned {policy[key]}'
    try:
        raw = repository.read_artifact(evaluation['evidence_sha256'])
    except (KeyError, ValueError, OSError):
        return 'corrupt', 'evaluation evidence artifact unreadable'
    if hashlib.sha256(raw).hexdigest() != evaluation['evidence_sha256']:
        return 'corrupt', 'evaluation evidence digest mismatch'
    try:
        evidence = json.loads(raw)
    except ValueError:
        return 'corrupt', 'evaluation evidence is not valid JSON'
    fixture = (evidence.get('verifier') or {}).get('fixture_sha256')
    if fixture != evaluation.get('suite_sha256'):
        return 'corrupt', 'evaluation suite digest does not match its evidence'
    quality = (evidence.get('gates') or {}).get('quality')
    if quality is False or (evidence.get('verdict') or {}).get('decision') == 'reject':
        return 'failed', 'bound evaluation records a quality regression'
    if quality is not True:
        return 'missing', 'bound evaluation has no quality result'
    return 'passed', 'bound held-out evaluation passed'
