"""Controller-owned campaign control (harness H3 step 2).

The planner proposes; the verifier judges; this module (deterministic, no model)
decides what each verified outcome means for the campaign:

- `diagnose_outcome`: turns a verified (or failed) trial into an evidence-linked
  diagnosis: what happened, which bottleneck the measurements point at, the
  remaining gap to the declared target, and guidance for the next hypothesis. Every
  diagnosis names the trial (iteration, treatment signature, evidence) it came from,
  so the next plan's lineage is traceable.
- `CampaignState`: tracks the best verified candidate, consecutive non-improving
  iterations and the stop decision.

A campaign stops on exactly one declared reason: `target_met`, `iteration_limit`,
`budget_exhausted`, `stagnation`, `deadline`, `cancelled` or `needs_implementation`.
"Continue until the target is reached" always means target reached OR a declared
limit reached; success is never assumed. Without a declared target the first `keep`
ends the campaign (the earlier behaviour).
"""
from __future__ import annotations

import math

STOP_REASONS = ('target_met', 'kept', 'iteration_limit', 'budget_exhausted', 'stagnation', 'deadline', 'cancelled',
                'needs_implementation')
PHASES = (('engine_queue_ms', 'queue_admission'), ('engine_prefill_ms', 'prefill'), ('engine_decode_ms', 'decode'))
DEFAULT_STAGNATION = 2


def _num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse_target(raw):
    """{metric: {'direction': 'min'|'max', 'value': number}} -> validated dict (or None)."""
    if raw in (None, {}):
        return None
    from inferstack.contracts.metrics import require_metric
    if not isinstance(raw, dict) or len(raw) > 8:
        raise ValueError('target must map up to 8 metrics to {direction, value}')
    out = {}
    for metric, spec in raw.items():
        require_metric(metric)
        if not isinstance(spec, dict) or spec.get('direction') not in ('min', 'max') or not _num(spec.get('value')):
            raise ValueError(f'target {metric} needs direction min|max and a finite value')
        out[metric] = {'direction': spec['direction'], 'value': float(spec['value'])}
    return out


def candidate_value(verdict, metric):
    entry = ((verdict or {}).get('metrics') or {}).get(metric) or {}
    value = (entry.get('candidate') or {}).get('median')
    return value if _num(value) else None


def gaps(verdict, target):
    """Per-metric remaining gap to the target (0 when satisfied, None when unmeasured)."""
    out = {}
    for metric, spec in (target or {}).items():
        value = candidate_value(verdict, metric)
        if value is None:
            out[metric] = None
        elif spec['direction'] == 'min':
            out[metric] = max(0.0, value - spec['value'])
        else:
            out[metric] = max(0.0, spec['value'] - value)
    return out


def target_met(verdict, target):
    if not target or (verdict or {}).get('decision') != 'keep':
        return False
    remaining = gaps(verdict, target)
    return all(g == 0.0 for g in remaining.values())


def relative_gain(verdict, metrics=None):
    """Mean relative gain over the objective metrics (positive = better); ranks verified candidates."""
    values = []
    for metric, entry in ((verdict or {}).get('metrics') or {}).items():
        if metrics and metric not in metrics:
            continue
        base = (entry.get('baseline') or {}).get('median')
        gain = entry.get('gain')
        if _num(base) and base and _num(gain):
            values.append(gain / abs(base))
    return sum(values) / len(values) if values else None


def _bottleneck(verdict):
    """Attribute the remaining time from the candidate's measured phases (no model, no text)."""
    medians = {name: candidate_value(verdict, name) for name, _ in PHASES}
    measured = {label: medians[name] for name, label in PHASES if medians[name] is not None}
    if len(measured) >= 2 and sum(measured.values()) > 0:
        label, value = max(measured.items(), key=lambda kv: kv[1])
        share = value / sum(measured.values())
        if share >= 0.5:
            return {'bottleneck': label, 'confidence': 'evidenced',
                    'signal': {'phase_share': round(share, 3), 'phases_ms': measured}}
        return {'bottleneck': 'uncertain', 'confidence': 'suspected',
                'signal': {'phase_share': round(share, 3), 'phases_ms': measured}}
    ttft, e2e = candidate_value(verdict, 'ttft_client_ms'), candidate_value(verdict, 'e2e_ms')
    if ttft is not None and e2e:
        share = ttft / e2e
        label = 'prefill' if share > 0.5 else 'decode'
        return {'bottleneck': label, 'confidence': 'suspected',
                'signal': {'ttft_share_of_e2e': round(share, 3)}}
    return {'bottleneck': 'uncertain', 'confidence': 'unknown', 'signal': None}


def diagnose_outcome(outcome, iteration, target=None, *, error=None):
    """Evidence-linked diagnosis of one trial, consumed by the next planning iteration."""
    plan = (outcome or {}).get('plan') or {}
    verdict = (outcome or {}).get('verdict') or {}
    decision = verdict.get('decision')
    reasons = [str(r) for r in verdict.get('reasons') or []]
    parent = {'iteration': iteration, 'treatment_signature': plan.get('treatment_signature'),
              'evidence': (verdict.get('evidence') or {}).get('sha256') if isinstance(verdict.get('evidence'), dict)
              else None, 'trial_refs': verdict.get('trial_refs')}
    attribution = _bottleneck(verdict)
    remaining = gaps(verdict, target) if target else None
    if error is not None:
        finding, guidance = 'implementation_failed', (
            'the treatment could not be built or launched; fix or replace it with a different, smaller change; '
            'the identical treatment will be refused')
    elif any(r.startswith('required gate failed') for r in reasons):
        finding, guidance = 'gate_failed', (
            'the treatment broke a required gate (correctness/quality/health); do not build on it; choose a '
            'treatment that preserves the failed gate')
    elif decision == 'retest':
        finding, guidance = 'noisy', 'measurements were too noisy to decide; repeat with more repeats or a ' \
                                     'longer workload before changing the treatment'
    elif decision == 'reject' and 'SLO failed' in reasons:
        finding, guidance = 'slo_failed', 'a declared limit was violated; reduce the change or target the ' \
                                          f"{attribution['bottleneck']} phase differently"
    elif decision == 'reject':
        finding, guidance = 'regressed', 'the treatment made the objective worse; revert direction'
    elif decision == 'inconclusive':
        finding, guidance = 'no_effect', (
            f"no material change; the measured bottleneck ({attribution['bottleneck']}) is likely elsewhere than "
            'this treatment acts; target that phase or gather a bounded profile first')
    elif decision == 'keep' and target and not target_met(verdict, target):
        finding, guidance = 'partial', (
            f"verified improvement but the target is not met (remaining gap {remaining}); the next hypothesis "
            f"should address the remaining {attribution['bottleneck']} time, compared against the original "
            'baseline')
    elif decision == 'keep':
        finding, guidance = 'target_met' if target else 'improved', 'verified improvement'
    else:
        finding, guidance = 'unknown', 'no verifier decision recorded'
    return {'version': 1, 'parent': parent, 'decision': decision, 'finding': finding,
            'reasons': reasons[:5], **attribution, 'remaining_gap': remaining,
            'gain': relative_gain(verdict, list(target) if target else None),
            'error': str(error)[:500] if error is not None else None, 'next': guidance}


class CampaignState:
    """Best verified candidate, stagnation counter and the stop decision."""

    def __init__(self, target=None, *, stagnation=DEFAULT_STAGNATION):
        self.target, self.stagnation_limit = target, stagnation
        self.best, self.best_gain, self.stalled = None, None, 0

    def observe(self, iteration, outcome):
        """Update with a verified outcome; returns a stop reason or None."""
        verdict = outcome.get('verdict') or {}
        if verdict.get('decision') == 'keep':
            gain = relative_gain(verdict, list(self.target) if self.target else None)
            if self.best is None or (gain is not None and (self.best_gain is None or gain > self.best_gain)):
                self.best = {'iteration': iteration, 'candidate': outcome.get('candidate'),
                             'treatment_signature': (outcome.get('plan') or {}).get('treatment_signature'),
                             'gain': gain, 'metrics': verdict.get('metrics'),
                             'remaining_gap': gaps(verdict, self.target) if self.target else None}
                self.best_gain, self.stalled = gain, 0
            else:
                self.stalled += 1
            if not self.target:
                return 'kept'
            if target_met(verdict, self.target):
                return 'target_met'
        else:
            self.stalled += 1
        if self.target and self.stagnation_limit and self.stalled >= self.stagnation_limit:
            return 'stagnation'
        return None

    def result(self, history, stop_reason, lineage):
        decision = 'keep' if self.best is not None else (
            history[-1]['verdict']['decision'] if history else 'inconclusive')
        return {'decision': decision, 'experiments': history, 'stop_reason': stop_reason,
                'target': self.target, 'target_met': stop_reason == 'target_met', 'best': self.best,
                'lineage': lineage}
