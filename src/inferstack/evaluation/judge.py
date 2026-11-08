from __future__ import annotations

from dataclasses import dataclass
import statistics
from .metrics import finite_number


@dataclass(frozen=True)
class Objective:
    metric: str
    direction: str
    limit: float | None = None
    epsilon: float = 0.02

    def __post_init__(self):
        if self.direction not in {'min', 'max'} or not self.metric:
            raise ValueError('invalid objective')
        if not finite_number(self.epsilon) or self.epsilon >= 1:
            raise ValueError('epsilon must be in [0,1)')
        if self.limit is not None and not finite_number(self.limit):
            raise ValueError('limit must be finite and nonnegative')


MANDATORY = frozenset({'correctness', 'quality', 'health', 'comparable', 'complete'})


def evaluate(baseline, candidate, objectives, *, gates, min_trials=3, max_cv=0.1, required=None):
    """Constrained epsilon comparison over repeated aggregate metrics.

    Caller supplies immutable comparable experiment identities separately. This
    function never accepts missing evidence or interprets an LLM verdict as a gate.
    """
    if min_trials < 2 or not finite_number(max_cv) or not objectives:
        raise ValueError('invalid evaluation policy')
    if len({o.metric for o in objectives})!=len(objectives):raise ValueError('duplicate objective metric')
    # Every gate of the bound verification policy must have run and passed; a gate
    # that is unsupported or could not be measured (None/absent) is inconclusive.
    required = MANDATORY | set(required or ())
    # A gate that ran and failed is decisive even when other gates are unknown.
    failed = sorted(k for k in required if k in gates and gates[k] is not None and gates[k] is not True)
    if failed:
        return {'decision': 'reject', 'reasons': ['required gate failed: '+','.join(failed)]}
    absent = sorted(k for k in required if k not in gates or gates[k] is None)
    if absent:
        return {'decision': 'inconclusive', 'reasons': ['missing gates: '+','.join(absent)]}
    if len(baseline) < min_trials or len(candidate) < min_trials:
        return {'decision': 'inconclusive', 'reasons': ['insufficient independent trials']}
    summary = {}
    improved = False
    worse = False
    noisy = False
    slo_failed = False
    for objective in objectives:
        groups = []
        for trials in (baseline, candidate):
            values = [trial.get(objective.metric) for trial in trials]
            if not all(finite_number(v) for v in values):
                return {'decision': 'inconclusive', 'reasons': ['missing or invalid metric: '+objective.metric]}
            median = statistics.median(values)
            mean = statistics.mean(values)
            cv = statistics.stdev(values)/mean if mean else (0 if all(v==0 for v in values) else float('inf'))
            mad = statistics.median(abs(v-median) for v in values)
            groups.append({'median': median, 'cv': cv, 'mad': mad, 'trials': len(values)})
            noisy |= cv > max_cv
        b,c = [g['median'] for g in groups]
        tolerance = abs(b)*objective.epsilon
        # An improvement must also exceed run-to-run spread (2x the larger MAD).
        noise_floor = 2*max(groups[0]['mad'], groups[1]['mad'])
        gain = b-c if objective.direction == 'min' else c-b
        improved |= gain > max(tolerance, noise_floor)
        worse |= gain < -tolerance
        if objective.limit is not None:
            slo_failed |= c > objective.limit if objective.direction == 'min' else c < objective.limit
        summary[objective.metric] = {'baseline': groups[0], 'candidate': groups[1], 'gain': gain,
                                     'tolerance': tolerance, 'noise_floor': noise_floor}
    decision = 'reject' if slo_failed else 'retest' if noisy else 'reject' if worse else 'keep' if improved else 'inconclusive'
    return {'decision': decision, 'metrics': summary,
            'reasons': ['SLO failed'] if slo_failed else ['noise exceeds policy'] if noisy else
            ['material regression'] if worse else ['epsilon improvement'] if improved else ['no material improvement'],
            'scope': 'repeated aggregate metrics; tail sample adequacy must be established by complete gate'}
