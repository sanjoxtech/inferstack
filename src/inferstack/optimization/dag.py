"""Typed experiment plans (DAG) and pre-execution policy validation.

A plan states a hypothesis and an executable DAG of actions. The model proposes;
`validate_plan` decides. Validation is structural (unique ids, known
dependencies, acyclic, exactly one terminal measurement), referential (pinned
baseline, evidence citations limited to what the job may cite, registered
candidates and techniques) and policy-based (no action targets the baseline or
any routed release deployment, gates include the mandatory set, estimates within
remaining budget, source patches confined to allowed roots, no identical retry of
a treatment that already failed).

The schema has no field for a verdict or pass flag: a plan can never declare its
own success.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from inferstack.contracts.metrics import require_metric

MANDATORY_GATES = {'correctness', 'quality', 'health', 'comparable', 'complete'}
KNOWN_GATES = MANDATORY_GATES | {'numerical_oracle', 'quality_budget', 'cancellation', 'cache_isolation', 'overload',
                                 'liveness', 'performance', 'slo'}
PATCH_ROOTS = ('src/inferstack/kernels/', 'src/inferstack/runtime/')
PROFILE_KEYS = {
    'vllm': {'max_num_seqs', 'max_num_batched_tokens', 'enable_prefix_caching', 'enable_chunked_prefill',
             'gpu_memory_utilization', 'block_size', 'kv_cache_dtype'},
    'sglang': {'max_running_requests', 'chunked_prefill_size', 'disable_radix_cache', 'mem_fraction_static',
               'schedule_policy', 'kv_cache_dtype'},
    'inferstack_native': {'max_num_batched_tokens', 'max_running_requests', 'max_prefill_chunk_tokens',
                          'decode_priority', 'forward_fusion'},
}


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Action(Strict):
    id: str = Field(pattern=r'^[a-z][a-z0-9_-]{0,39}$')
    type: Literal['config_change', 'plugin_install', 'source_patch', 'measure']
    depends_on: list[str] = Field(default_factory=list)
    params: dict


class Estimate(Strict):
    gpu_seconds: float = Field(ge=0)
    wall_seconds: float = Field(gt=0)
    provider_tokens: int = Field(default=0, ge=0)


class StopRule(Strict):
    max_iterations: int = Field(ge=1, le=20)
    stop_on: list[Literal['keep', 'budget_exhausted', 'reject_twice', 'inconclusive_twice']] = Field(min_length=1)


class ExperimentPlan(Strict):
    hypothesis: str = Field(min_length=10, max_length=4000)
    baseline_ref: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9._:/-]*@[1-9][0-9]*$')
    evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    actions: list[Action] = Field(min_length=1, max_length=20)
    expected_measurements: list[str] = Field(min_length=1)
    gates: list[str] = Field(min_length=1)
    estimate: Estimate
    stop_rule: StopRule

    @field_validator('expected_measurements')
    @classmethod
    def metrics_registered(cls, names):
        for name in names:
            require_metric(name)
        return names

    @field_validator('gates')
    @classmethod
    def gates_known(cls, gates):
        unknown = set(gates) - KNOWN_GATES
        if unknown:
            raise ValueError(f'unknown gates: {sorted(unknown)}')
        return gates


class PlanRejected(ValueError):
    pass


def treatment_signature(plan: ExperimentPlan) -> str:
    """Identity of the treatment, independent of wording and action ids."""
    body = sorted(json.dumps({'type': a.type, 'params': a.params}, sort_keys=True)
                  for a in plan.actions if a.type != 'measure')
    return hashlib.sha256('\n'.join(body).encode()).hexdigest()


def _acyclic(actions):
    ids = [a.id for a in actions]
    if len(set(ids)) != len(ids):
        raise PlanRejected('duplicate action ids')
    deps = {a.id: set(a.depends_on) for a in actions}
    for a in actions:
        missing = deps[a.id] - set(ids)
        if missing:
            raise PlanRejected(f'action {a.id} depends on unknown {sorted(missing)}')
    order, ready = [], [i for i in ids if not deps[i]]
    remaining = {i: set(d) for i, d in deps.items()}
    while ready:
        node = ready.pop(0)
        order.append(node)
        for other, d in remaining.items():
            if node in d:
                d.discard(node)
                if not d and other not in order and other not in ready:
                    ready.append(other)
    if len(order) != len(ids):
        raise PlanRejected('action graph has a cycle')
    return order


def validate_plan(raw: dict, *, repository, baseline_id, candidate_ids, allowed_evidence, history_signatures,
                  remaining_budget=None, production_ids=()):
    """Parse and validate a proposed plan. Returns (plan, normalized dict). Raises PlanRejected."""
    from pydantic import ValidationError
    try:
        plan = ExperimentPlan.model_validate(raw)
    except ValidationError as exc:
        raise PlanRejected('plan does not match schema: ' + '; '.join(
            f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:5])) from None
    order = _acyclic(plan.actions)
    terminal = [a for a in plan.actions if a.type == 'measure']
    if len(terminal) != 1 or any(terminal[0].id in a.depends_on for a in plan.actions):
        raise PlanRejected('plan needs exactly one terminal measure action')
    missing_gates = MANDATORY_GATES - set(plan.gates)
    if missing_gates:
        raise PlanRejected(f'plan omits mandatory gates {sorted(missing_gates)}')

    bid, brev = plan.baseline_ref.rsplit('@', 1)
    baseline = repository.read('deployment', baseline_id)
    if bid != baseline_id or int(brev) != baseline['revision']:
        raise PlanRejected(f'baseline_ref must be the current baseline snapshot {baseline_id}@{baseline["revision"]}')
    unknown = set(plan.evidence_ids) - set(allowed_evidence)
    if unknown:
        raise PlanRejected(f'citations not available to this job: {sorted(unknown)}')
    protected = {baseline_id, *production_ids}
    engine = baseline['body']['engine']
    for action in plan.actions:
        p = action.params
        if action.type == 'config_change':
            if set(p) == {'candidate_deployment_id'}:
                if p['candidate_deployment_id'] not in candidate_ids:
                    raise PlanRejected(f"{action.id}: {p['candidate_deployment_id']} is not a registered candidate")
                if p['candidate_deployment_id'] in protected:
                    raise PlanRejected(f'{action.id}: may not modify the baseline or a production deployment')
            elif set(p) == {'profile_patch'} and isinstance(p['profile_patch'], dict) and p['profile_patch']:
                bad = set(p['profile_patch']) - PROFILE_KEYS.get(engine, set())
                if bad:
                    raise PlanRejected(f'{action.id}: profile keys not allowed for {engine}: {sorted(bad)}')
            else:
                raise PlanRejected(f'{action.id}: config_change needs candidate_deployment_id or profile_patch')
        elif action.type in ('plugin_install', 'source_patch'):
            technique_id = p.get('technique_id')
            try:
                from inferstack.research.techniques import assess
                verdict = assess(repository, technique_id, baseline_id)
            except (KeyError, ValueError, TypeError):
                raise PlanRejected(f'{action.id}: unknown technique {technique_id!r}') from None
            if verdict['verdict'] == 'incompatible':
                raise PlanRejected(f'{action.id}: technique {technique_id} is incompatible: {verdict["blockers"]}')
            if action.type == 'plugin_install':
                if set(p) != {'technique_id', 'dependency'} or not verdict['executable']:
                    raise PlanRejected(f'{action.id}: plugin_install needs an executable technique with pinned '
                                       'dependencies and params {technique_id, dependency}')
                pinned = {f'{d["name"]}{d["version"]}' for d in
                          repository.read('technique', technique_id)['body'].get('dependencies', [])}
                if p['dependency'] not in pinned:
                    raise PlanRejected(f"{action.id}: dependency {p['dependency']!r} is not a pinned dependency "
                                       f'of {technique_id}')
            else:
                files = p.get('files')
                if set(p) - {'technique_id', 'files', 'description'} or not isinstance(files, list) or not files:
                    raise PlanRejected(f'{action.id}: source_patch needs technique_id, files, description')
                for path in files:
                    if not isinstance(path, str) or '..' in path.split('/') or not path.startswith(PATCH_ROOTS):
                        raise PlanRejected(f'{action.id}: patch path {path!r} outside {PATCH_ROOTS}')
        elif action.type == 'measure':
            if set(p) - {'workload_ref', 'objective_ref', 'repeats'}:
                raise PlanRejected(f'{action.id}: measure accepts workload_ref, objective_ref, repeats')

    signature = treatment_signature(plan)
    if signature in history_signatures:
        raise PlanRejected('identical treatment was already tried in this job; propose a different one')
    if remaining_budget:
        for key in ('gpu_seconds', 'provider_tokens'):
            limit = remaining_budget.get(key)
            if limit is not None and getattr(plan.estimate, key) > limit:
                raise PlanRejected(f'estimate {key}={getattr(plan.estimate, key)} exceeds remaining {limit}')
    normalized = plan.model_dump(mode='json')
    normalized.update({'action_order': order, 'treatment_signature': signature,
                       'baseline_id': baseline_id, 'baseline_revision': baseline['revision']})
    candidate = next((a.params['candidate_deployment_id'] for a in plan.actions
                      if a.type == 'config_change' and 'candidate_deployment_id' in a.params), None)
    if candidate is not None:  # compatibility with the endpoint executor
        normalized['candidate_id'] = candidate
        normalized['candidate_revision'] = repository.read('deployment', candidate)['revision']
    return plan, normalized
