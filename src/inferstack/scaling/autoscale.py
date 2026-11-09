"""Inference-driven autoscaling: policy, offline replay, KEDA objects and single ownership.

Modes (one owner per replica count, docs/CLOUD_SERVING_RESEARCH_PLAN.md P4):
- `manual`   InferStack's `KubernetesProvider.scale` is the only actuator.
- `advisory` nothing is actuated; `recommend()` replays a demand trace (or live
             samples) and records what a policy *would* do. Required before KEDA.
- `keda`     one ScaledObject (and the HPA KEDA generates) owns the replica count.
             The Deployment carries `inferstack.io/scaling-owner: keda`, and
             `KubernetesProvider.scale` refuses to act on it.

Signals: in-flight requests per release first (supported without Flow Control);
queue-based signals only after bounded Flow Control is qualified. GPU utilisation
is diagnostic, never a scaling signal. Autoscaling cannot create GPU capacity:
pending or unschedulable pods and quota limits are surfaced, not hidden.
Scale-to-zero is a separate opt-in qualification and is refused here.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

MODES = ('manual', 'advisory', 'keda')
OWNER_ANNOTATION = 'inferstack.io/scaling-owner'


@dataclass(frozen=True)
class ScalingPolicy:
    min_replicas: int = 1
    max_replicas: int = 4
    target_inflight_per_replica: float = 8.0
    scale_up_max_step: int = 2            # replicas added per evaluation at most
    scale_up_cooldown_s: float = 60.0
    scale_down_stabilization_s: float = 300.0
    startup_s: float = 120.0              # measured time-to-ready (P2 startup evidence)
    gpus_per_replica: int = 1
    max_gpus: int | None = None           # quota: autoscaling never exceeds it
    gateway_max_inflight: int | None = None

    def __post_init__(self):
        if self.min_replicas < 1:
            raise ValueError('min_replicas must be >= 1; scale-to-zero is a separate opt-in qualification')
        if self.max_replicas < self.min_replicas:
            raise ValueError('max_replicas must be >= min_replicas')
        if not self.target_inflight_per_replica > 0 or self.scale_up_max_step < 1:
            raise ValueError('target_inflight_per_replica must be > 0 and scale_up_max_step >= 1')
        if self.startup_s < 0 or self.scale_down_stabilization_s < 0:
            raise ValueError('durations must be non-negative')

    @property
    def replica_cap(self):
        if self.max_gpus is None or not self.gpus_per_replica:
            return self.max_replicas
        return min(self.max_replicas, self.max_gpus // self.gpus_per_replica)

    def warnings(self):
        out = []
        if self.replica_cap < self.max_replicas:
            out.append(f'GPU quota caps replicas at {self.replica_cap} (< max_replicas {self.max_replicas})')
        if self.gateway_max_inflight is not None:
            needed = math.ceil(self.replica_cap * self.target_inflight_per_replica)
            if self.gateway_max_inflight < needed:
                out.append(f'gateway max_inflight {self.gateway_max_inflight} rejects demand before the '
                           f'autoscaler sees it (needs >= {needed} at {self.replica_cap} replicas)')
        return out


def recommend(trace, policy: ScalingPolicy, *, initial=None, interval_s=15.0):
    """Replay a demand trace [(t, inflight or None), ...] through the policy.

    None samples (metrics outage) hold the current target: never scale on missing
    data. New replicas only add capacity after `startup_s`. Returns the decision log
    and summary metrics: time-to-capacity per burst, oscillations, replica-seconds,
    unmet demand (inflight above ready capacity) and capacity-capped intervals."""
    if not trace:
        raise ValueError('empty trace')
    cap = policy.replica_cap
    target = min(max(initial or policy.min_replicas, policy.min_replicas), cap)
    ready_at = [trace[0][0] - policy.startup_s] * target  # initial replicas are already ready
    last_up, below_since = -math.inf, None
    log, unmet, capped, replica_s, changes = [], 0.0, 0, 0.0, []
    shortfall_since, time_to_capacity = None, []
    prev_t = trace[0][0]
    for t, demand in trace:
        dt = t - prev_t
        prev_t = t
        ready = sum(1 for r in ready_at if r <= t)
        replica_s += len(ready_at) * dt
        action, reason = 'hold', 'metrics unavailable' if demand is None else 'within target'
        if demand is not None:
            desired = max(policy.min_replicas, math.ceil(demand / policy.target_inflight_per_replica))
            if desired > cap:
                capped += 1
                reason = f'demand needs {desired} replicas; capped at {cap} (max_replicas/GPU quota)'
                desired = cap
            capacity = ready * policy.target_inflight_per_replica
            unmet += max(0.0, demand - capacity) * dt
            if demand > capacity and shortfall_since is None:
                shortfall_since = t
            elif demand <= capacity and shortfall_since is not None:
                time_to_capacity.append(t - shortfall_since)
                shortfall_since = None
            if desired > target and t - last_up >= policy.scale_up_cooldown_s:
                step = min(desired - target, policy.scale_up_max_step)
                target += step
                ready_at += [t + policy.startup_s] * step
                last_up, below_since = t, None
                action = f'scale_up +{step}'
                changes.append(+1)
            elif desired < target:
                below_since = t if below_since is None else below_since
                if t - below_since >= policy.scale_down_stabilization_s:
                    remove = target - desired
                    target = desired
                    ready_at = sorted(ready_at)[:target]  # remove newest first (drain before termination)
                    below_since = None
                    action = f'scale_down -{remove}'
                    changes.append(-1)
                else:
                    reason = 'below target; within scale-down stabilization'
            else:
                below_since = None
        log.append({'t': t, 'inflight': demand, 'target': target, 'ready': ready, 'action': action, 'reason': reason})
    oscillations = sum(1 for a, b in zip(changes, changes[1:]) if a != b)
    return {'policy': asdict(policy), 'warnings': policy.warnings(), 'decisions': log,
            'summary': {'replica_seconds': replica_s, 'unmet_inflight_seconds': unmet,
                        'time_to_capacity_s': time_to_capacity, 'open_shortfall_since': shortfall_since,
                        'oscillations': oscillations, 'capped_intervals': capped,
                        'max_target': max(d['target'] for d in log)}}


def scaled_object(deployment_name, policy: ScalingPolicy, *, prometheus_url, query, release_key=None):
    """One KEDA ScaledObject per release Deployment (the only actuator in `keda` mode)."""
    labels = {'app.kubernetes.io/managed-by': 'inferstack', **({'inferstack.io/release': release_key}
                                                              if release_key else {})}
    return {'apiVersion': 'keda.sh/v1alpha1', 'kind': 'ScaledObject',
            'metadata': {'name': deployment_name, 'labels': labels,
                         'annotations': {OWNER_ANNOTATION: 'keda'}},
            'spec': {'scaleTargetRef': {'name': deployment_name},
                     'minReplicaCount': policy.min_replicas, 'maxReplicaCount': policy.replica_cap,
                     'pollingInterval': 15, 'cooldownPeriod': int(policy.scale_down_stabilization_s),
                     'fallback': {'failureThreshold': 3, 'replicas': policy.min_replicas},
                     'advanced': {'horizontalPodAutoscalerConfig': {'behavior': {
                         'scaleUp': {'stabilizationWindowSeconds': 0, 'selectPolicy': 'Min',
                                     'policies': [{'type': 'Pods', 'value': policy.scale_up_max_step,
                                                   'periodSeconds': int(policy.scale_up_cooldown_s)}]},
                         'scaleDown': {'stabilizationWindowSeconds': int(policy.scale_down_stabilization_s),
                                       'policies': [{'type': 'Pods', 'value': 1, 'periodSeconds': 60}]}}}},
                     'triggers': [{'type': 'prometheus', 'metadata': {
                         'serverAddress': prometheus_url, 'query': query,
                         'threshold': str(policy.target_inflight_per_replica),
                         'ignoreNullValues': 'false'}}]}}


def inflight_query(deployment_id):
    """In-flight requests for one deployment as exported by the InferStack gateway (/metrics)."""
    if not deployment_id.replace('-', '').replace('_', '').replace('.', '').isalnum():
        raise ValueError('invalid deployment id for a metric selector')
    return f'sum(inferstack_gateway_inflight{{deployment="{deployment_id}"}})'


class Autoscaler:
    """Applies a release's scaling mode through a KubernetesProvider; one actuator at a time."""

    def __init__(self, provider, *, prometheus_url):
        self.provider, self.prometheus_url = provider, prometheus_url

    def owner(self, external):
        code, out, _ = self.provider.run(self.provider.base + ['get', 'deployment', external['name'], '-o', 'json'],
                                         check=False)
        if code:
            return None
        import json
        return ((json.loads(out).get('metadata') or {}).get('annotations') or {}).get(OWNER_ANNOTATION, 'manual')

    def set_mode(self, external, mode, policy: ScalingPolicy, *, deployment_id, release_key=None, advisory=None):
        """Switch ownership. `keda` requires an accepted advisory replay (explicit handoff)."""
        import json
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        name = external['name']
        if mode == 'keda':
            if not advisory or advisory.get('accepted') is not True:
                raise ValueError('keda mode requires an accepted advisory replay of this policy')
            if advisory.get('policy') != asdict(policy):
                raise ValueError('the accepted advisory replay was for a different policy')
            doc = scaled_object(name, policy, prometheus_url=self.prometheus_url,
                                query=inflight_query(deployment_id), release_key=release_key)
            self.provider.run(self.provider.base + ['annotate', 'deployment', name, '--overwrite',
                                                    f'{OWNER_ANNOTATION}=keda'])
            self.provider.run(self.provider.base + ['apply', '-f', '-'], input_text=json.dumps(doc))
            return doc
        # Leaving keda: delete the ScaledObject first so no HPA fights the new owner.
        self.provider.run(self.provider.base + ['delete', 'scaledobject', name, '--ignore-not-found'], check=False)
        self.provider.run(self.provider.base + ['annotate', 'deployment', name, '--overwrite',
                                                f'{OWNER_ANNOTATION}={mode}'])
        return None


def reconcile_accounting(allocations, alloc_id, status, *, gpus_per_replica):
    """Make GPU accounting follow the replicas that actually exist (incl. surge), not the target."""
    if status is None:
        return None
    # Live pods (including surge and terminating-but-holding-GPU pods) when known, else the
    # largest of the controller's own counts. Never the autoscaler's target.
    observed = status['pods'] if status.get('pods') is not None else \
        max(status.get('replicas', 0), status.get('updated', 0), status.get('ready', 0))
    allocations.rescale(alloc_id, observed * gpus_per_replica)
    return observed
