"""P4: inference-driven scaling, advisory replay first, exactly one actuator (control logic, CPU only)."""
import json
from dataclasses import asdict

import pytest

from inferstack.infrastructure import AllocationManager
from inferstack.infrastructure.allocations import QuotaExceeded
from inferstack.infrastructure.remote import KubernetesProvider
from inferstack.scaling.autoscale import (OWNER_ANNOTATION, Autoscaler, ScalingPolicy, inflight_query,
                                          reconcile_accounting, recommend, scaled_object)
from inferstack.storage.repository import Repository


def trace(values, step=15.0):
    return [(i * step, v) for i, v in enumerate(values)]


def test_policy_refuses_scale_to_zero_and_reports_conflicting_limits():
    with pytest.raises(ValueError, match='scale-to-zero'):
        ScalingPolicy(min_replicas=0)
    policy = ScalingPolicy(max_replicas=8, gpus_per_replica=2, max_gpus=6, gateway_max_inflight=10)
    assert policy.replica_cap == 3
    warnings = ' '.join(policy.warnings())
    assert 'GPU quota caps replicas at 3' in warnings and 'gateway max_inflight 10 rejects demand' in warnings


def test_burst_scales_up_in_bounded_steps_and_capacity_arrives_after_startup():
    policy = ScalingPolicy(max_replicas=6, target_inflight_per_replica=10, scale_up_max_step=2,
                           scale_up_cooldown_s=30, startup_s=60)
    result = recommend(trace([5] * 4 + [55] * 30), policy)
    ups = [d for d in result['decisions'] if d['action'].startswith('scale_up')]
    assert all(int(d['action'].split('+')[1]) <= 2 for d in ups)
    assert result['summary']['max_target'] == 6 and result['summary']['unmet_inflight_seconds'] > 0
    assert result['summary']['time_to_capacity_s'] and result['summary']['time_to_capacity_s'][0] >= 60


def test_idle_scales_down_only_after_stabilization():
    policy = ScalingPolicy(min_replicas=1, max_replicas=4, target_inflight_per_replica=10,
                           scale_down_stabilization_s=300, startup_s=0)
    result = recommend(trace([40] * 2 + [0] * 30), policy, initial=4)
    first_down = next(d for d in result['decisions'] if d['action'].startswith('scale_down'))
    assert first_down['t'] - 30 >= 300 and result['decisions'][-1]['target'] == 1


def test_noisy_demand_does_not_oscillate():
    policy = ScalingPolicy(max_replicas=4, target_inflight_per_replica=10, scale_down_stabilization_s=300,
                           startup_s=0)
    result = recommend(trace([19, 21] * 40), policy, initial=2)
    assert result['summary']['oscillations'] <= 1


def test_metrics_outage_holds_the_current_target():
    policy = ScalingPolicy(max_replicas=4, target_inflight_per_replica=10, startup_s=0)
    result = recommend(trace([30, None, None, None, 30]), policy, initial=3)
    assert [d['target'] for d in result['decisions']] == [3, 3, 3, 3, 3]
    assert all(d['reason'] == 'metrics unavailable' for d in result['decisions'][1:4])


def test_quota_caps_replicas_and_is_reported():
    policy = ScalingPolicy(max_replicas=10, target_inflight_per_replica=10, gpus_per_replica=1, max_gpus=2,
                           startup_s=0, scale_up_cooldown_s=0)
    result = recommend(trace([100] * 10), policy)
    assert result['summary']['max_target'] == 2 and result['summary']['capped_intervals'] == 10
    assert result['summary']['open_shortfall_since'] is not None  # demand still unmet: surfaced, not hidden


def test_long_generation_sustained_load_holds_capacity():
    policy = ScalingPolicy(max_replicas=4, target_inflight_per_replica=8, startup_s=30, scale_up_cooldown_s=15)
    result = recommend(trace([30] * 60), policy)
    assert result['decisions'][-1]['target'] == 4 and result['summary']['oscillations'] == 0


def test_scaled_object_is_bounded_and_marks_ownership():
    policy = ScalingPolicy(min_replicas=1, max_replicas=8, gpus_per_replica=1, max_gpus=4,
                           scale_down_stabilization_s=300)
    doc = scaled_object('isx-a', policy, prometheus_url='http://prom:9090', query=inflight_query('cand'))
    spec = doc['spec']
    assert spec['minReplicaCount'] == 1 and spec['maxReplicaCount'] == 4
    assert doc['metadata']['annotations'][OWNER_ANNOTATION] == 'keda'
    assert spec['advanced']['horizontalPodAutoscalerConfig']['behavior']['scaleDown'][
        'stabilizationWindowSeconds'] == 300
    assert spec['triggers'][0]['metadata']['query'] == 'sum(inferstack_gateway_inflight{deployment="cand"})'
    with pytest.raises(ValueError):
        inflight_query('x"}) or vector(1')


class Kubectl:
    def __init__(self, annotations=None):
        self.calls, self.annotations = [], annotations or {}

    def __call__(self, argv, input_text=None, timeout=120):
        self.calls.append(argv)
        if 'get' in argv and 'deployment' in argv:
            return 0, json.dumps({'metadata': {'name': 'isx-a', 'uid': 'u', 'annotations': self.annotations},
                                  'spec': {'replicas': 1}, 'status': {'readyReplicas': 1}}), ''
        if 'get' in argv and 'pods' in argv:
            return 0, json.dumps({'items': []}), ''
        return 0, '{"items": []}', ''


def test_manual_scale_is_refused_while_keda_owns_replicas():
    runner = Kubectl({OWNER_ANNOTATION: 'keda'})
    provider = KubernetesProvider(namespace='inferstack', runner=runner)
    with pytest.raises(RuntimeError, match='owned by keda'):
        provider.scale({'name': 'isx-a', 'uid': 'u'}, 2, timeout=1)
    assert not any('scale' in c for c in runner.calls)


def test_keda_handoff_requires_accepted_replay_of_the_same_policy():
    runner = Kubectl()
    autoscaler = Autoscaler(KubernetesProvider(namespace='inferstack', runner=runner), prometheus_url='http://p')
    policy = ScalingPolicy(max_replicas=3)
    with pytest.raises(ValueError, match='accepted advisory'):
        autoscaler.set_mode({'name': 'isx-a'}, 'keda', policy, deployment_id='cand')
    with pytest.raises(ValueError, match='different policy'):
        autoscaler.set_mode({'name': 'isx-a'}, 'keda', policy, deployment_id='cand',
                            advisory={'accepted': True, 'policy': asdict(ScalingPolicy(max_replicas=4))})
    doc = autoscaler.set_mode({'name': 'isx-a'}, 'keda', policy, deployment_id='cand',
                              advisory={'accepted': True, 'policy': asdict(policy)})
    assert doc['kind'] == 'ScaledObject'
    runner.calls.clear()
    autoscaler.set_mode({'name': 'isx-a'}, 'manual', policy, deployment_id='cand')
    verbs = [c[c.index('-n') + 2] for c in runner.calls]
    assert verbs == ['delete', 'annotate']  # the HPA owner is removed before manual control resumes


def test_accounting_follows_live_pods_including_surge(tmp_path):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    repo.submit('j', {})
    claim = repo.claim(100)
    manager = AllocationManager(repo, 'w', quotas={'kubernetes': {'max_gpus': 4}})
    alloc = manager.intent(claim, 's', purpose='p', provider='kubernetes', resource_type='k8s_deployment',
                           spec={}, gpu_count=1)
    manager.activate(alloc['id'], claim, {'name': 'isx'})
    assert reconcile_accounting(manager, alloc['id'], {'replicas': 2, 'ready': 2, 'pods': 3}, gpus_per_replica=1) == 3
    assert manager.get(alloc['id'])['gpu_count'] == 3
    with pytest.raises(QuotaExceeded):
        reconcile_accounting(manager, alloc['id'], {'pods': 5}, gpus_per_replica=1)


def test_unschedulable_pods_are_surfaced():
    provider = KubernetesProvider(namespace='inferstack')
    provider.pods = lambda external: [{'metadata': {'name': 'p'}, 'status': {'conditions': [
        {'type': 'PodScheduled', 'status': 'False', 'reason': 'Unschedulable',
         'message': '0/3 nodes are available: 3 Insufficient nvidia.com/gpu.'}]}}]
    assert provider.pending_pods({}) == [{'pod': 'p', 'reason': 'Unschedulable',
                                          'message': '0/3 nodes are available: 3 Insufficient nvidia.com/gpu.'}]
