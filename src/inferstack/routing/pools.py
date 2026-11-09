"""Release-scoped InferencePools with identity-gated pod membership.

Ownership (docs/CLOUD_SERVING_RESEARCH_PLAN.md):
- InferStack decides which release a request goes to (its existing per-request
  stable/candidate split; exactly one splitter) and which pods may serve a release.
- The llm-d / GAIE endpoint picker (EPP) chooses a replica *within* that release's
  pool: prefix-cache affinity, load, queue. InferStack never implements a second
  cache scheduler.

Each immutable release gets its own InferencePool and an HTTPRoute that matches the
`x-inferstack-release` header the InferStack gateway adds upstream, so a request can
never be scheduled onto another release's replica, and a warm rollback pool stays
addressable until its release is retired.

Membership fails closed: the pool selector requires `inferstack.io/admitted=true`,
which no pod template sets. `admit_pods` enumerates *every* pod of the release
(a request to a Service or a shared command-line declaration proves nothing about
all replicas), compares each pod's observed image digest and launch arguments with
the release's frozen identity, and only then labels it admitted. Mismatched pods are
labelled quarantined with the reason; pods whose identity cannot be observed yet
(not ready, no imageID) stay out of the pool. Replacements and scale-out pods are
new instances and are admitted only after the same check.

Trust level: the image digest comes from the kubelet's imageID and the revision from
the container arguments the API reports. That proves which artifact was launched,
not which tensors are resident in accelerator memory.
"""
from __future__ import annotations

import hashlib
import json
import re

from inferstack.infrastructure.remote import LABEL, k8s_name

RELEASE_LABEL = 'inferstack.io/release'
ADMITTED_LABEL = 'inferstack.io/admitted'
QUARANTINE_LABEL = 'inferstack.io/quarantine'
RELEASE_HEADER = 'x-inferstack-release'
POOL_API = 'inference.networking.k8s.io/v1'
ROUTE_API = 'gateway.networking.k8s.io/v1'


def release_key(release_id):
    """DNS-label-safe stable key for an immutable release ID."""
    return 'r-' + hashlib.sha256(release_id.encode()).hexdigest()[:16]


def pool_name(release_id):
    return 'pool-' + release_key(release_id)


def pool_manifests(release_id, *, port, gateway, epp_service, epp_port=9002, failure_mode='FailClose',
                   request_timeout='600s'):
    """InferencePool (admitted pods of this release only) + HTTPRoute keyed on the release header."""
    key, name = release_key(release_id), pool_name(release_id)
    labels = {'app.kubernetes.io/managed-by': 'inferstack', RELEASE_LABEL: key}
    if failure_mode not in ('FailClose', 'FailOpen'):
        raise ValueError('failure_mode must be FailClose or FailOpen')
    pool = {'apiVersion': POOL_API, 'kind': 'InferencePool',
            'metadata': {'name': name, 'labels': labels, 'annotations': {'inferstack.io/release-id': release_id}},
            'spec': {'selector': {'matchLabels': {RELEASE_LABEL: key, ADMITTED_LABEL: 'true'}},
                     'targetPorts': [{'number': int(port)}],
                     'endpointPickerRef': {'name': epp_service, 'port': {'number': int(epp_port)},
                                           'failureMode': failure_mode}}}
    route = {'apiVersion': ROUTE_API, 'kind': 'HTTPRoute', 'metadata': {'name': name, 'labels': labels},
             'spec': {'parentRefs': [{'name': gateway}],
                      'rules': [{'matches': [{'headers': [{'type': 'Exact', 'name': RELEASE_HEADER, 'value': key}]}],
                                 'backendRefs': [{'group': 'inference.networking.k8s.io', 'kind': 'InferencePool',
                                                  'name': name}],
                                 'timeouts': {'request': request_timeout}}]}}
    return [pool, route]


def pod_identity(pod):
    """What the API reports about one pod: uid, node, readiness, image digests and engine args."""
    status = pod.get('status') or {}
    digests, images_known = set(), True
    for container in status.get('containerStatuses') or []:
        image_id = container.get('imageID') or ''
        if '@sha256:' in image_id:
            digests.add(image_id.split('@', 1)[1])
        else:
            images_known = False
    engine = next((c for c in (pod.get('spec') or {}).get('containers') or [] if c.get('name') == 'engine'), {})
    ready = any(c.get('type') == 'Ready' and c.get('status') == 'True' for c in status.get('conditions') or [])
    labels = (pod.get('metadata') or {}).get('labels') or {}
    return {'pod': pod['metadata']['name'], 'uid': pod['metadata'].get('uid'),
            'node': (pod.get('spec') or {}).get('nodeName'), 'ready': ready,
            'image_digests': sorted(digests) if images_known and digests else None,
            'args': list(engine.get('args') or []), 'admitted': labels.get(ADMITTED_LABEL)}


def _flag(args, name):
    return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else None


def check_pod(identity, expected):
    """(decision, reason): admit | quarantine | pending. `expected`: image_digest, model_revision."""
    if not identity['ready']:
        return 'pending', 'not ready'
    if identity['image_digests'] is None:
        return 'pending', 'image digest not yet reported'
    want = expected.get('image_digest')
    if want and identity['image_digests'] != [want]:
        return 'quarantine', f"image {identity['image_digests']} is not the release's {want}"
    revision = expected.get('model_revision')
    if revision:
        observed = _flag(identity['args'], '--revision')
        if observed != revision:
            return 'quarantine', f'model revision {observed!r} is not the release\'s {revision}'
    if not want and not revision:
        return 'quarantine', 'release has no pinned identity to admit against'
    return 'admit', 'identity matches the frozen release'


def admit_pods(provider, external, expected):
    """Reconcile pool membership for every pod of the allocation. Returns a per-pod report."""
    pods = provider.pods(external)
    if pods is None:
        return {'known': False, 'pods': []}
    report = []
    for pod in pods:
        identity = pod_identity(pod)
        decision, reason = check_pod(identity, expected)
        if decision == 'admit' and identity['admitted'] != 'true':
            provider.label_pod(identity['pod'], {ADMITTED_LABEL: 'true', QUARANTINE_LABEL: None})
        elif decision == 'quarantine' and identity['admitted'] != 'false':
            slug = re.sub(r'[^A-Za-z0-9_.-]', '-', reason)[:63].strip('-.') or 'mismatch'
            provider.label_pod(identity['pod'], {ADMITTED_LABEL: 'false', QUARANTINE_LABEL: slug})
        report.append({**identity, 'decision': decision, 'reason': reason})
    return {'known': True, 'pods': report,
            'admitted': sum(1 for r in report if r['decision'] == 'admit'),
            'quarantined': sum(1 for r in report if r['decision'] == 'quarantine')}


class PoolRouting:
    """Applies and retires release pools through a KubernetesProvider's kubectl runner."""

    def __init__(self, provider, *, gateway, epp_service, epp_port=9002, gateway_url=None):
        self.provider, self.gateway, self.epp_service, self.epp_port = provider, gateway, epp_service, epp_port
        self.gateway_url = gateway_url  # where the InferStack gateway sends pool-routed requests

    def apply(self, release_id, *, port):
        docs = pool_manifests(release_id, port=port, gateway=self.gateway, epp_service=self.epp_service,
                              epp_port=self.epp_port)
        self.provider.run(self.provider.base + ['apply', '-f', '-'],
                          input_text=json.dumps({'apiVersion': 'v1', 'kind': 'List', 'items': docs}))
        return docs

    def retire(self, release_id):
        name = pool_name(release_id)
        for kind in ('httproute', 'inferencepool'):
            self.provider.run(self.provider.base + ['delete', kind, name, '--ignore-not-found'], check=False)

    def upstream(self, release_id):
        """How the InferStack gateway reaches this release: the shared Gateway plus the release header."""
        return {'base_url': self.gateway_url, 'headers': {RELEASE_HEADER: release_key(release_id)}}


__all__ = ['RELEASE_LABEL', 'ADMITTED_LABEL', 'RELEASE_HEADER', 'release_key', 'pool_name', 'pool_manifests',
           'pod_identity', 'check_pod', 'admit_pods', 'PoolRouting', 'LABEL', 'k8s_name']
