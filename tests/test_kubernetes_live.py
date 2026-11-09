"""Live Kubernetes lifecycle (H): quota, launch, image digest, scale, graceful drain, verified stop.

Runs only with INFERSTACK_TEST_K8S_CONTEXT naming a reachable cluster context (for
example a local `kind` cluster). Creates and deletes its own namespace. CPU only:
GPU scheduling needs a GPU node pool and is a separate qualification.
"""
import json
import os
import shutil
import subprocess
import uuid

import pytest

from inferstack.infrastructure.remote import KubernetesProvider

CONTEXT = os.environ.get('INFERSTACK_TEST_K8S_CONTEXT')
IMAGE = os.environ.get('INFERSTACK_TEST_K8S_IMAGE',
                       'python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704')
pytestmark = pytest.mark.skipif(not (CONTEXT and shutil.which('kubectl')),
                                reason='needs kubectl and INFERSTACK_TEST_K8S_CONTEXT (a reachable cluster context)')


def kubectl(*args, check=True):
    return subprocess.run(['kubectl', '--context', CONTEXT, *args], capture_output=True, text=True, check=check)


@pytest.fixture
def namespace():
    name = 'inferstack-test-' + uuid.uuid4().hex[:8]
    kubectl('create', 'namespace', name)
    try:
        yield name
    finally:
        kubectl('delete', 'namespace', name, '--wait=false', check=False)


def test_cluster_lifecycle_with_quota_scale_and_drain(namespace):
    provider = KubernetesProvider(namespace=namespace, context=CONTEXT)
    provider.ensure_quota(max_pods=2, max_gpus=0, max_cpu=2, max_memory='2Gi')
    alloc = 'alloc-k8slive' + uuid.uuid4().hex[:6]
    spec = {'image': IMAGE, 'container_port': 8000, 'ready_path': '/', 'cpu': '250m', 'memory': '256Mi',
            'drain_seconds': 2, 'shutdown_seconds': 5,
            'args': ['python', '-m', 'http.server', '8000', '--directory', '/tmp']}
    external = provider.launch(alloc, spec)
    try:
        assert provider.scale(external, 1, timeout=300)['ready'] == 1
        assert provider.observed_image_digest(external) == IMAGE.split('@', 1)[1]
        assert provider.scale(external, 2, timeout=180)['pods'] == 2
        # The namespace quota (2 pods) bounds scale-up: a third replica is never admitted.
        with pytest.raises(TimeoutError):
            provider.scale(external, 3, timeout=20)
        events = kubectl('get', 'events', '-n', namespace, '-o', 'json').stdout
        assert 'exceeded quota' in events
        drained = provider.drain(external, timeout=120)
        assert drained['pods'] == 0 and provider.pods(external) == []
    finally:
        assert provider.stop(external, grace_seconds=120) is True
    assert not provider.alive(external) and provider.pods(external) == []
    assert json.loads(kubectl('get', 'all', '-n', namespace, '-l', f'inferstack.allocation={alloc}',
                              '-o', 'json').stdout)['items'] == []


def test_identity_admission_labels_real_pods(namespace):
    """P3 on a real API server: every pod is enumerated and labelled; a wrong pin quarantines."""
    from inferstack.routing.pools import ADMITTED_LABEL, RELEASE_LABEL, admit_pods, release_key
    provider = KubernetesProvider(namespace=namespace, context=CONTEXT)
    alloc = 'alloc-admit' + uuid.uuid4().hex[:6]
    spec = {'image': IMAGE, 'container_port': 8000, 'ready_path': '/', 'cpu': '100m', 'memory': '128Mi',
            'drain_seconds': 1, 'shutdown_seconds': 2, 'replicas': 2,
            'pod_labels': {RELEASE_LABEL: release_key('rel-live')},
            'args': ['python', '-m', 'http.server', '8000', '--directory', '/tmp']}
    external = provider.launch(alloc, spec)
    try:
        provider.scale(external, 2, timeout=300)
        digest = IMAGE.split('@', 1)[1]
        report = admit_pods(provider, external, {'image_digest': digest})
        assert report['admitted'] == 2, report
        admitted = json.loads(kubectl('get', 'pods', '-n', namespace, '-l',
                                      f'{ADMITTED_LABEL}=true,{RELEASE_LABEL}={release_key("rel-live")}',
                                      '-o', 'json').stdout)['items']
        assert len(admitted) == 2
        wrong = admit_pods(provider, external, {'image_digest': 'sha256:' + '0' * 64})
        assert wrong['quarantined'] == 2
        assert json.loads(kubectl('get', 'pods', '-n', namespace, '-l', f'{ADMITTED_LABEL}=true',
                                  '-o', 'json').stdout)['items'] == []
    finally:
        assert provider.stop(external, grace_seconds=60) is True
