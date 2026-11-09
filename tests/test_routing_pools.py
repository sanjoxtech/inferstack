"""P3: release-scoped InferencePools with identity-gated membership (object/control logic, CPU only)."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from inferstack.engines.managed import ManagedEngine
from inferstack.infrastructure.remote import KubernetesProvider
from inferstack.releases.manager import ReleaseManager
from inferstack.releases.runtime import ReleaseRuntime, RuntimeUnavailable
from inferstack.routing.pools import (ADMITTED_LABEL, RELEASE_HEADER, RELEASE_LABEL, PoolRouting, admit_pods,
                                      check_pod, pod_identity, pool_manifests, pool_name, release_key)
from inferstack.serving.evidence import EvidenceStore
from inferstack.serving.gateway import create_gateway
from inferstack.storage.repository import Repository

DIGEST = 'sha256:' + 'a' * 64
REV = 'b' * 40


def pod(name, *, ready=True, digest=DIGEST, revision=REV, labels=None):
    return {'metadata': {'name': name, 'uid': 'uid-' + name, 'labels': labels or {}},
            'spec': {'nodeName': 'gpu-1', 'containers': [{'name': 'engine',
                                                          'args': ['vllm', 'serve', 'm', '--revision', revision]}]},
            'status': {'phase': 'Running', 'conditions': [{'type': 'Ready', 'status': 'True' if ready else 'False'}],
                       'containerStatuses': [{'imageID': f'registry/vllm@{digest}' if digest else ''}]}}


class FakeProvider:
    name = 'kubernetes'

    def __init__(self, pods):
        self._pods, self.labels, self.base, self.applied, self.deleted = pods, {}, ['kubectl'], [], []

    def pods(self, external):
        return self._pods

    def label_pod(self, name, labels):
        self.labels.setdefault(name, {}).update(labels)

    def run(self, argv, input_text=None, check=True):
        if 'apply' in argv:
            self.applied.append(json.loads(input_text))
        if 'delete' in argv:
            self.deleted.append(argv[-2:])
        return 0, '', ''


def test_pool_selects_only_admitted_pods_of_one_release():
    pool, route = pool_manifests('rel-abc', port=8000, gateway='inference', epp_service='epp')
    key = release_key('rel-abc')
    assert pool['spec']['selector']['matchLabels'] == {RELEASE_LABEL: key, ADMITTED_LABEL: 'true'}
    assert pool['spec']['endpointPickerRef']['failureMode'] == 'FailClose'
    match = route['spec']['rules'][0]['matches'][0]['headers'][0]
    assert match == {'type': 'Exact', 'name': RELEASE_HEADER, 'value': key}
    assert route['spec']['rules'][0]['backendRefs'][0] == {'group': 'inference.networking.k8s.io',
                                                          'kind': 'InferencePool', 'name': pool_name('rel-abc')}
    assert release_key('rel-abc') != release_key('rel-abd')


def test_pod_template_never_pre_admits():
    provider = KubernetesProvider(namespace='inferstack')
    labels = {RELEASE_LABEL: release_key('rel')}
    deployment = provider.manifests('alloc-1', {'image': 'r/vllm@' + DIGEST, 'container_port': 8000,
                                                'pod_labels': labels})[0]
    template = deployment['spec']['template']['metadata']['labels']
    assert template[RELEASE_LABEL] == labels[RELEASE_LABEL] and ADMITTED_LABEL not in template
    with pytest.raises(ValueError, match='reserved'):
        provider.manifests('alloc-1', {'image': 'r/vllm@' + DIGEST, 'container_port': 8000,
                                       'pod_labels': {ADMITTED_LABEL: 'true'}})


@pytest.mark.parametrize('kwargs,decision', [
    ({}, 'admit'), ({'ready': False}, 'pending'), ({'digest': None}, 'pending'),
    ({'digest': 'sha256:' + 'c' * 64}, 'quarantine'), ({'revision': 'd' * 40}, 'quarantine')])
def test_each_pod_is_checked_against_the_frozen_identity(kwargs, decision):
    identity = pod_identity(pod('p', **kwargs))
    assert check_pod(identity, {'image_digest': DIGEST, 'model_revision': REV})[0] == decision


def test_unpinned_release_admits_nothing():
    assert check_pod(pod_identity(pod('p')), {'image_digest': None, 'model_revision': None})[0] == 'quarantine'


def test_admission_labels_good_pods_and_quarantines_mismatched_replicas():
    provider = FakeProvider([pod('good'), pod('wrong', digest='sha256:' + 'c' * 64), pod('booting', ready=False)])
    report = admit_pods(provider, {}, {'image_digest': DIGEST, 'model_revision': REV})
    assert report['admitted'] == 1 and report['quarantined'] == 1
    assert provider.labels['good'][ADMITTED_LABEL] == 'true'
    assert provider.labels['wrong'][ADMITTED_LABEL] == 'false' and provider.labels['wrong']['inferstack.io/quarantine']
    assert 'booting' not in provider.labels  # unknown identity stays out of the pool


def test_scale_out_pod_is_admitted_only_after_its_own_check():
    provider = FakeProvider([pod('a', labels={ADMITTED_LABEL: 'true'})])
    admit_pods(provider, {}, {'image_digest': DIGEST, 'model_revision': REV})
    assert 'a' not in provider.labels  # already admitted: no relabel churn
    provider._pods.append(pod('b', ready=False))
    admit_pods(provider, {}, {'image_digest': DIGEST, 'model_revision': REV})
    assert 'b' not in provider.labels
    provider._pods[-1] = pod('b')
    admit_pods(provider, {}, {'image_digest': DIGEST, 'model_revision': REV})
    assert provider.labels['b'][ADMITTED_LABEL] == 'true'


def test_unknown_pod_listing_admits_nothing():
    provider = FakeProvider(None)
    assert admit_pods(provider, {}, {'image_digest': DIGEST}) == {'known': False, 'pods': []}


def test_pool_apply_and_retire():
    provider = FakeProvider([])
    routing = PoolRouting(provider, gateway='inference', epp_service='epp', gateway_url='http://gw')
    routing.apply('rel-x', port=8000)
    assert [d['kind'] for d in provider.applied[0]['items']] == ['InferencePool', 'HTTPRoute']
    routing.retire('rel-x')
    assert provider.deleted == [[pool_name('rel-x'), '--ignore-not-found']] * 2
    assert routing.upstream('rel-x') == {'base_url': 'http://gw', 'headers': {RELEASE_HEADER: release_key('rel-x')}}


def test_gateway_sends_release_header_and_records_selected_endpoint(tmp_path):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    key = release_key('rel-pool')
    repo.put_record('deployment', 'd', {'id': 'd', 'engine': 'generic_openai', 'model': 'm', 'base_url': 'http://gw',
                                        'profile': {'routing': {'release_header': key}}})
    repo.put('release', 'rel-pool', {'deployment': repo.read('deployment', 'd'), 'evidence': {'sha256': 'f' * 64,
             'bytes': 0}, 'job_id': 'x', 'verification_scope': 'x'})
    ReleaseManager(repo).route('m', 'rel-pool')
    seen = []

    def handler(request):
        seen.append(request.headers.get(RELEASE_HEADER))
        return httpx.Response(200, json={'choices': [{'text': 'ok'}]},
                              headers={'x-gateway-destination-endpoint': '10.0.0.7:8000'})
    client = TestClient(create_gateway(repo, 't', engine_factory=lambda spec: ManagedEngine(
        spec, transport=httpx.MockTransport(handler))))
    assert client.post('/v1/completions', headers={'Authorization': 'Bearer t'},
                       json={'model': 'm', 'prompt': 'x'}).status_code == 200
    assert seen == [key]
    [row] = EvidenceStore(repo).rows(release_id='rel-pool', since=0)
    assert row['endpoint'] == '10.0.0.7:8000'


@pytest.mark.asyncio
async def test_pooled_kubernetes_release_is_ready_only_after_identity_admission(tmp_path):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    image = 'registry/vllm@' + DIGEST
    container = {'provider': 'kubernetes', 'image': image, 'container_port': 8000}
    repo.put_record('deployment', 'd', {'id': 'd', 'engine': 'vllm', 'model': 'm', 'base_url': 'http://gw',
                                        'profile': {'container': container}})
    repo.put('release', 'rel', {'deployment': repo.read('deployment', 'd'), 'evidence': {'sha256': 'f' * 64,
             'bytes': 0}, 'job_id': 'x', 'verification_scope': 'x', 'identity': {'model_revision': REV}})

    class K8s(FakeProvider):
        launches_remotely, resource_type = True, 'k8s_deployment'

        def launch(self, alloc_id, spec):
            self.spec = spec
            return {'provider': 'kubernetes', 'name': 'isx', 'uid': 'u', 'allocation_id': alloc_id}

        def alive(self, external):
            return True

        def stop(self, external):
            return True

    good = K8s([pod('p1')])
    runtime = ReleaseRuntime(repo, good, providers={'kubernetes': good}, log_root=tmp_path / 'l', ready_timeout=2,
                             routing=PoolRouting(good, gateway='inference', epp_service='epp'))
    result = await runtime.ensure('rel')
    assert result['replaced'] and good.spec['pod_labels'][RELEASE_LABEL] == release_key('rel')
    assert good.labels['p1'][ADMITTED_LABEL] == 'true' and good.applied
    await runtime.retire('rel')
    assert good.deleted  # pool removed on retirement

    bad = K8s([pod('p1', revision='d' * 40)])
    runtime = ReleaseRuntime(repo, bad, providers={'kubernetes': bad}, log_root=tmp_path / 'l2', ready_timeout=1,
                             routing=PoolRouting(bad, gateway='inference', epp_service='epp'))
    with pytest.raises(RuntimeUnavailable, match='identity admission'):
        await runtime.ensure('rel')
