"""Package H: container and Kubernetes providers, hardening, reconciliation by
label and UID, remote launch-and-attach, and price-based spend accounting."""
import json
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from inferstack.infrastructure import AllocationManager, reconcile
from inferstack.infrastructure.pricing import PriceTable
from inferstack.infrastructure.remote import LABEL, DockerProvider, KubernetesProvider, k8s_name
from inferstack.optimization.experiment import EndpointExecutor
from inferstack.storage.repository import Repository

FIXTURES = Path(__file__).parent / 'fixtures'
IMAGE = 'vllm/vllm-openai@sha256:' + 'a' * 64


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class FakeDocker:
    """Docker CLI double whose containers are real local engine processes."""

    def __init__(self):
        self.containers, self.calls = {}, []

    def __call__(self, argv, input_text=None, timeout=120):
        self.calls.append(argv)
        cmd = argv[1]
        if cmd == 'run':
            cid = uuid.uuid4().hex * 2
            label = argv[argv.index('--label') + 1].split('=', 1)[1]
            host_port = argv[argv.index('-p') + 1].split(':')[1]
            proc = subprocess.Popen([sys.executable, str(FIXTURES / 'fake_openai_server.py'), '--port', host_port])
            self.containers[cid] = {'label': label, 'proc': proc}
            return 0, cid + '\n', ''
        if cmd == 'inspect':
            c = self.containers.get(argv[-1])
            if c is None:
                return 1, '', 'No such object'
            return 0, ('true' if c['proc'].poll() is None else 'false') + '\n', ''
        if cmd == 'ps':
            label = argv[argv.index('--filter') + 1].split('=', 2)[2]
            return 0, ''.join(cid + '\n' for cid, c in self.containers.items() if c['label'] == label), ''
        if cmd in ('stop', 'rm'):
            c = self.containers.get(argv[-1])
            if c:
                c['proc'].kill()
                c['proc'].wait()
                if cmd == 'rm':
                    del self.containers[argv[-1]]
            return 0, '', ''
        return 1, '', f'unsupported {cmd}'

    def close(self):
        for c in self.containers.values():
            c['proc'].kill()


class FakeKubectl:
    def __init__(self):
        self.objects, self.calls = {}, []

    def __call__(self, argv, input_text=None, timeout=120):
        self.calls.append((argv, input_text))
        args = argv[argv.index('-n') + 2:]
        verb = args[0]
        if verb == 'apply':
            for item in json.loads(input_text)['items']:
                key = (item['kind'].lower(), item['metadata']['name'])
                self.objects[key] = {**item, 'metadata': {**item['metadata'], 'uid': uuid.uuid4().hex}}
            return 0, 'applied', ''
        if verb == 'get' and '-l' in args:
            label = args[args.index('-l') + 1].split('=', 1)[1]
            wanted = {'pods': 'pod', 'replicasets': 'replicaset'}.get(args[1], args[1])
            items = [o for (kind, _), o in self.objects.items()
                     if kind == wanted and o['metadata']['labels'].get(LABEL) == label]
            return 0, json.dumps({'items': items}), ''
        if verb == 'get':
            obj = self.objects.get((args[1], args[2]))
            return (0, json.dumps(obj), '') if obj else (1, '', 'NotFound')
        if verb == 'delete':
            self.objects.pop((args[1], args[2]), None)
            return 0, '', ''
        return 1, '', 'unsupported'


# --- hardening -------------------------------------------------------------------------

def test_docker_launch_is_hardened_and_digest_pinned():
    argv = DockerProvider(runner=None).launch_argv('alloc-1', {'image': IMAGE, 'host_port': 8100,
                                                               'container_port': 8000, 'gpus': [0, 1],
                                                               'args': ['--model', 'm']})
    for flag in ('--read-only', '--cap-drop', 'no-new-privileges', '--pids-limit', '--memory'):
        assert flag in argv
    assert f'{LABEL}=alloc-1' in argv and '127.0.0.1:8100:8000' in argv and 'device=0,1' in argv
    assert argv[-3:] == [IMAGE, '--model', 'm'] and '--privileged' not in argv and '--network=host' not in argv
    with pytest.raises(ValueError, match='pinned by digest'):
        DockerProvider(runner=None).launch_argv('a', {'image': 'vllm/vllm-openai:latest', 'host_port': 1,
                                                       'container_port': 2})


def test_kubernetes_manifests_are_hardened():
    provider = KubernetesProvider(namespace='inferstack-exp', runner=None, egress_cidrs=['10.20.0.0/16'])
    deployment, service, policy = provider.manifests('alloc-ABC_1', {'image': IMAGE, 'container_port': 8000,
                                                                     'gpus': 1, 'args': ['--port', '8000']})
    pod = deployment['spec']['template']['spec']
    container = pod['containers'][0]
    assert deployment['metadata']['name'] == 'isx-alloc-abc-1' and pod['automountServiceAccountToken'] is False
    assert container['securityContext'] == {'runAsNonRoot': True, 'runAsUser': 65532, 'allowPrivilegeEscalation': False,
                                            'readOnlyRootFilesystem': True, 'capabilities': {'drop': ['ALL']}}
    assert container['resources']['limits']['nvidia.com/gpu'] == '1'
    assert policy['spec']['policyTypes'] == ['Ingress', 'Egress']
    assert policy['spec']['egress'][0] == {'ports': [{'protocol': 'UDP', 'port': 53}, {'protocol': 'TCP', 'port': 53}]}
    assert policy['spec']['egress'][1]['to'] == [{'ipBlock': {'cidr': '10.20.0.0/16'}}]
    assert service['spec']['selector'] == {LABEL: 'alloc-ABC_1'}
    assert pod['terminationGracePeriodSeconds'] == 45 and container['lifecycle']['preStop'] == {'sleep': {'seconds': 15}}
    with pytest.raises(ValueError, match='namespace'):
        KubernetesProvider(namespace='Bad_NS', runner=None)


# --- remote launch -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_container_candidate_launch_attach_and_cleanup(tmp_path):
    docker = FakeDocker()
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    port = free_port()
    repo.put_record('deployment', 'base', {'id': 'base', 'engine': 'vllm', 'model': 'm', 'base_url': 'http://base'})
    repo.put_record('deployment', 'cand', {'id': 'cand', 'engine': 'vllm', 'model': 'm',
                                           'base_url': f'http://127.0.0.1:{port}', 'image_digest': 'sha256:' + 'a' * 64,
                                           'profile': {'container': {'image': IMAGE, 'host_port': port,
                                                                     'container_port': 8000, 'gpus': [0]}}})
    job = repo.submit('j', {})
    claim = repo.claim(60, worker_id='w')
    pricing = PriceTable({'docker/NVIDIA A100-SXM4-40GB': {'usd_per_gpu_hour': 3.6, 'source': 'operator quote',
                                                           'as_of': '2026-11-01'}})
    executor = EndpointExecutor(repo, tmp_path / 'logs', allocations=AllocationManager(repo, 'w'),
                                provider=DockerProvider(runner=docker), pricing=pricing,
                                gpu_name='NVIDIA A100-SXM4-40GB', ready_timeout=20)
    context = {'job': {**claim, 'id': job, 'step_attempt': 1}, 'step_key': 'iter0.experiment'}
    plan = {'candidate_id': 'cand', 'candidate_revision': 1, 'baseline_id': 'base', 'baseline_revision': 1}
    try:
        candidate = await executor.execute(plan, 'exp1', context)
        assert candidate['external']['provider'] == 'docker' and await executor.active['exp1'].ready()
        [alloc] = AllocationManager(repo, 't').list(job_id=job)
        assert alloc['state'] == 'active' and alloc['gpu_count'] == 1
        assert alloc['spec']['price']['usd_per_gpu_hour'] == 3.6
        await executor.cleanup(candidate)
        assert AllocationManager(repo, 't').get(alloc['id'])['state'] == 'released' and not docker.containers
        assert repo.remaining(job) is None
    finally:
        docker.close()


# --- reconciliation ------------------------------------------------------------------------

def test_kubernetes_reconciliation_by_label_and_uid(tmp_path):
    kubectl = FakeKubectl()
    provider = KubernetesProvider(namespace='exp', runner=kubectl)
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    job = repo.submit('j', {})
    claim = repo.claim(60, worker_id='dead-worker')
    manager = AllocationManager(repo, 'dead-worker')
    ctx = {**claim, 'id': job, 'step_attempt': 1}
    spec = {'image': IMAGE, 'container_port': 8000}
    recorded = manager.intent(ctx, 's', purpose='a', provider='kubernetes', resource_type='k8s_deployment', spec=spec)
    external = provider.launch(recorded['id'], spec)
    manager.record_external(recorded['id'], external)
    unrecorded = manager.intent(ctx, 's', purpose='b', provider='kubernetes', resource_type='k8s_deployment', spec=spec)
    provider.launch(unrecorded['id'], spec)  # crash before identity was recorded
    replaced = manager.intent(ctx, 's', purpose='c', provider='kubernetes', resource_type='k8s_deployment', spec=spec)
    old = provider.launch(replaced['id'], spec)
    manager.record_external(replaced['id'], old)
    provider.launch(replaced['id'], spec)  # someone recreated the same name: new UID, not ours to delete
    survivor = kubectl.objects[('deployment', k8s_name(replaced['id']))]['metadata']['uid']
    outcomes = {a['allocation_id']: a['outcome'] for a in reconcile(repo, provider)}
    assert outcomes == {recorded['id']: 'orphan_stopped', unrecorded['id']: 'unrecorded_launch_stopped',
                        replaced['id']: 'already_exited'}
    assert ('deployment', k8s_name(recorded['id'])) not in kubectl.objects
    assert ('networkpolicy', k8s_name(recorded['id'])) not in kubectl.objects
    assert kubectl.objects[('deployment', k8s_name(replaced['id']))]['metadata']['uid'] == survivor


# --- pricing ----------------------------------------------------------------------------------

def test_spend_is_derived_and_unknown_price_exhausts_spend_budget(tmp_path):
    now = [1000.0]
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    repo.clock = lambda: now[0]
    priced = repo.submit('p', {}, budget={'spend_usd': 1.0})
    claim = repo.claim(100000)
    m = AllocationManager(repo, 'w')
    table = PriceTable({'local/a100': {'usd_per_gpu_hour': 3.6, 'source': 'quote', 'as_of': '2026-11-01'}})
    a = m.intent({**claim, 'id': priced}, 's', purpose='x', provider='local', resource_type='process',
                 spec=table.stamp('local', 'A100', 1), gpu_count=1)
    m.activate(a['id'], {**claim, 'id': priced}, {'host': 'h', 'pid': 1})
    now[0] += 600  # 10 min x 1 GPU x $3.6/h = $0.60
    assert repo.remaining(priced)['spend_usd'] == pytest.approx(0.4)
    now[0] += 600
    assert repo.exhausted(priced) == ['spend_usd']
    unpriced = repo.submit('u', {}, budget={'spend_usd': 100.0})
    claim2 = repo.claim(100000)
    b = m.intent({**claim2, 'id': unpriced}, 's', purpose='x', provider='local', resource_type='process',
                 spec=table.stamp('local', 'H100', 1), gpu_count=1)
    m.activate(b['id'], {**claim2, 'id': unpriced}, {'host': 'h', 'pid': 2})
    assert repo.exhausted(unpriced) == ['spend_usd'] and repo.remaining(unpriced)['spend_usd'] == 0.0
    with pytest.raises(ValueError, match='source'):
        PriceTable({'x/y': {'usd_per_gpu_hour': 1, 'source': '', 'as_of': '2026'}})
    assert table.stamp('local', 'A100', 0)['price']['usd_per_gpu_hour'] == 0.0


def test_allocation_quotas_and_rescale_accounting(tmp_path):
    """Quotas bound live allocations and GPUs; GPU-seconds follow the scale history."""
    from inferstack.infrastructure.allocations import QuotaExceeded
    now = [1000.0]
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    repo.clock = lambda: now[0]
    job = repo.submit('q', {}, budget={'gpu_seconds': 10_000})
    claim = repo.claim(600)
    ctx = {**claim, 'id': job, 'step_attempt': 1}
    manager = AllocationManager(repo, 'w', quotas={'kubernetes': {'max_live': 2, 'max_gpus': 3}})
    spec = {'image': IMAGE, 'container_port': 8000}
    a = manager.intent(ctx, 's', purpose='a', provider='kubernetes', resource_type='k8s_deployment', spec=spec, gpu_count=2)
    with pytest.raises(QuotaExceeded, match='GPUs'):
        manager.intent(ctx, 's', purpose='b', provider='kubernetes', resource_type='k8s_deployment', spec=spec,
                       gpu_count=2)
    manager.intent(ctx, 's', purpose='c', provider='kubernetes', resource_type='k8s_deployment', spec=spec, gpu_count=1)
    with pytest.raises(QuotaExceeded, match='live allocations'):
        manager.intent(ctx, 's', purpose='d', provider='kubernetes', resource_type='k8s_deployment', spec=spec)
    manager.activate(a['id'], ctx, {'provider': 'kubernetes'})
    now[0] += 100           # 100 s at 2 GPUs
    with pytest.raises(QuotaExceeded):
        manager.rescale(a['id'], 3)  # 3 + the other allocation's 1 > 3
    manager.rescale(a['id'], 1)
    now[0] += 100           # 100 s at 1 GPU
    manager.release(a['id'], 'done')
    assert repo._usage(repo.get(job))['gpu_seconds'] == pytest.approx(300.0)
