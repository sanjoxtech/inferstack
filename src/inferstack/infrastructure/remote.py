"""Container (Docker) and Kubernetes providers behind the allocation contract.

Both satisfy the provider interface used by the executor and reconciler:
`launch(alloc_id, spec)`, `external_for(...)`, `owns`, `alive`, `stop`,
`find_tagged`. Every resource carries the label `inferstack.allocation=<id>` and
is identified by its runtime UID, so a recreated object with the same name is
never mistaken for ours.

Hardening defaults:
- Docker: read-only root filesystem, all capabilities dropped,
  no-new-privileges, PID and memory limits, explicit GPU devices, no host
  network, tmpfs scratch.
- Kubernetes: non-root, no privilege escalation, all capabilities dropped,
  read-only root, no service-account token, GPU limits, and a NetworkPolicy
  allowing ingress to the serving port and DNS-only egress (model weights must
  be pre-staged or served from an allowlisted cache; set `egress_cidrs`).
CLIs are invoked through an injectable runner so tests can use fakes; live
cluster qualification is a Package J gate.
"""
from __future__ import annotations

import json
import re
import subprocess
import time

LABEL = 'inferstack.allocation'
INSTANCE_LABEL = 'inferstack.instance'
# Every kind an allocation may own; discovery and cleanup cover all of them.
K8S_KINDS = ('deployment', 'replicaset', 'pod', 'service', 'networkpolicy')
_NAME = re.compile(r'[^a-z0-9-]')


def k8s_name(alloc_id):
    return ('isx-' + _NAME.sub('-', alloc_id.lower()))[:63].rstrip('-')


class CommandFailed(RuntimeError):
    pass


def default_runner(argv, input_text=None, timeout=120):
    out = subprocess.run(argv, input=input_text, capture_output=True, text=True, timeout=timeout)
    return out.returncode, out.stdout, out.stderr


class _CLIProvider:
    launches_remotely = True

    def __init__(self, runner=default_runner):
        self.runner = runner

    def run(self, argv, input_text=None, check=True, timeout=120):
        code, out, err = self.runner(argv, input_text, timeout)
        if check and code:
            raise CommandFailed(f'{argv[0]} {argv[1]} failed: {err.strip()[-400:]}')
        return code, out, err


class DockerProvider(_CLIProvider):
    name = 'docker'
    resource_type = 'container'

    def __init__(self, *, runner=default_runner, network='bridge', memory='32g', pids=512, docker='docker'):
        super().__init__(runner)
        self.network, self.memory, self.pids, self.docker = network, memory, pids, docker

    def launch_argv(self, alloc_id, spec):
        image = spec['image']
        if not re.search(r'@sha256:[0-9a-f]{64}$', image):
            raise ValueError('container image must be pinned by digest (image@sha256:...)')
        argv = [self.docker, 'run', '-d', '--name', k8s_name(alloc_id), '--label', f'{LABEL}={alloc_id}',
                '--read-only', '--tmpfs', '/tmp:rw,size=1g', '--cap-drop', 'ALL',
                '--security-opt', 'no-new-privileges', '--pids-limit', str(self.pids), '--memory', self.memory,
                '--network', self.network, '-p', f"127.0.0.1:{spec['host_port']}:{spec['container_port']}"]
        if spec.get('gpus'):
            argv += ['--gpus', 'device=' + ','.join(str(g) for g in spec['gpus'])]
        for mount in spec.get('read_only_mounts', []):
            argv += ['-v', f"{mount['source']}:{mount['target']}:ro"]
        for key, value in sorted((spec.get('env') or {}).items()):
            argv += ['-e', f'{key}={value}']
        return argv + [image, *spec.get('args', [])]

    def launch(self, alloc_id, spec):
        _, out, _ = self.run(self.launch_argv(alloc_id, spec))
        return {'provider': self.name, 'container_id': out.strip().splitlines()[-1], 'name': k8s_name(alloc_id)}

    def owns(self, external):
        return isinstance(external, dict) and external.get('provider') == self.name

    def external_for(self, container_id):
        return {'provider': self.name, 'container_id': container_id}

    def alive(self, external):
        code, out, _ = self.run([self.docker, 'inspect', '--format', '{{.State.Running}}', external['container_id']],
                                check=False)
        return code == 0 and out.strip() == 'true'

    def find_tagged(self, alloc_id):
        _, out, _ = self.run([self.docker, 'ps', '-aq', '--no-trunc', '--filter', f'label={LABEL}={alloc_id}'])
        return [line for line in out.split() if line]

    def stop(self, external, grace_seconds=10.0):
        self.run([self.docker, 'stop', '-t', str(int(grace_seconds)), external['container_id']], check=False)
        self.run([self.docker, 'rm', '-f', external['container_id']], check=False)
        code, _, _ = self.run([self.docker, 'inspect', external['container_id']], check=False)
        return code != 0


class KubernetesProvider(_CLIProvider):
    name = 'kubernetes'
    resource_type = 'k8s_deployment'

    def __init__(self, *, namespace, runner=default_runner, kubectl='kubectl', egress_cidrs=(), context=None):
        super().__init__(runner)
        if not re.fullmatch(r'[a-z0-9]([-a-z0-9]*[a-z0-9])?', namespace):
            raise ValueError('invalid namespace')
        self.namespace, self.kubectl, self.egress_cidrs = namespace, kubectl, tuple(egress_cidrs)
        self.base = [kubectl] + (['--context', context] if context else []) + ['-n', namespace]

    def manifests(self, alloc_id, spec):
        image = spec['image']
        if not re.search(r'@sha256:[0-9a-f]{64}$', image):
            raise ValueError('container image must be pinned by digest (image@sha256:...)')
        name = k8s_name(alloc_id)
        labels = {LABEL: alloc_id, INSTANCE_LABEL: name, 'app.kubernetes.io/managed-by': 'inferstack'}
        port = int(spec['container_port'])
        resources = {'limits': {'memory': spec.get('memory', '32Gi'), 'cpu': str(spec.get('cpu', 8))},
                     'requests': {'memory': spec.get('memory', '32Gi'), 'cpu': str(spec.get('cpu', 8))}}
        if spec.get('gpus'):
            resources['limits']['nvidia.com/gpu'] = str(spec['gpus'])
            resources['requests']['nvidia.com/gpu'] = str(spec['gpus'])
        container = {'name': 'engine', 'image': image, 'args': list(spec.get('args', [])),
                     'ports': [{'containerPort': port}], 'resources': resources,
                     'env': [{'name': k, 'value': str(v)} for k, v in sorted((spec.get('env') or {}).items())],
                     'readinessProbe': {'httpGet': {'path': spec.get('ready_path', '/v1/models'), 'port': port},
                                        'periodSeconds': 5, 'failureThreshold': 120},
                     'securityContext': {'runAsNonRoot': True, 'allowPrivilegeEscalation': False,
                                         'readOnlyRootFilesystem': True, 'capabilities': {'drop': ['ALL']}},
                     'volumeMounts': [{'name': 'scratch', 'mountPath': '/tmp'}]}
        egress = [{'ports': [{'protocol': 'UDP', 'port': 53}, {'protocol': 'TCP', 'port': 53}]}]
        if self.egress_cidrs:
            egress.append({'to': [{'ipBlock': {'cidr': c}} for c in self.egress_cidrs]})
        return [
            {'apiVersion': 'apps/v1', 'kind': 'Deployment', 'metadata': {'name': name, 'labels': labels},
             'spec': {'replicas': 1, 'selector': {'matchLabels': {LABEL: alloc_id}},
                      'template': {'metadata': {'labels': labels},
                                   'spec': {'automountServiceAccountToken': False, 'enableServiceLinks': False,
                                            'containers': [container],
                                            'volumes': [{'name': 'scratch', 'emptyDir': {'sizeLimit': '2Gi'}}],
                                            **({'nodeSelector': spec['node_selector']} if spec.get('node_selector') else {})}}}},
            {'apiVersion': 'v1', 'kind': 'Service', 'metadata': {'name': name, 'labels': labels},
             'spec': {'selector': {LABEL: alloc_id}, 'ports': [{'port': port, 'targetPort': port}]}},
            {'apiVersion': 'networking.k8s.io/v1', 'kind': 'NetworkPolicy', 'metadata': {'name': name, 'labels': labels},
             'spec': {'podSelector': {'matchLabels': {LABEL: alloc_id}}, 'policyTypes': ['Ingress', 'Egress'],
                      'ingress': [{'ports': [{'port': port}]}], 'egress': egress}},
        ]

    def launch(self, alloc_id, spec):
        docs = self.manifests(alloc_id, spec)
        self.run(self.base + ['apply', '-f', '-'], input_text=json.dumps({'apiVersion': 'v1', 'kind': 'List',
                                                                           'items': docs}))
        _, out, _ = self.run(self.base + ['get', 'deployment', k8s_name(alloc_id), '-o', 'json'])
        meta = json.loads(out)['metadata']
        return {'provider': self.name, 'namespace': self.namespace, 'name': meta['name'], 'uid': meta['uid'],
                'service': f"{meta['name']}.{self.namespace}.svc", 'allocation_id': alloc_id}

    def owns(self, external):
        return isinstance(external, dict) and external.get('provider') == self.name and \
            external.get('namespace') == self.namespace

    def external_for(self, name):
        _, out, _ = self.run(self.base + ['get', 'deployment', name, '-o', 'json'])
        meta = json.loads(out)['metadata']
        return {'provider': self.name, 'namespace': self.namespace, 'name': meta['name'], 'uid': meta['uid']}

    def alive(self, external):
        code, out, _ = self.run(self.base + ['get', 'deployment', external['name'], '-o', 'json'], check=False)
        return code == 0 and json.loads(out)['metadata']['uid'] == external['uid']

    def find_tagged(self, alloc_id):
        _, out, _ = self.run(self.base + ['get', 'deployment', '-l', f'{LABEL}={alloc_id}', '-o', 'json'])
        return [item['metadata']['name'] for item in json.loads(out).get('items', [])]

    def _selector(self, external):
        if external.get('allocation_id'):
            return f"{LABEL}={external['allocation_id']}"
        return f"{INSTANCE_LABEL}={external['name']}"

    def _list(self, kind, external):
        code, out, _ = self.run(self.base + ['get', kind, '-l', self._selector(external), '-o', 'json'], check=False)
        if code != 0:
            return None
        try:
            return json.loads(out).get('items', [])
        except (ValueError, AttributeError):
            return None

    def pods(self, external):
        """Non-terminated Pods of this allocation's Deployment UID; None when the API cannot answer.

        A Pod whose owner chain (Pod -> ReplicaSet -> Deployment) leads to a
        different Deployment UID belongs to a same-name replacement, not to us.
        Pods without owner references are counted as ours (conservative).
        """
        pods = self._list('pods', external)
        if pods is None:
            return None
        try:
            sets = self._list('replicasets', external) or []
        except Exception:  # noqa: BLE001 - unknown ownership only makes cleanup more conservative
            sets = []
        foreign_sets = {r['metadata'].get('uid') for r in sets
                        if any(o.get('kind') == 'Deployment' and o.get('uid') != external['uid']
                               for o in r['metadata'].get('ownerReferences') or [])}
        live = []
        for pod in pods:
            owners = pod.get('metadata', {}).get('ownerReferences') or []
            if any(o.get('kind') == 'ReplicaSet' and o.get('uid') in foreign_sets for o in owners):
                continue
            if (pod.get('status') or {}).get('phase') not in ('Succeeded', 'Failed'):
                live.append(pod)
        return live

    def residual(self, external):
        """True while the Deployment or any of its Pods (GPU work) still exists, or when unknown."""
        pods = self.pods(external)
        return self.alive(external) or pods is None or bool(pods)

    def stop(self, external, grace_seconds=30.0):
        """Delete the allocation's objects and wait for its Pods to terminate.

        Deleting a Deployment only starts garbage collection; ReplicaSets and Pods
        (holding GPUs) may keep running. Release is confirmed only when the
        Deployment is gone and no non-terminated Pod with the allocation label remains.
        A same-name object with a different UID is not ours and is never deleted.
        """
        code, out, _ = self.run(self.base + ['get', 'deployment', external['name'], '-o', 'json'], check=False)
        if code == 0 and json.loads(out)['metadata']['uid'] == external['uid']:
            for kind in ('deployment', 'service', 'networkpolicy'):
                self.run(self.base + ['delete', kind, external['name'], '--ignore-not-found', '--wait=false',
                                      '--cascade=foreground'], check=False)
        deadline = time.monotonic() + grace_seconds
        while True:
            pods = self.pods(external)
            if not self.alive(external) and pods == []:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
