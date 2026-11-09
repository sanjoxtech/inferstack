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
        # A registry digest or a local content-addressed image ID (built candidate) are both immutable.
        if not (re.search(r'@sha256:[0-9a-f]{64}$', image) or re.fullmatch(r'sha256:[0-9a-f]{64}', image)):
            raise ValueError('container image must be pinned by digest (image@sha256:...) or image ID')
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

    def observed_image_id(self, external):
        code, image_id, _ = self.run([self.docker, 'inspect', '--format', '{{.Image}}', external['container_id']],
                                     check=False)
        return image_id.strip() if code == 0 and image_id.strip() else None

    def observed_image_digest(self, external):
        """Repository digest of the image the container actually runs (attestation evidence), or None."""
        code, image_id, _ = self.run([self.docker, 'inspect', '--format', '{{.Image}}', external['container_id']],
                                     check=False)
        if code != 0:
            return None
        code, out, _ = self.run([self.docker, 'image', 'inspect', '--format', '{{json .RepoDigests}}',
                                 image_id.strip()], check=False)
        try:
            digests = json.loads(out) if code == 0 else []
        except ValueError:
            return None
        found = sorted({d.split('@', 1)[1] for d in digests or [] if '@sha256:' in d})
        return found[0] if len(found) == 1 else None

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
    schedules_devices = True  # GPU requests are placed by the cluster scheduler (nvidia.com/gpu)

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
        extra = dict(spec.get('pod_labels') or {})
        for key, value in extra.items():  # e.g. inferstack.io/release; never the admission label
            if key in ('inferstack.io/admitted', LABEL, INSTANCE_LABEL) or \
                    not re.fullmatch(r'([a-z0-9.-]+/)?[A-Za-z0-9_.-]{1,63}', key) or \
                    not re.fullmatch(r'[A-Za-z0-9_.-]{0,63}', str(value)):
                raise ValueError(f'invalid or reserved pod label {key}')
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
                     # Drain: endpoints are removed first, the container keeps serving in-flight
                     # requests for drain_seconds, then receives SIGTERM within the grace period.
                     'lifecycle': {'preStop': {'sleep': {'seconds': int(spec.get('drain_seconds', 15))}}},
                     'securityContext': {'runAsNonRoot': True, 'runAsUser': int(spec.get('run_as_user', 65532)),
                                         'allowPrivilegeEscalation': False,
                                         'readOnlyRootFilesystem': True, 'capabilities': {'drop': ['ALL']}},
                     'volumeMounts': [{'name': 'scratch', 'mountPath': '/tmp'}]}
        volumes = [{'name': 'scratch', 'emptyDir': {'sizeLimit': spec.get('scratch_size', '2Gi')}}]
        init = []
        cache = spec.get('model_cache')
        if cache:
            # Immutable artifact cache: a PVC mounted read-only (serving never writes trusted
            # weights), separate from scratch. Readiness is gated on a verify init-container
            # when an image with the inferstack verifier is supplied.
            claim, mount = cache['claim'], cache.get('mount_path', '/models')
            if not re.fullmatch(r'[a-z0-9]([-a-z0-9]*[a-z0-9])?', claim) or not mount.startswith('/'):
                raise ValueError('model_cache needs a valid PVC name and absolute mount_path')
            volumes.append({'name': 'model-cache', 'persistentVolumeClaim': {'claimName': claim, 'readOnly': True}})
            container['volumeMounts'].append({'name': 'model-cache', 'mountPath': mount, 'readOnly': True})
            if cache.get('verify_image'):
                if not re.search(r'@sha256:[0-9a-f]{64}$', cache['verify_image']):
                    raise ValueError('model_cache.verify_image must be pinned by digest')
                init.append({'name': 'verify-model-cache', 'image': cache['verify_image'],
                             'args': ['python', '-m', 'inferstack.infrastructure.model_cache', 'verify',
                                      '--root', mount, '--model-json', json.dumps(cache['model'], sort_keys=True)],
                             'resources': {'limits': {'cpu': '1', 'memory': '1Gi'},
                                           'requests': {'cpu': '1', 'memory': '1Gi'}},
                             'securityContext': container['securityContext'],
                             'volumeMounts': [{'name': 'model-cache', 'mountPath': mount, 'readOnly': True},
                                              {'name': 'scratch', 'mountPath': '/tmp'}]})
        if spec.get('compile_cache_size'):
            # Bounded writable location for engine compilation caches (torch/triton/vLLM).
            volumes.append({'name': 'compile-cache', 'emptyDir': {'sizeLimit': spec['compile_cache_size']}})
            container['volumeMounts'].append({'name': 'compile-cache', 'mountPath': '/var/cache/engine'})
            container['env'] += [{'name': 'XDG_CACHE_HOME', 'value': '/var/cache/engine'},
                                 {'name': 'VLLM_CACHE_ROOT', 'value': '/var/cache/engine/vllm'}]
        egress = [{'ports': [{'protocol': 'UDP', 'port': 53}, {'protocol': 'TCP', 'port': 53}]}]
        if self.egress_cidrs:
            egress.append({'to': [{'ipBlock': {'cidr': c}} for c in self.egress_cidrs]})
        return [
            {'apiVersion': 'apps/v1', 'kind': 'Deployment', 'metadata': {'name': name, 'labels': labels},
             'spec': {'replicas': int(spec.get('replicas', 1)), 'selector': {'matchLabels': {LABEL: alloc_id}},
                      'template': {'metadata': {'labels': {**labels, **extra}},
                                   'spec': {'automountServiceAccountToken': False, 'enableServiceLinks': False,
                                            'terminationGracePeriodSeconds': int(spec.get('drain_seconds', 15)) +
                                            int(spec.get('shutdown_seconds', 30)),
                                            'containers': [container], **({'initContainers': init} if init else {}),
                                            'volumes': volumes,
                                            **({'nodeSelector': spec['node_selector']} if spec.get('node_selector') else {})}}}},
            {'apiVersion': 'v1', 'kind': 'Service', 'metadata': {'name': name, 'labels': labels},
             'spec': {'selector': {LABEL: alloc_id}, 'ports': [{'port': port, 'targetPort': port}]}},
            {'apiVersion': 'networking.k8s.io/v1', 'kind': 'NetworkPolicy', 'metadata': {'name': name, 'labels': labels},
             'spec': {'podSelector': {'matchLabels': {LABEL: alloc_id}}, 'policyTypes': ['Ingress', 'Egress'],
                      'ingress': [{'ports': [{'port': port}]}], 'egress': egress}},
        ]

    def cache_job_manifest(self, name, *, image, claim, model, credentials_secret=None, source='huggingface',
                           mount_path='/models', active_deadline_seconds=7200):
        """One-shot downloader Job: the only workload with artifact credentials and write access.

        It fetches the pinned files into staging on the PVC, verifies every checksum and
        publishes atomically (infrastructure/model_cache.py). Serving pods only read."""
        if not re.search(r'@sha256:[0-9a-f]{64}$', image):
            raise ValueError('downloader image must be pinned by digest')
        container = {'name': 'publish', 'image': image,
                     'args': ['python', '-m', 'inferstack.infrastructure.model_cache', 'publish', '--root', mount_path,
                              '--source', source, '--model-json', json.dumps(model, sort_keys=True)],
                     'resources': {'limits': {'cpu': '2', 'memory': '4Gi'}, 'requests': {'cpu': '2', 'memory': '4Gi'}},
                     'securityContext': {'runAsNonRoot': True, 'runAsUser': 65532, 'allowPrivilegeEscalation': False,
                                         'readOnlyRootFilesystem': True, 'capabilities': {'drop': ['ALL']}},
                     'env': [{'name': 'HF_HUB_DISABLE_TELEMETRY', 'value': '1'}, {'name': 'HOME', 'value': '/tmp'}],
                     'volumeMounts': [{'name': 'model-cache', 'mountPath': mount_path},
                                      {'name': 'scratch', 'mountPath': '/tmp'}]}
        if credentials_secret:
            container['envFrom'] = [{'secretRef': {'name': credentials_secret}}]
        labels = {'app.kubernetes.io/managed-by': 'inferstack', 'inferstack.io/role': 'model-cache-publish'}
        return {'apiVersion': 'batch/v1', 'kind': 'Job', 'metadata': {'name': name, 'labels': labels},
                'spec': {'backoffLimit': 2, 'activeDeadlineSeconds': active_deadline_seconds,
                         'ttlSecondsAfterFinished': 3600,
                         'template': {'metadata': {'labels': labels},
                                      'spec': {'restartPolicy': 'Never', 'automountServiceAccountToken': False,
                                               'containers': [container],
                                               'volumes': [{'name': 'model-cache',
                                                            'persistentVolumeClaim': {'claimName': claim}},
                                                           {'name': 'scratch', 'emptyDir': {'sizeLimit': '4Gi'}}]}}}}

    def label_pod(self, name, labels):
        """Set (value) or remove (None) labels on one pod; used for identity-gated pool membership."""
        args = [f'{k}={v}' if v is not None else f'{k}-' for k, v in sorted(labels.items())]
        self.run(self.base + ['label', 'pod', name, '--overwrite', *args])

    def startup_phases(self, external):
        """Per-pod startup breakdown from the API's own timestamps (seconds after creation):
        scheduled, initialized (incl. cache verification), container started, ready.
        Image pull time is inside scheduled->started; engine load/compile inside started->ready."""
        from datetime import datetime
        stamp = lambda s: datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').timestamp() if s else None
        out = []
        for pod in self.pods(external) or []:
            created = stamp(pod['metadata'].get('creationTimestamp'))
            conditions = {c['type']: stamp(c.get('lastTransitionTime')) for c in (pod.get('status') or {}).get('conditions') or []
                          if c.get('status') == 'True'}
            started = None
            for status in (pod.get('status') or {}).get('containerStatuses') or []:
                started = stamp(((status.get('state') or {}).get('running') or {}).get('startedAt')) or started
            rel = lambda t: None if t is None or created is None else t - created
            out.append({'pod': pod['metadata']['name'], 'uid': pod['metadata'].get('uid'),
                        'node': (pod.get('spec') or {}).get('nodeName'),
                        'scheduled_s': rel(conditions.get('PodScheduled')), 'initialized_s': rel(conditions.get('Initialized')),
                        'started_s': rel(started), 'ready_s': rel(conditions.get('Ready'))})
        return out

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

    def namespace_manifests(self, *, max_pods, max_gpus, max_cpu, max_memory, default_cpu='1', default_memory='2Gi'):
        """ResourceQuota + LimitRange bounding everything InferStack may run in its namespace."""
        labels = {'app.kubernetes.io/managed-by': 'inferstack'}
        return [
            {'apiVersion': 'v1', 'kind': 'ResourceQuota', 'metadata': {'name': 'inferstack-quota', 'labels': labels},
             'spec': {'hard': {'pods': str(max_pods), 'requests.nvidia.com/gpu': str(max_gpus),
                               'limits.cpu': str(max_cpu), 'limits.memory': str(max_memory),
                               'requests.cpu': str(max_cpu), 'requests.memory': str(max_memory)}}},
            {'apiVersion': 'v1', 'kind': 'LimitRange', 'metadata': {'name': 'inferstack-defaults', 'labels': labels},
             'spec': {'limits': [{'type': 'Container', 'default': {'cpu': default_cpu, 'memory': default_memory},
                                  'defaultRequest': {'cpu': default_cpu, 'memory': default_memory}}]}},
        ]

    def ensure_quota(self, **limits):
        docs = self.namespace_manifests(**limits)
        self.run(self.base + ['apply', '-f', '-'], input_text=json.dumps({'apiVersion': 'v1', 'kind': 'List',
                                                                           'items': docs}))
        return docs

    def deployment_status(self, external):
        code, out, _ = self.run(self.base + ['get', 'deployment', external['name'], '-o', 'json'], check=False)
        if code != 0:
            return None
        body = json.loads(out)
        if body['metadata']['uid'] != external['uid']:
            return None
        status = body.get('status') or {}
        return {'replicas': body['spec'].get('replicas', 0), 'ready': status.get('readyReplicas', 0) or 0,
                'updated': status.get('updatedReplicas', 0) or 0, 'available': status.get('availableReplicas', 0) or 0}

    def scale(self, external, replicas, *, timeout=600.0, poll=1.0):
        """Scale the allocation's Deployment and wait until exactly `replicas` are ready and the rest are gone."""
        if isinstance(replicas, bool) or not isinstance(replicas, int) or not 0 <= replicas <= 64:
            raise ValueError('replicas must be an integer in [0, 64]')
        if self.deployment_status(external) is None:
            raise RuntimeError(f"deployment {external['name']} is not ours or does not exist")
        owner = self.scaling_owner(external)
        if owner not in (None, 'manual'):
            raise RuntimeError(f"deployment {external['name']} replicas are owned by {owner}; switch it to manual first")
        self.run(self.base + ['scale', 'deployment', external['name'], f'--replicas={replicas}'])
        deadline = time.monotonic() + timeout
        while True:
            status, pods = self.deployment_status(external), self.pods(external)
            if status and status['ready'] >= replicas and pods is not None and len(pods) == replicas:
                return {**status, 'pods': len(pods)}
            if time.monotonic() >= deadline:
                raise TimeoutError(f"scale to {replicas} not reached: {status}, pods={None if pods is None else len(pods)}")
            time.sleep(poll)

    def scaling_owner(self, external):
        """Who owns the replica count: 'manual' (InferStack), 'advisory' or 'keda'."""
        code, out, _ = self.run(self.base + ['get', 'deployment', external['name'], '-o', 'json'], check=False)
        if code:
            return None
        annotations = (json.loads(out).get('metadata') or {}).get('annotations') or {}
        return annotations.get('inferstack.io/scaling-owner', 'manual')

    def pending_pods(self, external):
        """Pods that cannot be scheduled (e.g. no free GPU, quota) with the scheduler's reason."""
        out = []
        for pod in self.pods(external) or []:
            for condition in (pod.get('status') or {}).get('conditions') or []:
                if condition.get('type') == 'PodScheduled' and condition.get('status') == 'False':
                    out.append({'pod': pod['metadata']['name'], 'reason': condition.get('reason'),
                                'message': (condition.get('message') or '')[:300]})
        return out

    def drain(self, external, *, timeout=600.0):
        """Graceful drain: scale to zero (preStop keeps serving in-flight work) and wait for all Pods to exit."""
        return self.scale(external, 0, timeout=timeout)

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

    def observed_image_digest(self, external):
        """Single image digest every running container of the allocation reports (Pod imageID), or None."""
        pods = self.pods(external)
        if not pods:
            return None
        digests = set()
        for pod in pods:
            for status in (pod.get('status') or {}).get('containerStatuses') or []:
                image_id = status.get('imageID') or ''
                if '@sha256:' not in image_id:
                    return None
                digests.add(image_id.split('@', 1)[1])
        return digests.pop() if len(digests) == 1 else None

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
