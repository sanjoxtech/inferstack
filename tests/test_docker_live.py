"""Live Docker lifecycle (H): launch, identity, discovery, attestation evidence, verified stop.

Runs only against a real Docker daemon with a locally present digest-pinned image
(INFERSTACK_TEST_DOCKER_IMAGE=repo@sha256:..., default: local python:3.12-slim).
Never pulls. Skips name the missing requirement.
"""
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.request

import pytest

from inferstack.infrastructure.remote import DockerProvider


def local_image():
    configured = os.environ.get('INFERSTACK_TEST_DOCKER_IMAGE')
    if configured:
        return configured
    if not shutil.which('docker'):
        return None
    out = subprocess.run(['docker', 'image', 'inspect', '--format', '{{json .RepoDigests}}', 'python:3.12-slim'],
                         capture_output=True, text=True)
    digests = json.loads(out.stdout) if out.returncode == 0 and out.stdout.strip() else []
    return digests[0] if digests else None


def daemon_up():
    return shutil.which('docker') and subprocess.run(['docker', 'info'], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(not (daemon_up() and local_image()),
                                reason='needs a running Docker daemon and a local digest-pinned image '
                                       '(INFERSTACK_TEST_DOCKER_IMAGE)')


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def test_real_container_lifecycle():
    image, port = local_image(), free_port()
    provider = DockerProvider(memory='256m', pids=64)
    alloc = 'alloc-livetest' + str(os.getpid())
    external = provider.launch(alloc, {'image': image, 'host_port': port, 'container_port': 8000,
                                       'args': ['python', '-m', 'http.server', '8000', '--directory', '/tmp']})
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=2).read()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.3)
        assert provider.alive(external)
        assert provider.find_tagged(alloc) == [external['container_id']]
        assert provider.observed_image_digest(external) == image.split('@', 1)[1]
        assert provider.stop(external, grace_seconds=2) is True
        assert not provider.alive(external) and provider.find_tagged(alloc) == []
    finally:
        subprocess.run(['docker', 'rm', '-f', external['container_id']], capture_output=True)


SERVER = ('import json, http.server, tinyplug\n'
          'class H(http.server.BaseHTTPRequestHandler):\n'
          '    def do_GET(self):\n'
          '        body = json.dumps({"data": [{"id": "m"}], "plugin": tinyplug.VERSION}).encode()\n'
          '        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()\n'
          '        self.wfile.write(body)\n'
          'http.server.HTTPServer(("0.0.0.0", 8000), H).serve_forever()\n')


@pytest.mark.asyncio
async def test_plugin_candidate_is_built_into_an_image_and_served_by_docker(tmp_path):
    """V16: a plugin experiment on a container provider builds a pinned candidate image offline and runs it."""
    import subprocess as sp

    from inferstack.execution.executor import IsolatedExecutor
    from inferstack.infrastructure import AllocationManager
    from inferstack.infrastructure.images import ImageBuilder
    from inferstack.storage.repository import Repository
    from test_isolated_executor import make_wheel

    image = local_image()
    root = tmp_path / 'src'
    root.mkdir()
    (root / 'README').write_text('x')
    for args in (['init', '-q'], ['add', '-A'], ['-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'b']):
        sp.run(['git', '-C', str(root), *args], check=True, capture_output=True)
    revision = sp.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    index = tmp_path / 'index'
    index.mkdir()
    make_wheel(index, 'tinyplug', '1.0.0')
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    port = free_port()
    repo.put_record('deployment', 'base', {
        'id': 'base', 'engine': 'vllm', 'model': 'm', 'base_url': f'http://127.0.0.1:{port}', 'engine_version': '0.11.0',
        'profile': {'container': {'image': image, 'container_port': 8000, 'host_port': port,
                                  'args': ['python', '-c', SERVER]}}})
    repo.put_record('technique', 'fused', {'name': 'plug', 'summary': 's', 'engines': [{'engine': 'vllm'}],
                                           'dependencies': [{'name': 'tinyplug', 'version': '==1.0.0'}]})
    job = repo.submit('j', {})
    claim = repo.claim(600, worker_id='w')
    allocations = AllocationManager(repo, 'w')
    executor = IsolatedExecutor(repo, tmp_path / 'logs', work_root=tmp_path / 'work', source_root=root,
                                source_revision=revision, allocations=allocations,
                                provider=DockerProvider(memory='256m', pids=64), ready_timeout=60,
                                pip_index_args=('--no-index', '--find-links', str(index)),
                                image_builder=ImageBuilder())
    plan = {'baseline_id': 'base', 'baseline_revision': 1, 'action_order': ['plug'],
            'actions': [{'id': 'plug', 'type': 'plugin_install',
                         'params': {'technique_id': 'fused', 'dependency': 'tinyplug==1.0.0'}}]}
    context = {'job': {**claim, 'id': job, 'step_attempt': 1}, 'step_key': 'iter0.experiment'}
    candidate = await executor.execute(plan, 'imgexp', context)
    try:
        record = repo.read('deployment', candidate['deployment_id'])['body']
        built = record['profile']['container']['image']
        assert built.startswith('sha256:') and record['image_digest'] == built
        assert DockerProvider().observed_image_id(candidate['external']) == built  # runs exactly the built image
        url = record['base_url'] + '/v1/models'
        assert json.loads(urllib.request.urlopen(url, timeout=5).read())['plugin'] == '1.0.0'
        assert 'image.Dockerfile' in candidate['artifacts']
    finally:
        await executor.cleanup(candidate)
    assert [a['state'] for a in allocations.list(job_id=job) if a['provider'] == 'docker'] == ['released']
