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
