"""V05: release attestation compares the runtime with every pinned manifest field."""
import httpx
import pytest

from inferstack.releases.attestation import compare, release_attestor
from inferstack.storage.repository import Repository

REV = 'b' * 40
EXPECTED = {'model': 'm', 'engine_version': '0.9.1', 'image_digest': 'sha256:' + 'a' * 64, 'plugin_refs': [],
            'build_sha256': None, 'model_revision': REV, 'tokenizer_revision': REV}
IDENTITY = {'served_models': ['m'], 'version': '0.9.1'}
LAUNCHED = {'model_revision': REV, 'tokenizer_revision': REV, 'image_digest': 'sha256:' + 'a' * 64}


def test_complete_identity_is_attested():
    assert compare(EXPECTED, IDENTITY, LAUNCHED)[0] == 'attested'


@pytest.mark.parametrize('field,value', [('model_revision', 'c' * 40), ('image_digest', 'sha256:' + 'd' * 64)])
def test_any_mismatch_is_mismatch(field, value):
    assert compare(EXPECTED, IDENTITY, {**LAUNCHED, field: value})[0] == 'mismatch'


def test_missing_observation_or_unpinned_manifest_is_unverified():
    assert compare(EXPECTED, IDENTITY, {})[0] == 'unverified'
    assert compare({**EXPECTED, 'model_revision': None}, IDENTITY, LAUNCHED)[0] == 'unverified'
    assert compare({**EXPECTED, 'image_digest': None}, IDENTITY, LAUNCHED)[0] == 'unverified'


@pytest.mark.asyncio
async def test_attestor_reads_runtime_and_launch_observation(tmp_path):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    repo.put_record('model', 'qwen', {'source': 'huggingface', 'repo_id': 'Qwen/x', 'revision': REV})
    deployment = {'id': 'd', 'engine': 'vllm', 'model': 'm', 'base_url': 'http://cand', 'model_ref': 'qwen@1',
                  'engine_version': '0.9.1', 'image_digest': 'sha256:' + 'a' * 64}
    repo.put_record('deployment', 'd', deployment)
    release = {'body': {'deployment': repo.read('deployment', 'd')}}

    def handler(request):
        if request.url.path == '/v1/models':
            return httpx.Response(200, json={'data': [{'id': 'm'}]})
        if request.url.path == '/version':
            return httpx.Response(200, json={'version': '0.9.1'})
        return httpx.Response(404)

    async def launched(_):
        return LAUNCHED
    attest = release_attestor(repo, observe_launch=launched, transport=httpx.MockTransport(handler))
    assert (await attest(release))['status'] == 'attested'
    attest = release_attestor(repo, transport=httpx.MockTransport(handler))
    assert (await attest(release))['status'] == 'unverified'


@pytest.mark.asyncio
async def test_live_runtime_is_attested_from_observed_launch(tmp_path):
    """V05 end to end: the running process's command line supplies the model/tokenizer revisions."""
    import socket
    import sys
    from pathlib import Path

    from inferstack.infrastructure import LocalProcessProvider
    from inferstack.releases.attestation import launch_observer
    from inferstack.releases.runtime import ReleaseRuntime

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    repo.put_record('model', 'qwen', {'source': 'huggingface', 'repo_id': 'Qwen/x', 'revision': REV})
    fixture = str(Path(__file__).parent / 'fixtures/fake_openai_server.py')
    repo.put_record('deployment', 'd', {
        'id': 'd', 'engine': 'generic_openai', 'model': 'm', 'base_url': f'http://127.0.0.1:{port}', 'model_ref': 'qwen@1',
        'command': [sys.executable, fixture, '--port', str(port), '--revision', REV, '--tokenizer-revision', REV],
        'profile': {'build_sha256': 'c' * 64}})
    repo.put('release', 'rel', {'deployment': repo.read('deployment', 'd'), 'evidence': {'sha256': 'f' * 64, 'bytes': 0},
                                'job_id': 'j', 'verification_scope': 's'})
    provider = LocalProcessProvider()
    runtime = ReleaseRuntime(repo, provider, log_root=tmp_path / 'logs', ready_timeout=20)
    attest = release_attestor(repo, observe_launch=launch_observer(repo, {'local': provider}))
    try:
        await runtime.ensure('rel')
        result = await attest(repo.read('release', 'rel'))
        assert result['launched']['model_revision'] == REV and result['launched']['replicas'] == 1
        assert result['checks']['model_revision'] is True and result['checks']['tokenizer_revision'] is True
        # generic engine reports no version and the build digest is not observable for a plain process
        assert result['status'] == 'unverified' and result['checks']['artifact_digest'] is None
    finally:
        await runtime.retire('rel')
