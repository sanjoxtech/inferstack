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
    deployment = {'id': 'd', 'engine': 'vllm', 'model': 'm', 'base_url': 'http://cand', 'model_ref': 'qwen',
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
