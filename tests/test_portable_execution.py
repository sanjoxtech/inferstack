"""E1.0 ExecutionManifest identity and E1.1 CompatibilityResolver acceptance.

No GPU and no provider: environments are recorded HardwareEnvironment fixtures.
"""
import json
import time

import pytest
from pydantic import ValidationError

from inferstack.contracts.records import TechniqueV2
from inferstack.execution.compatibility import resolve
from inferstack.execution.manifest import ExecutionManifestV1, canonical_json, from_records
from inferstack.storage.repository import Repository

GIB = 1 << 30
REV = 'b1485b2fa6dfa1287294f269f5fb618e03d52d7c'
SRC = 'a' * 40


def body(**over):
    base = {
        'refs': {'model': 'qwen@1', 'deployment': 'dep@3', 'workload': 'wl@1', 'techniques': []},
        'model': {'source': 'huggingface', 'repo_id': 'Qwen/Qwen3.5-2B-Base', 'revision': REV,
                  'tokenizer_revision': REV, 'config_revision': REV, 'architecture': 'qwen3_5',
                  'parameter_count': 2_000_000_000},
        'engine': {'name': 'vllm', 'version': '0.11.0'},
        'dtype': 'bf16',
        'profile': {'max_num_seqs': 8},
        'hardware': {'gpu_count': 1, 'min_gpu_memory_bytes': 16 * GIB},
        'workload': {'name': 'scorecard', 'fixture_sha256': 'f' * 64, 'max_context_tokens': 8192,
                     'max_output_tokens': 256, 'concurrency': [8]},
        'measurement': {'evaluator_sha256': 'e' * 64, 'rounds': 3, 'gates': ['correctness']},
        'limits': {'max_runtime_s': 3600, 'max_cost_usd': 5.0},
    }
    for key, value in over.items():
        if isinstance(value, dict) and key in ('model', 'workload', 'measurement', 'limits', 'refs'):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


def manifest(**over):
    return ExecutionManifestV1.model_validate(body(**over))


def gpu(i, name, gib, cc):
    return {'uuid': f'GPU-{name}-{i}', 'name': name, 'memory_bytes': gib * GIB, 'compute_capability': cc}


def env(*gpus, libraries=None):
    return {'provider': 'colab', 'gpus': list(gpus), 'observed_at': time.time(), 'libraries': libraries or {}}


A100 = env(gpu(0, 'NVIDIA A100-SXM4-40GB', 40, '8.0'))
T4 = env(gpu(0, 'Tesla T4', 15, '7.5'))
T4X2 = env(gpu(0, 'Tesla T4', 16, '7.5'), gpu(1, 'Tesla T4', 16, '7.5'))
SUPPORTED = {('vllm', 'Qwen/Qwen3.5-2B-Base'): 'supported'}


# --- E1.0 manifest identity -------------------------------------------------------

def test_digest_is_deterministic_and_ignores_key_order():
    a = manifest()
    shuffled = json.loads(json.dumps(body()), object_pairs_hook=lambda kv: dict(reversed(kv)))
    b = ExecutionManifestV1.model_validate(shuffled)
    assert a.digest == b.digest and a.experiment_id == b.experiment_id
    assert ExecutionManifestV1.from_json(a.to_json()).digest == a.digest
    assert canonical_json({'b': 1, 'a': [2, {'d': 1, 'c': 2}]}) == '{"a":[2,{"c":2,"d":1}],"b":1}'


@pytest.mark.parametrize('change', [
    {'model': {'revision': 'c' * 40}},
    {'model': {'tokenizer_revision': 'c' * 40}},
    {'model': {'config_revision': 'c' * 40}},
    {'engine': {'name': 'vllm', 'version': '0.11.1'}},
    {'engine': {'name': 'vllm', 'version': '0.11.0', 'image_digest': 'sha256:' + '1' * 64}},
    {'dtype': 'fp16'},
    {'kv_cache_dtype': 'fp8'},
    {'profile': {'max_num_seqs': 16}},
    {'hardware': {'gpu_count': 1, 'min_gpu_memory_bytes': 80 * GIB}},
    {'hardware': {'gpu_count': 1, 'min_compute_capability': '9.0'}},
    {'workload': {'fixture_sha256': '0' * 64}},
    {'workload': {'concurrency': [16]}},
    {'workload': {'max_context_tokens': 4096}},
    {'measurement': {'evaluator_sha256': '0' * 64}},
    {'refs': {'model': 'qwen@1', 'techniques': ['fuse@2']}},
])
def test_every_identity_field_changes_the_experiment_id(change):
    assert manifest(**change).experiment_id != manifest().experiment_id


def test_budget_and_lineage_change_the_digest_but_not_the_experiment():
    base = manifest()
    richer = manifest(limits={'max_runtime_s': 7200, 'max_cost_usd': 20.0},
                      lineage={'campaign_job_id': 'job-7', 'iteration': 2})
    assert richer.experiment_id == base.experiment_id and richer.digest != base.digest


def test_native_identity_includes_source_and_validated_patch():
    native = manifest(engine={'name': 'inferstack_native', 'version': '2.0.0rc1', 'source_revision': SRC},
                      profile={'forward_fusion': True})
    patched = manifest(engine={'name': 'inferstack_native', 'version': '2.0.0rc1', 'source_revision': SRC,
                               'candidate_id': '1' * 64, 'patch_sha256': '2' * 64}, profile={'forward_fusion': True})
    assert native.experiment_id != patched.experiment_id
    with pytest.raises(ValidationError, match='source_revision'):
        manifest(engine={'name': 'inferstack_native', 'version': '2.0.0rc1'})
    with pytest.raises(ValidationError, match='together'):
        manifest(engine={'name': 'inferstack_native', 'version': '2.0.0rc1', 'source_revision': SRC,
                         'candidate_id': '1' * 64})


@pytest.mark.parametrize('version', ['*', '>=0.10', 'latest', '0.11.*', '~=0.11'])
def test_engine_version_must_be_exact(version):
    with pytest.raises(ValidationError, match='exact'):
        manifest(engine={'name': 'vllm', 'version': version})


def test_mutable_model_revision_and_unpinned_refs_are_refused():
    with pytest.raises(ValidationError):
        manifest(model={'revision': 'main'})
    with pytest.raises(ValidationError, match='pin'):
        manifest(refs={'model': 'qwen'})


@pytest.mark.parametrize('leak', [
    {'profile': {'api_key': 'not-a-real-secret-value'}},
    {'profile': {'max_num_seqs': 8, 'note': 'sk-' + 'x' * 30}},
    {'lineage': {'campaign_job_id': 'https://user:hunter2@example.com/x'}},
])
def test_credentials_cannot_be_serialized(leak):
    with pytest.raises((ValidationError, ValueError)):
        manifest(**leak)


def test_manifest_is_frozen_and_fallbacks_never_mutate_it():
    m = manifest(fallbacks=[{'dtype': 'fp16'}])
    with pytest.raises(ValidationError):
        m.dtype = 'fp16'
    before = m.digest
    derived = m.derive({'dtype': 'fp16'})
    assert m.dtype == 'bf16' and m.digest == before
    assert derived.experiment_id != m.experiment_id and derived.lineage.parent_experiment_id == m.experiment_id
    assert derived.fallbacks == ()
    with pytest.raises(ValueError):
        m.derive({'engine': 'sglang'})


def test_bound_snapshot_survives_newer_record_revisions(tmp_path):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    repo.put_record('model', 'qwen', {'source': 'huggingface', 'repo_id': 'Qwen/Qwen3.5-2B-Base', 'revision': REV,
                                      'tokenizer_revision': REV, 'architecture': 'qwen3_5'})
    repo.put_record('deployment', 'dep', {'id': 'dep', 'engine': 'vllm', 'model': 'Qwen/Qwen3.5-2B-Base',
                                          'base_url': 'http://x', 'model_ref': 'qwen@1', 'engine_version': '0.11.0',
                                          'profile': {'dtype': 'bf16', 'max_num_seqs': 8}})
    repo.put_record('workload', 'wl', {'name': 'wl', 'cases': [{'prompt': 'hi'}], 'concurrency': 8})
    kwargs = dict(deployment_id='dep', workload_id='wl', workload_fixture_sha256='f' * 64,
                  measurement={'evaluator_sha256': 'e' * 64}, limits={'max_runtime_s': 600},
                  hardware={'gpu_count': 1}, max_context_tokens=8192, max_output_tokens=256, concurrency=(8,))
    m = from_records(repo, **kwargs)
    assert m.refs.deployment == 'dep@1' and m.refs.model == 'qwen@1' and m.dtype == 'bf16'
    assert m.engine.version == '0.11.0' and 'dtype' not in m.profile
    frozen = m.to_json()
    repo.put_record('deployment', 'dep', {'id': 'dep', 'engine': 'vllm', 'model': 'Qwen/Qwen3.5-2B-Base',
                                          'base_url': 'http://x', 'model_ref': 'qwen@1', 'engine_version': '0.12.0',
                                          'profile': {'dtype': 'bf16', 'max_num_seqs': 8}}, expected_revision=1)
    assert m.to_json() == frozen  # the old manifest is unchanged
    newer = from_records(repo, **kwargs)
    assert newer.refs.deployment == 'dep@2' and newer.experiment_id != m.experiment_id


# --- E1.1 compatibility -----------------------------------------------------------

def test_a100_bf16_is_supported_with_conformance_evidence():
    result = resolve(manifest(), A100, model_support=SUPPORTED)
    assert result.status == 'supported', result.reasons
    assert result.experiment_id == manifest().experiment_id


def test_unknown_engine_model_intersection_requires_a_probe():
    result = resolve(manifest(), A100)
    assert result.status == 'probe_required'
    assert any('conformance' in p for p in result.probes)


def test_t4_fp16_is_supported_when_declared():
    m = manifest(dtype='fp16', hardware={'gpu_count': 1, 'min_gpu_memory_bytes': 14 * GIB})
    assert resolve(m, T4, model_support=SUPPORTED).status == 'supported'


def test_t4_bf16_is_refused_before_spend_with_fp16_alternative():
    result = resolve(manifest(hardware={'gpu_count': 1}), T4, model_support=SUPPORTED)
    assert result.status == 'unsupported' and any('8.0' in r for r in result.reasons)
    fp16 = [a for a in result.alternatives if a['change'] == {'dtype': 'fp16'}]
    assert fp16 and fp16[0]['new_identity'] and fp16[0]['experiment_id'] != result.experiment_id


def test_t4_fp8_request_is_unsupported_with_fp16_and_hardware_alternatives():
    m = manifest(dtype='fp8_e4m3', hardware={'gpu_count': 1})
    result = resolve(m, T4, model_support={('vllm', 'Qwen/Qwen3.5-2B-Base'): 'supported'})
    assert result.status == 'unsupported'
    assert any('8.9' in r and '7.5' in r for r in result.reasons)
    kinds = {(a['kind'], json.dumps(a['change'])) for a in result.alternatives}
    assert ('manifest', json.dumps({'dtype': 'fp16'})) in kinds
    assert any(a['kind'] == 'environment' for a in result.alternatives)
    assert any(a['kind'] == 'technique' and 'weight-only' in a['reason'] for a in result.alternatives)
    assert m.dtype == 'fp8_e4m3'  # the request itself was not rewritten


def test_two_t4s_are_not_one_32gb_gpu():
    m = manifest(model={'parameter_count': None}, hardware={'gpu_count': 1, 'min_gpu_memory_bytes': 24 * GIB},
                 dtype='fp16')
    result = resolve(m, T4X2, model_support=SUPPORTED)
    assert result.status == 'unsupported'
    assert any('memory is per device' in r for r in result.reasons)
    tp = [a for a in result.alternatives if (a['change'] or {}).get('tensor_parallel') == 2]
    assert tp and tp[0]['new_identity']


def test_declared_tensor_parallel_is_only_as_good_as_the_engine_model_evidence():
    m = manifest(model={'parameter_count': None}, dtype='fp16',
                 hardware={'gpu_count': 2, 'tensor_parallel': 2, 'min_gpu_memory_bytes': 12 * GIB})
    assert resolve(m, T4X2, model_support=SUPPORTED).status == 'probe_required'  # TP=2 support unknown
    proven = {('vllm', 'Qwen/Qwen3.5-2B-Base', 'tp2'): 'supported'}
    assert resolve(m, T4X2, model_support=proven).status == 'probe_required'  # model size still unknown
    sized = manifest(dtype='fp16', hardware={'gpu_count': 2, 'tensor_parallel': 2})
    assert resolve(sized, T4X2, model_support=proven).status == 'supported'
    native = manifest(engine={'name': 'inferstack_native', 'version': '2.0.0rc1', 'source_revision': SRC},
                      profile={}, dtype='fp16', hardware={'gpu_count': 2, 'tensor_parallel': 2})
    assert resolve(native, T4X2, model_support=proven).status == 'unsupported'


def test_declared_fallback_is_a_separate_experiment():
    m = manifest(hardware={'gpu_count': 1}, fallbacks=[{'dtype': 'fp16'}])
    result = resolve(m, T4, model_support=SUPPORTED)
    assert result.status == 'supported_with_fallback'
    assert result.fallback['change'] == {'dtype': 'fp16'}
    assert result.fallback['experiment_id'] != m.experiment_id == result.experiment_id
    assert ExecutionManifestV1.model_validate(result.fallback['manifest']).dtype == 'fp16'
    assert m.dtype == 'bf16'


def test_model_that_does_not_fit_one_device_is_refused_and_near_fit_needs_probe():
    big = manifest(model={'parameter_count': 9_000_000_000}, dtype='fp16', hardware={'gpu_count': 1})
    assert resolve(big, T4, model_support=SUPPORTED).status == 'unsupported'
    near = manifest(model={'parameter_count': 5_000_000_000}, dtype='fp16', hardware={'gpu_count': 1})
    result = resolve(near, T4, model_support=SUPPORTED)
    assert result.status == 'probe_required' and any('fit probe' in p for p in result.probes)
    unknown = manifest(model={'parameter_count': None})
    assert resolve(unknown, A100, model_support=SUPPORTED).status == 'probe_required'


def test_invalid_profile_option_is_unsupported():
    result = resolve(manifest(profile={'max_num_seqs': 8, 'not_an_option': 1}), A100, model_support=SUPPORTED)
    assert result.status == 'unsupported' and any('profile' in r for r in result.reasons)


def test_capability_states_are_respected():
    unknown = manifest(engine={'name': 'inferstack_native', 'version': '2.0.0rc1', 'source_revision': SRC},
                       profile={}, required_capabilities=['cuda_graph_replay'])
    native_ok = {('inferstack_native', 'Qwen/Qwen3.5-2B-Base'): 'supported'}
    assert resolve(unknown, A100, model_support=native_ok).status == 'probe_required'
    missing = manifest(engine={'name': 'inferstack_native', 'version': '2.0.0rc1', 'source_revision': SRC},
                       profile={}, required_capabilities=['paged_kv'])
    assert resolve(missing, A100, model_support=native_ok).status == 'unsupported'


def technique(**over):
    fields = {'name': 'fused-x', 'summary': 's', 'engines': [{'engine': 'vllm'}], 'dtypes': ['bf16'],
              'min_compute_capability': '9.0', 'min_gpu_memory_bytes': 30 * GIB}
    return TechniqueV2.model_validate({**fields, **over})


def test_technique_compute_capability_and_memory_floors_are_enforced():
    result = resolve(manifest(), A100, techniques=[technique()], model_support=SUPPORTED)
    assert result.status == 'unsupported' and any('technique fused-x' in r and '9.0' in r for r in result.reasons)
    small = env(gpu(0, 'NVIDIA H100', 24, '9.0'))
    result = resolve(manifest(), small, techniques=[technique()], model_support=SUPPORTED)
    assert result.status == 'unsupported' and any('GiB' in r for r in result.reasons)


def test_a_proven_label_does_not_override_unknown_facts_here():
    proven = technique(min_compute_capability=None, min_gpu_memory_bytes=None, status='proven',
                       proven_by=['trial-1@1'], requires_capabilities=['request_trace'])
    result = resolve(manifest(), A100, techniques=[proven], model_support=SUPPORTED)
    assert result.status == 'probe_required'  # vLLM request_trace is unknown, whatever the label says
    assert 'proof_of' in result.checked['proof']


def test_unpinned_technique_dependency_and_wrong_installed_engine_are_refused():
    loose = technique(min_compute_capability=None, min_gpu_memory_bytes=None,
                      dependencies=[{'name': 'flashinfer', 'version': '>=0.2'}])
    assert resolve(manifest(), A100, techniques=[loose], model_support=SUPPORTED).status == 'unsupported'
    stale = env(gpu(0, 'NVIDIA A100-SXM4-40GB', 40, '8.0'), libraries={'vllm': '0.10.2'})
    result = resolve(manifest(), stale, model_support=SUPPORTED)
    assert result.status == 'unsupported' and any('0.10.2' in r for r in result.reasons)


def test_unrecorded_compute_capability_requires_a_probe():
    blind = env({'uuid': 'GPU-x', 'name': 'NVIDIA A100', 'memory_bytes': 40 * GIB})
    assert resolve(manifest(), blind, model_support=SUPPORTED).status == 'probe_required'
