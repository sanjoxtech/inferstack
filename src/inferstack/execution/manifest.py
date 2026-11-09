"""ExecutionManifest v1: one sealed, content-addressed description of an inference experiment (E1.0).

The manifest is a *binding layer* over existing records, not a new source of truth.
It carries pinned references (`id@N`) to the model, deployment, workload, objective
and technique records, plus a resolved snapshot of the exact identity those
references stood for when the manifest was made. The snapshot exists to prove what
was executed; the records stay authoritative and may gain newer revisions without
changing an existing manifest.

Two digests:

- `experiment_id`: what the experiment *is*: model/tokenizer/config revisions,
  engine identity (version, image, native source revision and validated patch),
  precision, context/generation limits, profile, hardware requirement, workload
  fixture and measurement policy, technique references. Changing any of these makes
  a different experiment (BF16 -> FP16, A100 -> T4, a different vLLM version, ...).
- `digest`: the whole manifest, additionally covering spend limits, declared
  fallbacks and campaign lineage. The same experiment run under a larger budget or
  from a different campaign keeps its `experiment_id`.

Manifests are frozen. A fallback or alternative is always a *new* manifest with its
own identity (`derive`), never an in-place edit.

No credentials: any credential-looking key or value is refused at construction.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from inferstack.contracts.records import MUTABLE_REVISIONS, SHA256, parse_ref
from inferstack.contracts.redaction import reject_secrets, scrub_text

SCHEMA = 'inferstack.execution_manifest/v1'
ENGINES = ('inferstack_native', 'vllm', 'sglang', 'generic_openai')
# Bytes per weight element; used only for the static model-fit estimate.
DTYPE_BYTES = {'fp32': 4.0, 'bf16': 2.0, 'fp16': 2.0, 'fp8_e4m3': 1.0, 'fp8_e5m2': 1.0, 'int8': 1.0, 'int4': 0.5}
DTYPES = tuple(DTYPE_BYTES)
FALLBACK_KEYS = ('dtype', 'kv_cache_dtype', 'tensor_parallel', 'gpu_count')
_HEX40 = re.compile(r'^[0-9a-f]{40}$')
_UNPINNED_VERSION = re.compile(r'[*<>=~^,]|^(latest|main|master|nightly|dev)$', re.I)


def canonical_json(value) -> str:
    """Deterministic JSON: sorted keys, no whitespace, no NaN/Infinity."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def sha256_canonical(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class _Frozen(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class ModelIdentity(_Frozen):
    source: Literal['huggingface', 'foundry', 'local']
    repo_id: str = Field(min_length=1, max_length=300)
    revision: str = Field(min_length=1, max_length=200)
    tokenizer_revision: str | None = None
    config_revision: str | None = None
    architecture: str | None = Field(default=None, max_length=200)
    parameter_count: int | None = Field(default=None, gt=0)   # total weights, for the static fit estimate

    @model_validator(mode='after')
    def immutable(self):
        for name in ('revision', 'tokenizer_revision', 'config_revision'):
            value = getattr(self, name)
            if value is None:
                continue
            if value in MUTABLE_REVISIONS:
                raise ValueError(f'model {name}={value!r} is mutable; pin an immutable revision')
            if self.source == 'huggingface' and not _HEX40.match(value):
                raise ValueError(f'model {name} must be a 40-hex Hugging Face commit')
        return self


class EngineIdentity(_Frozen):
    name: Literal['inferstack_native', 'vllm', 'sglang', 'generic_openai']
    version: str = Field(min_length=1, max_length=100)
    image_digest: str | None = Field(default=None, pattern=r'^sha256:[0-9a-f]{64}$')
    # InferStack-native identity is its source, not a version string: the pinned commit
    # plus (for a patched candidate) the validated implementation's content address.
    source_revision: str | None = None
    candidate_id: str | None = Field(default=None, pattern=SHA256)
    patch_sha256: str | None = Field(default=None, pattern=SHA256)
    plugins: tuple[str, ...] = ()

    @model_validator(mode='after')
    def exact(self):
        if _UNPINNED_VERSION.search(self.version):
            raise ValueError(f'engine version {self.version!r} is a range or moving tag; pin an exact version')
        if self.source_revision is not None and not _HEX40.match(self.source_revision):
            raise ValueError('engine source_revision must be a 40-hex git commit')
        if self.name == 'inferstack_native' and self.source_revision is None:
            raise ValueError('inferstack_native engine identity requires source_revision (its git commit)')
        if (self.candidate_id is None) != (self.patch_sha256 is None):
            raise ValueError('candidate_id and patch_sha256 are declared together')
        if self.candidate_id is not None and self.source_revision is None:
            raise ValueError('a patched candidate requires its source_revision')
        return self


class HardwareRequirement(_Frozen):
    """What the experiment needs. GPU memory is per device: devices never add up
    unless the manifest declares tensor_parallel across them."""
    gpu_count: int = Field(default=1, ge=1, le=64)
    min_gpu_memory_bytes: int | None = Field(default=None, gt=0)
    min_compute_capability: str | None = Field(default=None, pattern=r'^[0-9]+\.[0-9]+$')
    gpu_names: tuple[str, ...] = ()          # accepted product names (substring match), empty = any
    tensor_parallel: int = Field(default=1, ge=1, le=64)

    @model_validator(mode='after')
    def parallelism_fits(self):
        if self.tensor_parallel > self.gpu_count:
            raise ValueError('tensor_parallel cannot exceed gpu_count')
        return self


class WorkloadIdentity(_Frozen):
    name: str = Field(min_length=1, max_length=200)
    fixture_sha256: str = Field(pattern=SHA256)
    max_context_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    concurrency: tuple[int, ...] = Field(default=(1,), min_length=1)

    @field_validator('concurrency')
    @classmethod
    def positive(cls, value):
        if any(c < 1 for c in value):
            raise ValueError('concurrency levels must be >= 1')
        return tuple(sorted(set(value)))


class MeasurementPolicy(_Frozen):
    evaluator_sha256: str = Field(pattern=SHA256)
    rounds: int = Field(default=3, ge=1, le=100)
    warmup_requests: int = Field(default=2, ge=0)
    gates: tuple[str, ...] = ()


class Limits(_Frozen):
    max_runtime_s: int = Field(gt=0)
    max_gpu_hours: float | None = Field(default=None, gt=0)
    max_cost_usd: float | None = Field(default=None, ge=0)
    max_output_tokens_total: int | None = Field(default=None, gt=0)


class Lineage(_Frozen):
    campaign_job_id: str | None = None
    plan_ref: str | None = None
    implementation_id: str | None = None
    iteration: int | None = Field(default=None, ge=0)
    parent_experiment_id: str | None = Field(default=None, pattern=SHA256)


class Refs(_Frozen):
    """Pinned references to the authoritative records (`id@N`)."""
    model: str | None = None
    deployment: str | None = None
    workload: str | None = None
    objective: str | None = None
    techniques: tuple[str, ...] = ()

    @model_validator(mode='after')
    def pinned(self):
        for name in ('model', 'deployment', 'workload', 'objective'):
            value = getattr(self, name)
            if value is not None and parse_ref(value)[1] is None:
                raise ValueError(f'refs.{name} must pin a revision (id@N)')
        for value in self.techniques:
            if parse_ref(value)[1] is None:
                raise ValueError('technique refs must pin a revision (id@N)')
        return self


class ExecutionManifestV1(_Frozen):
    schema_: Literal['inferstack.execution_manifest/v1'] = Field(default=SCHEMA, alias='schema')
    refs: Refs = Refs()
    model: ModelIdentity
    engine: EngineIdentity
    dtype: Literal['fp32', 'bf16', 'fp16', 'fp8_e4m3', 'fp8_e5m2', 'int8', 'int4']
    kv_cache_dtype: str | None = Field(default=None, max_length=40)
    profile: dict = Field(default_factory=dict)
    required_capabilities: tuple[str, ...] = ()
    hardware: HardwareRequirement
    workload: WorkloadIdentity
    measurement: MeasurementPolicy
    limits: Limits
    fallbacks: tuple[dict, ...] = ()
    lineage: Lineage = Lineage()

    model_config = ConfigDict(extra='forbid', frozen=True, populate_by_name=True)

    @model_validator(mode='before')
    @classmethod
    def no_credentials(cls, data):
        if isinstance(data, dict):
            reject_secrets(data)
            text = canonical_json(_plain(data))
            if scrub_text(text) != text:
                raise ValueError('manifest contains credential-looking text; reference secrets by *_env name only')
        return data

    @model_validator(mode='after')
    def consistent(self):
        if not isinstance(self.profile, dict) or 'dtype' in self.profile:
            raise ValueError('profile must be an object; precision belongs in manifest.dtype')
        if self.workload.max_output_tokens >= self.workload.max_context_tokens:
            raise ValueError('max_output_tokens must be below max_context_tokens')
        for fallback in self.fallbacks:
            if not fallback or set(fallback) - set(FALLBACK_KEYS):
                raise ValueError(f'fallbacks may only override {FALLBACK_KEYS}')
        return self

    # --- identity -------------------------------------------------------------

    def identity_body(self) -> dict:
        body = self.model_dump(mode='json', by_alias=True)
        return {k: body[k] for k in ('schema', 'model', 'engine', 'dtype', 'kv_cache_dtype', 'profile',
                                     'required_capabilities', 'hardware', 'workload', 'measurement')} | {
            'techniques': sorted(body['refs']['techniques'])}

    @property
    def experiment_id(self) -> str:
        return sha256_canonical(self.identity_body())

    @property
    def digest(self) -> str:
        return sha256_canonical(self.model_dump(mode='json', by_alias=True))

    def to_json(self) -> str:
        return canonical_json(self.model_dump(mode='json', by_alias=True))

    @classmethod
    def from_json(cls, text: str) -> 'ExecutionManifestV1':
        return cls.model_validate(json.loads(text))

    def derive(self, change: dict) -> 'ExecutionManifestV1':
        """A new manifest (new identity) with `change` applied; self is never modified.

        Keys: dtype, kv_cache_dtype, tensor_parallel, gpu_count. The derived manifest
        records this experiment as its parent and declares no further fallbacks.
        """
        unknown = set(change) - set(FALLBACK_KEYS)
        if unknown:
            raise ValueError(f'cannot derive by changing {sorted(unknown)}')
        body = self.model_dump(mode='json', by_alias=True)
        for key in ('dtype', 'kv_cache_dtype'):
            if key in change:
                body[key] = change[key]
        for key in ('tensor_parallel', 'gpu_count'):
            if key in change:
                body['hardware'][key] = change[key]
        if 'tensor_parallel' in change:
            old_tp, new_tp = self.hardware.tensor_parallel, change['tensor_parallel']
            if body['hardware']['gpu_count'] < new_tp:
                body['hardware']['gpu_count'] = new_tp
            # The per-device floor was one shard of the model; resharding divides it.
            if self.hardware.min_gpu_memory_bytes:
                body['hardware']['min_gpu_memory_bytes'] = -(-self.hardware.min_gpu_memory_bytes * old_tp // new_tp)
        body['fallbacks'] = []
        body['lineage'] = {**body['lineage'], 'parent_experiment_id': self.experiment_id}
        return type(self).model_validate(body)


def _plain(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode='json')
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)


# --- binding over existing records ----------------------------------------------

def _pinned(repository, kind, identifier):
    record = repository.read(kind, identifier)
    return f"{identifier}@{record['revision']}", record['body']


def from_records(repository, *, deployment_id: str, workload_id: str, workload_fixture_sha256: str,
                 measurement: dict, limits: dict, hardware: dict, max_context_tokens: int,
                 max_output_tokens: int, concurrency=(1,), objective_id: str | None = None,
                 technique_ids=(), dtype: str | None = None, required_capabilities=(), candidate: dict | None = None,
                 source_revision: str | None = None, fallbacks=(), lineage: dict | None = None) -> ExecutionManifestV1:
    """Bind the current revisions of existing records into a sealed manifest.

    The deployment must reference a typed model record; precision comes from the
    deployment profile unless given. `candidate` is a validated H3 implementation
    candidate ({'candidate_id', 'patch_sha256', 'source_revision'}).
    """
    deployment_ref, deployment = _pinned(repository, 'deployment', deployment_id)
    if not deployment.get('model_ref'):
        raise ValueError('deployment has no model_ref; register the model record first')
    model_id, model_rev = parse_ref(deployment['model_ref'])
    model_record = repository.read('model', model_id, model_rev)
    model_ref = f"{model_id}@{model_record['revision']}"
    model = model_record['body']
    workload_ref, _ = _pinned(repository, 'workload', workload_id)
    objective_ref = _pinned(repository, 'objective', objective_id)[0] if objective_id else None
    technique_refs = tuple(_pinned(repository, 'technique', t)[0] for t in technique_ids)
    profile = dict(deployment.get('profile') or {})
    precision = dtype or profile.pop('dtype', None)
    profile.pop('dtype', None)
    if precision is None:
        raise ValueError('precision unknown: set deployment profile.dtype or pass dtype')
    kv_cache_dtype = profile.pop('kv_cache_dtype', None)
    for key in [k for k in profile if k.startswith('supported_')]:
        profile.pop(key)
    candidate = candidate or {}
    engine = {'name': deployment['engine'], 'version': deployment.get('engine_version') or '',
              'image_digest': deployment.get('image_digest'),
              'source_revision': candidate.get('source_revision') or source_revision,
              'candidate_id': candidate.get('candidate_id'), 'patch_sha256': candidate.get('patch_sha256'),
              'plugins': sorted(deployment.get('plugin_refs') or [])}
    return ExecutionManifestV1.model_validate({
        'refs': {'model': model_ref, 'deployment': deployment_ref, 'workload': workload_ref,
                 'objective': objective_ref, 'techniques': technique_refs},
        'model': {'source': model['source'], 'repo_id': model['repo_id'], 'revision': model['revision'],
                  'tokenizer_revision': model.get('tokenizer_revision'),
                  'config_revision': model.get('config_revision'), 'architecture': model.get('architecture')},
        'engine': engine, 'dtype': precision, 'kv_cache_dtype': kv_cache_dtype, 'profile': profile,
        'required_capabilities': tuple(required_capabilities), 'hardware': hardware,
        'workload': {'name': workload_id, 'fixture_sha256': workload_fixture_sha256,
                     'max_context_tokens': max_context_tokens, 'max_output_tokens': max_output_tokens,
                     'concurrency': tuple(concurrency)},
        'measurement': measurement, 'limits': limits, 'fallbacks': tuple(fallbacks), 'lineage': lineage or {}})
