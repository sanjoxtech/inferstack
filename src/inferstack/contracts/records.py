"""Versioned record schemas for the InferStack control plane.

Each record kind has a numbered schema. The schema version is stored beside the
body (not inside it) so existing consumers of record bodies stay unchanged.

Reference fields name another record as `id` (current revision) or `id@N`
(pinned revision N). The repository verifies that every reference resolves before
a typed write commits. Pinned references are required where reproducibility
depends on an exact snapshot (plans, trials).
"""
from __future__ import annotations

import re
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .metrics import require_metric

ID_PATTERN = r'^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$'
_ID = re.compile(ID_PATTERN)
_REF = re.compile(r'^(?P<id>[A-Za-z0-9][A-Za-z0-9._:/-]{0,199})(?:@(?P<rev>[1-9][0-9]*))?$')
SHA256 = r'^[0-9a-f]{64}$'
MUTABLE_REVISIONS = {'main', 'master', 'latest', 'head', 'HEAD', 'dev', 'trunk'}


class IncompatibleSchema(ValueError):
    pass


def valid_id(identifier: str) -> bool:
    return isinstance(identifier, str) and bool(_ID.match(identifier))


def parse_ref(value: str) -> tuple[str, int | None]:
    match = _REF.match(value) if isinstance(value, str) else None
    if not match:
        raise ValueError(f'invalid record reference: {value!r}')
    return match['id'], int(match['rev']) if match['rev'] else None


class Record(BaseModel):
    model_config = ConfigDict(extra='forbid')
    KIND: ClassVar[str]
    VERSION: ClassVar[int] = 1
    # field name -> referenced kind; field value is a ref string, list of refs or None.
    REFS: ClassVar[dict[str, str]] = {}
    PINNED: ClassVar[set[str]] = set()

    def references(self) -> list[tuple[str, str, int | None]]:
        out = []
        for field, kind in self.REFS.items():
            value = getattr(self, field)
            for item in (value if isinstance(value, list) else [value]):
                if item is None:
                    continue
                identifier, revision = parse_ref(item)
                if field in self.PINNED and revision is None:
                    raise ValueError(f'{field} must pin a revision (id@N)')
                out.append((kind, identifier, revision))
        return out


# --- model, hardware, deployment ------------------------------------------------

class ModelArtifact(Record):
    KIND = 'model'
    source: Literal['huggingface', 'foundry', 'local']
    repo_id: str = Field(min_length=1, max_length=300)
    revision: str = Field(min_length=1, max_length=200)
    tokenizer_revision: str | None = None
    config_revision: str | None = None
    task: Literal['text-generation', 'chat', 'embedding', 'other'] = 'text-generation'
    architecture: str | None = None
    dtype: str | None = None
    checksums: dict[str, str] = Field(default_factory=dict)
    license: str | None = None

    @model_validator(mode='after')
    def immutable_revisions(self):
        for name in ('revision', 'tokenizer_revision', 'config_revision'):
            value = getattr(self, name)
            if value is None:
                continue
            if value in MUTABLE_REVISIONS:
                raise ValueError(f'{name}={value!r} is mutable; pin an immutable revision')
            if self.source == 'huggingface' and not re.fullmatch(r'[0-9a-f]{40}', value):
                raise ValueError(f'{name} must be a 40-hex Hugging Face commit')
        for path, digest in self.checksums.items():
            if not re.fullmatch(SHA256, digest):
                raise ValueError(f'checksum for {path} must be sha256 hex')
        return self


class GPU(BaseModel):
    model_config = ConfigDict(extra='forbid')
    uuid: str = Field(min_length=1)
    name: str
    memory_bytes: int = Field(gt=0)
    compute_capability: str | None = None


class HardwareEnvironment(Record):
    KIND = 'hardware'
    provider: str = Field(min_length=1)        # attached | colab | gke | eks | ...
    region: str | None = None
    gpus: list[GPU] = Field(default_factory=list)
    topology: str | None = None
    cpu_model: str | None = None
    cpu_count: int | None = Field(default=None, gt=0)
    ram_bytes: int | None = Field(default=None, gt=0)
    driver_version: str | None = None
    cuda_version: str | None = None
    libraries: dict[str, str] = Field(default_factory=dict)
    observed_at: float = Field(gt=0)

    @field_validator('gpus')
    @classmethod
    def unique_gpus(cls, gpus):
        if len({g.uuid for g in gpus}) != len(gpus):
            raise ValueError('duplicate GPU UUID')
        return gpus


class Deployment(Record):
    """v1 keeps the legacy DeploymentSpec fields and adds optional identity refs."""
    KIND = 'deployment'
    REFS = {'model_ref': 'model', 'hardware_ref': 'hardware', 'plugin_refs': 'technique'}
    id: str
    engine: Literal['inferstack_native', 'vllm', 'sglang', 'generic_openai']
    model: str = Field(min_length=1)
    base_url: str
    api_key_env: str | None = None
    command: list[str] = Field(default_factory=list)
    model_ref: str | None = None
    hardware_ref: str | None = None
    engine_version: str | None = None
    image_digest: str | None = Field(default=None, pattern=r'^sha256:[0-9a-f]{64}$')
    plugin_refs: list[str] = Field(default_factory=list)
    profile: dict = Field(default_factory=dict)

    @model_validator(mode='after')
    def legacy_spec(self):
        from inferstack.engines.managed import DeploymentSpec
        DeploymentSpec.from_record(self.model_dump())
        return self


# --- workload, objective --------------------------------------------------------

class Distribution(BaseModel):
    model_config = ConfigDict(extra='forbid')
    p50: int = Field(ge=0)
    p95: int = Field(ge=0)
    max: int = Field(gt=0)


class WorkloadSpec(Record):
    KIND = 'workload'
    name: str = Field(min_length=1)
    dataset_sha256: str | None = Field(default=None, pattern=SHA256)
    cases: list[dict] = Field(default_factory=list, max_length=10000)
    prompt_tokens: Distribution | None = None
    output_tokens: Distribution | None = None
    arrival: Literal['closed_loop', 'poisson', 'trace'] = 'closed_loop'
    rate_rps: float | None = Field(default=None, gt=0)
    concurrency: int = Field(default=1, ge=1, le=100000)
    cache_condition: Literal['cold', 'warm', 'mixed'] = 'cold'
    seed: int = 0

    @model_validator(mode='after')
    def has_prompts(self):
        if not self.cases and self.dataset_sha256 is None:
            raise ValueError('workload requires inline cases or dataset_sha256')
        if self.arrival == 'poisson' and self.rate_rps is None:
            raise ValueError('poisson arrival requires rate_rps')
        return self


class SLO(BaseModel):
    model_config = ConfigDict(extra='forbid')
    metric: str
    direction: Literal['min', 'max']
    limit: float | None = None
    percentile: float | None = Field(default=None, gt=0, lt=100)
    epsilon: float = Field(default=0.02, ge=0, lt=1)

    @field_validator('metric')
    @classmethod
    def registered(cls, value):
        require_metric(value)
        return value


class Objective(Record):
    KIND = 'objective'
    name: str = Field(min_length=1)
    slos: list[SLO] = Field(min_length=1)
    quality_budget: dict | None = None
    cost_limit_usd: float | None = Field(default=None, gt=0)

    @field_validator('slos')
    @classmethod
    def unique(cls, slos):
        keys = [(s.metric, s.percentile) for s in slos]
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate SLO metric/percentile')
        return slos


# --- evidence, technique, plan, trial -------------------------------------------

class Evidence(Record):
    KIND = 'evidence'
    source_type: Literal['text', 'url', 'telemetry', 'experiment', 'operator']
    uri: str | None = None
    retrieved_at: float | None = Field(default=None, gt=0)
    content_sha256: str = Field(pattern=SHA256)
    artifact_sha256: str | None = Field(default=None, pattern=SHA256)
    provenance: Literal['user_supplied', 'retrieved', 'internal']
    verification: Literal['unverified', 'verified', 'failed', 'unavailable'] = 'unverified'
    title: str | None = None
    license: str | None = None
    code_revision: str | None = None

    @model_validator(mode='after')
    def provenance_consistency(self):
        if self.source_type == 'url' and not self.uri:
            raise ValueError('url evidence requires uri')
        if self.provenance == 'retrieved' and self.retrieved_at is None:
            raise ValueError('retrieved evidence requires retrieved_at')
        if self.code_revision in MUTABLE_REVISIONS:
            raise ValueError('code_revision must be immutable')
        return self


class EvidenceV2(Evidence):
    """v2: source availability outcome and canonical source identity.

    A blocked or unavailable source is still recorded (content is empty and its
    hash is the empty-content hash) so a job keeps a truthful source trail.
    """
    VERSION = 2
    source_key: str = Field(min_length=1, max_length=2100)
    availability: Literal['available', 'blocked', 'unavailable', 'unsupported', 'too_large', 'invalid'] = 'available'
    availability_reason: str | None = Field(default=None, max_length=1000)
    content_bytes: int = Field(ge=0)
    content_type: str | None = None

    @model_validator(mode='after')
    def availability_consistency(self):
        empty = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
        if self.availability == 'available' and (self.content_bytes == 0 or self.content_sha256 == empty):
            raise ValueError('available evidence must carry content')
        if self.availability != 'available' and (self.content_sha256 != empty or self.verification != 'unavailable'):
            raise ValueError('unavailable evidence must have empty content and verification=unavailable')
        return self


class Dependency(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    version: str
    revision: str | None = None
    license: str | None = None


class EngineRequirement(BaseModel):
    model_config = ConfigDict(extra='forbid')
    engine: Literal['inferstack_native', 'vllm', 'sglang', 'generic_openai']
    versions: str = '*'


class Technique(Record):
    KIND = 'technique'
    REFS = {'evidence_refs': 'evidence', 'proven_by': 'trial'}
    name: str = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=20000)
    evidence_refs: list[str] = Field(default_factory=list)
    engines: list[EngineRequirement] = Field(min_length=1)
    operations: list[str] = Field(default_factory=list)
    dtypes: list[str] = Field(default_factory=list)
    hardware: list[str] = Field(default_factory=list)
    dependencies: list[Dependency] = Field(default_factory=list)
    expected_benefit: str | None = None
    risks: list[str] = Field(default_factory=list)
    compatibility_claims: list[str] = Field(default_factory=list)
    proven_by: list[str] = Field(default_factory=list)


class TechniqueV2(Technique):
    """v2: shapes, runtime pins and lifecycle status. Claims are never 'proven'
    without trial references; `proven` requires proven_by."""
    VERSION = 2
    shapes: list[str] = Field(default_factory=list)
    min_compute_capability: str | None = Field(default=None, pattern=r'^[0-9]+\.[0-9]+$')
    min_gpu_memory_bytes: int | None = Field(default=None, gt=0)
    requires_capabilities: list[str] = Field(default_factory=list)
    status: Literal['claim', 'candidate', 'proven', 'rejected'] = 'claim'

    @model_validator(mode='after')
    def proof_required(self):
        if self.status == 'proven' and not self.proven_by:
            raise ValueError('proven techniques must reference the trials that proved them')
        return self


class Plan(Record):
    KIND = 'plan'
    REFS = {'baseline_ref': 'deployment', 'candidate_ref': 'deployment', 'evidence_refs': 'evidence',
            'objective_ref': 'objective', 'workload_ref': 'workload'}
    PINNED = {'baseline_ref', 'candidate_ref', 'evidence_refs', 'objective_ref', 'workload_ref'}
    job_id: str
    iteration: int = Field(ge=0)
    hypothesis: str = Field(min_length=1, max_length=10000)
    baseline_ref: str
    candidate_ref: str
    evidence_refs: list[str] = Field(default_factory=list)
    objective_ref: str | None = None
    workload_ref: str | None = None
    gates: list[str] = Field(default_factory=list)
    stop_rule: str | None = None


class Trial(Record):
    KIND = 'trial'
    REFS = {'deployment_ref': 'deployment', 'workload_ref': 'workload', 'hardware_ref': 'hardware',
            'plan_ref': 'plan'}
    PINNED = {'deployment_ref', 'workload_ref', 'hardware_ref', 'plan_ref'}
    experiment_id: str
    plan_ref: str | None = None
    deployment_ref: str
    workload_ref: str | None = None
    hardware_ref: str | None = None
    role: Literal['baseline', 'candidate']
    repeat_index: int = Field(ge=0)
    metrics: dict[str, float | None]
    raw_artifact_sha256: str = Field(pattern=SHA256)
    errors: list[str] = Field(default_factory=list)
    started_at: float = Field(gt=0)
    ended_at: float = Field(gt=0)

    @model_validator(mode='after')
    def check(self):
        for name in self.metrics:
            require_metric(name)
        if self.ended_at < self.started_at:
            raise ValueError('ended_at precedes started_at')
        return self


# --- release, route, rollout ----------------------------------------------------

class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sha256: str = Field(pattern=SHA256)
    bytes: int = Field(ge=0)


class Release(Record):
    """v1 is the manifest ReleaseManager produces today (frozen deployment snapshot)."""
    KIND = 'release'
    deployment: dict
    evidence: ArtifactRef
    job_id: str
    verification_scope: str

    def references(self):
        snapshot = self.deployment
        if not isinstance(snapshot.get('id'), str) or not isinstance(snapshot.get('revision'), int):
            raise ValueError('release must embed a deployment snapshot with id and revision')
        return [('deployment', snapshot['id'], snapshot['revision'])]


class Route(Record):
    KIND = 'route'
    REFS = {'stable': 'release', 'candidate': 'release', 'previous': 'release'}
    stable: str
    candidate: str | None = None
    weight: int = Field(ge=0, le=100)
    previous: str | None = None


class RolloutStage(BaseModel):
    model_config = ConfigDict(extra='forbid')
    weight: int = Field(ge=1, le=100)
    min_duration_s: float = Field(ge=0)
    max_error_rate: float | None = Field(default=None, ge=0, le=1)
    max_latency_regression: float | None = Field(default=None, ge=0)


class Rollout(Record):
    KIND = 'rollout'
    REFS = {'release_ref': 'release', 'previous_release_ref': 'release'}
    model: str
    release_ref: str
    previous_release_ref: str | None = None
    stages: list[RolloutStage] = Field(min_length=1)
    state: Literal['pending', 'progressing', 'paused', 'promoted', 'rolled_back', 'failed'] = 'pending'
    current_stage: int = Field(default=0, ge=0)
    updated_at: float = Field(gt=0)

    @model_validator(mode='after')
    def monotonic(self):
        weights = [s.weight for s in self.stages]
        if weights != sorted(weights) or weights[-1] != 100:
            raise ValueError('rollout stage weights must be non-decreasing and end at 100')
        if self.current_stage >= len(self.stages):
            raise ValueError('current_stage out of range')
        return self


class RolloutV2(Rollout):
    """v2: controller state for progressive delivery (each transition is a revision)."""
    VERSION = 2
    REFS = {**Rollout.REFS, 'stable_ref': 'release'}
    stable_ref: str | None = None
    stage_started_at: float | None = Field(default=None, gt=0)
    min_requests: int = Field(default=20, ge=1)
    reason: str | None = Field(default=None, max_length=2000)
    observations: dict = Field(default_factory=dict)


# --- allocation, feedback -------------------------------------------------------

class Budget(BaseModel):
    model_config = ConfigDict(extra='forbid')
    wall_seconds: float | None = Field(default=None, gt=0)
    gpu_seconds: float | None = Field(default=None, gt=0)
    provider_tokens: int | None = Field(default=None, gt=0)
    spend_usd: float | None = Field(default=None, gt=0)


class ResourceAllocation(Record):
    KIND = 'allocation'
    REFS = {'hardware_ref': 'hardware'}
    worker_id: str
    job_id: str | None = None
    provider: str
    resource_type: Literal['process', 'container', 'gpu', 'cloud_instance', 'k8s_pod', 'k8s_deployment']
    external_id: str | None = None
    hardware_ref: str | None = None
    gpu_uuids: list[str] = Field(default_factory=list)
    budget: Budget = Field(default_factory=Budget)
    state: Literal['intent', 'active', 'releasing', 'released', 'orphaned']
    created_at: float = Field(gt=0)


class InferenceIssue(Record):
    """FactoryOS / operator / telemetry feedback. Contract version is the schema version."""
    KIND = 'feedback'
    REFS = {'deployment_ref': 'deployment', 'objective_ref': 'objective'}
    issue_id: str = Field(min_length=1, max_length=200)
    idempotency_key: str = Field(min_length=1, max_length=200)
    source: Literal['factoryos', 'operator', 'telemetry']
    trace_ids: list[str] = Field(default_factory=list, max_length=1000)
    deployment_ref: str | None = None
    category: Literal['latency', 'throughput', 'error', 'quality', 'capacity', 'cost', 'unknown'] = 'unknown'
    observations: dict = Field(default_factory=dict)
    objective_ref: str | None = None
    reported_at: float = Field(gt=0)


SCHEMAS: dict[str, dict[int, type[Record]]] = {}
for _cls in (ModelArtifact, HardwareEnvironment, Deployment, WorkloadSpec, Objective, Evidence, EvidenceV2,
             Technique, TechniqueV2,
             Plan, Trial, Release, Route, Rollout, RolloutV2, ResourceAllocation, InferenceIssue):
    SCHEMAS.setdefault(_cls.KIND, {})[_cls.VERSION] = _cls
CURRENT_VERSIONS = {kind: max(versions) for kind, versions in SCHEMAS.items()}


def schema_for(kind: str, version: int | None = None) -> type[Record]:
    if kind not in SCHEMAS:
        raise ValueError(f'unknown record kind: {kind}')
    version = CURRENT_VERSIONS[kind] if version is None else version
    if isinstance(version, bool) or not isinstance(version, int) or version not in SCHEMAS[kind]:
        raise IncompatibleSchema(f'{kind} schema version {version} is not supported by this build '
                                 f'(supported: {sorted(SCHEMAS[kind])})')
    return SCHEMAS[kind][version]


def validate(kind: str, body: dict, version: int | None = None) -> Record:
    return schema_for(kind, version).model_validate(body)
