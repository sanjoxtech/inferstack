"""Narrow, read-only tools for the planner.

Every tool validates its arguments with a strict schema, runs with a bounded
result size, and never exposes endpoints, launch commands or credentials. There
are no write tools: the planner cannot change deployments, routes or releases,
and cannot record a verdict. Text derived from external sources is returned
inside the untrusted-evidence envelope.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

CODE_ROOTS = ('src/inferstack/kernels', 'src/inferstack/runtime', 'src/inferstack/engines',
              'src/inferstack/serving', 'configs/profiles')
MAX_RESULT_CHARS = 12000


class ToolError(ValueError):
    pass


class _Args(BaseModel):
    model_config = ConfigDict(extra='forbid')


class NoArgs(_Args):
    pass


class DeploymentArgs(_Args):
    deployment_id: str = Field(min_length=1, max_length=200)


class TelemetryArgs(_Args):
    deployment_id: str = Field(min_length=1, max_length=200)
    metrics: list[str] = Field(default_factory=list, max_length=20)


class CodeArgs(_Args):
    path: str = Field(min_length=1, max_length=300)
    start_line: int = Field(default=1, ge=1)
    max_lines: int = Field(default=120, ge=1, le=200)


class EvidenceSearchArgs(_Args):
    query: str = Field(min_length=2, max_length=200)


class ExperienceArgs(_Args):
    treatment_signature: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    limit: int = Field(default=10, ge=1, le=50)


class TechniqueArgs(_Args):
    technique_id: str = Field(min_length=1, max_length=200)


class PlannerTools:
    def __init__(self, repository, *, deployment_ids, allowed_evidence, history, code_root=None, telemetry=None):
        self.repository = repository
        self.deployment_ids = set(deployment_ids)
        self.allowed_evidence = list(allowed_evidence)
        self.history = history
        self.code_root = Path(code_root or Path(__file__).resolve().parents[3]).resolve()
        self.telemetry = telemetry  # async (deployment_id) -> normalized snapshot, or None
        self.specs = {
            'get_deployment': (DeploymentArgs, self.get_deployment,
                               'Engine, model, versions and profile of a baseline or candidate deployment.'),
            'list_experiments': (NoArgs, self.list_experiments,
                                 'Previous plans and verifier outcomes in this job, including failures.'),
            'query_telemetry': (TelemetryArgs, self.query_telemetry,
                                'Latest normalized engine metrics for a deployment (null when unavailable).'),
            'read_code': (CodeArgs, self.read_code, f'Read source lines under {", ".join(CODE_ROOTS)}.'),
            'search_evidence': (EvidenceSearchArgs, self.search_evidence,
                                'Search evidence available to this job; returns ids and untrusted snippets.'),
            'list_techniques': (NoArgs, self.list_techniques,
                                'Registered techniques that are not incompatible with the baseline.'),
            'check_compatibility': (TechniqueArgs, self.check_compatibility,
                                    'Claim-vs-proof compatibility of one technique with the baseline.'),
            'get_measurements': (DeploymentArgs, self.get_measurements,
                                 'Latest measured bottleneck of a deployment: goodput knee per case (which SLO '
                                 'broke first, TTFT split and lane counters there) and kernel-family GPU time '
                                 'shares from a decode-step profile; stale when made on another revision.'),
            'search_experience': (ExperienceArgs, self.search_experience,
                                  'Outcomes of earlier campaigns for this model: applicable (same setup: real '
                                  'evidence) vs transferable (different setup: hypothesis only), with evidence '
                                  'digests. Optionally filter by treatment_signature.'),
        }
        self.baseline_id = next(iter(deployment_ids))

    def openai_tools(self):
        return [{'type': 'function', 'function': {'name': name, 'description': desc,
                                                  'parameters': schema.model_json_schema()}}
                for name, (schema, _, desc) in self.specs.items()]

    async def call(self, name, arguments: str):
        if name not in self.specs:
            raise ToolError(f'unknown tool {name!r}; available: {sorted(self.specs)}')
        schema, fn, _ = self.specs[name]
        try:
            args = schema.model_validate(json.loads(arguments or '{}'))
        except (ValueError, ValidationError) as exc:
            raise ToolError(f'invalid arguments for {name}: {str(exc)[:300]}') from None
        result = await fn(**args.model_dump())
        text = json.dumps(result, sort_keys=True, default=str)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + '...[truncated]'
        return text

    def _deployment(self, deployment_id):
        if deployment_id not in self.deployment_ids:
            raise ToolError(f'deployment {deployment_id!r} is not part of this job')
        return self.repository.read('deployment', deployment_id)

    async def get_deployment(self, deployment_id):
        record = self._deployment(deployment_id)
        b = record['body']
        return {'id': record['id'], 'revision': record['revision'], 'engine': b['engine'], 'model': b['model'],
                'engine_version': b.get('engine_version'), 'model_ref': b.get('model_ref'),
                'hardware_ref': b.get('hardware_ref'), 'plugin_refs': b.get('plugin_refs', []),
                'profile': b.get('profile', {}), 'role': 'baseline' if record['id'] == self.baseline_id else 'candidate'}

    async def list_experiments(self):
        return [{'iteration': i, 'hypothesis': e['plan'].get('hypothesis'),
                 'treatment_signature': e['plan'].get('treatment_signature'),
                 'candidate': e['plan'].get('candidate_id'), 'decision': e['verdict'].get('decision'),
                 'reasons': e['verdict'].get('reasons'),
                 # Controller diagnosis of this trial (H3): finding, bottleneck, remaining gap, guidance.
                 'diagnosis': {k: (e.get('diagnosis') or {}).get(k) for k in
                               ('finding', 'bottleneck', 'confidence', 'remaining_gap', 'next')}
                 if e.get('diagnosis') else None} for i, e in enumerate(self.history)]

    async def query_telemetry(self, deployment_id, metrics):
        self._deployment(deployment_id)
        if self.telemetry is None:
            return {'available': False, 'reason': 'telemetry not configured for this planner'}
        snapshot = await self.telemetry(deployment_id)
        selected = snapshot.get('metrics', {})
        if metrics:
            selected = {k: v for k, v in selected.items() if k in metrics}
        return {'available': snapshot.get('available', False), 'mapping_status': snapshot.get('mapping_status'),
                'metrics': selected, 'reason': snapshot.get('reason')}

    async def read_code(self, path, start_line, max_lines):
        target = (self.code_root / path).resolve()
        allowed = [(self.code_root / root).resolve() for root in CODE_ROOTS]
        if not any(target.is_relative_to(root) for root in allowed) or not target.is_file():
            raise ToolError(f'path {path!r} is not a readable file under {CODE_ROOTS}')
        lines = target.read_text(errors='replace').splitlines()
        chunk = lines[start_line - 1:start_line - 1 + max_lines]
        return {'path': path, 'start_line': start_line, 'total_lines': len(lines),
                'text': '\n'.join(f'{start_line + i}: {line}' for i, line in enumerate(chunk))}

    async def search_evidence(self, query):
        from inferstack.research.evidence import EvidenceStore, untrusted_block
        store, hits = EvidenceStore(self.repository), []
        for identifier in self.allowed_evidence:
            try:
                snapshot = store.read(identifier)
            except (KeyError, ValueError):
                continue
            content = snapshot.get('content') or ''
            at = content.lower().find(query.lower())
            title = snapshot['body'].get('title') or ''
            if at >= 0 or query.lower() in title.lower():
                excerpt = content[max(0, at - 300): at + 700] if at >= 0 else content[:600]
                hits.append({'evidence_id': identifier,
                             'snippet': untrusted_block({**snapshot, 'content': excerpt}, max_chars=1200)})
        return {'hits': hits[:10]}

    async def get_measurements(self, deployment_id):
        from inferstack.evaluation.bottleneck import latest_measurements
        self._deployment(deployment_id)
        found = latest_measurements(self.repository, deployment_id)
        return {kind: {k: m.get(k) for k in ('summary', 'stale', 'measured_at', 'raw_sha256')}
                for kind, m in found.items()} or {'available': False, 'reason': 'no measurement recorded'}

    async def search_experience(self, treatment_signature=None, limit=10):
        from inferstack.memory.experience import experience
        found = experience(self.repository, self.baseline_id, limit=limit, treatment_signature=treatment_signature)
        return {k: found[k] for k in ('summary', 'failed_here_signatures', 'items', 'truncated')} | \
            {'excluded': {'unrelated': found['excluded']['unrelated'],
                          'inaccessible': len(found['excluded']['inaccessible'])}}

    async def list_techniques(self):
        from inferstack.research.techniques import candidates
        return candidates(self.repository, self.baseline_id)

    async def check_compatibility(self, technique_id):
        from inferstack.research.techniques import assess
        try:
            return assess(self.repository, technique_id, self.baseline_id)
        except KeyError:
            raise ToolError(f'unknown technique {technique_id!r}') from None
