"""ResultBundle v1 (E1.2): the verifiable package an execution target returns.

A bundle is one zip file:

    bundle.json          index: manifest + digests, observed hardware, preflight,
                         launch identity, steps, gates, terminal status, file checksums
    <run files>          raw benchmark records (status.json, requests.jsonl, server.log,
                         scorecard.json, ...) exactly as the runner wrote them

`open_bundle` refuses anything it cannot fully account for: unsafe member names,
files not listed in the index (or listed but absent), checksum mismatches, a
manifest whose recomputed digest/experiment id differs from the index, a manifest
other than the one the caller expected, and credential-looking content.

`import_bundle` stores the zip as an immutable artifact and records what actually
ran: the observed HardwareEnvironment, one Evidence record for the bundle, and, for
a completed run bound to a deployment record, one Trial per (round, case,
concurrency) cell. Gates that were not run stay `missing`; a failed or missing gate
is written into each trial's errors so it can never count as proof
(`research.techniques.proof_of` rejects trials with errors).
"""
from __future__ import annotations

import hashlib
import io
import json
import posixpath
import time
import zipfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from inferstack.contracts.records import HardwareEnvironment
from inferstack.contracts.redaction import reject_secrets, scrub_text

from .manifest import ExecutionManifestV1, canonical_json

SCHEMA = 'inferstack.result_bundle/v1'
INDEX = 'bundle.json'
MAX_BUNDLE_BYTES = 512 << 20
MAX_MEMBERS = 5000
TEXT_SUFFIXES = ('.json', '.jsonl', '.log', '.txt', '.md', '.csv')
GATE_STATES = ('passed', 'failed', 'missing')
# scorecard per-round metric -> registered trial metric (unit conversion where needed)
TRIAL_METRICS = {'ttft_p50_ms': ('ttft_client_ms', 1.0), 'ttft_p95_ms': ('ttft_client_p95_ms', 1.0),
                 'e2e_p50_ms': ('e2e_ms', 1.0), 'output_tps': ('output_tokens_per_s', 1.0),
                 'peak_memory_mb': ('gpu_memory_sampled_peak_bytes', float(1 << 20))}


class BundleRejected(ValueError):
    pass


class Step(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    ok: bool
    started_at: float
    ended_at: float
    detail: str | None = Field(default=None, max_length=4000)


class BundleIndex(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_: Literal['inferstack.result_bundle/v1'] = Field(default=SCHEMA, alias='schema')
    experiment_id: str = Field(pattern=r'^[0-9a-f]{64}$')
    manifest_digest: str = Field(pattern=r'^[0-9a-f]{64}$')
    manifest: dict
    environment: dict
    preflight: dict
    status: Literal['completed', 'refused', 'probe_failed', 'failed', 'partial']
    failure_reason: str | None = Field(default=None, max_length=4000)
    launch: dict = Field(default_factory=dict)
    steps: list[Step] = Field(default_factory=list)
    gates: dict[str, Literal['passed', 'failed', 'missing']] = Field(default_factory=dict)
    runner: dict = Field(default_factory=dict)
    run_dir: str | None = None
    started_at: float
    ended_at: float
    files: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra='forbid', populate_by_name=True)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe(name: str) -> bool:
    return (bool(name) and not name.startswith('/') and '\\' not in name and ':' not in name.split('/')[0]
            and posixpath.normpath(name) == name and not name.startswith('..') and '/../' not in f'/{name}/')


def write_bundle(root: Path, index: dict, out: Path) -> Path:
    """Zip every file under `root` with an index whose `files` lists their sha256."""
    root, out = Path(root), Path(out)
    members = {}
    for path in sorted(p for p in root.rglob('*') if p.is_file()):
        name = path.relative_to(root).as_posix()
        if name == INDEX:
            continue
        members[name] = path.read_bytes()
    body = BundleIndex.model_validate({**index, 'files': {n: _sha(d) for n, d in members.items()}})
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'x', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(INDEX, json.dumps(body.model_dump(mode='json', by_alias=True), indent=2, sort_keys=True))
        for name, data in members.items():
            zf.writestr(name, data)
    return out


def open_bundle(source, *, expected: ExecutionManifestV1 | None = None):
    """Verify a bundle; returns (index: BundleIndex, manifest, members: {name: bytes}, raw zip bytes)."""
    raw = Path(source).read_bytes() if not isinstance(source, (bytes, bytearray)) else bytes(source)
    if len(raw) > MAX_BUNDLE_BYTES:
        raise BundleRejected('bundle exceeds the size limit')
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise BundleRejected(f'not a zip file: {exc}') from exc
    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_MEMBERS:
            raise BundleRejected('too many bundle members')
        names = [i.filename for i in infos if not i.is_dir()]
        if len(set(names)) != len(names):
            raise BundleRejected('duplicate bundle member names')
        unsafe = [n for n in names if not _safe(n)]
        if unsafe:
            raise BundleRejected(f'unsafe member names: {unsafe[:5]}')
        if sum(i.file_size for i in infos) > 4 * MAX_BUNDLE_BYTES:
            raise BundleRejected('bundle expands beyond the size limit')
        if INDEX not in names:
            raise BundleRejected(f'{INDEX} missing')
        members = {n: zf.read(n) for n in names}
    try:
        index = BundleIndex.model_validate(json.loads(members.pop(INDEX)))
    except (ValueError, TypeError) as exc:
        raise BundleRejected(f'invalid {INDEX}: {exc}') from exc
    listed, present = set(index.files), set(members)
    if listed != present:
        raise BundleRejected(f'unlisted members {sorted(present - listed)[:5]}; missing members '
                             f'{sorted(listed - present)[:5]}')
    bad = [n for n, d in members.items() if _sha(d) != index.files[n]]
    if bad:
        raise BundleRejected(f'checksum mismatch: {bad[:5]}')
    try:
        manifest = ExecutionManifestV1.model_validate(index.manifest)
    except ValueError as exc:
        raise BundleRejected(f'bundle manifest is invalid: {exc}') from exc
    if manifest.digest != index.manifest_digest or manifest.experiment_id != index.experiment_id:
        raise BundleRejected('bundle manifest does not match its declared digest/experiment id')
    if expected is not None and (expected.digest != manifest.digest):
        raise BundleRejected(f'manifest mismatch: expected experiment {expected.experiment_id[:12]} digest '
                             f'{expected.digest[:12]}, bundle has {manifest.experiment_id[:12]} '
                             f'{manifest.digest[:12]}')
    try:
        reject_secrets(index.model_dump(mode='json', by_alias=True))
    except ValueError as exc:
        raise BundleRejected(str(exc)) from exc
    for name, data in members.items():
        if name.endswith(TEXT_SUFFIXES):
            text = data.decode('utf-8', errors='replace')
            if scrub_text(text) != text:
                raise BundleRejected(f'{name} contains credential-looking text; scrub it at the source')
    try:
        HardwareEnvironment.model_validate(index.environment)
    except ValueError as exc:
        raise BundleRejected(f'observed environment is invalid: {exc}') from exc
    return index, manifest, members, raw


def _gate_states(index: BundleIndex, manifest: ExecutionManifestV1) -> dict:
    states = {g: 'missing' for g in manifest.measurement.gates}
    states.update(index.gates)
    return states


def _put_once(repository, kind, identifier, body):
    try:
        existing = repository.read(kind, identifier)
    except KeyError:
        repository.put_record(kind, identifier, body)
        return f'{identifier}@1'
    return f"{identifier}@{existing['revision']}"


def _cells(members, run_dir):
    """Per-(round, case, concurrency) metrics from the raw request records, via the scorecard rules."""
    from inferstack.execution.runner import scorecard_module
    scorecard = scorecard_module()
    prefix = f'{run_dir}/' if run_dir else ''
    out = []
    rounds = sorted({n[len(prefix):].split('/')[0] for n in members
                     if n.startswith(prefix + 'round-') and n.endswith('/status.json')})
    for round_name in rounds:
        for status_name in sorted(n for n in members if n.startswith(f'{prefix}{round_name}/')
                                  and n.endswith('/status.json')):
            status = json.loads(members[status_name])
            if not status.get('ready'):
                continue
            requests_name = status_name[:-len('status.json')] + 'requests.jsonl'
            rows = [json.loads(line) for line in members.get(requests_name, b'').decode().splitlines() if line]
            by_cell = {}
            for row in rows:
                by_cell.setdefault((row['case'], row['concurrency']), []).append(row)
            cells = {(c['case'], c['concurrency']): c for c in status.get('cells') or []}
            for key, cell_rows in sorted(by_cell.items()):
                metrics = scorecard._round_metrics(cell_rows, cells.get(key))
                out.append({'round': int(round_name.split('-')[1]), 'case': key[0], 'concurrency': key[1],
                            'metrics': metrics, 'started_at': status.get('started_at'),
                            'ended_at': status.get('stopped_at')})
    return out


def import_bundle(repository, source, *, expected: ExecutionManifestV1 | None = None) -> dict:
    """Verify and record a bundle. Idempotent: re-importing the same zip changes nothing."""
    index, manifest, members, raw = open_bundle(source, expected=expected)
    stored = repository.artifact(raw)
    bundle_sha = stored['sha256']
    environment = HardwareEnvironment.model_validate(index.environment)
    hardware_body = environment.model_dump(mode='json')
    hardware_id = 'hw-' + hashlib.sha256(canonical_json(hardware_body).encode()).hexdigest()[:24]
    hardware_ref = _put_once(repository, 'hardware', hardware_id, hardware_body)
    evidence_id = f'bundle-{bundle_sha[:32]}'
    evidence_ref = _put_once(repository, 'evidence', evidence_id, {
        'source_type': 'experiment', 'provenance': 'internal', 'content_sha256': bundle_sha,
        'artifact_sha256': bundle_sha, 'verification': 'verified', 'source_key': f'result-bundle:{bundle_sha}',
        'content_bytes': len(raw), 'content_type': 'application/zip', 'retrieved_at': time.time(),
        'title': f'{manifest.engine.name} {manifest.model.repo_id} {manifest.dtype} '
                 f'[{index.status}] experiment {index.experiment_id[:12]}'})
    gates = _gate_states(index, manifest)
    gate_errors = [f'gate {g}: {s}' for g, s in sorted(gates.items()) if s != 'passed']
    trials = []
    skipped = None
    if index.status != 'completed':
        skipped = f'bundle status {index.status}: no trials recorded'
    elif not manifest.refs.deployment:
        skipped = 'manifest is not bound to a deployment record: no trials recorded'
    else:
        role = 'candidate' if manifest.engine.candidate_id else 'baseline'
        for cell in _cells(members, index.run_dir):
            m = cell['metrics']
            metrics = {name: (m[key] * scale if m.get(key) is not None else None)
                       for key, (name, scale) in TRIAL_METRICS.items()}
            metrics['failure_rate'] = None if m.get('success_rate') is None else 1.0 - m['success_rate']
            errors = list(gate_errors)
            if m.get('success_rate') is not None and m['success_rate'] < 1.0:
                errors.append(f"{m['requests'] - round(m['success_rate'] * m['requests'])} request(s) failed")
            trial_id = (f"x-{index.experiment_id[:16]}-{bundle_sha[:8]}-r{cell['round']}-{cell['case']}"
                        f"-c{cell['concurrency']}")
            started = cell['started_at'] or index.started_at
            trials.append(_put_once(repository, 'trial', trial_id, {
                'experiment_id': f"{index.experiment_id}:{cell['case']}:c{cell['concurrency']}",
                'deployment_ref': manifest.refs.deployment, 'hardware_ref': hardware_ref,
                'workload_ref': manifest.refs.workload, 'role': role, 'repeat_index': cell['round'],
                'metrics': metrics, 'raw_artifact_sha256': bundle_sha, 'errors': errors,
                'started_at': started, 'ended_at': max(cell['ended_at'] or index.ended_at, started)}))
    return {'experiment_id': index.experiment_id, 'manifest_digest': index.manifest_digest,
            'status': index.status, 'failure_reason': index.failure_reason, 'bundle_sha256': bundle_sha,
            'hardware_ref': hardware_ref, 'evidence_ref': evidence_ref, 'trials': trials,
            'trials_skipped': skipped, 'gates': gates, 'preflight_status': index.preflight.get('status')}
