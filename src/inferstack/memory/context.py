"""Versioned deployment/workload context snapshot for planning (harness H1).

Answers, from existing records only (bounded, read-only):
- what exactly is deployed: the deployment revision, its applicability identity
  (model/tokenizer/config, engine/build/plugins, precision, hardware/runtime) and
  engine capability claims with their basis;
- how it is served: routes and releases that reference this deployment;
- what the workload looks like: the supplied workload fixture and objectives;
- what live evidence exists: a recent serving-evidence window when gateways record it;
- which techniques are claimed / proven / incompatible here;
- what earlier campaigns found (memory/experience.py);
- what is missing, as an explicit list rather than silent defaults.

The snapshot is canonical JSON stored as a content-addressed artifact: the same state
produces the same digest, so a plan can cite exactly the context it was made from.
"""
from __future__ import annotations

import json
import time

CONTEXT_VERSION = 1


def _routes_for(repository, deployment_id):
    out = []
    for route in repository.list_records('route', limit=1000):
        body = route['body']
        for role in ('stable', 'candidate', 'previous'):
            release_id = body.get(role)
            if not release_id:
                continue
            try:
                release = repository.read('release', release_id)['body']
            except KeyError:
                continue
            if (release.get('deployment') or {}).get('id') == deployment_id:
                out.append({'model': route['id'], 'route_revision': route['revision'], 'role': role,
                            'release_id': release_id, 'weight': body.get('weight') if role == 'candidate' else None,
                            'release_identity': release.get('identity')})
    return out


def _live_window(repository, deployment_id, seconds):
    try:
        from inferstack.serving.evidence import EvidenceStore
        now = time.time()
        window = EvidenceStore(repository).window(deployment_id=deployment_id, since=now - seconds, until=now)
    except Exception:  # noqa: BLE001 - a store without serving evidence simply has none
        return None
    if not window.get('requests') and not window.get('rejected'):
        return None
    keep = ('requests', 'errors', 'timeouts', 'cancelled', 'rejected', 'samples', 'ttft_p50_ms', 'ttft_p95_ms',
            'e2e_p50_ms', 'e2e_p95_ms', 'stall_p95_ms', 'streams_without_content')
    return {k: window.get(k) for k in keep} | {'window_seconds': seconds}


def context_snapshot(repository, deployment_id, *, workload=None, objectives=None, live_window_s=3600,
                     experience_limit=10, store=True, exclude_jobs=()):
    """Build (and by default store) the context snapshot. Returns {'sha256', 'snapshot'}."""
    from inferstack.engines.capabilities import _DEFAULTS
    from inferstack.memory.experience import experience
    from inferstack.research.techniques import applicability, candidates
    record = repository.read('deployment', deployment_id)
    body = record['body']
    identity = applicability(repository, body)
    missing = []
    if not body.get('model_ref'):
        missing.append('model_ref: no pinned model record (weights/tokenizer revisions unknown)')
    if not body.get('hardware_ref'):
        missing.append('hardware_ref: no hardware record (GPU, driver, CUDA unknown)')
    if not body.get('engine_version'):
        missing.append('engine_version unknown')
    if not (body.get('profile') or {}).get('dtype'):
        missing.append('dtype unknown')
    if workload is None:
        missing.append('workload: no fixture supplied (cases, lengths, concurrency unknown)')
    capabilities = {op: {'state': state.value, 'basis': basis}
                    for op, (state, basis, _) in (_DEFAULTS.get(body['engine']) or {}).items()}
    techniques = []
    try:
        for result in candidates(repository, deployment_id):
            techniques.append({'technique_id': result['technique_id'], 'verdict': result['verdict'],
                               'proven_by': result['proven_by'], 'unknowns': result['unknowns'][:3]})
    except Exception as exc:  # noqa: BLE001 - recorded as missing context, not fatal
        missing.append(f'techniques unavailable: {type(exc).__name__}')
    workload_digest = None
    if workload is not None:
        import hashlib
        workload_digest = hashlib.sha256(repository.encode(workload).encode()).hexdigest()
    live = _live_window(repository, deployment_id, live_window_s)
    if live is None:
        missing.append('live serving evidence: none recorded for this deployment in the window')
    prior = experience(repository, deployment_id, limit=experience_limit, exclude_jobs=exclude_jobs)
    snapshot = {
        'version': CONTEXT_VERSION, 'deployment_ref': f"{deployment_id}@{record['revision']}",
        'deployment': {k: body.get(k) for k in ('engine', 'model', 'engine_version', 'image_digest', 'plugin_refs',
                                               'model_ref', 'hardware_ref', 'resources')}
                      | {'profile_keys': sorted((body.get('profile') or {}).keys())},
        'identity': identity, 'capabilities': capabilities, 'routes': _routes_for(repository, deployment_id),
        'workload': {'fixture_sha256': workload_digest, 'cases': len(workload) if isinstance(workload, list) else None,
                     'objectives': objectives},
        'live_evidence': live, 'techniques': techniques,
        'experience': {'summary': prior['summary'], 'failed_here_signatures': prior['failed_here_signatures'],
                       'items': prior['items'], 'excluded': prior['excluded'], 'truncated': prior['truncated']},
        'missing': missing,
    }
    encoded = json.dumps(snapshot, sort_keys=True, separators=(',', ':'), default=str, allow_nan=False)
    sha = repository.artifact(encoded.encode())['sha256'] if store else None
    return {'sha256': sha, 'snapshot': snapshot}
