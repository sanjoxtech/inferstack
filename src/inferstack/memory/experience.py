"""Bounded cross-campaign retrieval of prior experiment outcomes with applicability checks.

Source of truth is the existing durable store: terminal jobs carry their experiments
(plan, candidate, verdict with an evidence artifact digest). Nothing is summarised
into a separate memory database, so every retrieved item keeps its exact references:
job ID, iteration, treatment signature, candidate deployment revision and the
digest-verified evidence artifact.

Each prior experiment is classified against the target deployment:
- `applicable`: same applicability identity (research/techniques.applicability) as the
  experiment's baseline, so its outcome is evidence *for this setup*: a `keep` is a
  matching success, a `reject` a known failure of that treatment here;
- `transferable`: the same model family (served model or model repo) on a different
  setup; usable only as a hypothesis, with the differing identity components listed;
- excluded and only counted: unrelated models, unreadable or tampered evidence
  (reported as `inaccessible`), non-terminal jobs and the current job itself.

Bounded: at most `scan_jobs` most recent terminal jobs are examined and `limit` items
returned; raw model outputs and workload contents are never included.
"""
from __future__ import annotations

import hashlib
import json

EXPERIENCE_VERSION = 1
TERMINAL = ('completed', 'failed', 'cancelled')


def _identity(repository, record_body, pins):
    """Applicability identity of a deployment as it was measured; evidence pins win over refs."""
    from inferstack.research.techniques import applicability
    body = dict(record_body)
    for key in ('model_ref', 'hardware_ref'):
        pinned = (pins or {}).get(key)
        if isinstance(pinned, list) and pinned and pinned[0]:
            body[key] = pinned[0]  # verifier records [baseline, candidate] pins
    return applicability(repository, body, strict_refs=True)


def _model_family(repository, body):
    from inferstack.contracts.records import parse_ref
    family = {'served_model': body.get('model'), 'repo_id': None}
    if body.get('model_ref'):
        try:
            identifier, revision = parse_ref(body['model_ref'])
            family['repo_id'] = repository.read('model', identifier, revision)['body'].get('repo_id')
        except (KeyError, ValueError):
            pass
    return family


def _evidence(repository, ref):
    """(evidence dict or None, problem or None) after digest verification."""
    sha = (ref or {}).get('sha256')
    if not sha:
        return None, 'no evidence reference'
    try:
        raw = repository.read_artifact(sha)
    except ValueError:
        return None, 'evidence digest mismatch'
    except (KeyError, OSError):
        return None, 'evidence artifact missing'
    if hashlib.sha256(raw).hexdigest() != sha:
        return None, 'evidence digest mismatch'
    try:
        return json.loads(raw), None
    except ValueError:
        return None, 'evidence is not valid JSON'


def _terminal_jobs(repository, scan_jobs, exclude):
    placeholders = ','.join('?' * len(TERMINAL))
    with repository.connection() as db:
        rows = db.execute(f'SELECT id,state,request_key,updated,result FROM jobs WHERE state IN ({placeholders}) '
                          'ORDER BY updated DESC LIMIT ?', (*TERMINAL, scan_jobs)).fetchall()
    for row in rows:
        if row['id'] in exclude:
            continue
        result = json.loads(row['result']) if row['result'] else {}
        experiments = result.get('experiments') or (result.get('partial') or {}).get('experiments') or []
        yield row, experiments


def experience(repository, deployment_id, *, limit=20, scan_jobs=200, exclude_jobs=(), treatment_signature=None):
    """Prior outcomes relevant to a deployment, newest first within each class."""
    from inferstack.research.techniques import applicability, identity_differences
    if not 1 <= limit <= 200 or not 1 <= scan_jobs <= 2000:
        raise ValueError('limit must be in [1,200] and scan_jobs in [1,2000]')
    target_record = repository.read('deployment', deployment_id)
    target = target_record['body']
    target_identity = applicability(repository, target)
    target_family = _model_family(repository, target)
    items, excluded = [], {'unrelated': 0, 'inaccessible': [], 'malformed': 0}
    scanned = 0
    for job, experiments in _terminal_jobs(repository, scan_jobs, set(exclude_jobs)):
        scanned += 1
        for index, experiment in enumerate(experiments):
            plan, verdict = experiment.get('plan') or {}, experiment.get('verdict') or {}
            if not isinstance(plan, dict) or not isinstance(verdict, dict) or not verdict.get('decision'):
                excluded['malformed'] += 1
                continue
            signature = plan.get('treatment_signature')
            if treatment_signature and signature != treatment_signature:
                continue
            evidence, problem = _evidence(repository, verdict.get('evidence'))
            if evidence is None:
                excluded['inaccessible'].append({'job_id': job['id'], 'iteration': index, 'reason': problem})
                continue
            baseline = evidence.get('baseline') or {}
            body = baseline.get('body') if isinstance(baseline, dict) else None
            if not isinstance(body, dict):
                excluded['malformed'] += 1
                continue
            family = _model_family(repository, body)
            same_family = family['served_model'] == target_family['served_model'] or (
                family['repo_id'] is not None and family['repo_id'] == target_family['repo_id'])
            if not same_family:
                excluded['unrelated'] += 1
                continue
            differing = identity_differences(_identity(repository, body, evidence.get('identity')), target_identity)
            decision = verdict['decision']
            item = {
                'job_id': job['id'], 'job_state': job['state'], 'iteration': index, 'at': job['updated'],
                'treatment_signature': signature, 'hypothesis': (plan.get('hypothesis') or '')[:300],
                'actions': [{'type': a.get('type'), 'params': a.get('params')} for a in plan.get('actions') or []
                            if isinstance(a, dict) and a.get('type') != 'measure'],
                'baseline_ref': f"{baseline.get('id')}@{baseline.get('revision')}",
                'candidate_ref': (f"{experiment['candidate'].get('deployment_id')}@"
                                  f"{experiment['candidate'].get('revision')}")
                                 if isinstance(experiment.get('candidate'), dict) else None,
                'decision': decision, 'reasons': (verdict.get('reasons') or [])[:5],
                'evidence_sha256': verdict['evidence']['sha256'],
                'workload_fixture_sha256': (evidence.get('verifier') or {}).get('fixture_sha256'),
                'evaluator_sha256': (evidence.get('verifier') or {}).get('code_sha256'),
                'applicability': 'applicable' if not differing else 'transferable',
                'differences': differing,
            }
            if not differing:
                item['meaning'] = {'keep': 'verified success on this setup', 'reject': 'known failure on this setup'
                                   }.get(decision, f'{decision} on this setup (not proof)')
            else:
                item['meaning'] = f'{decision} on a different setup: hypothesis only, not proof here'
            items.append(item)
    order = {'applicable': 0, 'transferable': 1}
    items.sort(key=lambda i: (order[i['applicability']], -(i['at'] or 0)))
    applicable = [i for i in items if i['applicability'] == 'applicable']
    return {
        'version': EXPERIENCE_VERSION, 'deployment_ref': f"{deployment_id}@{target_record['revision']}",
        'target_identity': target_identity, 'scanned_jobs': scanned, 'scan_limit': scan_jobs,
        'items': items[:limit], 'truncated': len(items) > limit,
        'summary': {
            'applicable_successes': sum(1 for i in applicable if i['decision'] == 'keep'),
            'applicable_failures': sum(1 for i in applicable if i['decision'] == 'reject'),
            'transferable': sum(1 for i in items if i['applicability'] == 'transferable'),
        },
        # Treatments that already failed on this exact setup: a planner must not repeat them blindly.
        'failed_here_signatures': sorted({i['treatment_signature'] for i in applicable
                                          if i['decision'] == 'reject' and i['treatment_signature']}),
        'excluded': {**excluded, 'inaccessible': excluded['inaccessible'][:20]},
    }
