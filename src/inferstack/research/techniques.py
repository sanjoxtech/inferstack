"""Technique registry: compatibility claims vs. proven compatibility.

A technique manifest is a claim. `assess` checks it against a concrete
deployment (engine and version, dtype, hardware, required engine capabilities,
dependency pinning) and returns one of:

- incompatible: a declared requirement is definitely not met -> excluded.
- unknown: a requirement cannot be checked (missing version/hardware data);
  eligible only as a candidate experiment, never as an accepted optimization.
- compatible_claimed: all checkable requirements match, but no trial proves it.
- compatible_proven: a referenced trial on a matching engine/model proved it.

Executable means the technique could enter the executor: it is not incompatible
and every dependency is pinned to an exact version with a known license.
"""
from __future__ import annotations

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from inferstack.engines.capabilities import Support, lookup


def _version_ok(spec: str, version: str | None):
    if spec in ('', '*'):
        return True
    if version is None:
        return None
    try:
        return Version(version) in SpecifierSet(spec)
    except (InvalidSpecifier, InvalidVersion):
        return None


def _cc(text):
    major, minor = text.split('.')
    return int(major), int(minor)


def assess(repository, technique_id: str, deployment_id: str) -> dict:
    technique = repository.read_typed('technique', technique_id)
    deployment_record = repository.read('deployment', deployment_id)
    deployment = deployment_record['body']
    hardware = None
    if deployment.get('hardware_ref'):
        from inferstack.contracts.records import parse_ref
        hid, hrev = parse_ref(deployment['hardware_ref'])
        hardware = repository.read('hardware', hid, hrev)['body']
    blockers, unknowns, notes = [], [], []

    requirement = next((e for e in technique.engines if e.engine == deployment['engine']), None)
    if requirement is None:
        blockers.append(f"engine {deployment['engine']} not in {[e.engine for e in technique.engines]}")
    else:
        ok = _version_ok(requirement.versions, deployment.get('engine_version'))
        if ok is False:
            blockers.append(f"engine version {deployment.get('engine_version')} outside {requirement.versions}")
        elif ok is None:
            unknowns.append(f'engine version unknown or unparsable for requirement {requirement.versions}')

    dtype = (deployment.get('profile') or {}).get('dtype')
    if technique.dtypes:
        if dtype is None:
            unknowns.append(f'deployment dtype unknown; technique requires {technique.dtypes}')
        elif dtype not in technique.dtypes:
            blockers.append(f'dtype {dtype} not in {technique.dtypes}')

    for capability in getattr(technique, 'requires_capabilities', []):
        try:
            state = lookup(deployment['engine'], capability).state
        except ValueError:
            unknowns.append(f'capability {capability} is not in the capability matrix')
            continue
        if state is Support.UNSUPPORTED:
            blockers.append(f"{deployment['engine']} does not support {capability}")
        elif state is Support.UNKNOWN:
            unknowns.append(f"{deployment['engine']} {capability} is unknown")

    min_cc = getattr(technique, 'min_compute_capability', None)
    min_mem = getattr(technique, 'min_gpu_memory_bytes', None)
    if technique.hardware or min_cc or min_mem:
        gpus = (hardware or {}).get('gpus') or []
        if not gpus:
            unknowns.append('deployment hardware is not recorded')
        else:
            if technique.hardware and not any(any(h.lower() in g['name'].lower() for h in technique.hardware)
                                              for g in gpus):
                blockers.append(f"GPU {[g['name'] for g in gpus]} not in {technique.hardware}")
            if min_cc:
                ccs = [g.get('compute_capability') for g in gpus]
                if any(c is None for c in ccs):
                    unknowns.append('GPU compute capability not recorded')
                elif min(_cc(c) for c in ccs) < _cc(min_cc):
                    blockers.append(f'compute capability {min(ccs)} below {min_cc}')
            if min_mem and min(g['memory_bytes'] for g in gpus) < min_mem:
                blockers.append(f'GPU memory below {min_mem} bytes')

    unpinned = [d.name for d in technique.dependencies if not d.version.startswith('==') or d.license is None]
    if unpinned:
        notes.append('dependencies need an exact version and license before execution: ' + ', '.join(unpinned))

    proven = []
    for ref in technique.proven_by:
        from inferstack.contracts.records import parse_ref
        tid, trev = parse_ref(ref)
        trial = repository.read('trial', tid, trev)['body']
        did, drev = parse_ref(trial['deployment_ref'])
        proved_on = repository.read('deployment', did, drev)['body']
        if proved_on['engine'] == deployment['engine'] and proved_on['model'] == deployment['model']:
            proven.append(f'{tid}@{trev}' if trev else tid)

    if blockers or getattr(technique, 'status', 'claim') == 'rejected':
        verdict = 'incompatible'
    elif unknowns:
        verdict = 'unknown'
    elif proven:
        verdict = 'compatible_proven'
    else:
        verdict = 'compatible_claimed'
    return {'technique_id': technique_id, 'deployment_id': deployment_id,
            'deployment_revision': deployment_record['revision'], 'verdict': verdict,
            'blockers': blockers, 'unknowns': unknowns, 'notes': notes, 'proven_by': proven,
            'executable': verdict != 'incompatible' and not unpinned,
            'claims': list(technique.compatibility_claims)}


def candidates(repository, deployment_id: str) -> list[dict]:
    """Techniques eligible as candidate experiments for a deployment (incompatible excluded)."""
    out = []
    for record in repository.list_records('technique', limit=1000):
        if record['schema_version'] is None:
            continue
        result = assess(repository, record['id'], deployment_id)
        if result['verdict'] != 'incompatible':
            out.append(result)
    order = {'compatible_proven': 0, 'compatible_claimed': 1, 'unknown': 2}
    return sorted(out, key=lambda r: (order[r['verdict']], r['technique_id']))
