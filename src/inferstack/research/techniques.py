"""Technique registry: compatibility claims vs. proven compatibility.

A technique manifest is a claim. `assess` checks it against a concrete
deployment (engine and version, dtype, hardware, required engine capabilities,
dependency pinning) and returns one of:

- incompatible: a declared requirement is definitely not met -> excluded.
- unknown: a requirement cannot be checked (missing version/hardware data);
  eligible only as a candidate experiment, never as an accepted optimization.
- compatible_claimed: all checkable requirements match, but no trial proves it.
- compatible_proven: a referenced trial is valid proof for *this* deployment (see
  `proof_of`): a successful candidate trial whose verification evidence is intact and
  records `keep` with every required gate passed, whose treatment demonstrably is
  this technique, and whose reuse identity (engine and version, model and pinned
  revision, dtype, hardware) equals the target's. Anything weaker is reported in
  `transferable` as a hypothesis with the reason, never as proof.

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


APPLICABILITY_VERSION = 1
# Components that must be known on both sides for any proof; everything else in the
# identity must still be equal (both unknown counts as equal, one-sided does not).
REQUIRED = ('engine', 'engine_version', 'model.revision', 'model.tokenizer_revision', 'dtype', 'hardware.gpus')


def _hash(value):
    import hashlib
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def applicability(repository, deployment, *, strict_refs=False):
    """Versioned applicability identity of a deployment: what a measured result is valid for.

    strict_refs: an unpinned model/hardware reference resolves to *no* identity (a
    trial's deployment must not silently become evidence about a record's current
    revision). Targets are resolved as they are now."""
    from inferstack.contracts.records import parse_ref
    identity = {'version': APPLICABILITY_VERSION, 'engine': deployment.get('engine'),
                'engine_version': deployment.get('engine_version'), 'image_digest': deployment.get('image_digest'),
                'plugin_refs': sorted(deployment.get('plugin_refs') or []),
                'dtype': (deployment.get('profile') or {}).get('dtype'), 'model': None, 'hardware': None}
    for field, kind in (('model_ref', 'model'), ('hardware_ref', 'hardware')):
        ref = deployment.get(field)
        if not ref:
            continue
        identifier, revision = parse_ref(ref)
        if strict_refs and revision is None:
            continue
        try:
            body = repository.read(kind, identifier, revision)['body']
        except KeyError:
            continue
        if kind == 'model':
            identity['model'] = {'repo_id': body.get('repo_id'), 'revision': body.get('revision'),
                                 'tokenizer_revision': body.get('tokenizer_revision') or body.get('revision'),
                                 'config_revision': body.get('config_revision'),
                                 'checksums': _hash(body.get('checksums') or {})}
        else:
            identity['hardware'] = {
                'gpus': sorted([g.get('name'), g.get('memory_bytes'), g.get('compute_capability')]
                               for g in body.get('gpus') or []) or None,
                'topology': body.get('topology'), 'driver_version': body.get('driver_version'),
                'cuda_version': body.get('cuda_version'), 'libraries': dict(sorted((body.get('libraries') or {}).items()))}
    return identity


def _get(identity, dotted):
    value = identity
    for part in dotted.split('.'):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def identity_differences(proved, target, *, treatment=None):
    """Components that differ, or that are required but unknown on either side. The
    technique under test (`treatment`) is excluded from the plugin comparison: it is the
    difference the proof is about; any other plugin difference still counts."""
    if treatment is not None:
        proved = {**proved, 'plugin_refs': [p for p in proved.get('plugin_refs') or [] if p != treatment]}
        target = {**target, 'plugin_refs': [p for p in target.get('plugin_refs') or [] if p != treatment]}
    out = [k for k in REQUIRED if _get(proved, k) is None or _get(target, k) is None]
    for key in ('engine', 'engine_version', 'image_digest', 'plugin_refs', 'dtype'):
        if proved.get(key) != target.get(key):
            out.append(key)
    for group in ('model', 'hardware'):
        a, b = proved.get(group) or {}, target.get(group) or {}
        out += [f'{group}.{k}' for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    return sorted(set(out))


def reuse_key(repository, deployment):
    """Backward-compatible alias: the applicability identity of a deployment."""
    return applicability(repository, deployment)


def proof_of(repository, technique_id, ref, target_key):
    """(proven: bool, reason). A trial reference alone is never proof."""
    import hashlib
    import json
    from inferstack.contracts.records import parse_ref
    try:
        tid, trev = parse_ref(ref)
        trial = repository.read('trial', tid, trev)['body']
    except (KeyError, ValueError):
        return False, 'trial record missing'
    if trial.get('role') != 'candidate':
        return False, 'not a candidate trial'
    if trial.get('errors'):
        return False, 'trial recorded errors: ' + '; '.join(trial['errors'])[:200]
    try:
        raw = repository.read_artifact(trial['raw_artifact_sha256'])
    except ValueError:
        return False, 'verification evidence digest mismatch (corrupt or tampered artifact)'
    except (KeyError, OSError):
        return False, 'verification evidence artifact missing'
    if hashlib.sha256(raw).hexdigest() != trial['raw_artifact_sha256']:
        return False, 'verification evidence digest mismatch'
    try:
        evidence = json.loads(raw)
    except ValueError:
        return False, 'verification evidence is not valid JSON'
    verdict = (evidence.get('verdict') or {}).get('decision')
    required = (evidence.get('policy') or {}).get('required') or []
    gates = evidence.get('gates') or {}
    if verdict != 'keep':
        return False, f'authoritative verdict is {verdict!r}, not keep'
    if not required or any(gates.get(g) is not True for g in required):
        return False, 'required verification gates did not all pass'
    did, drev = parse_ref(trial['deployment_ref'])
    try:
        proved_on = repository.read('deployment', did, drev)['body']
    except KeyError:
        return False, 'trial deployment revision missing'
    lineage = technique_id in (proved_on.get('plugin_refs') or [])
    if not lineage and trial.get('plan_ref'):
        try:
            pid, prev = parse_ref(trial['plan_ref'])
            plan = repository.read('plan', pid, prev)['body']
            lineage = any((a.get('params') or {}).get('technique_id') == technique_id
                          for a in plan.get('actions') or [])
        except (KeyError, ValueError):
            lineage = False
    if not lineage:
        return False, 'treatment lineage to this technique not established'
    differing = identity_differences(applicability(repository, proved_on, strict_refs=True), target_key,
                                     treatment=technique_id)
    if differing:
        return False, 'different or unknown identity: ' + ', '.join(differing)
    return True, 'successful verified trial on the same applicability identity'


def proof_scope(repository, ref):
    """The workload and evaluator a proving trial was measured under (its evidence scope)."""
    import json
    from inferstack.contracts.records import parse_ref
    tid, trev = parse_ref(ref)
    trial = repository.read('trial', tid, trev)['body']
    evidence = json.loads(repository.read_artifact(trial['raw_artifact_sha256']))
    verifier = evidence.get('verifier') or {}
    return {'workload_fixture_sha256': verifier.get('fixture_sha256') or (evidence.get('policy') or {}).get('fixture_sha256'),
            'evaluator_sha256': verifier.get('code_sha256'), 'workload_ref': trial.get('workload_ref')}


def assess(repository, technique_id: str, deployment_id: str, *, workload_fixture_sha256=None) -> dict:
    """workload_fixture_sha256: when given, a proof only counts if it was measured on that
    workload fixture; proofs always report their workload/evaluator scope."""
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

    proven, transferable, scopes = [], [], {}
    target_key = applicability(repository, deployment)
    for ref in technique.proven_by:
        ok, reason = proof_of(repository, technique_id, ref, target_key)
        if ok:
            scope = proof_scope(repository, ref)
            if workload_fixture_sha256 and scope['workload_fixture_sha256'] != workload_fixture_sha256:
                ok, reason = False, 'proved on a different workload fixture'
            else:
                scopes[ref] = scope
        (proven if ok else transferable).append(ref if ok else {'trial': ref, 'reason': reason})

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
            'transferable': transferable, 'proof_scope': scopes,
            'applicability_version': APPLICABILITY_VERSION,
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
