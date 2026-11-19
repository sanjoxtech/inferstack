#!/usr/bin/env python
"""G3: import a hand-run campaign into the harness record store, so experiment memory knows it.

Reads a campaign manifest (docs/verification/campaigns/*.json) and writes, through the Repository API:
  - the model and hardware records (pinned);
  - one deployment record per profile used (the profile TOML as the record's profile);
  - per experiment, one content-addressed evidence artifact: baseline/candidate records, pinned identity,
    the verifier description, and the digests + summary of the evidence files under
    docs/verification/evidence/<dir>/;
  - one terminal job (`completed`, payload kind `backfill`) whose result lists the experiments as
    {plan, candidate, verdict} in campaign order: the shape `memory.experience` reads.

Treatment signatures follow optimization.dag.treatment_signature (config_change with a profile_patch), so
the planner's no-identical-retry rule applies to treatments that failed here. Idempotent: the job's
request key is the campaign id plus the manifest digest; re-running with the same manifest is a no-op.
Nothing here claims more than the evidence: `inconclusive` stays inconclusive.

  python scripts/backfill_experiments.py docs/verification/campaigns/l4-qwen35-p9-p13.json --state .inferstack
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE_VERSION = None   # set in main(): inferstack package version (exact commits stay in the evidence)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def signature(actions):
    body = sorted(json.dumps({'type': a['type'], 'params': a['params']}, sort_keys=True)
                  for a in actions if a['type'] != 'measure')
    return hashlib.sha256('\n'.join(body).encode()).hexdigest()


def put_once(repo, kind, identifier, body):
    try:
        return f"{identifier}@{repo.read(kind, identifier)['revision']}"
    except KeyError:
        return f'{identifier}@{repo.put_record(kind, identifier, body)}'


def deployment(repo, profile_name, model_ref, hardware_ref):
    path = ROOT / 'configs' / 'profiles' / f'{profile_name}.toml'
    ident = 'native-' + profile_name.replace('production-', '').replace('.', '-')
    body = {'id': ident, 'engine': 'inferstack_native', 'model': 'factorylm-2b', 'base_url': 'http://127.0.0.1:8000',
            'command': ['inferstack', 'serve', '--profile', f'configs/profiles/{profile_name}.toml'],
            'model_ref': model_ref, 'hardware_ref': hardware_ref,
            'engine_version': ENGINE_VERSION,
            # applicability reads profile['dtype']; the native TOML keeps it in a section
            'profile': {'dtype': 'bf16', **tomllib.loads(path.read_text())}}
    ref = put_once(repo, 'deployment', ident, body)
    rid, rev = ref.split('@')
    return {'id': rid, 'revision': int(rev), 'body': repo.read('deployment', rid, int(rev))['body']}


def evidence_files(name):
    d = ROOT / 'docs' / 'verification' / 'evidence' / name
    if not d.is_dir():
        raise SystemExit(f'evidence directory missing: {d}')
    files = {p.name: sha(p.read_bytes()) for p in sorted(d.iterdir()) if p.is_file()}
    summary = (d / 'goodput.md').read_text()[:4000] if (d / 'goodput.md').is_file() else None
    return {'dir': f'docs/verification/evidence/{name}', 'files': files, 'summary': summary}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('manifest', type=Path)
    ap.add_argument('--state', default='.inferstack')
    ap.add_argument('--database', help='database URL; default STATE/control.db')
    ap.add_argument('--dry-run', action='store_true', help='validate the manifest and evidence, write nothing')
    a = ap.parse_args()
    raw = a.manifest.read_bytes()
    m = json.loads(raw)
    for e in m['experiments']:
        evidence_files(e['evidence'])
        if e['decision'] not in ('keep', 'reject', 'inconclusive'):
            raise SystemExit(f"{e['id']}: bad decision {e['decision']}")
    if a.dry_run:
        print(json.dumps({'campaign': m['campaign'], 'experiments': len(m['experiments']), 'ok': True}))
        return
    global ENGINE_VERSION
    import inferstack
    from inferstack.storage.repository import Repository
    ENGINE_VERSION = f'inferstack {inferstack.__version__}'
    state = Path(a.state)
    repo = Repository(a.database or state / 'control.db', state / 'artifacts')
    mm = {k: v for k, v in m['model'].items() if k != 'id'}
    model_ref = put_once(repo, 'model', m['model']['id'], mm)
    hw = {k: v for k, v in m['hardware'].items() if k != 'id'}
    hardware_ref = put_once(repo, 'hardware', m['hardware']['id'], {**hw, 'observed_at': time.time()})
    goodput_sha = sha((ROOT / m['verifier']['code']).read_bytes())
    slo_sha = sha((ROOT / m['verifier']['slo']).read_bytes())
    experiments = []
    for e in m['experiments']:
        base = deployment(repo, e['baseline'], model_ref, hardware_ref)
        cand = deployment(repo, e['candidate'], model_ref, hardware_ref) if e.get('candidate') else None
        if e.get('patch'):
            actions = [{'id': 'change', 'type': 'config_change', 'params': {'profile_patch': e['patch']}}]
        else:
            actions = [{'id': 'change', 'type': 'source_patch', 'params': {'design': e['id']}}]
        actions.append({'id': 'measure', 'type': 'measure', 'depends_on': ['change'], 'params': {}})
        ev = {'imported': True, 'campaign': m['campaign'], 'experiment': e['id'],
              'baseline': base, 'candidate': cand,
              'identity': {'model_ref': [model_ref, model_ref], 'hardware_ref': [hardware_ref, hardware_ref]},
              'verifier': {**m['verifier'], 'code_sha256': goodput_sha, 'fixture_sha256': slo_sha},
              'source': evidence_files(e['evidence'])}
        digest = repo.artifact(json.dumps(ev, sort_keys=True).encode())
        digest = digest if isinstance(digest, str) else digest.get('sha256', digest)
        experiments.append({
            'plan': {'hypothesis': e['hypothesis'], 'baseline_ref': f"{base['id']}@{base['revision']}",
                     'actions': actions, 'treatment_signature': signature(actions)},
            'candidate': {'deployment_id': cand['id'], 'revision': cand['revision']} if cand else None,
            'verdict': {'decision': e['decision'], 'reasons': e['reasons'], 'evidence': {'sha256': digest}}})
    key = f"backfill:{m['campaign']}:{sha(raw)[:16]}"
    with repo.transaction() as db:
        job_id = repo.submit_in(db, key, {'kind': 'backfill', 'campaign': m['campaign'],
                                          'manifest_sha256': sha(raw)})
        row = db.execute('SELECT state FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row['state'] == 'pending':
            result = {'experiments': experiments, 'backfill': {'manifest': str(a.manifest)}}
            db.execute("UPDATE jobs SET state='completed',result=?,updated=? WHERE id=?",
                       (repo.encode(result), repo.clock(), job_id))
            repo.event(db, job_id, 'completed', {'backfill': m['campaign'], 'experiments': len(experiments)})
            status = 'imported'
        else:
            status = f"already {row['state']}"
    print(json.dumps({'job_id': job_id, 'status': status, 'experiments': len(experiments),
                      'model_ref': model_ref, 'hardware_ref': hardware_ref}, indent=2))


if __name__ == '__main__':
    main()
