"""G3: the P9-P13 campaign imports into harness memory as same-setup evidence, idempotently."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs' / 'verification' / 'campaigns' / 'l4-qwen35-p9-p13.json'


def run(state):
    out = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'backfill_experiments.py'), str(MANIFEST),
                          '--state', str(state)], capture_output=True, text=True, cwd=ROOT, check=True)
    return json.loads(out.stdout)


def test_backfill_feeds_experience_and_is_idempotent(tmp_path):
    from inferstack.memory import experience
    from inferstack.storage.repository import Repository
    first = run(tmp_path)
    second = run(tmp_path)
    assert first['status'] == 'imported' and second['status'] == 'already completed'
    assert first['job_id'] == second['job_id']
    repo = Repository(tmp_path / 'control.db', tmp_path / 'artifacts')
    out = experience(repo, 'native-v1-36-0', limit=20)
    manifest = json.loads(MANIFEST.read_text())
    decisions = [e['decision'] for e in manifest['experiments']]
    assert out['summary']['applicable_successes'] == decisions.count('keep')
    assert out['summary']['applicable_failures'] == decisions.count('reject')
    assert out['summary']['transferable'] == 0 and not out['excluded']['inaccessible']
    by_hyp = {i['hypothesis'][:40]: i for i in out['items']}
    two_drafts = next(i for h, i in by_hyp.items() if 'second draft' in i['hypothesis'])
    assert two_drafts['decision'] == 'reject'
    assert two_drafts['treatment_signature'] in out['failed_here_signatures']
    inconclusive = [i for i in out['items'] if i['decision'] == 'inconclusive']
    assert len(inconclusive) == 1 and inconclusive[0]['treatment_signature'] not in out['failed_here_signatures']
