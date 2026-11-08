"""Package J: retention enforcement and qualification verdict semantics."""
import json
import os
import subprocess
import sys
from pathlib import Path

from inferstack.contracts.redaction import RetentionPolicy
from inferstack.control.maintenance import enforce
from inferstack.storage.repository import Repository

ROOT = Path(__file__).parents[1]


def test_retention_prunes_only_old_terminal_evidence(tmp_path):
    now = [1_000_000.0]
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    repo.clock = lambda: now[0]
    done = repo.submit('done', {})
    lease = repo.claim(10_000)
    repo.finish(done, lease['lease_token'], {'decision': 'keep', 'evidence': repo.artifact(b'kept evidence')})
    running = repo.submit('running', {})
    repo.claim(10_000)
    orphan = repo.artifact(b'unreferenced')
    old = now[0] - 40 * 86400
    os.utime(tmp_path / 'a' / orphan['sha256'], (old, old))
    os.utime(tmp_path / 'a' / repo.get(done)['result']['evidence']['sha256'], (old, old))
    now[0] += 31 * 86400
    policy = RetentionPolicy(event_ttl_days=30, artifact_ttl_days=30)
    preview = enforce(repo, policy, dry_run=True)
    assert preview['events_deleted'] == 3 and preview['artifacts_deleted'] == 1
    assert len(repo.get(done)['events']) == 3  # dry run changed nothing
    report = enforce(repo, policy)
    assert report == {'events_deleted': 3, 'artifacts_deleted': 1, 'artifacts_kept_referenced': 1, 'dry_run': False}
    assert repo.get(done)['events'] == [] and repo.get(done)['result']['decision'] == 'keep'
    assert len(repo.get(running)['events']) == 2  # active jobs untouched
    assert not (tmp_path / 'a' / orphan['sha256']).exists()


def test_pinned_foundry_manifests():
    for stage in ('candidate', 'v0-baseline'):
        manifest = json.loads((ROOT / f'examples/model_foundry/factorylm-2b/{stage}/manifest.json').read_text())
        for part in ('source', 'tokenizer'):
            assert len(manifest[part]['revision']) == 40 and int(manifest[part]['revision'], 16) >= 0
