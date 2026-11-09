"""Fetch GitHub Actions results for one exact commit as qualification evidence.

    python scripts/ci_evidence.py --commit <full sha> --out results/ci-evidence-<sha>.json

Uses the authenticated GitHub CLI (`gh api`). The file records the repository, commit,
fetch time and every check run (name, conclusion, head SHA, URL). It is consumed by
`v2_qualify.py --sandbox-evidence` / `scripts/requalify.py`, which accept it only for
the same commit and only when the required CI jobs succeeded.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def fetch(repo, commit):
    out = subprocess.run(['gh', 'api', '-X', 'GET', f'repos/{repo}/commits/{commit}/check-runs',
                          '-f', 'per_page=100'], capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise SystemExit(f'gh api failed: {out.stderr.strip()[-300:]}')
    body = json.loads(out.stdout)
    runs = body.get('check_runs', [])
    if body.get('total_count', len(runs)) > len(runs):
        raise SystemExit('more than 100 check runs for this commit; refusing a partial evidence file')
    return {'repository': repo, 'commit': commit, 'fetched_at': time.time(), 'source': 'github check-runs API',
            'check_runs': [{'name': r['name'], 'status': r['status'], 'conclusion': r['conclusion'],
                            'head_sha': r['head_sha'], 'url': r['html_url']} for r in runs]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--repo', default='sanjoxtech/inferstack')
    p.add_argument('--commit', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if len(a.commit) != 40:
        raise SystemExit('use the full 40-character commit SHA')
    evidence = fetch(a.repo, a.commit)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(evidence, indent=2))
    print(json.dumps({r['name']: r['conclusion'] for r in evidence['check_runs']}, indent=2))


if __name__ == '__main__':
    main()
