#!/usr/bin/env python
"""G7 real run: one code idea -> model coding agent -> desk -> Colab L4 A/B -> PR bundle (never published).

The provider key is read from the environment variable named by --key-env and is never printed or stored.
The Colab VM clones the pinned commit (HEAD, which must be pushed or at least committed) and applies the
validated diff to a pristine export with tests and the evaluator restored.

  export OPENROUTER_API_KEY=...        # in your own terminal; never pasted into chat
  python scripts/code_change_run.py --kind openrouter --model <vendor/model> --key-env OPENROUTER_API_KEY
  python scripts/code_change_run.py --dry-run        # build the handoff and show what the agent sees; no calls
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from inferstack.execution import Sandbox
from inferstack.execution.gpu_backend import ColabBackend
from inferstack.implementation import ImplementationDesk
from inferstack.implementation.coder import ModelImplementer
from inferstack.optimization.code_change import CodeChange, evaluator_fingerprint
from inferstack.optimization.remote_verify import RemoteABVerifier
from inferstack.providers.base import create_provider
from inferstack.storage.repository import Repository

GDN_FILE = 'src/inferstack/kernels/gdn_banked.py'
GDN = {
    'hypothesis': ('G4 on L4 (native-v1-36-0, interactive): GPU busy 98%, decode is compute-bound; at 48 rows the '
                   'banked GDN decode kernel is 34% of eager step time (7% at 8 rows, 24% at 32) and grows with '
                   'rows, GEMM is 55%. G5 found settings at their limit. A faster GDN banked decode kernel (better '
                   'tiling over V, fewer state loads/stores per step, launch config tuned for L4 sm_89) lowers '
                   'ms/step at high rows and lifts goodput@99 and SLO attainment at 8-16 req/s.'),
    'description': ('Speed up the Triton banked GDN decode kernel in gdn_banked.py (gdn_banked / gdn_banked_fixup '
                    'paths) for L4. Keep the math exactly the same: identical recurrence, gating, L2 norm, scale '
                    'and state update order, so greedy outputs stay token-identical. Allowed: block sizes (BV), '
                    'num_warps/num_stages, loop structure, fusing loads, avoiding redundant global memory '
                    'traffic. Keep the reference() functions and the public signatures unchanged.'),
    'files': [GDN_FILE],
    'tests': ['tests/test_gdn_banked.py', 'tests/test_kernels_reference.py'],
}


def head():
    return subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                          check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['openrouter', 'deepseek'], default='openrouter')
    ap.add_argument('--model', help='provider model id (configured, never hard-coded)')
    ap.add_argument('--key-env', default='OPENROUTER_API_KEY')
    ap.add_argument('--baseline', default='native-v1-36-0', help='deployment record id')
    ap.add_argument('--baseline-profile', default='native_lazy', help='scorecard profile of the baseline')
    ap.add_argument('--name', default='g7-gdn')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--max-gain', type=float, default=1.5)
    ap.add_argument('--max-output-tokens', type=int, default=16000)
    ap.add_argument('--state', default=str(ROOT / '.inferstack'))
    ap.add_argument('--out', default=str(ROOT / 'results' / 'code_change'))
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    state, out = Path(a.state), Path(a.out)
    repo = Repository(state / 'control.db', state / 'artifacts')
    rev = head()
    dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        sys.exit('tracked files have uncommitted changes: the VM runs HEAD, commit first')
    desk = ImplementationDesk(repo, source_root=ROOT, sandbox=Sandbox(), work_root=out / 'work', test_timeout=900)
    verifier = RemoteABVerifier(repo, ColabBackend(), out / 'runs', repeats=a.repeats, source_revision=rev)
    estimate = 2 * a.repeats * verifier.hours_per_trial * verifier.backend.price_per_hour
    proposal = {'action_id': 'patch', 'technique_id': 'gdn-banked-kernel', 'files': GDN['files'],
                'description': GDN['description']}
    plan = {'hypothesis': GDN['hypothesis'], 'baseline_ref': f'{a.baseline}@1', 'evidence_ids': [],
            'actions': [{'id': 'patch', 'type': 'source_patch', 'depends_on': [],
                         'params': {'files': GDN['files'], 'tests': GDN['tests']}},
                        {'id': 'measure', 'type': 'measure', 'depends_on': ['patch'], 'params': {}}]}
    print(json.dumps({'source_revision': rev, 'evaluator_fingerprint': evaluator_fingerprint()[:16],
                      'baseline': a.baseline, 'baseline_profile': a.baseline_profile,
                      'gpu_estimate_usd': round(estimate, 2), 'files': GDN['files'], 'tests': GDN['tests']}, indent=2))
    if a.dry_run:
        impl = desk.request(proposal, plan, baseline_id=a.baseline, source_revision=rev, job_id=a.name + '-dry',
                            independent_tests=GDN['tests'])
        msgs = ModelImplementer(None, source_root=ROOT, name='dry').messages(impl['body']['handoff'], None)
        task = json.loads(msgs[1]['content'])
        print('agent sees:', {k: (len(v) if isinstance(v, (str, dict)) else v) for k, v in task.items()})
        return
    if not a.model:
        sys.exit('--model is required for a real run')
    provider = create_provider({'kind': a.kind, 'model': a.model, 'api_key_env': a.key_env,
                                'max_output_tokens': a.max_output_tokens, 'timeout': 600})
    cc = CodeChange(desk, ModelImplementer(provider, source_root=ROOT), verifier, out_root=out / 'prs',
                    max_gain=a.max_gain)
    result = asyncio.run(cc.run(proposal, plan, baseline_id=a.baseline, baseline_profile=a.baseline_profile,
                                source_revision=rev, name=a.name, tests=GDN['tests'], job_id=a.name))
    (out / f'{a.name}.result.json').parent.mkdir(parents=True, exist_ok=True)
    (out / f'{a.name}.result.json').write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: result.get(k) for k in ('decision', 'attempts', 'reason', 'cost_usd', 'bundle', 'title')},
                     indent=2, default=str))


if __name__ == '__main__':
    main()
