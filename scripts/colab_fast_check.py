#!/usr/bin/env python3
"""Fast native-only L4 check through the Google Colab CLI (one command, no browser).

Pipeline stage for native treatments: install + smoke + a reduced scorecard for the chosen
native profiles only, then download the evidence and stop the runtime. vLLM/SGLang are not
re-measured; compare against the rivals' rows of the last full run (same GPU class, pinned
versions) with `engine_scorecard.py aggregate`. A fast check screens a treatment. Only a
full 3-round scorecard (docs/notebooks/colab_a100_engine_scorecard.ipynb) promotes one.

  python scripts/colab_fast_check.py --profiles native_static native_budget \
      --workload configs/scorecard/workload-fast.json --rounds 2

The repository is shipped as a git bundle of the current branch, so no credential reaches
the runtime. The pinned commit is HEAD (it must be committed). Requires `colab` (google-colab-cli)
signed in.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'docs/notebooks/colab_a100_engine_scorecard.ipynb'
SETUP_CELLS = 7          # GPU check + clone, native env, stage helpers (markdown included)


def sh(*argv, check=True, capture=False):
    print('+', ' '.join(map(str, argv)), flush=True)
    return subprocess.run(list(map(str, argv)), check=check, text=True,
                          capture_output=capture)


def build_notebook(commit, profiles, workload, rounds, out_path):
    nb = json.loads(NOTEBOOK.read_text())
    cells = nb['cells'][:SETUP_CELLS]
    setup = ''.join(cells[2]['source'])
    marker = "COMMIT = '"
    start = setup.index(marker) + len(marker)
    cells[2]['source'] = setup[:start] + commit + setup[setup.index("'", start):]
    code = f'''
ONLY = {profiles!r}
for p in ONLY:
    stage(p)
only = [p for p in ONLY if STAGE.get(p) == 'passed']
print('fast check profiles:', only, '| stages:', STAGE)
assert only, 'no native profile passed its smoke test'
(OUT/'workload-fast.json').write_text({json.dumps(json.dumps(workload))})
require_empty_gpu()
code = logged([PY,'-u','scripts/engine_scorecard.py','run','--profiles',PROFILES,
               '--workload',str(OUT/'workload-fast.json'),
               '--out',str((OUT/'scorecard').relative_to(REPO)),'--rounds','{rounds}',
               '--run-id',f'{{GPU_LABEL}}-fast-{{int(time.time())}}','--only',*only,
               '--ready-timeout',str(READY_TIMEOUT)], 'scorecard-runner.log', env=SCORE_ENV, timeout=4*60*60)
print('scorecard exit:', code)
(OUT/'stages.json').write_text(json.dumps(STAGE, indent=2))
import shutil as _sh
print('Evidence archive:', _sh.make_archive(str(BASE/'fast-check'), 'zip', root_dir=OUT.parent, base_dir=OUT.name))
'''
    cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                  'source': code})
    nb['cells'] = cells
    out_path.write_text(json.dumps(nb, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profiles', nargs='+', required=True, help='native profile names from the scorecard JSON')
    ap.add_argument('--workload', default=str(ROOT / 'configs/scorecard/workload-fast.json'))
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--gpu', default='L4')
    ap.add_argument('--session', default=f'fast-{int(time.time())}')
    ap.add_argument('--out', default=str(ROOT / 'results/fast-check'))
    ap.add_argument('--keep', action='store_true', help='leave the runtime running afterwards')
    args = ap.parse_args()
    colab = shutil.which('colab') or str(Path.home() / '.local/bin/colab')
    if sh('git', 'status', '--porcelain', '--untracked-files=no', capture=True).stdout.strip():
        sys.exit('commit tracked changes first: the runtime gets HEAD, not the working tree')
    commit = sh('git', 'rev-parse', 'HEAD', capture=True).stdout.strip()
    branch = sh('git', 'rev-parse', '--abbrev-ref', 'HEAD', capture=True).stdout.strip()
    workload = json.loads(Path(args.workload).read_text())
    out = Path(args.out) / f'{args.session}-{commit[:7]}'
    out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bundle, nb, prep = tmp / 'inferstack.bundle', tmp / 'fast_check.ipynb', tmp / 'prep.py'
        sh('git', 'bundle', 'create', bundle, branch)
        build_notebook(commit, args.profiles, workload, args.rounds, nb)
        prep.write_text(
            'import subprocess as sp\n'
            f'r=sp.run("cd /content && rm -rf inferstack && git clone -q -b {branch} inferstack.bundle inferstack '
            f'&& git -C inferstack cat-file -t {commit}", shell=True, capture_output=True, text=True)\n'
            'print(r.stdout, r.stderr); assert r.returncode == 0\n')
        sh(colab, 'new', '--gpu', args.gpu, '-s', args.session)
        try:
            sh(colab, 'upload', '-s', args.session, bundle, '/content/inferstack.bundle')
            sh(colab, 'exec', '-s', args.session, '-f', prep, '--timeout', '300')
            with (out / 'run.log').open('w') as log:
                proc = subprocess.run([colab, 'exec', '-s', args.session, '-f', str(nb), '--timeout', '0'],
                                      stdout=log, stderr=subprocess.STDOUT, text=True)
            print('notebook exit:', proc.returncode, '| log:', out / 'run.log')
            got = sh(colab, 'download', '-s', args.session, '/content/fast-check.zip', out / 'evidence.zip', check=False)
            if got.returncode == 0:
                shutil.unpack_archive(out / 'evidence.zip', out)
                for md in sorted(out.glob('*/scorecard/*/scorecard.md')):
                    print(md.read_text())
        finally:
            if not args.keep:
                sh(colab, 'stop', '-s', args.session, check=False)
            stray = ROOT / 'docs/notebooks' / f'{nb.stem}_output.ipynb'
            if stray.exists():
                stray.replace(out / stray.name)
    print('evidence:', out)


if __name__ == '__main__':
    main()
