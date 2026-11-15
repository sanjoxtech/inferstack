#!/usr/bin/env python3
"""L4 scorecard runs through the Google Colab CLI, robust to a lost local connection.

The job runs on the Colab VM as a detached process (setup cells of
docs/notebooks/colab_a100_engine_scorecard.ipynb + stages + scorecard), so a dropped
network on this machine does not stop it. This driver polls the VM, downloads the evidence
archive when the job is done, and stops the runtime only after the archive is safely
downloaded. If the download fails, the runtime is left running and the command to fetch
it is printed.

Fast check (screening, native profiles, rivals compared from the last full run):
  python scripts/colab_fast_check.py --profiles native_static native_budget --rounds 2
Full scorecard (release decision; rivals re-measured):
  python scripts/colab_fast_check.py --profiles native_unfused native_full native_static \\
      native_budget vllm sglang --rounds 3 --workload default

The repository is shipped as a git bundle of the current branch, so no credential reaches
the runtime. The pinned commit is HEAD (commit first).
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'docs/notebooks/colab_a100_engine_scorecard.ipynb'
SETUP_CELLS = 7          # GPU check + clone, native env, stage helpers (markdown included)
RIVALS = ('vllm', 'sglang')
JOB_LOG = '/content/job.log'
ARCHIVE = '/content/job-evidence.zip'


def sh(*argv, check=True, capture=False, timeout=None):
    print('+', ' '.join(map(str, argv)), flush=True)
    try:
        return subprocess.run(list(map(str, argv)), check=check, text=True, capture_output=capture, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f'timed out after {timeout}s', flush=True)
        return subprocess.CompletedProcess(argv, 124, '', '')


def build_job(commit, profiles, workload, rounds, install_fla=False, gdn_check=False, tests=(), steps=()):
    """One Python script: the notebook's setup cells, optional pre-steps, stages, scorecard."""
    nb = json.loads(NOTEBOOK.read_text())
    cells = [''.join(c['source']) for c in nb['cells'][:SETUP_CELLS] if c['cell_type'] == 'code']
    marker = "COMMIT = '"
    start = cells[0].index(marker) + len(marker)
    cells[0] = cells[0][:start] + commit + cells[0][cells[0].index("'", start):]
    pre = ''
    if tests:   # engine tests need torch + transformers (runtime only)
        pre += ("assert logged([PY,'-m','pip','install','-q','pytest>=8','pytest-asyncio>=0.24'],'install-pytest.log',"
                "env=ENV,timeout=900)==0\n"
                "def run_tests(paths, log, *extra):\n"
                "    t = logged([PY,'-m','pytest','-q','-rs','-p','no:cacheprovider',*extra,*paths],log,env=ENV,timeout=3600)\n"
                "    print(log, 'exit:', t, flush=True)\n"
                "    if t != 0:   # archive the failure so the driver can fetch it and stop the VM\n"
                "        import shutil as _sh\n"
                f"        os.replace(_sh.make_archive('/content/job-evidence-tmp','zip',root_dir=OUT.parent,base_dir=OUT.name),{ARCHIVE!r})\n"
                "        raise SystemExit('engine tests failed: see '+log)\n"
                # before FLA: once it is installed, transformers routes even CPU test tensors to it
                f"run_tests({list(tests)!r}, 'pytest.log')\n")
    if install_fla:
        pre += ("assert logged([PY,'-m','pip','install','--no-deps','fla-core==0.6.0','flash-linear-attention==0.6.0',"
                "'einops'],'install-fla.log',env=ENV,timeout=1800)==0\n")
        if tests:   # the CUDA FLA-vs-oracle check only runs once FLA is present
            pre += "run_tests(['tests/test_gdn_primitive.py'], 'pytest-fla.log', '-k', 'fla')\n"
    if gdn_check:
        pre += ("print('gdn backend check exit:', logged([PY,'-u','scripts/gdn_backend_check.py','--out',"
                "str(OUT/'gdn-backend-check.json')],'gdn-backend-check.log',env=ENV,timeout=3600), flush=True)\n")
    for i, step in enumerate(steps):   # diagnosis scripts: logged, recorded, not fatal
        argv = shlex.split(step)
        pre += (f"print('step {i} exit:', logged([PY,'-u',*[a.replace('{{OUT}}',str(OUT)) for a in {argv!r}]],"
                f"'step-{i}.log',env=ENV,timeout=7200), flush=True)\n")   # {OUT}: the evidence directory
    if not profiles:   # diagnosis only: archive and stop, no scorecard
        return '\n\n'.join(cells + [pre, f'''
import shutil as _sh
os.replace(_sh.make_archive('/content/job-evidence-tmp','zip',root_dir=OUT.parent,base_dir=OUT.name),{ARCHIVE!r})
print('Evidence archive:', {ARCHIVE!r}, flush=True)
'''])
    workload_arg = '' if workload is None else (
        f"(OUT/'workload.json').write_text({json.dumps(json.dumps(workload))})\n")
    wl = "[] if not (OUT/'workload.json').exists() else ['--workload',str(OUT/'workload.json')]"
    tail = f'''
ONLY = {list(profiles)!r}
for p in ONLY:
    stage(p, engine=p) if p in {RIVALS!r} else stage(p)
only = [p for p in ONLY if STAGE.get(p) == 'passed']
print('profiles:', only, '| stages:', STAGE, flush=True)
assert only, 'no profile passed its smoke test'
{workload_arg}require_empty_gpu()
code = logged([PY,'-u','scripts/engine_scorecard.py','run','--profiles',PROFILES,*({wl}),
               '--out',str((OUT/'scorecard').relative_to(REPO)),'--rounds','{rounds}',
               '--run-id',f'{{GPU_LABEL}}-{{int(time.time())}}','--only',*only,
               '--ready-timeout',str(READY_TIMEOUT)], 'scorecard-runner.log', env=SCORE_ENV, timeout=12*60*60)
print('scorecard exit:', code, flush=True)
(OUT/'stages.json').write_text(json.dumps(STAGE, indent=2))
import shutil as _sh
tmp = _sh.make_archive('/content/job-evidence-tmp', 'zip', root_dir=OUT.parent, base_dir=OUT.name)
os.replace(tmp, {ARCHIVE!r})   # atomic: the archive only appears when complete
print('Evidence archive:', {ARCHIVE!r}, flush=True)
'''
    return '\n\n'.join(cells + [pre, tail])


LAUNCH = f'''
import subprocess, sys, os
if os.path.exists('/content/job.pid'):
    pid = int(open('/content/job.pid').read())
    if os.path.exists(f'/proc/{{pid}}'):
        print('job already running, pid', pid); raise SystemExit(0)
for f in ({ARCHIVE!r}, {JOB_LOG!r}):
    if os.path.exists(f): os.remove(f)
p = subprocess.Popen([sys.executable, '-u', '/content/job.py'], stdout=open({JOB_LOG!r}, 'w'),
                     stderr=subprocess.STDOUT, cwd='/content', start_new_session=True)
open('/content/job.pid', 'w').write(str(p.pid)); print('job pid', p.pid)
'''

POLL = f'''
import os, json
pid = int(open('/content/job.pid').read())
alive = os.path.exists(f'/proc/{{pid}}') and open(f'/proc/{{pid}}/stat').read().split()[2] != 'Z'
tail = open({JOB_LOG!r}).read()[-1500:] if os.path.exists({JOB_LOG!r}) else ''
print('POLL ' + json.dumps({{'alive': alive, 'archive': os.path.exists({ARCHIVE!r}), 'tail': tail}}))
'''


def retry(*argv, tries=5, wait=20, timeout=900):
    """colab CLI steps are idempotent here; retry them through network drops and hangs."""
    for i in range(tries):
        if sh(*argv, check=False, timeout=timeout).returncode == 0:
            return
        print(f'retry {i + 1}/{tries} after failure', flush=True)
        time.sleep(wait)
    raise SystemExit(f'failed after {tries} tries: {argv}')


def poll(colab, session, script):
    try:
        r = subprocess.run([colab, 'exec', '-s', session, '-f', str(script), '--timeout', '120'],
                           capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        return None
    line = next((x for x in r.stdout.splitlines() if x.startswith('POLL ')), None)
    return json.loads(line[5:]) if line else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profiles', nargs='*', default=[], help='profile names from the scorecard JSON (none: steps only)')
    ap.add_argument('--workload', default=str(ROOT / 'configs/scorecard/workload-fast.json'),
                    help="workload JSON, or 'default' for the full scorecard workload")
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--gpu', default='L4')
    ap.add_argument('--session', default=f'run-{int(time.time())}')
    ap.add_argument('--out', default=str(ROOT / 'results/fast-check'))
    ap.add_argument('--poll-s', type=float, default=120)
    ap.add_argument('--keep', action='store_true', help='leave the runtime running afterwards')
    ap.add_argument('--install-fla', action='store_true', help='pip install pinned flash-linear-attention (--no-deps)')
    ap.add_argument('--gdn-check', action='store_true', help='run scripts/gdn_backend_check.py before the profiles')
    ap.add_argument('--step', action='append', default=[], help="script + args run on the runtime, e.g. \"scripts/x.py --out OUT/x.json\"")
    ap.add_argument('--tests', nargs='*', default=(), help='pytest paths to run on the runtime first (job stops on failure)')
    args = ap.parse_args()
    colab = shutil.which('colab') or str(Path.home() / '.local/bin/colab')
    if sh('git', 'status', '--porcelain', '--untracked-files=no', capture=True).stdout.strip():
        sys.exit('commit tracked changes first: the runtime gets HEAD, not the working tree')
    commit = sh('git', 'rev-parse', 'HEAD', capture=True).stdout.strip()
    branch = sh('git', 'rev-parse', '--abbrev-ref', 'HEAD', capture=True).stdout.strip()
    workload = None if args.workload == 'default' else json.loads(Path(args.workload).read_text())
    out = Path(args.out) / f'{args.session}-{commit[:7]}'
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='colab-run-'))
    bundle, job, prep, launch, pollf = (tmp / n for n in ('inferstack.bundle', 'job.py', 'prep.py', 'launch.py',
                                                          'poll.py'))
    sh('git', 'bundle', 'create', bundle, branch)
    job.write_text(build_job(commit, args.profiles, workload, args.rounds, args.install_fla, args.gdn_check, args.tests, args.step))
    prep.write_text(
        'import subprocess as sp\n'
        f'r=sp.run("cd /content && rm -rf inferstack && git clone -q -b {branch} inferstack.bundle inferstack '
        f'&& git -C inferstack cat-file -t {commit}", shell=True, capture_output=True, text=True)\n'
        'print(r.stdout, r.stderr); assert r.returncode == 0\n')
    launch.write_text(LAUNCH)
    pollf.write_text(POLL)
    (out / 'job.py').write_text(job.read_text())
    have = sh(colab, 'sessions', capture=True, check=False, timeout=120).stdout
    if f'[{args.session}]' not in have:
        retry(colab, 'new', '--gpu', args.gpu, '-s', args.session)
    retry(colab, 'upload', '-s', args.session, bundle, '/content/inferstack.bundle')
    retry(colab, 'upload', '-s', args.session, job, '/content/job.py')
    retry(colab, 'exec', '-s', args.session, '-f', prep, '--timeout', '300')
    retry(colab, 'exec', '-s', args.session, '-f', launch, '--timeout', '120')
    state, misses = None, 0
    while True:
        time.sleep(args.poll_s)
        state = poll(colab, args.session, pollf)
        if state is None:          # network or CLI hiccup: the job keeps running on the VM
            misses += 1
            print(f'poll failed ({misses}); retrying', flush=True)
            continue
        misses = 0
        (out / 'job.log.tail').write_text(state['tail'])
        last = [x for x in state['tail'].splitlines() if x.strip()][-1:] or ['']
        print(time.strftime('%H:%M:%S'), 'alive' if state['alive'] else 'exited',
              '| archive' if state['archive'] else '', '|', last[0][:160], flush=True)
        if state['archive'] or not state['alive']:
            break
    fetched = False
    if state and state['archive']:
        for attempt in range(5):
            if sh(colab, 'download', '-s', args.session, ARCHIVE, out / 'evidence.zip', check=False,
                  timeout=900).returncode == 0:
                fetched = True
                break
            time.sleep(30)
    sh(colab, 'download', '-s', args.session, JOB_LOG, out / 'job.log', check=False, timeout=300)
    if fetched:
        shutil.unpack_archive(out / 'evidence.zip', out)
        for md in sorted(out.glob('*/scorecard/*/scorecard.md')):
            print(md.read_text())
        if not args.keep:
            sh(colab, 'stop', '-s', args.session, check=False)
    else:
        print(f'NO EVIDENCE DOWNLOADED: runtime {args.session!r} left running. Fetch with\n'
              f'  colab download -s {args.session} {ARCHIVE} {out / "evidence.zip"}\n'
              f'then: colab stop -s {args.session}', flush=True)
    print('evidence:', out)
    sys.exit(0 if fetched else 2)


if __name__ == '__main__':
    main()
