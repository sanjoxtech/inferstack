#!/usr/bin/env python
"""G2: one A/B experiment on the GPU host itself: baseline vs candidate, interleaved, open-loop goodput.

Runs where the GPU is (Colab VM, rented box, this machine), because latency must be measured next to the
engine. Each trial launches one side fresh, waits until ready, checks a few greedy completions, runs the
goodput sweep (inferstack.evaluation.goodput), and stops the server. Order alternates per repeat (AB, BA,
AB, ...), so drift on the host hits both sides alike. The harness judges ab.json with its own rules
(optimization/remote_verify.py); this script only measures.

The candidate is a named profile, the baseline plus an allowlisted native profile patch
(engines.profiles.native_overlay), or the baseline run on patched source (G7: `--candidate-diff-b64`, a
unified diff confined to the patchable roots, applied to a pristine export of `--source-revision` with tests
and the evaluator restored; the candidate server gets that tree first on PYTHONPATH). Prompts and seeds are fixed, so both sides get identical traffic.

  python scripts/remote_ab.py --profiles configs/scorecard/a100-qwen3.5-2b.json --baseline native_lazy \\
      --patch '{"static_decode_admit_every_steps": 3}' --repeats 3 --out OUT/ab
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / 'src'))
import engine_scorecard as es  # noqa: E402

from inferstack.engines.managed import DeploymentSpec, ManagedEngine  # noqa: E402
from inferstack.evaluation.goodput import GoodputCase, cases_from_slo, sweep  # noqa: E402

CHECK_PROMPTS = ('The capital of France is', 'def fibonacci(n):', '2 + 2 =', 'List three primary colors:')


def side_profile(profiles, name, patch, workdir):
    p = dict(next(x for x in profiles if x['name'] == name))
    if patch:
        from inferstack.engines.profiles import native_overlay
        if p.get('engine') != 'inferstack_native':
            raise SystemExit('profile patches are only defined for inferstack_native profiles')
        argv, path = native_overlay(p['argv'], patch, workdir, source_root=ROOT)
        p = {**p, 'name': f'{name}+patch', 'argv': argv, 'overlay': str(path)}
    return p


def source_side(profile, diff, source_root, revision, workdir):
    """The baseline profile served from a patched export of the pinned source (G7)."""
    import os
    import shutil

    from inferstack.execution.patches import check_patch, materialize_source, touched_paths
    check_patch(diff, touched_paths(diff))                    # confined to the patchable roots, again, on the host
    target = Path(workdir) / 'candidate-src'
    shutil.rmtree(target, ignore_errors=True)        # one diff per A/B: a rerun in the same dir starts pristine
    tree = materialize_source(source_root, revision, diff, target)
    env = dict(profile.get('env') or {})
    env['PYTHONPATH'] = os.pathsep.join(x for x in (str(tree / 'src'), env.get('PYTHONPATH')
                                                    or os.environ.get('PYTHONPATH')) if x)
    return {**profile, 'name': f"{profile['name']}+source", 'env': env, 'source_tree': str(tree)}


def source_tests(profile, tests, timeout=1800):
    """G7: the controller's tests (restored from the pinned revision) against the patched tree, on the GPU host,
    before any A/B time. A kernel bug that CPU-only validation skipped fails here in seconds."""
    import importlib.util
    import os
    tree = Path(profile['source_tree'])
    if importlib.util.find_spec('pytest') is None:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pytest>=8', 'pytest-asyncio>=0.24'],
                       check=False)
    env = {**os.environ, **(profile.get('env') or {})}
    try:
        r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *tests], cwd=tree,
                           env=env, capture_output=True, text=True, timeout=timeout, check=False)
        code, text = r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        code, text = 124, f'timed out after {timeout}s'
    result = {'tests': list(tests), 'exit_code': code, 'passed': code == 0, 'tail': text[-3000:]}
    print(json.dumps({k: result[k] for k in ('tests', 'exit_code', 'passed')}), flush=True)
    return result


async def greedy_check(base, model):
    out = []
    async with es.httpx.AsyncClient(timeout=120) as client:
        for prompt in CHECK_PROMPTS:
            try:
                r = await client.post(base + '/v1/completions', json={'model': model, 'prompt': prompt,
                                                                      'max_tokens': 16, 'temperature': 0})
                r.raise_for_status()
                out.append(r.json()['choices'][0]['text'])
            except Exception as exc:  # noqa: BLE001 - recorded, fails the correctness gate
                out.append(None)
                print('check failed:', type(exc).__name__, str(exc)[:200], flush=True)
    return out


async def trial(profile, side, repeat, sweep_args, out, ready_timeout):
    port = es.free_port()
    base = f'http://127.0.0.1:{port}'
    proc, log, argv = es.launch(profile, port, out / f'{side}-r{repeat}.server.log')
    row = {'side': side, 'repeat': repeat, 'profile': profile['name'], 'argv': argv}
    t0 = time.monotonic()
    try:
        ready, detail = await es.wait_ready(base, profile['model'], proc, ready_timeout)
        row.update(ready=ready, load_s=round(time.monotonic() - t0, 2))
        if not ready:
            row['error'] = str(detail)[:500]
            return row
        row['check'] = await greedy_check(base, profile['model'])
        engine = ManagedEngine(DeploymentSpec(id=f'{side}-{repeat}', engine='generic_openai',
                                              model=profile['model'], base_url=base))
        try:
            # G4: native engines expose cumulative counters; the sweep stores their change per rate step
            probe = (lambda: es.ttft_breakdown(base)) if profile.get('engine') == 'inferstack_native' else None
            row['goodput'] = await sweep(engine.stream, **sweep_args, probe=probe)
        finally:
            await engine.client.aclose()
    finally:
        row['exit_code'] = es.stop(proc)
        log.close()
    print(json.dumps({k: row.get(k) for k in ('side', 'repeat', 'ready', 'load_s')} |
                     {'goodput99_rps': (row.get('goodput') or {}).get('goodput99_rps')}), flush=True)
    return row


def gpu_name():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True,
                              text=True, timeout=30).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001 - CPU host (tests)
        return None


async def run(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    profiles = json.loads(Path(a.profiles).read_text())['profiles']
    patch = json.loads(a.patch) if a.patch else None
    diff = base64.b64decode(a.candidate_diff_b64).decode() if a.candidate_diff_b64 else None
    if sum(map(bool, (a.candidate, patch, diff))) != 1:
        raise SystemExit('give exactly one of --candidate, --patch or --candidate-diff-b64')
    sides = {'baseline': side_profile(profiles, a.baseline, None, out)}
    if diff:
        revision = a.source_revision or subprocess.run(['git', '-C', a.source_root, 'rev-parse', 'HEAD'],
                                                       capture_output=True, text=True).stdout.strip()
        sides['candidate'] = source_side(sides['baseline'], diff, a.source_root, revision, out)
    else:
        sides['candidate'] = side_profile(profiles, a.candidate or a.baseline, patch, out)
    tested = source_tests(sides['candidate'], a.source_tests) if diff and a.source_tests else None
    workload = json.loads(Path(a.workload).read_text())
    cases = [c for c in workload['cases'] if not a.cases or c['name'] in a.cases]
    cases = cases_from_slo(cases, json.loads(Path(a.slo).read_text())) if a.slo else \
        [GoodputCase(**c) for c in cases]
    sweep_args = {'cases': cases, 'rates': tuple(a.rates), 'seconds_per_rate': a.seconds_per_rate,
                  'min_requests': a.min_requests, 'warmup': a.warmup}
    rows = []
    for repeat in range(a.repeats if tested is None or tested['passed'] else 0):   # failed tests: no A/B time
        for side in (('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')):
            rows.append(await trial(sides[side], side, repeat, sweep_args, out, a.ready_timeout))
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    result = {'version': 1, 'commit': commit or None, 'host': platform.node(), 'gpu': gpu_name(),
              'patch': patch, 'sides': sides,
              'source_diff_sha256': hashlib.sha256(diff.encode()).hexdigest() if diff else None,
              'source_tests': tested,
              'repeats': a.repeats, 'order': 'alternating AB/BA per repeat',
              'sweep': {**sweep_args, 'cases': [c.__dict__ for c in cases]}, 'trials': rows}
    (out / 'ab.json').write_text(json.dumps(result, indent=2, default=str))
    print('ab.json written:', out / 'ab.json', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profiles', required=True, help='scorecard-style profiles JSON')
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--candidate', help='candidate profile name')
    ap.add_argument('--patch', help='JSON native profile patch applied to the baseline')
    ap.add_argument('--candidate-diff-b64', help='base64 unified diff: candidate = baseline on patched source')
    ap.add_argument('--source-root', default=str(ROOT), help='git checkout the diff applies to')
    ap.add_argument('--source-revision', help='pinned revision of --source-root (default HEAD)')
    ap.add_argument('--source-tests', nargs='+', default=(),
                    help='pytest selectors run against the patched tree before the A/B (with --candidate-diff-b64)')
    ap.add_argument('--workload', default=str(ROOT / 'configs/scorecard/workload-fast.json'))
    ap.add_argument('--cases', nargs='*', help='workload case names (default: all)')
    ap.add_argument('--slo', default=str(ROOT / 'configs/tune/slo-v2-l4.json'),
                    help="SLO JSON with industry_limits; '' to read limits from the workload cases")
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--rates', type=float, nargs='+', default=[1, 2, 4, 6, 8, 12, 16])
    ap.add_argument('--seconds-per-rate', type=float, default=20.0)
    ap.add_argument('--min-requests', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--ready-timeout', type=float, default=1200)
    ap.add_argument('--out', required=True)
    asyncio.run(run(ap.parse_args()))


if __name__ == '__main__':
    main()
