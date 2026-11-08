#!/usr/bin/env python
"""InferStack v2 consolidated qualification run (GPU host, e.g. Colab A100).

Runs every gate that this environment can run and records each as passed,
failed or skipped (with the reason). The overall verdict is `qualified` only
when every REQUIRED gate passed; any required skip or failure yields
`not_qualified`. Optional gates (vLLM conformance, live providers, PostgreSQL)
never upgrade or hide a required result.

  python scripts/v2_qualify.py --out results/v2-qualification
  # optional: DEEPSEEK_API_KEY/OPENROUTER_API_KEY + --deepseek-model/--openrouter-model
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ('environment', 'model_identity', 'test_suite', 'kernel_equivalence', 'native_smoke')
OPTIONAL = ('vllm_conformance', 'sglang_conformance', 'provider_deepseek', 'provider_openrouter', 'postgresql',
            'docker', 'kubernetes')
PINNED_MODEL = ('Qwen/Qwen3.5-2B-Base', 'b1485b2fa6dfa1287294f269f5fb618e03d52d7c')


def gpu_profile(gpus):
    """T4 and A100 are separate qualification profiles; results never transfer between them."""
    names = ' '.join(g.get('name', '') for g in gpus).upper()
    for key in ('H100', 'A100', 'L4', 'T4', 'A10'):
        if key in names:
            return key.lower()
    return 'other' if gpus else 'cpu'
# Every check the kernel-equivalence script must report, per seed. A report with a
# missing, extra-empty or non-finite check is not evidence of equivalence.
KERNEL_SEEDS = (123, 7, 2026)
KERNEL_CHECKS = ('rmsnorm', 'add_rmsnorm.sum', 'add_rmsnorm.norm', 'gated_rmsnorm', 'silu_mul', 'beta_g.beta',
                 'beta_g.g', 'grouped_gdn.output', 'grouped_gdn.state')
EXPECTED_KERNEL_CHECKS = frozenset(f'{c}/seed{s}' for s in KERNEL_SEEDS for c in KERNEL_CHECKS)
# Skip reasons that are acceptable on a GPU qualification host; any other skip
# means required coverage did not run there.
ACCEPTED_SKIP_MARKERS = ('PostgreSQL', 'initdb', 'pg_ctl', 'INFERSTACK_TEST_PG')


def run(argv, log, timeout, env=None):
    started = time.time()
    try:
        proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, 'PYTHONPATH': str(ROOT / 'src'), **(env or {})})
        code, out = proc.returncode, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        code, out = 124, f'timeout after {timeout}s\n{exc.stdout or ""}{exc.stderr or ""}'
    log.write_text(f'$ {" ".join(map(str, argv))}\nexit {code}\n\n{out}')
    return code, out, round(time.time() - started, 1)


def gate(status, **detail):
    return {'status': status, **detail}


def kernel_gate(code, text):
    """Kernel equivalence passes only with exit 0, the complete check manifest and finite passing results."""
    try:
        report = json.loads(text[text.index('{'):text.rindex('}') + 1])
        checks = report['checks']
    except (ValueError, KeyError, TypeError):
        return gate('failed', reason='no report produced', exit=code), None
    import math
    names = [c.get('check') for c in checks if isinstance(c, dict)]
    missing = sorted(EXPECTED_KERNEL_CHECKS - set(names))
    unexpected = sorted(set(names) - EXPECTED_KERNEL_CHECKS)
    duplicate = sorted({n for n in names if names.count(n) > 1})
    def finite(c):
        keys = [k for k in ('max_ulp', 'mean_ulp', 'max_rel', 'mean_rel') if k in c]
        return bool(keys) and all(isinstance(c[k], (int, float)) and math.isfinite(c[k]) for k in keys)
    nonfinite = [c.get('check') for c in checks if isinstance(c, dict) and not finite(c)]
    failed_checks = [c.get('check') for c in checks if not (isinstance(c, dict) and c.get('pass') is True)]
    reasons = []
    if code != 0:
        reasons.append(f'kernel process exited {code}')
    if missing:
        reasons.append('missing checks')
    if unexpected or duplicate:
        reasons.append('unexpected or duplicate checks')
    if nonfinite:
        reasons.append('non-finite error statistics')
    if failed_checks or report.get('pass') is not True:
        reasons.append('checks failed')
    status = 'failed' if reasons else 'passed'
    return gate(status, exit=code, reasons=reasons, failed=failed_checks, missing=missing, unexpected=unexpected,
                duplicate=duplicate, nonfinite=nonfinite, method=report.get('method'), checks=len(checks)), report


def suite_gate(code, text, has_gpu):
    summary = [l for l in text.splitlines() if ' passed' in l or ' failed' in l or ' error' in l][-1:]
    skips = [l for l in text.splitlines() if l.startswith('SKIPPED')]
    accepted = [l for l in skips if any(m in l for m in ACCEPTED_SKIP_MARKERS)]
    blocking = [l for l in skips if l not in accepted] if has_gpu else []
    reasons = ([f'pytest exited {code}'] if code != 0 else []) + \
        (['required coverage skipped on GPU host'] if blocking else [])
    return gate('failed' if reasons else 'passed', reasons=reasons, summary=summary[0] if summary else None,
                skipped_accepted=accepted[:50], skipped_blocking=blocking[:50])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='results/v2-qualification')
    p.add_argument('--model-repo', default='Qwen/Qwen3.5-2B-Base')
    p.add_argument('--deepseek-model')
    p.add_argument('--openrouter-model')
    p.add_argument('--require', action='append', default=[], choices=OPTIONAL,
                   help='make an optional integration gate required for this deployment (repeatable); '
                        'a required gate that is skipped blocks qualification')
    a = p.parse_args()
    required = tuple(REQUIRED) + tuple(g for g in a.require if g not in REQUIRED)
    out = (ROOT / a.out)
    out.mkdir(parents=True, exist_ok=True)
    gates = {}

    from inferstack.infrastructure.inventory import collect
    hardware, notes = collect(provider=os.environ.get('INFERSTACK_PROVIDER', 'attached'))
    (out / 'hardware.json').write_text(json.dumps({'hardware': hardware, 'notes': notes}, indent=2))
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    gates['environment'] = gate('passed' if hardware['gpus'] else 'failed', commit=commit or 'unknown',
                                gpus=[g['name'] for g in hardware['gpus']], notes=notes,
                                libraries=hardware['libraries'])

    code, text, secs = run([sys.executable, '-m', 'pytest', '-rs', '-p', 'no:cacheprovider'],
                           out / 'pytest.log', 3600)
    gates['test_suite'] = suite_gate(code, text, bool(hardware['gpus'])) | {'seconds': secs}

    code, text, secs = run([sys.executable, 'scripts/v160_kernel_equivalence.py'], out / 'kernel-equivalence.log', 1800)
    gates['kernel_equivalence'], report = kernel_gate(code, text)
    if report is not None:
        (out / 'kernel-equivalence.json').write_text(json.dumps(report, indent=2))

    manifest_path = ROOT / 'examples/model_foundry/factorylm-2b/candidate/manifest.json'
    from inferstack.contracts.records import MUTABLE_REVISIONS
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    revisions = {part: (manifest.get(part) or {}).get('revision') for part in ('source', 'tokenizer')}
    import re
    mutable = {k: v for k, v in revisions.items() if not v or v in MUTABLE_REVISIONS or not re.fullmatch(r'[0-9a-f]{40}', v)}
    gates['model_identity'] = gate('failed' if mutable or not revisions else 'passed', manifest=str(manifest_path),
                                   revisions=revisions, mutable=mutable,
                                   fix='pin the Foundry manifest to a 40-hex Hugging Face commit' if mutable else None)

    if not hardware['gpus']:
        gates['native_smoke'] = gate('skipped', reason='no GPU')
    else:
        code, text, secs = run([sys.executable, 'scripts/v160_colab_matrix.py', '--results',
                                str((out / 'matrix').relative_to(ROOT)), '--skip-operator-profile'],
                               out / 'matrix.log', 7200)
        analysis = out / 'matrix' / 'analysis.json'
        lanes = {}
        if analysis.exists():
            lanes = {l['lane']: l['verdict']['decision'] for l in json.loads(analysis.read_text())['ablations_c8']}
        gates['native_smoke'] = gate('passed' if code == 0 else 'failed', liveness=code == 0, seconds=secs,
                                     lane_verdicts=lanes, note='liveness only; performance claims need lane keep '
                                     'verdicts from repeated runs')

    import importlib.util
    for engine, module in (('vllm', 'vllm'), ('sglang', 'sglang')):
        key = f'{engine}_conformance'
        if importlib.util.find_spec(module) is None:
            gates[key] = gate('skipped', requirement=f'{module} installed', reason=f'{module} not installed')
            continue
        if not hardware['gpus']:
            gates[key] = gate('skipped', requirement='GPU', reason='no GPU on this host')
            continue
        code, text, secs = run([sys.executable, 'scripts/engine_conformance.py', '--engine', engine,
                                '--model', PINNED_MODEL[0], '--revision', PINNED_MODEL[1],
                                '--out', str((out / f'conformance-{engine}.json').relative_to(ROOT))],
                               out / f'conformance-{engine}.runner.log', 3600)
        detail = {}
        try:
            detail = json.loads((out / f'conformance-{engine}.json').read_text())
        except (OSError, ValueError):
            pass
        gates[key] = gate('passed' if code == 0 and detail.get('pass') is True else 'failed', seconds=secs,
                          checks=detail.get('checks'), error=detail.get('error'))

    for key, test, requirement, env_key in (
            ('postgresql', ['tests/test_control_store.py', 'tests/test_execution_store.py'],
             'INFERSTACK_TEST_POSTGRES_URL', 'INFERSTACK_TEST_POSTGRES_URL'),
            ('docker', ['tests/test_docker_live.py'], 'Docker daemon with a local pinned image', None),
            ('kubernetes', ['tests/test_kubernetes_live.py'], 'INFERSTACK_TEST_K8S_CONTEXT', 'INFERSTACK_TEST_K8S_CONTEXT')):
        if env_key and not os.environ.get(env_key):
            gates[key] = gate('skipped', requirement=requirement, reason=f'{env_key} not set')
            continue
        code, text, secs = run([sys.executable, '-m', 'pytest', '-o', 'addopts=', '-q', '-rs', '-p', 'no:cacheprovider',
                                *test], out / f'{key}.log', 3600)
        skipped = [l for l in text.splitlines() if l.startswith('SKIPPED')]
        status = 'failed' if code != 0 else 'skipped' if skipped and ' passed' not in text else 'passed'
        gates[key] = gate(status, seconds=secs, summary=(text.strip().splitlines() or [''])[-1],
                          skipped=skipped[:10] or None, requirement=requirement)

    for kind, model, env in (('deepseek', a.deepseek_model, 'DEEPSEEK_API_KEY'),
                             ('openrouter', a.openrouter_model, 'OPENROUTER_API_KEY')):
        if not model or not os.environ.get(env):
            gates[f'provider_{kind}'] = gate('skipped', reason=f'--{kind}-model and {env} required')
            continue
        code, text, secs = run([sys.executable, 'scripts/provider_smoke.py', '--kind', kind, '--model', model,
                                '--key-env', env], out / f'provider-{kind}.log', 600)
        gates[f'provider_{kind}'] = gate('passed' if code == 0 else 'failed', seconds=secs)

    required_failed = [g for g in required if gates[g]['status'] != 'passed']
    from inferstack import __version__
    result = {'version': __version__, 'commit': commit, 'finished_at': time.time(), 'required': list(required),
              'gpu_profile': gpu_profile(hardware['gpus']),
              'optional_not_required': [g for g in OPTIONAL if g not in required],
              'verdict': 'qualified' if not required_failed else 'not_qualified',
              'blocking': required_failed, 'gates': gates}
    (out / 'qualification.json').write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: result[k] for k in ('verdict', 'blocking')} |
                     {'gates': {k: v['status'] for k, v in gates.items()}}, indent=2))
    raise SystemExit(0 if result['verdict'] == 'qualified' else 2)


if __name__ == '__main__':
    main()
