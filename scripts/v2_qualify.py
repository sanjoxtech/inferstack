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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='results/v2-qualification')
    p.add_argument('--model-repo', default='Qwen/Qwen3.5-2B-Base')
    p.add_argument('--deepseek-model')
    p.add_argument('--openrouter-model')
    a = p.parse_args()
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
    summary = [l for l in text.splitlines() if ' passed' in l or ' failed' in l or ' error' in l][-1:]
    skips = [l for l in text.splitlines() if l.startswith('SKIPPED')]
    gates['test_suite'] = gate('passed' if code == 0 else 'failed', summary=summary[0] if summary else None,
                               skipped=skips[:50], seconds=secs)

    code, text, secs = run([sys.executable, 'scripts/v160_kernel_equivalence.py'], out / 'kernel-equivalence.log', 1800)
    try:
        report = json.loads(text[text.index('{'):text.rindex('}') + 1])
        gates['kernel_equivalence'] = gate('passed' if report['pass'] else 'failed', failed=report['failed'],
                                           method=report['method'], checks=len(report['checks']))
        (out / 'kernel-equivalence.json').write_text(json.dumps(report, indent=2))
    except (ValueError, KeyError):
        gates['kernel_equivalence'] = gate('failed', reason='no report produced', exit=code)

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

    try:
        import vllm  # noqa: F401
        gates['vllm_conformance'] = gate('skipped', reason='vllm importable; launch-and-attest conformance must be '
                                         'run with a pinned model revision (see COLAB_V2_QUALIFICATION.md)')
    except ImportError:
        gates['vllm_conformance'] = gate('skipped', reason='vllm not installed')

    for kind, model, env in (('deepseek', a.deepseek_model, 'DEEPSEEK_API_KEY'),
                             ('openrouter', a.openrouter_model, 'OPENROUTER_API_KEY')):
        if not model or not os.environ.get(env):
            gates[f'provider_{kind}'] = gate('skipped', reason=f'--{kind}-model and {env} required')
            continue
        code, text, secs = run([sys.executable, 'scripts/provider_smoke.py', '--kind', kind, '--model', model,
                                '--key-env', env], out / f'provider-{kind}.log', 600)
        gates[f'provider_{kind}'] = gate('passed' if code == 0 else 'failed', seconds=secs)

    required_failed = [g for g in REQUIRED if gates[g]['status'] != 'passed']
    result = {'version': '2.0.0rc1', 'commit': commit, 'finished_at': time.time(), 'required': list(REQUIRED),
              'verdict': 'qualified' if not required_failed else 'not_qualified',
              'blocking': required_failed, 'gates': gates}
    (out / 'qualification.json').write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: result[k] for k in ('verdict', 'blocking')} |
                     {'gates': {k: v['status'] for k, v in gates.items()}}, indent=2))
    raise SystemExit(0 if result['verdict'] == 'qualified' else 2)


if __name__ == '__main__':
    main()
