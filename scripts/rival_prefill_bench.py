#!/usr/bin/env python
"""P5 rival study: SGLang's prefill latency on the same shapes as packed_prefill_profile.py.

Installs SGLang (pinned) into /content/venv-sglang if it is missing, then runs SGLang's own
offline single-batch benchmark (python -m sglang.benchmark.one_batch, no HTTP, no scheduler) for
each (batch, input_len) shape with output_len 2, and records its reported prefill latency.
Compared with our prefill_packed timings for the same shapes this says whether the scorecard's
prefill_heavy gap is per-token prefill cost (kernel + host) or scheduling.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REV = 'b1485b2fa6dfa1287294f269f5fb618e03d52d7c'
SHAPES = [(1, 300), (1, 1200), (8, 300), (16, 1200)]


def run(cmd, log, timeout=3600):
    with open(log, 'a') as f:
        f.write('$ ' + ' '.join(map(str, cmd)) + '\n')
        f.flush()
        return subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--spec', default='sglang[all]==0.5.23')
    ap.add_argument('--venv', default='/content/venv-sglang')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix('.log')
    venv = Path(a.venv)
    py = venv / 'bin/python'
    res = {'spec': a.spec, 'shapes': []}
    if not py.exists():   # SGLang needs Python 3.12 on Colab (outlines_core has no 3.13 wheel)
        tools = Path('/content/venv-tools')
        if not (tools / 'bin/uv').exists():
            run([sys.executable, '-m', 'venv', tools], log)
            run([tools / 'bin/python', '-m', 'pip', 'install', '-q', 'uv'], log)
        run([tools / 'bin/uv', 'venv', '--seed', '--python', '3.12', venv], log)
        if run([py, '-m', 'pip', 'install', '-q', '--upgrade', 'pip', a.spec], log):
            res['error'] = 'install failed'
            out.write_text(json.dumps(res, indent=2) + '\n')
            return
    res['version'] = subprocess.run([py, '-c', 'import sglang;print(sglang.__version__)'],
                                    capture_output=True, text=True).stdout.strip()
    import os   # full environment (CUDA driver paths such as LD_LIBRARY_PATH), engine venv first on PATH
    env = {**os.environ, 'PATH': f"{venv / 'bin'}:{os.environ.get('PATH', '')}"}
    for bs, n in SHAPES:
        rf = out.parent / f'sglang-bench-{bs}x{n}.jsonl'
        text_log = out.parent / f'sglang-bench-{bs}x{n}.log'
        cmd = [py, '-m', 'sglang.benchmark.one_batch', '--model-path', 'Qwen/Qwen3.5-2B-Base', '--revision', REV,
               '--dtype', 'bfloat16', '--context-length', '8192', '--batch-size', bs, '--input-len', n,
               '--output-len', 2, '--run-name', f'{bs}x{n}', '--result-filename', rf]
        with open(text_log, 'w') as f:
            code = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, timeout=1800,
                                  env=env).returncode
        txt = text_log.read_text(errors='replace')
        lat = re.findall(r'Prefill\.\s+latency:\s+([0-9.]+)\s*s', txt)
        row = {'batch': bs, 'input_len': n, 'exit': code,
               'prefill_latency_ms_runs': [float(x) * 1e3 for x in lat],
               'tail': txt[-1500:] if code else ''}
        if rf.exists():
            row['result_lines'] = rf.read_text().splitlines()[-2:]
        print(json.dumps({k: v for k, v in row.items() if k != 'tail'}), flush=True)
        res['shapes'].append(row)
    out.write_text(json.dumps(res, indent=2) + '\n')


if __name__ == '__main__':
    main()
