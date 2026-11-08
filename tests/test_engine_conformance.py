"""The engine conformance runner's own logic, against an emulated `vllm` executable (not GPU evidence)."""
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
REV = 'b' * 40


def test_conformance_runner_end_to_end_with_emulated_engine(tmp_path):
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    shim = bindir / 'vllm'
    shim.write_text(f'''#!/bin/sh
# emulated "vllm serve MODEL ... --port P": forwards to the fake OpenAI server in vLLM flavor
model="$2"; shift 2
exec "{sys.executable}" "{ROOT / 'tests/fixtures/fake_openai_server.py'}" --flavor vllm --model "$model" "$@"
''')
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    out = tmp_path / 'report.json'
    env = {**os.environ, 'PATH': f'{bindir}{os.pathsep}{os.environ["PATH"]}'}
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/engine_conformance.py'), '--engine', 'vllm',
                           '--model', 'm', '--revision', REV, '--out', str(out), '--ready-timeout', '30'],
                          capture_output=True, text=True, env=env, timeout=120)
    report = json.loads(out.read_text())
    checks = report['checks']
    assert checks['ready'] and checks['revision_in_running_argv'] and checks['stopped_cleanly'], report
    assert checks['deterministic_completion'] and checks['usage_reported'] and checks['live_after_abandoned_stream']
    # Families the emulator does not export are reported, and fail the run: missing telemetry is not a pass.
    assert report['pass'] is (not report.get('metric_families_missing')) and (proc.returncode == 0) is report['pass']
