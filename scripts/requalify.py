"""Re-evaluate an existing qualification.json with CI sandbox evidence for the same commit.

    python scripts/ci_evidence.py --commit <sha> --out ci.json
    python scripts/requalify.py path/to/qualification.json --sandbox-evidence ci.json

Writes `qualification-with-ci-evidence.json` next to the input; the original file is
never modified. The rule (v2_qualify.apply_sandbox_evidence) only accepts evidence for
the same commit, with the CI suite jobs successful, for a test_suite gate blocked
solely by sandbox skips.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('qualification', type=Path)
    p.add_argument('--sandbox-evidence', type=Path, required=True)
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location('v2_qualify', ROOT / 'scripts/v2_qualify.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = json.loads(a.qualification.read_text())
    result = module.finalize(original, json.loads(a.sandbox_evidence.read_text()))
    result['amended_from'] = {'file': a.qualification.name, 'verdict': original['verdict'],
                              'blocking': original['blocking']}
    target = a.qualification.with_name('qualification-with-ci-evidence.json')
    target.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({'verdict': result['verdict'], 'blocking': result['blocking'],
                      'test_suite': result['gates']['test_suite'].get('sandbox_evidence')}, indent=2))
    raise SystemExit(0 if result['verdict'] == 'qualified' else 2)


if __name__ == '__main__':
    main()
