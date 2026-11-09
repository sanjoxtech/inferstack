"""CI sandbox evidence may complete a GPU qualification only under strict conditions."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v2_qualify_ev', ROOT / 'scripts/v2_qualify.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
SHA = 'd' * 40
SKIP = 'SKIPPED [1] tests/test_isolated_executor.py:137: no filesystem isolation backend (need sandbox-exec or bwrap)'


def result(**suite):
    gates = {g: {'status': 'passed'} for g in q.REQUIRED}
    gates['test_suite'] = {'status': 'failed', 'reasons': ['required coverage skipped'], 'skipped_blocking': [SKIP],
                           **suite}
    return {'commit': SHA, 'required': list(q.REQUIRED), 'gates': gates, 'verdict': 'not_qualified',
            'blocking': ['test_suite']}


def evidence(commit=SHA, conclusion='success'):
    return {'commit': commit, 'repository': 'o/r', 'check_runs': [
        {'name': job, 'conclusion': conclusion, 'head_sha': commit, 'url': 'https://ci/' + job}
        for job in q.SANDBOX_CI_JOBS]}


def test_same_commit_green_ci_completes_the_gate():
    final = q.finalize(result(), evidence())
    assert final['verdict'] == 'qualified' and final['gates']['test_suite']['sandbox_evidence']['accepted']


@pytest.mark.parametrize('ev,problem', [(evidence(commit='e' * 40), 'not ' + SHA),
                                        (evidence(conclusion='failure'), 'did not succeed')])
def test_other_commit_or_red_ci_is_refused(ev, problem):
    final = q.finalize(result(), ev)
    assert final['verdict'] == 'not_qualified'
    assert any(problem in p for p in final['gates']['test_suite']['sandbox_evidence']['problems'])


def test_real_failures_are_never_excused():
    final = q.finalize(result(reasons=['pytest exited 1', 'required coverage skipped']), evidence())
    assert final['verdict'] == 'not_qualified'


def test_non_sandbox_skips_are_never_excused():
    final = q.finalize(result(skipped_blocking=[SKIP, 'SKIPPED [1] tests/test_x.py:1: needs a GPU']), evidence())
    assert final['verdict'] == 'not_qualified'


def test_other_required_failures_still_block():
    bad = result()
    bad['gates']['kernel_equivalence'] = {'status': 'failed'}
    assert q.finalize(bad, evidence())['blocking'] == ['kernel_equivalence']
