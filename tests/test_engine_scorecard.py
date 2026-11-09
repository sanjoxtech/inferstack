"""N0 scorecard flow on CPU with real fake-engine processes: immutable per-round records,
rotated order, unsupported profiles, equal-work check and no claim below 3 rounds."""
import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FAKE = str(ROOT / 'tests' / 'fixtures' / 'fake_openai_server.py')
spec = importlib.util.spec_from_file_location('engine_scorecard', ROOT / 'scripts' / 'engine_scorecard.py')
scorecard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorecard)

WORKLOAD = {'version': 1, 'cases': [{'name': 'short', 'prompt_words': 8, 'max_tokens': 6},
                                    {'name': 'shared', 'prompt_words': 30, 'max_tokens': 6, 'shared_prefix': True}],
            'concurrency': [1, 2], 'requests_per_cell': {'min': 3, 'per_concurrency': 1}, 'warmup_requests': 1,
            'temperature': 0.0, 'seed': 1}


def engine(name, chunk_delay, *extra):
    return {'name': name, 'engine': 'fake', 'model': 'm',
            'argv': [sys.executable, FAKE, '--port', '{port}', '--usage', '--stream-tokens', '6',
                     '--chunk-delay', str(chunk_delay), *extra]}


@pytest.fixture
def profiles(tmp_path):
    path = tmp_path / 'profiles.json'
    path.write_text(json.dumps({'baseline': 'slow', 'profiles': [
        engine('slow', 0.03), engine('fast', 0.005),
        {'name': 'broken', 'engine': 'fake', 'model': 'm', 'argv': [sys.executable, '-c', 'raise SystemExit(3)']}]}))
    workload = tmp_path / 'workload.json'
    workload.write_text(json.dumps(WORKLOAD))
    return path, workload


def args(profiles, out, rounds, run_id):
    return argparse.Namespace(profiles=str(profiles[0]), workload=str(profiles[1]), out=str(out), rounds=rounds,
                              run_id=run_id, only=None, ready_timeout=20, request_timeout=30)


def test_three_rounds_give_relations_and_records_are_immutable(profiles, tmp_path):
    out = asyncio.run(scorecard.run(args(profiles, tmp_path / 'r', 3, 'run1')))
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['workload_sha256'] and manifest['rounds'] == 3
    statuses = [json.loads((out / f'round-{r}' / p / 'status.json').read_text())
                for r in range(3) for p in ('slow', 'fast')]
    assert all(s['ready'] and s['exit_code'] is not None for s in statuses)
    rows = (out / 'round-0' / 'fast' / 'requests.jsonl').read_text().splitlines()
    assert len(rows) == 2 * (3 + 3) and all(json.loads(r)['output_tokens'] == 6 for r in rows)
    card = scorecard.aggregate([out], baseline='slow')
    assert 'broken' in card['unsupported_or_failed'] and card['profiles']['broken']['ready_rounds'] == 0
    fast = next(c for c in card['cells'] if c['profile'] == 'fast' and c['case'] == 'short' and c['concurrency'] == 1)
    assert fast['rounds'] == 3
    assert fast['vs_baseline']['output_tps']['relation'] == 'ahead'
    assert fast['vs_baseline']['e2e_p50_ms']['relation'] == 'ahead' and fast['vs_baseline']['equal_work'] is True
    assert fast['metrics']['client_decode_p50_ms']['median'] < 20
    assert '| short | 1 | fast | 3 |' in scorecard.markdown(card)
    with pytest.raises(FileExistsError):  # a run id is never reused or overwritten
        asyncio.run(scorecard.run(args(profiles, tmp_path / 'r', 1, 'run1')))


def test_fewer_than_three_rounds_never_claim_a_relation(profiles, tmp_path):
    out = asyncio.run(scorecard.run(args(profiles, tmp_path / 'r', 1, 'one')))
    card = scorecard.aggregate([out], baseline='slow')
    relations = {v['relation'] for c in card['cells'] for k, v in (c.get('vs_baseline') or {}).items()
                 if isinstance(v, dict)}
    assert relations <= {'insufficient_rounds', 'missing'}


def test_missing_usage_is_missing_not_zero(tmp_path):
    path = tmp_path / 'p.json'
    path.write_text(json.dumps({'profiles': [{'name': 'nousage', 'engine': 'fake', 'model': 'm', 'argv': [
        sys.executable, FAKE, '--port', '{port}', '--stream-tokens', '4', '--chunk-delay', '0.001']}]}))
    workload = tmp_path / 'w.json'
    workload.write_text(json.dumps({**WORKLOAD, 'cases': WORKLOAD['cases'][:1], 'concurrency': [1]}))
    out = asyncio.run(scorecard.run(args((path, workload), tmp_path / 'r', 1, 'x')))
    cell = scorecard.aggregate([out])['cells'][0]
    assert cell['metrics']['output_tps'] == {'median': None, 'missing': True, 'rounds': 0}
    assert cell['metrics']['client_decode_p50_ms']['median'] is None and cell['metrics']['ttft_p50_ms']['median']


def test_rounds_rotate_profile_order(profiles, tmp_path):
    out = asyncio.run(scorecard.run(args(profiles, tmp_path / 'r', 2, 'rot')))
    first = {r: min((json.loads((out / f'round-{r}' / p / 'status.json').read_text())['started_at'], p)
                    for p in ('slow', 'fast', 'broken'))[1] for r in range(2)}
    assert first[0] != first[1]


def test_prompts_are_deterministic_and_shared_prefixes_share():
    case = {'name': 'c', 'prompt_words': 20, 'shared_prefix': True}
    a, b = scorecard.prompts(case, 3, 7), scorecard.prompts(case, 3, 7)
    assert a == b and len({p.split('\n')[0] for p in a}) == 1
    unique = scorecard.prompts({'name': 'u', 'prompt_words': 20}, 3, 7)
    assert len(set(unique)) == 3
