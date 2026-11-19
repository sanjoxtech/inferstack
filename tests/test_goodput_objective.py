"""G1: open-loop goodput as a harness objective (evaluation/goodput.py + EndpointVerifier hook)."""
import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from inferstack.evaluation.goodput import GoodputCase, cases_from_slo, is_good, summarize, sweep

FIXTURES = Path(__file__).parent / 'fixtures'
CASE = GoodputCase('chat', prompt_words=8, max_tokens=4, ttft_ms=80.0, tpot_ms=20.0)
FAST = dict(rates=(10, 20, 40, 80), seconds_per_rate=0.5, min_requests=12, warmup=0)


class SlotEngine:
    """In-process engine with `slots` concurrent requests: queueing past capacity shows up as TTFT."""
    def __init__(self, slots, first_s=0.01, gap_s=0.005, usage=True):
        self.sem, self.first_s, self.gap_s, self.usage = asyncio.Semaphore(slots), first_s, gap_s, usage

    async def stream(self, payload):
        async with self.sem:
            await asyncio.sleep(self.first_s)
            n = payload['max_tokens']
            for i in range(n):
                yield f'data: {json.dumps({"choices": [{"text": str(i)}]})}\n\n'.encode()
                await asyncio.sleep(self.gap_s)
            if self.usage:
                yield f'data: {json.dumps({"choices": [], "usage": {"completion_tokens": n}})}\n\n'.encode()
            yield b'data: [DONE]\n\n'


def test_is_good_needs_every_slo_and_token_counts():
    ok = {'success': True, 'ttft_ms': 50, 'e2e_ms': 50 + 3 * 10, 'output_tokens': 4}
    assert is_good(ok, CASE)
    assert not is_good({**ok, 'ttft_ms': 90, 'e2e_ms': 120}, CASE)          # TTFT over
    assert not is_good({**ok, 'e2e_ms': 50 + 3 * 25}, CASE)                 # per-token over
    assert not is_good({**ok, 'output_tokens': None}, CASE)                 # unverifiable
    assert not is_good({**ok, 'success': False}, CASE)
    assert summarize([{'rate': 1, 'good_frac': 1.0}, {'rate': 2, 'good_frac': 0.95}, {'rate': 4, 'good_frac': 0.4}]) == \
        {'goodput99': 1, 'goodput90': 2}


def test_cases_from_slo_use_the_p99_ttft_class_limit():
    slo = {'industry_limits': {'interactive': {'ttft_p95_ms': 250, 'ttft_p99_ms': 500, 'client_decode_p95_ms': 30}}}
    (c,) = cases_from_slo([{'name': 'interactive', 'prompt_words': 48, 'max_tokens': 128}], slo)
    assert (c.ttft_ms, c.tpot_ms, c.shared_prefix) == (500.0, 30.0, False)


@pytest.mark.asyncio
async def test_more_capacity_gives_more_goodput_and_missing_usage_is_flagged():
    small = await sweep(SlotEngine(1).stream, [CASE], **FAST)
    large = await sweep(SlotEngine(16).stream, [CASE], **FAST)
    assert large['goodput99_rps'] > small['goodput99_rps'] and not large['token_counts_missing']
    steps = small['cases']['chat']['steps']
    assert steps[-1]['good_frac'] < 0.5 or len(steps) == len(FAST['rates'])   # sweep stops past the knee
    blind = await sweep(SlotEngine(16, usage=False).stream, [CASE], **FAST)
    assert blind['token_counts_missing'] > 0 and blind['goodput99_rps'] == 0


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture
def usage_server():
    procs = []

    def start():
        port = free_port()
        procs.append(subprocess.Popen([sys.executable, str(FIXTURES / 'fake_openai_server.py'), '--port', str(port),
                                       '--usage', '--stream-tokens', '4', '--chunk-delay', '0.002']))
        for _ in range(200):
            with socket.socket() as s:
                if s.connect_ex(('127.0.0.1', port)) == 0:
                    return f'http://127.0.0.1:{port}'
            time.sleep(0.05)
        raise AssertionError('server did not start')
    yield start
    for p in procs:
        p.kill()
        p.wait()


@pytest.mark.asyncio
async def test_verifier_measures_goodput_on_every_repeat(tmp_path, usage_server):
    from inferstack.optimization.experiment import EndpointExecutor, EndpointVerifier
    from inferstack.storage.repository import Repository
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    repo.put_record('hardware', 'l4', {'provider': 'colab', 'observed_at': 1.0, 'gpus': []})
    repo.put_record('model', 'qwen', {'source': 'huggingface', 'repo_id': 'Qwen/x', 'revision': 'b' * 40})
    for name in ('base', 'cand'):
        repo.put_record('deployment', name, {'id': name, 'engine': 'generic_openai', 'model': 'm',
                                             'base_url': usage_server(), 'hardware_ref': 'l4', 'model_ref': 'qwen'})
    executor = EndpointExecutor(repo, tmp_path / 'logs')
    # the fake server streams numbers, not the answer: correctness is not what this test is about
    cases = [{'prompt': '2+2', 'reference': 'x', 'match': 'contains', 'max_tokens': 4}]
    goodput = {'cases': [GoodputCase('chat', 8, 4, ttft_ms=500.0, tpot_ms=50.0)], 'rates': (5, 10),
               'seconds_per_rate': 0.4, 'min_requests': 4, 'warmup': 0}
    verifier = EndpointVerifier(repo, executor, 'base', cases, [{'metric': 'goodput99_rps', 'direction': 'max'}],
                                repeats=3, goodput=goodput, phase_timing=False)
    plan = {'candidate_id': 'cand', 'candidate_revision': 1, 'baseline_id': 'base', 'baseline_revision': 1}
    candidate = await executor.execute(plan, 'exp')
    try:
        verdict = await verifier.verify(plan, candidate)
    finally:
        await executor.cleanup(candidate)
    evidence = json.loads(repo.read_artifact(verdict['evidence']['sha256']))
    values = [t['goodput99_rps'] for side in evidence['trials'] for t in side]
    assert len(values) == 6 and all(v == 10 for v in values)            # both sides meet the SLO at every rate
    assert [len(side) for side in evidence['goodput']] == [3, 3]
    assert evidence['policy']['goodput']['cases'][0]['name'] == 'chat'
    assert verdict['decision'] in {'inconclusive', 'reject', 'keep', 'retest'}
    assert verdict['metrics']['goodput99_rps']['gain'] == 0               # identical servers: no gain claimed
