"""G4: measured bottlenecks (goodput knee + engine counters, kernel-family profile) reach the planner."""
import asyncio
import json

import httpx
import pytest

from inferstack.evaluation.bottleneck import (bottleneck_terms, diagnose_case, diagnose_trials, kernel_family,
                                              latest_measurements, names_bottleneck, record_measurement,
                                              summarize_kernels)
from inferstack.evaluation.goodput import GoodputCase, sweep
from inferstack.storage.repository import Repository

KEY = 'sk-test-0123456789abcdefghij'
CASE = GoodputCase('chat', prompt_words=8, max_tokens=4, ttft_ms=60.0, tpot_ms=20.0)


class QueueEngine:
    """One slot, so past capacity requests queue: TTFT misses, with a counter dict like /v1/native/status."""
    def __init__(self):
        self.sem, self.counters = asyncio.Semaphore(1), {'requests': 0, 'queue_ms': 0.0, 'wait_ms': 0.0,
                                                         'prefill_ms': 0.0}

    async def stream(self, payload):
        t0 = asyncio.get_running_loop().time()
        async with self.sem:
            queued = (asyncio.get_running_loop().time() - t0) * 1000
            await asyncio.sleep(0.01)
            self.counters['requests'] += 1
            self.counters['queue_ms'] += queued
            self.counters['prefill_ms'] += 10.0
            for i in range(payload['max_tokens']):
                yield f'data: {json.dumps({"choices": [{"text": str(i)}]})}\n\n'.encode()
                await asyncio.sleep(0.001)
            yield f'data: {json.dumps({"choices": [], "usage": {"completion_tokens": payload["max_tokens"]}})}\n\n'.encode()
            yield b'data: [DONE]\n\n'

    async def probe(self):
        return dict(self.counters)


@pytest.mark.asyncio
async def test_sweep_records_misses_and_counter_deltas_and_the_knee_is_queueing():
    engine = QueueEngine()
    out = await sweep(engine.stream, [CASE], rates=(5, 160), seconds_per_rate=0.5, min_requests=10, warmup=0,
                      probe=engine.probe)
    steps = out['cases']['chat']['steps']
    assert all(s['counters']['requests'] == s['requests'] for s in steps)      # per-step delta, not cumulative
    knee = next(s for s in steps if s['good_frac'] < 0.99)
    assert knee['misses'].get('ttft', 0) > 0 and 'tpot' not in knee['misses']
    d = diagnose_case(out['cases']['chat'])
    assert d['bottleneck'] == 'queue_admission' and d['confidence'] == 'evidenced'
    assert d['knee_rps'] == knee['rate'] and d['signal']['ttft_split_ms_mean']['queue_ms'] > 10


def _case(misses, counters=None, rates=(4, 8, 12), fracs=(1.0, 1.0, 0.6)):
    steps = [{'rate': r, 'requests': 40, 'good_frac': f, 'wall_s': 10.0, 'misses': {} if f >= 0.99 else misses,
              **({'counters': counters} if counters and f < 0.99 else {})} for r, f in zip(rates, fracs)]
    return {'case': {'ttft_ms': 500, 'tpot_ms': 30}, 'steps': steps}


def test_knee_labels():
    assert diagnose_case(_case({'tpot': 12, 'ttft': 2}))['bottleneck'] == 'decode'
    assert diagnose_case(_case({'error': 9}))['bottleneck'] == 'capacity'
    assert diagnose_case(_case({'ttft': 9}))['confidence'] == 'suspected'          # no TTFT split exposed
    pf = diagnose_case(_case({'ttft': 9}, {'requests': 10, 'queue_ms': 100, 'wait_ms': 50, 'prefill_ms': 4000,
                                           'lane_spec_steps': 100, 'lane_spec_row_steps': 2400,
                                           'lane_spec_accepted': 1200, 'lane_spec_step_ms': 1500,
                                           'lane_gpu_thread_ms': 9000}))
    assert pf['bottleneck'] == 'prefill' and pf['signal']['lane'] == {
        'rows_per_step': 24.0, 'spec_step_ms': 15.0, 'spec_acceptance': 0.5, 'gpu_busy_frac': 0.9}
    assert diagnose_case(_case({}, fracs=(1.0, 1.0, 1.0)))['bottleneck'] == 'none'
    # L4 v1.36 shape: 98% busy GPU, TTFT and per-token misses mixed -> capacity, not a decode/queue flip
    hot = {'requests': 40, 'queue_ms': 11000, 'wait_ms': 0, 'prefill_ms': 1700, 'lane_gpu_thread_ms': 9840}
    sat = diagnose_case(_case({'ttft': 39, 'tpot': 45}, hot))
    assert sat['bottleneck'] == 'capacity' and sat['signal']['saturated']
    assert diagnose_case(_case({'ttft': 2, 'tpot': 45}, hot))['bottleneck'] == 'decode'      # one SLO dominates
    cool = dict(hot, lane_gpu_thread_ms=5000)
    assert diagnose_case(_case({'ttft': 39, 'tpot': 45}, cool))['bottleneck'] == 'decode'    # not saturated
    many = diagnose_trials([{'cases': {'c': _case({'tpot': 9})}}, {'cases': {'c': _case({'tpot': 9})}},
                            {'cases': {'c': _case({'error': 9})}}])
    assert many['bottleneck'] == 'decode' and many['cases']['c']['agreement'] == 0.67


DECODE_STEP = {'meta': {'gpu': 'NVIDIA L4', 'profile': 'p.toml'},
               'scaling': [{'rows': 8, 'position': 200, 'graph_ms': 9.0, 'ms_per_row': 1.125}],
               'kernels': {'48': {'eager_gpu_ms_sum': 20.0, 'top_kernels_ms': [
                   ['fused_recurrent_gated_delta_rule_fwd_kernel', 9.0], ['sm89_xmma_gemm_bf16bf16', 6.0],
                   ['flash_fwd_splitkv_kernel', 3.0], ['triton_poi_fused_rms_norm_0', 1.0],
                   ['void at::native::vectorized_elementwise_kernel', 1.0]]}}}


def test_kernel_families_and_shares():
    assert kernel_family('chunk_gated_delta_rule_fwd_kernel_h') == 'gdn'
    assert kernel_family('ampere_bf16_s16816gemm_bf16_128x64') == 'gemm'
    assert kernel_family('_conv1d_update_kernel') == 'conv'
    # names seen in the real L4 profile (P10a, decode_step_profile.py at 48 rows)
    assert kernel_family('_banked_kernel') == 'gdn' and kernel_family('_split_kernel') == 'attention'
    assert kernel_family('void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128>') == 'gemm'
    assert kernel_family('void at::native::reduce_kernel<512, 1, ReduceOp<c10::BFloat16, ArgMaxOps<float>>>') == 'sampling'
    s = summarize_kernels(DECODE_STEP)
    assert s['dominant'] == {'family': 'gdn', 'share': 0.45, 'rows': 48}
    assert s['by_rows']['48']['shares']['gemm'] == 0.3 and s['gpu'] == 'NVIDIA L4'


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv('OPT_KEY', KEY)
    r = Repository(tmp_path / 'db', tmp_path / 'a')
    for name in ('base', 'cand'):
        r.put_record('deployment', name, {'id': name, 'engine': 'vllm', 'model': 'm', 'base_url': f'http://{name}',
                                          'engine_version': '0.11.0', 'profile': {'dtype': 'bfloat16'}})
    return r


def test_measurements_are_stored_per_deployment_and_go_stale(repo):
    from inferstack.memory.context import context_snapshot
    record_measurement(repo, 'base', 'kernels', summarize_kernels(DECODE_STEP), DECODE_STEP, source={'job': 'kp'})
    found = latest_measurements(repo, 'base')
    assert found['kernels']['stale'] is False and found['kernels']['raw_sha256']
    assert bottleneck_terms(found) == {'gdn', 'gemm', 'attention'}
    snap = context_snapshot(repo, 'base', store=False)['snapshot']
    assert snap['measurements']['kernels']['summary']['dominant']['family'] == 'gdn'
    repo.put_record('deployment', 'base', {'id': 'base', 'engine': 'vllm', 'model': 'm', 'base_url': 'http://b',
                                           'engine_version': '0.12.0', 'profile': {'dtype': 'bfloat16'}},
                    expected_revision=1)
    assert latest_measurements(repo, 'base')['kernels']['stale'] is True
    record_measurement(repo, 'base', 'kernels', summarize_kernels(DECODE_STEP))      # re-measure replaces
    assert latest_measurements(repo, 'base')['kernels']['stale'] is False


def test_names_bottleneck():
    assert names_bottleneck('Cut GDN kernel share at 48 rows', {'gdn'})
    assert names_bottleneck('reduce queue admission wait', {'queue_admission'})
    assert not names_bottleneck('raise batch size', {'gdn', 'decode'})


def _reply(content):
    return httpx.Response(200, json={'choices': [{'message': {'role': 'assistant', 'content': content},
                                                  'finish_reason': 'stop'}],
                                     'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120}})


def _plan(hypothesis):
    return json.dumps({'hypothesis': hypothesis, 'baseline_ref': 'base@1', 'evidence_ids': [], 'actions': [
        {'id': 'cfg', 'type': 'config_change', 'depends_on': [], 'params': {'candidate_deployment_id': 'cand'}},
        {'id': 'measure', 'type': 'measure', 'depends_on': ['cfg'], 'params': {}}],
        'expected_measurements': ['e2e_ms', 'ttft_client_ms'],
        'gates': ['correctness', 'quality', 'health', 'comparable', 'complete', 'performance'],
        'estimate': {'gpu_seconds': 600, 'wall_seconds': 900, 'provider_tokens': 0},
        'stop_rule': {'max_iterations': 3, 'stop_on': ['keep', 'budget_exhausted']}})


@pytest.mark.asyncio
async def test_planner_must_cite_the_measured_bottleneck(repo):
    """Exit test: with a kernel profile on record, a plan that ignores it is sent back; the accepted plan
    names the measured family (GDN) and the planner saw the shares in its context."""
    from inferstack.optimization.planner import ToolPlanner
    from inferstack.providers import create_provider
    record_measurement(repo, 'base', 'kernels', summarize_kernels(DECODE_STEP), DECODE_STEP)
    requests = []
    replies = [_reply(_plan('Larger batch improves goodput.')),
               _reply(_plan('GDN kernels take 45% of decode GPU time at 48 rows; the candidate cuts the gdn share.'))]

    def script(request):
        requests.append(json.loads(request.content))
        return replies.pop(0)
    provider = create_provider({'kind': 'openrouter', 'model': 'configured-model', 'api_key_env': 'OPT_KEY'},
                               transport=httpx.MockTransport(script))
    planner = ToolPlanner(repo, provider, candidate_ids=['cand'])
    try:
        plan = await planner.plan({'deployment_id': 'base', 'goal': 'g'}, [])
    finally:
        await planner.close()
    assert plan['planner']['corrections'] == 1 and 'gdn' in plan['hypothesis'].lower()
    context = json.loads(requests[0]['messages'][1]['content'])
    assert context['measurements']['kernels']['summary']['dominant']['family'] == 'gdn'
    assert 'does not name the measured bottleneck' in requests[1]['messages'][-1]['content']


@pytest.mark.asyncio
async def test_planner_tool_reads_measurements(repo):
    from inferstack.optimization.tools import PlannerTools
    tools = PlannerTools(repo, deployment_ids=['base', 'cand'], allowed_evidence=[], history=[])
    assert json.loads(await tools.call('get_measurements', '{"deployment_id": "base"}'))['available'] is False
    record_measurement(repo, 'base', 'kernels', summarize_kernels(DECODE_STEP))
    got = json.loads(await tools.call('get_measurements', '{"deployment_id": "base"}'))
    assert got['kernels']['summary']['dominant']['family'] == 'gdn'
