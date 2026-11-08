"""Per-phase telemetry and quality beyond exact match in the endpoint verifier."""
import json

import httpx
import pytest

from inferstack.engines.managed import ManagedEngine
from inferstack.optimization.experiment import EndpointExecutor, EndpointVerifier, case_score
from inferstack.storage.repository import Repository

VLLM_BEFORE = ('# TYPE vllm:time_to_first_token_seconds histogram\n'
               'vllm:time_to_first_token_seconds_bucket{le="0.1"} 0\nvllm:time_to_first_token_seconds_bucket{le="+Inf"} 0\n'
               'vllm:time_to_first_token_seconds_sum 0\nvllm:time_to_first_token_seconds_count 0\n')


def engine_app(answer, *, logprobs=None, delta_ttft=0.05, stream_chunks=4):
    """MockTransport handler: completions, SSE stream, and a vLLM histogram that advances per request."""
    state = {'count': 0}

    def handler(request):
        path = request.url.path
        if path == '/v1/models':
            return httpx.Response(200, json={'data': [{'id': 'm'}]})
        if path == '/metrics':
            n = state['count']
            text = ('# TYPE vllm:time_to_first_token_seconds histogram\n'
                    f'vllm:time_to_first_token_seconds_bucket{{le="0.1"}} {n}\n'
                    f'vllm:time_to_first_token_seconds_bucket{{le="+Inf"}} {n}\n'
                    f'vllm:time_to_first_token_seconds_sum {n * delta_ttft}\n'
                    f'vllm:time_to_first_token_seconds_count {n}\n')
            return httpx.Response(200, text=text)
        body = json.loads(request.content)
        state['count'] += 1
        if body.get('stream'):
            chunks = b''.join(f'data: {json.dumps({"choices": [{"text": str(i)}]})}\n\n'.encode()
                              for i in range(stream_chunks))
            return httpx.Response(200, content=chunks, headers={'Content-Type': 'text/event-stream'})
        choice = {'text': answer, 'finish_reason': 'stop'}
        if body.get('logprobs') and logprobs is not None:
            choice['logprobs'] = {'tokens': ['the', 'cat'], 'token_logprobs': logprobs}
        return httpx.Response(200, json={'choices': [choice], 'usage': {'completion_tokens': 2}})
    return handler


async def verify(tmp_path, cand_answer, *, cand_logprobs=(-0.1, -0.2), gates=(), quality_budget=0.0, cases=None):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    repo.put_record('hardware', 'hw', {'provider': 'attached', 'observed_at': 1.0, 'gpus': []})
    repo.put_record('model', 'mdl', {'source': 'huggingface', 'repo_id': 'org/m', 'revision': 'b' * 40})
    handlers = {'base': engine_app('the cat sat on the mat', logprobs=[-0.1, -0.2]),
                'cand': engine_app(cand_answer, logprobs=list(cand_logprobs) if cand_logprobs else None)}
    for name in handlers:
        repo.put_record('deployment', name, {'id': name, 'engine': 'vllm', 'model': 'm', 'base_url': f'http://{name}',
                                             'hardware_ref': 'hw', 'model_ref': 'mdl'})
    factory = lambda spec: ManagedEngine(spec, transport=httpx.MockTransport(handlers[spec.id]))
    executor = EndpointExecutor(repo, tmp_path / 'logs', engine_factory=factory)
    cases = cases or [{'prompt': 'describe', 'reference': 'the cat sat on the mat', 'max_tokens': 8}]
    verifier = EndpointVerifier(repo, executor, 'base', cases, [{'metric': 'e2e_ms', 'direction': 'min'}],
                                quality_budget=quality_budget)
    plan = {'baseline_id': 'base', 'baseline_revision': 1, 'candidate_id': 'cand', 'candidate_revision': 1,
            'gates': ['correctness', 'quality', 'health', 'comparable', 'complete', *gates]}
    candidate = await executor.execute(plan, 'exp')
    try:
        verdict = await verifier.verify(plan, candidate)
    finally:
        await executor.cleanup(candidate)
    return verdict, json.loads(repo.read_artifact(verdict['evidence']['sha256']))


def test_graded_case_scores():
    assert case_score({'reference': 'the cat sat'}, 'the cat sat') == 1.0
    assert 0 < case_score({'reference': 'the cat sat on the mat'}, 'a cat sat') < 1
    assert case_score({'reference': 'Paris', 'match': 'contains'}, 'It is paris.') == 1.0
    assert case_score({'expected': 'four'}, 'four ') == 0.0  # exact stays exact


@pytest.mark.asyncio
async def test_phase_metrics_come_from_stream_and_engine_histograms(tmp_path):
    verdict, evidence = await verify(tmp_path, 'the cat sat on the mat')
    trial = evidence['trials'][1][0]
    assert trial['ttft_client_ms'] is not None and trial['chunk_gap_client_ms'] is not None
    assert trial['ttft_engine_ms'] == pytest.approx(50.0)  # histogram delta mean, seconds -> ms
    assert trial['ttft_client_p95_ms'] is None  # 1 sample per trial: tail not reported
    assert trial['output_tokens_per_s'] > 0  # from authoritative usage
    assert evidence['metric_sources']['ttft_engine_ms'] == 'engine'
    assert evidence['metric_sources']['ttft_client_ms'] == 'client'


@pytest.mark.asyncio
async def test_quality_budget_controls_acceptance(tmp_path):
    worse, evidence = await verify(tmp_path / 'a', 'a dog ran', gates=('quality_budget',))
    assert worse['gates']['quality'] is False and worse['gates']['quality_budget'] is False
    assert worse['decision'] == 'reject' and evidence['quality']['candidate'] < evidence['quality']['baseline']
    slight, _ = await verify(tmp_path / 'b', 'the cat sat on a mat', gates=('quality_budget',), quality_budget=0.3)
    assert slight['gates']['quality_budget'] is True  # within the declared budget, not "numerically identical"


@pytest.mark.asyncio
async def test_logit_agreement_gate(tmp_path):
    same, _ = await verify(tmp_path / 'a', 'the cat sat on the mat', gates=('logit_agreement',))
    assert same['gates']['logit_agreement'] is True
    drift, _ = await verify(tmp_path / 'b', 'the cat sat on the mat', cand_logprobs=(-0.9, -1.4),
                            gates=('logit_agreement',))
    assert drift['gates']['logit_agreement'] is False and drift['decision'] == 'reject'
    blind, _ = await verify(tmp_path / 'c', 'the cat sat on the mat', cand_logprobs=None, gates=('logit_agreement',))
    assert blind['gates']['logit_agreement'] is None and blind['decision'] == 'inconclusive'
