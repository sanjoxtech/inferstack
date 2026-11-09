"""P1: live serving evidence and defensible rollout gates (CLOUD_SERVING_RESEARCH_PLAN.md)."""
import hashlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from inferstack.engines.managed import ManagedEngine
from inferstack.releases.manager import ReleaseManager
from inferstack.releases.rollout import RolloutController, gateway_observer
from inferstack.serving.evidence import EvidenceStore, LiveStream, RequestRecord, summarize
from inferstack.serving.gateway import create_gateway
from inferstack.storage.repository import Repository


@pytest.fixture
def repo(tmp_path):
    return Repository(tmp_path / 'db', tmp_path / 'artifacts')


def releases(repo, evaluation=None):
    for name in ('base', 'cand'):
        repo.put_record('deployment', name, {'id': name, 'engine': 'generic_openai', 'model': 'm',
                                             'base_url': 'http://' + name})
        body = {'deployment': repo.read('deployment', name), 'evidence': {'sha256': 'f' * 64, 'bytes': 0},
                'job_id': 'fixture', 'verification_scope': 'fixture'}
        if name == 'cand' and evaluation is not None:
            body['evaluation'] = evaluation
        repo.put('release', 'rel-' + name, body)
    ReleaseManager(repo).route('m', 'rel-base')


def fill(store, release, n, *, ttft, e2e=200.0, stall=10.0, outcome='ok', at=1000.0):
    for i in range(n):
        value = ttft(i) if callable(ttft) else ttft
        store.record(RequestRecord(release, release[4:], outcome, True, at=at + i * 0.001, upstream_first_byte_ms=1.0,
                                   first_content_ms=value, e2e_ms=e2e, max_gap_ms=stall(i) if callable(stall) else stall))


def controller(repo, clock):
    return RolloutController(repo, observe=gateway_observer(repo), clock=lambda: clock[0])


def test_metadata_only_stream_has_no_first_content():
    live = LiveStream(started=0.0)
    live.feed(b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n', 0.001)
    live.feed(b'data: {"choices":[],"usage":{"completion_tokens":0}}\n\ndata: [DONE]\n\n', 0.002)
    phases = live.phases(0.003)
    assert phases['first_content_ms'] is None and phases['upstream_first_byte_ms'] == pytest.approx(1.0)
    rows = [{'outcome': 'ok', 'stream': 1, 'rejection_source': None, **phases}]
    summary = summarize(rows)
    assert summary['streams_without_content'] == 1 and summary['samples']['ttft'] == 0


def test_itl_only_from_authoritative_token_counts():
    live = LiveStream(started=0.0)
    live.feed(b'data: {"choices":[{"text":"a"}]}\n\n', 0.010)
    live.feed(b'data: {"choices":[{"text":"bcd"}]}\n\n', 0.040)
    assert live.phases(0.05)['itl_ms'] is None  # two content events are not two tokens
    live.feed(b'data: {"choices":[],"usage":{"completion_tokens":4}}\n\ndata: [DONE]\n\n', 0.041)
    assert live.phases(0.05)['itl_ms'] == pytest.approx(10.0)


def test_percentiles_need_their_minimum_samples(repo):
    store = EvidenceStore(repo)
    fill(store, 'rel-base', 150, ttft=50.0)
    window = store.window(release_id='rel-base', since=0, until=10**6)
    assert window['ttft_p50_ms'] == 50.0 and window['ttft_p95_ms'] is None and window['samples']['ttft'] == 150


@pytest.mark.asyncio
async def test_tail_only_regression_rolls_back_despite_unchanged_median(repo):
    releases(repo)
    store, clock = EvidenceStore(repo), [999.0]
    c = controller(repo, clock)
    await c.start('r', 'm', 'rel-cand', [{'weight': 50, 'min_duration_s': 0, 'ttft_p95_regression': 0.2,
                                          'max_latency_regression': 0.2}, {'weight': 100, 'min_duration_s': 0}], min_requests=200)
    fill(store, 'rel-base', 300, ttft=50.0)
    fill(store, 'rel-cand', 300, ttft=lambda i: 50.0 if i % 10 else 400.0)  # 10% slow tail, same p50
    clock[0] = 2000.0
    assert await c.step('r') == 'rolled_back'
    body = repo.read('rollout', 'r')['body']
    assert 'ttft_p95_regression' in body['reason']
    ref = body['evidence_refs'][-1]
    evidence = json.loads(repo.read_artifact(ref['sha256']))
    assert evidence['checks']['max_latency_regression']['pass'] is True  # the median gate alone would pass
    assert evidence['policy_sha256'] == body['policy_sha256'] and evidence['window']['until'] == 2000.0


@pytest.mark.asyncio
async def test_token_stall_rolls_back(repo):
    releases(repo)
    store, clock = EvidenceStore(repo), [999.0]
    c = controller(repo, clock)
    await c.start('r', 'm', 'rel-cand', [{'weight': 50, 'min_duration_s': 0, 'stall_p95_ms': 500}, {'weight': 100, 'min_duration_s': 0}], min_requests=200)
    fill(store, 'rel-cand', 250, ttft=50.0, stall=lambda i: 2000.0 if i % 5 == 0 else 20.0)
    clock[0] = 2000.0
    assert await c.step('r') == 'rolled_back'
    assert 'stall_p95_ms' in repo.read('rollout', 'r')['body']['reason']


@pytest.mark.asyncio
async def test_low_volume_holds_then_ends_inconclusive_never_passes_on_a_timer(repo):
    releases(repo)
    store, clock = EvidenceStore(repo), [999.0]
    c = controller(repo, clock)
    await c.start('r', 'm', 'rel-cand', [{'weight': 50, 'min_duration_s': 0, 'max_duration_s': 600,
                                          'ttft_p95_ms': 1000}, {'weight': 100, 'min_duration_s': 0}], min_requests=200)
    fill(store, 'rel-cand', 50, ttft=50.0)
    clock[0] = 1100.0
    assert (await c.step('r')).startswith('holding: only 50 candidate requests')
    clock[0] = 1700.0
    assert await c.step('r') == 'rolled_back'
    assert repo.read('rollout', 'r')['body']['reason'].startswith('inconclusive:')


@pytest.mark.asyncio
async def test_telemetry_loss_holds(repo):
    releases(repo)

    async def broken(release_id, since, until):
        raise ConnectionError('prometheus down')
    c = RolloutController(repo, observe=broken, clock=lambda: 2000.0)
    await c.start('r', 'm', 'rel-cand', [{'weight': 50, 'min_duration_s': 0}, {'weight': 100, 'min_duration_s': 0}])
    assert await c.step('r') == 'holding: telemetry unavailable (ConnectionError)'


def evaluation_for(repo, *, quality=True, fixture='a' * 64, tamper=False):
    evidence = {'verifier': {'fixture_sha256': fixture, 'code_sha256': 'c' * 64},
                'gates': {'quality': quality}, 'verdict': {'decision': 'keep' if quality else 'reject'}}
    artifact = repo.artifact(json.dumps(evidence).encode())
    sha = artifact['sha256']
    if tamper:
        (repo.artifact_root / sha).write_bytes(b'{"tampered": true}')
    return {'suite_sha256': fixture, 'evaluator_sha256': 'c' * 64, 'evidence_sha256': sha}


@pytest.mark.parametrize('case,expected', [('missing', 'holding: evaluation missing'),
                                           ('corrupt', 'holding: evaluation corrupt'),
                                           ('failed', 'rolled_back'), ('passed', 'promoted')])
@pytest.mark.asyncio
async def test_bound_evaluation_gates_promotion(repo, case, expected):
    evaluation = None if case == 'missing' else evaluation_for(repo, quality=case != 'failed',
                                                               tamper=case == 'corrupt')
    releases(repo, evaluation)
    store, clock = EvidenceStore(repo), [999.0]
    c = controller(repo, clock)
    await c.start('r', 'm', 'rel-cand', [{'weight': 100, 'min_duration_s': 0}], min_requests=5,
                  evaluation={'required': True})
    fill(store, 'rel-cand', 10, ttft=50.0)
    clock[0] = 2000.0
    assert (await c.step('r')).startswith(expected)


@pytest.mark.asyncio
async def test_pinned_evaluator_mismatch_holds(repo):
    releases(repo, evaluation_for(repo))
    store, clock = EvidenceStore(repo), [999.0]
    c = controller(repo, clock)
    await c.start('r', 'm', 'rel-cand', [{'weight': 100, 'min_duration_s': 0}], min_requests=5,
                  evaluation={'required': True, 'evaluator_sha256': 'd' * 64})
    fill(store, 'rel-cand', 10, ttft=50.0)
    clock[0] = 2000.0
    assert 'evaluation corrupt' in await c.step('r')


@pytest.mark.asyncio
async def test_evidence_survives_restart_and_spans_gateway_workers(repo, tmp_path):
    releases(repo)
    a, b = EvidenceStore(repo, worker='gw-a'), EvidenceStore(repo, worker='gw-b')
    fill(a, 'rel-cand', 120, ttft=50.0)
    fill(b, 'rel-cand', 120, ttft=50.0, at=1500.0)
    # A fresh process opening the same store (control-plane restart) sees every worker's records.
    reopened = Repository(tmp_path / 'db', tmp_path / 'artifacts')
    window = await gateway_observer(reopened)('rel-cand', 0, 10**6)
    assert window['requests'] == 240 and window['ttft_p95_ms'] == 50.0
    assert {r['worker'] for r in EvidenceStore(reopened).rows(release_id='rel-cand', since=0)} == {'gw-a', 'gw-b'}


@pytest.mark.asyncio
async def test_hundred_percent_stage_uses_retained_baseline_metrics(repo):
    releases(repo)
    store, clock = EvidenceStore(repo), [999.0]
    c = controller(repo, clock)
    await c.start('r', 'm', 'rel-cand', [{'weight': 50, 'min_duration_s': 0},
                                         {'weight': 100, 'min_duration_s': 0, 'ttft_p95_regression': 0.2}],
                  min_requests=200)
    fill(store, 'rel-base', 250, ttft=50.0)
    fill(store, 'rel-cand', 250, ttft=50.0)
    clock[0] = 2000.0
    assert await c.step('r') == 'advanced'
    fill(store, 'rel-cand', 250, ttft=200.0, at=2500.0)  # no stable traffic at 100%
    clock[0] = 3000.0
    assert await c.step('r') == 'rolled_back'
    checks = repo.read('rollout', 'r')['body']['observations']['checks']
    assert checks['ttft_p95_regression']['basis'] == 'retained_stable_stage_0'


def engine_factory(chunks):
    def handler(request):
        if request.url.path.endswith('/models'):
            return httpx.Response(200, json={'data': [{'id': 'm'}]})
        body = json.loads(request.content)
        if body.get('stream'):
            return httpx.Response(200, content=b''.join(chunks), headers={'Content-Type': 'text/event-stream'})
        return httpx.Response(200, json={'choices': [{'text': 'four'}], 'usage': {'completion_tokens': 1}})
    return lambda spec: ManagedEngine(spec, transport=httpx.MockTransport(handler))


def test_gateway_records_phases_outcomes_and_metrics(repo):
    releases(repo)
    stream = [b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n',
              b'data: {"choices":[{"delta":{"content":"four"}}]}\n\n',
              b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n', b'data: [DONE]\n\n']
    client = TestClient(create_gateway(repo, 't', engine_factory=engine_factory(stream)))
    h = {'Authorization': 'Bearer t'}
    assert client.post('/v1/completions', headers=h, json={'model': 'm', 'prompt': 'x'}).status_code == 200
    with client.stream('POST', '/v1/completions', headers=h, json={'model': 'm', 'prompt': 'x', 'stream': True}) as r:
        assert b'four' in b''.join(r.iter_bytes())
    window = client.get('/gateway/observe', headers=h, params={'release_id': 'rel-base', 'since': 0}).json()
    assert window['requests'] == 2 and window['errors'] == 0
    assert window['samples']['ttft'] == 1 and window['samples']['e2e'] == 2 and window['streams_without_content'] == 0
    rows = EvidenceStore(repo).rows(release_id='rel-base', since=0)
    streamed = next(r for r in rows if r['stream'])
    assert streamed['content_events'] == 1 and streamed['output_tokens'] == 1
    plain = next(r for r in rows if not r['stream'])
    assert plain['first_content_ms'] is None and plain['e2e_ms'] is not None  # no fake first-content phase
    text = client.get('/metrics', headers=h).text
    assert 'inferstack_gateway_requests_total{release="rel-base",deployment="base",outcome="ok"} 2' in text
    assert 'inferstack_gateway_ttft_ms_count{release="rel-base",deployment="base"} 1' in text
    assert 'prompt' not in text and 'four' not in text  # no request content in labels


def test_gateway_records_rejections_and_stream_without_terminal_event(repo):
    releases(repo)
    client = TestClient(create_gateway(repo, 't', engine_factory=engine_factory(
        [b'data: {"choices":[{"text":"four"}]}\n\n'])))  # upstream ends without [DONE]
    h = {'Authorization': 'Bearer t'}
    with client.stream('POST', '/v1/completions', headers=h, json={'model': 'm', 'prompt': 'x', 'stream': True}) as r:
        b''.join(r.iter_bytes())
    client.app.state.admission.state('base').draining = True
    assert client.post('/v1/completions', headers=h, json={'model': 'm', 'prompt': 'x'}).status_code == 503
    window = client.get('/gateway/observe', headers=h, params={'release_id': 'rel-base', 'since': 0}).json()
    assert window['errors'] == 1 and window['rejected'] == 1 and window['rejection_sources'] == {'draining': 1}
