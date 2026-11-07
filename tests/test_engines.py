"""Package C: adapters, telemetry normalization, inventory, per-model health and
admission, stream disconnect, overload, trace propagation, onboarding."""
import asyncio
import os
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from inferstack.contracts import validate
from inferstack.engines.adapters import AdapterError, SGLangAdapter, VLLMAdapter, adapter_for, attest
from inferstack.foundry.onboarding import OnboardingError, onboard
from inferstack.infrastructure.inventory import collect
from inferstack.serving.gateway import create_gateway
from inferstack.serving.health import AdmissionController
from inferstack.storage.repository import Repository
from inferstack.telemetry import normalize, quantile
from inferstack.telemetry.prometheus import Scrape

FIXTURES = Path(__file__).parent / 'fixtures'
COMMIT = 'a' * 40


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture
def servers(tmp_path):
    procs = []

    def start(**kw):
        port = free_port()
        argv = [sys.executable, str(FIXTURES / 'fake_openai_server.py'), '--port', str(port)]
        for k, v in kw.items():
            argv += ['--' + k.replace('_', '-'), str(v)]
        procs.append(subprocess.Popen(argv))
        deadline = time.monotonic() + 10
        while True:
            with socket.socket() as s:
                if s.connect_ex(('127.0.0.1', port)) == 0:
                    return f'http://127.0.0.1:{port}', port
            assert time.monotonic() < deadline
            time.sleep(0.05)
    yield start
    for p in procs:
        p.kill()
        p.wait()


# --- telemetry ---------------------------------------------------------------------

VLLM_T0 = '''# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="m",le="0.1"} 10
vllm:time_to_first_token_seconds_bucket{model_name="m",le="0.5"} 10
vllm:time_to_first_token_seconds_bucket{model_name="m",le="+Inf"} 10
vllm:time_to_first_token_seconds_count{model_name="m"} 10
vllm:time_to_first_token_seconds_sum{model_name="m"} 0.5
vllm:generation_tokens_total{model_name="m"} 1000
vllm:num_requests_running{model_name="m"} 1
'''
VLLM_T1 = VLLM_T0.replace('le="0.5"} 10', 'le="0.5"} 20').replace('le="+Inf"} 10', 'le="+Inf"} 20') \
    .replace('_count{model_name="m"} 10', '_count{model_name="m"} 20').replace('_sum{model_name="m"} 0.5', '_sum{model_name="m"} 3.5') \
    .replace('1000', '1600').replace('running{model_name="m"} 1', 'running{model_name="m"} 4')


def test_normalization_windows_semantics_and_missing_metrics():
    snap = normalize('vllm', VLLM_T1, previous_text=VLLM_T0, engine_version='0.11.0', clock=lambda: 5.0)
    m = snap['metrics']
    assert m['engine_ttft_ms']['value'] == pytest.approx(300.0)  # (3.5-0.5)/(20-10) s in the window
    assert m['engine_ttft_ms']['window'] == 'delta' and m['engine_ttft_ms']['semantics'].startswith('engine arrival')
    assert 100 <= m['engine_ttft_ms']['p50_estimate'] <= 500
    assert m['generation_tokens_total']['value'] == 600 and m['requests_running']['value'] == 4
    assert m['engine_tpot_ms']['value'] is None and 'absent' in m['engine_tpot_ms']['reason']  # never zero
    assert snap['mapping_status'] == 'declared' and len(snap['raw_sha256']) == 64
    reset = normalize('vllm', VLLM_T0, previous_text=VLLM_T1)['metrics']
    assert reset['engine_ttft_ms']['value'] is None and reset['generation_tokens_total']['value'] is None
    assert normalize('tensorrt', '')['available'] is False
    assert quantile({'count': 4, 'sum': 1, 'buckets': [(1.0, 1), (float('inf'), 4)]}, .99) is None
    with pytest.raises(ValueError, match='unparsable'):
        Scrape('not a metric line at all {')


# --- adapters ---------------------------------------------------------------------------

def test_launch_plans_pin_immutable_revisions():
    vllm = VLLMAdapter(None).launch_argv(model='Qwen/Qwen3.5-2B-Base', revision=COMMIT, port=8100)
    assert vllm[:4] == ['vllm', 'serve', 'Qwen/Qwen3.5-2B-Base', '--revision'] and COMMIT in vllm
    assert vllm[vllm.index('--tokenizer-revision') + 1] == COMMIT
    for adapter in (VLLMAdapter(None), SGLangAdapter(None)):
        with pytest.raises(AdapterError, match='immutable'):
            adapter.launch_argv(model='x', revision='main', port=1)
    with pytest.raises(AdapterError, match='must match'):
        SGLangAdapter(None).launch_argv(model='x', revision=COMMIT, tokenizer_revision='b' * 40, port=1)


@pytest.mark.asyncio
@pytest.mark.parametrize('flavor,engine', [('vllm', 'vllm'), ('sglang', 'sglang'), ('native', 'inferstack_native')])
async def test_runtime_identity_features_and_attestation(servers, flavor, engine):
    url, _ = servers(flavor=flavor, version='0.11.0')
    async with httpx.AsyncClient(base_url=url + '/') as client:
        adapter = adapter_for(engine, client)
        identity = await adapter.runtime_identity()
        assert identity['version'] == '0.11.0' and identity['served_models'] == ['m']
        good = await attest({'model': 'm', 'engine_version': '0.11.0'}, adapter)
        assert good['status'] == 'attested'
        assert (await attest({'model': 'm', 'engine_version': '0.12.0'}, adapter))['status'] == 'mismatch'
        assert (await attest({'model': 'other', 'engine_version': '0.11.0'}, adapter))['status'] == 'mismatch'
        assert (await attest({'model': 'm'}, adapter))['status'] == 'unverified'
        features = await adapter.observed_features(identity)
        telemetry = await adapter.telemetry(engine_version='0.11.0')
    if flavor == 'vllm':
        assert features['prefix_cache']['effective'] == 'active'
        assert telemetry['metrics']['engine_ttft_ms']['value'] == pytest.approx(100.0)
    elif flavor == 'sglang':
        assert features['cuda_graph_replay']['effective'] == 'disabled'
        assert telemetry['metrics']['kv_cache_usage_ratio']['value'] == 0.25
        assert telemetry['metrics']['engine_ttft_ms']['value'] is None
    else:
        assert features['m.paged_kv']['effective'] == 'inactive'
        assert features['m.cuda_graph_replay']['effective'] == 'not_proven'
        assert telemetry['metrics'] == {} and telemetry['available']


@pytest.mark.asyncio
async def test_identity_of_unreachable_engine_is_unknown_not_assumed():
    async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{free_port()}/') as client:
        adapter = VLLMAdapter(client)
        result = await attest({'model': 'm', 'engine_version': '0.11.0'}, adapter)
        assert result['status'] == 'mismatch' and result['identity']['version'] is None
        assert (await adapter.telemetry())['available'] is False


# --- inventory ----------------------------------------------------------------------

def test_inventory_records_every_gpu(tmp_path):
    fake = tmp_path / 'nvidia-smi'
    fake.write_text('#!/bin/sh\nif [ "$1" = "topo" ]; then echo "GPU0 X NV12"; exit 0; fi\n'
                    'echo "GPU-aaa, NVIDIA A100-SXM4-40GB, 40960, 550.54, 8.0, 00000000:07:00.0"\n'
                    'echo "GPU-bbb, NVIDIA A100-SXM4-40GB, 40960, 550.54, 8.0, 00000000:0F:00.0"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    body, notes = collect(provider='colab', nvidia_smi=str(fake))
    assert [g['uuid'] for g in body['gpus']] == ['GPU-aaa', 'GPU-bbb'] and body['gpus'][0]['memory_bytes'] == 40 * 2**30
    assert body['driver_version'] == '550.54' and 'NV12' in body['topology'] and notes == {}
    validate('hardware', body)
    missing, notes = collect(nvidia_smi=str(tmp_path / 'absent'))
    assert missing['gpus'] == [] and 'not found' in notes['gpu']
    validate('hardware', missing)


# --- gateway health / admission ---------------------------------------------------------------

class GatewayServer:
    def __init__(self, app):
        self.port = free_port()
        self.server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=self.port, log_level='error'))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        while not self.server.started:
            time.sleep(0.02)
        return f'http://127.0.0.1:{self.port}'

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(5)


def route(repo, model, deployment_id, base_url):
    from inferstack.releases.manager import ReleaseManager
    repo.put_record('deployment', deployment_id, {'id': deployment_id, 'engine': 'generic_openai', 'model': model,
                                                  'base_url': base_url})
    snapshot = repo.read('deployment', deployment_id)
    release = 'rel-' + deployment_id
    repo.put('release', release, {'deployment': snapshot, 'evidence': {'sha256': 'f' * 64, 'bytes': 0},
                                  'job_id': 'j', 'verification_scope': 'test'})
    ReleaseManager(repo).route(model, release)


H = {'Authorization': 'Bearer t'}


def test_unhealthy_model_does_not_block_healthy_model_and_recovers(tmp_path, servers):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    good_url, _ = servers()
    down_port = free_port()
    route(repo, 'good-model', 'good', good_url)
    route(repo, 'flaky-model', 'flaky', f'http://127.0.0.1:{down_port}')
    clock = [0.0]
    admission = AdmissionController(failure_threshold=2, cooldown_seconds=5, clock=lambda: clock[0])
    with GatewayServer(create_gateway(repo, 't', admission=admission)) as gw:
        req = lambda model: httpx.post(gw + '/v1/completions', headers=H, json={'model': model, 'prompt': 'x'})
        assert [req('flaky-model').status_code for _ in range(2)] == [502, 502]
        down = req('flaky-model')
        assert down.status_code == 503 and down.headers['Retry-After']
        assert req('good-model').status_code == 200  # isolation
        subprocess_proc = subprocess.Popen([sys.executable, str(FIXTURES / 'fake_openai_server.py'),
                                            '--port', str(down_port)])
        try:
            for _ in range(200):
                with socket.socket() as s:
                    if s.connect_ex(('127.0.0.1', down_port)) == 0:
                        break
                time.sleep(0.05)
            clock[0] += 6  # cooldown elapsed -> one half-open probe
            assert req('flaky-model').status_code == 200
            status = httpx.get(gw + '/gateway/status', headers=H).json()['deployments']
            assert status['flaky']['breaker'] == 'closed' and status['good']['failed'] == 0
        finally:
            subprocess_proc.kill()
            subprocess_proc.wait()


def test_overload_is_rejected_per_deployment_with_retry_after(tmp_path, servers):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    slow_url, _ = servers(delay=1.0)
    route(repo, 'slow-model', 'slow', slow_url)
    admission = AdmissionController(max_inflight=1)
    with GatewayServer(create_gateway(repo, 't', admission=admission)) as gw:
        results = {}

        def call(name):
            results[name] = httpx.post(gw + '/v1/completions', headers=H, timeout=10,
                                       json={'model': 'slow-model', 'prompt': 'x'})
        first = threading.Thread(target=call, args=('first',))
        first.start()
        time.sleep(0.3)
        call('second')
        first.join()
        assert results['first'].status_code == 200
        assert results['second'].status_code == 429 and results['second'].headers['Retry-After'] == '1'


def test_upstream_errors_keep_meaning_and_trace_ids_propagate(tmp_path, servers):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    events = tmp_path / 'events'
    ok_url, _ = servers(events=events)
    bad_url, _ = servers(fail_status=400)
    broken_url, _ = servers(fail_status=503)
    route(repo, 'ok', 'ok', ok_url)
    route(repo, 'bad', 'bad', bad_url)
    route(repo, 'broken', 'broken', broken_url)
    trace = '00-' + '1' * 32 + '-' + '2' * 16 + '-01'
    with GatewayServer(create_gateway(repo, 't')) as gw:
        r = httpx.post(gw + '/v1/completions', headers={**H, 'X-Request-Id': 'req-42', 'traceparent': trace},
                       json={'model': 'ok', 'prompt': 'x'})
        assert r.status_code == 200 and r.headers['X-Request-Id'] == 'req-42' and r.json()['echo_request_id'] == 'req-42'
        assert r.headers['traceparent'] == trace and f'request req-42 {trace}' in events.read_text()
        r = httpx.post(gw + '/v1/completions', headers={**H, 'X-Request-Id': 'bad id with spaces'},
                       json={'model': 'ok', 'prompt': 'x'})
        assert r.headers['X-Request-Id'] != 'bad id with spaces'
        assert httpx.post(gw + '/v1/completions', headers=H, json={'model': 'bad', 'prompt': 'x'}).status_code == 400
        assert httpx.post(gw + '/v1/completions', headers=H, json={'model': 'broken', 'prompt': 'x'}).status_code == 502
        streamed = httpx.post(gw + '/v1/completions', headers=H, json={'model': 'broken', 'prompt': 'x', 'stream': True})
        assert streamed.status_code == 502  # primed stream: an upstream error is never a 200
        status = httpx.get(gw + '/gateway/status', headers=H).json()['deployments']
        assert status['bad']['failed'] == 0 and status['broken']['failed'] == 2


def test_client_disconnect_cancels_upstream_stream_and_releases_capacity(tmp_path, servers):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    events = tmp_path / 'events'
    url, _ = servers(events=events)
    route(repo, 'm1', 'd1', url)
    admission = AdmissionController()
    with GatewayServer(create_gateway(repo, 't', admission=admission)) as gw:
        with httpx.stream('POST', gw + '/v1/completions', headers=H,
                          json={'model': 'm1', 'prompt': 'x', 'stream': True}) as response:
            assert response.status_code == 200
            lines = response.iter_lines()
            assert next(lines).startswith('data: ')
        # Client went away after one chunk; the upstream must observe the disconnect.
        for _ in range(100):
            if 'stream disconnected' in (events.read_text() if events.exists() else ''):
                break
            time.sleep(0.05)
        assert 'stream disconnected' in events.read_text() and 'stream completed' not in events.read_text()
        status = admission.status('d1')
        assert status['inflight'] == 0 and status['failed'] == 0


@pytest.mark.asyncio
async def test_drain_stops_admission_and_waits_for_inflight():
    admission = AdmissionController()
    admission.admit('d')
    task = asyncio.create_task(admission.drain('d', 2))
    await asyncio.sleep(0.05)
    with pytest.raises(Exception, match='draining'):
        admission.admit('d')
    admission.release('d', ok=True)
    assert await task == {'drained': True, 'remaining': 0}
    admission.undrain('d')
    admission.admit('d')
    assert await admission.drain('d', 0.1) == {'drained': False, 'remaining': 1}


# --- onboarding -------------------------------------------------------------------------

QWEN = {'model_type': 'qwen3_5', 'architectures': ['Qwen3_5ForConditionalGeneration'], 'torch_dtype': 'bfloat16',
        'text_config': {'model_type': 'qwen3_5_text', 'max_position_embeddings': 262144}}


def test_onboarding_supported_unsupported_and_immutable_identity(tmp_path):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    resolved = []

    def resolver(repo_id, revision):
        resolved.append(revision)
        return COMMIT, QWEN
    out = onboard(repo, model_id='qwen35-2b', repo_id='Qwen/Qwen3.5-2B-Base', revision='main', resolver=resolver)
    assert resolved == ['main'] and out['revision'] == COMMIT and out['status'] == 'supported'
    assert out['reference_engine'] == 'inferstack_native' and out['deployment_created'] is False
    assert out['execution_profile']['max_model_len'] == 262144 and out['model_record'] == 'qwen35-2b@1'
    assert repo.read('model', 'qwen35-2b')['body']['revision'] == COMMIT
    assert onboard(repo, model_id='qwen35-2b', repo_id='Qwen/Qwen3.5-2B-Base', revision=COMMIT,
                   config=QWEN)['model_record'] == 'qwen35-2b@1'  # idempotent
    with pytest.raises(OnboardingError, match='immutable commit'):
        onboard(repo, model_id='x', repo_id='r', revision='main', resolver=lambda r, v: ('main', QWEN))
    llama = onboard(repo, model_id='llama', repo_id='meta/llama', revision=COMMIT,
                    config={'model_type': 'llama', 'architectures': ['LlamaForCausalLM']})
    assert llama['status'] == 'needs_engine_probe' and llama['engines']['vllm']['status'] == 'needs_engine_probe'
    assert 'execution_profile' not in llama
    bert = onboard(repo, model_id='bert', repo_id='google/bert', revision=COMMIT,
                   config={'model_type': 'bert', 'architectures': ['BertForMaskedLM']})
    assert bert['status'] == 'unsupported' and 'not text generation' in bert['gaps'][0]
    with pytest.raises(KeyError):
        repo.read('model', 'bert')  # no record, no fabricated deployment
    remote = onboard(repo, model_id='custom', repo_id='x/custom', revision=COMMIT,
                     config={'model_type': 'custom', 'architectures': ['CustomForCausalLM'], 'auto_map': {'x': 'y'}})
    assert remote['status'] == 'unsupported' and 'trust_remote_code' in remote['gaps'][0]
    with pytest.raises(OnboardingError, match='mutable'):
        onboard(repo, model_id='f', repo_id='factorylm-2b', revision='main', source='foundry', config=QWEN)
