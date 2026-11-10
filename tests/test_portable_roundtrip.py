"""E1.2 notebook round trip on CPU: manifest -> generated notebook; runner (fake engine
process) -> ResultBundle -> verified import into Hardware/Evidence/Trial records.
Tampered bundles, manifest mismatches and credential leaks are refused; gates that were
not run stay missing; an unsupported preflight launches nothing."""
import asyncio
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import pytest

from inferstack.execution.manifest import ExecutionManifestV1, from_records, sha256_canonical
from inferstack.execution.notebooks import export_notebook
from inferstack.execution.results import BundleRejected, import_bundle, open_bundle
from inferstack.execution.runner import launch_argv, run
from inferstack.storage.repository import Repository

ROOT = Path(__file__).resolve().parents[1]
FAKE = str(ROOT / 'tests' / 'fixtures' / 'fake_openai_server.py')
REV = 'b1485b2fa6dfa1287294f269f5fb618e03d52d7c'
SRC = 'a' * 40
GIB = 1 << 30
MODEL = 'Qwen/Qwen3.5-2B-Base'
WORKLOAD = {'version': 1, 'cases': [{'name': 'short', 'prompt_words': 8, 'max_tokens': 6},
                                    {'name': 'long', 'prompt_words': 40, 'max_tokens': 6}],
            'concurrency': [1, 2], 'requests_per_cell': {'min': 2, 'per_concurrency': 1}, 'warmup_requests': 1,
            'temperature': 0.0, 'seed': 3}
A100 = {'provider': 'colab', 'observed_at': time.time(), 'libraries': {},
        'gpus': [{'uuid': 'GPU-a100-0', 'name': 'NVIDIA A100-SXM4-40GB', 'memory_bytes': 40 * GIB,
                  'compute_capability': '8.0'}]}
T4 = {**A100, 'gpus': [{'uuid': 'GPU-t4-0', 'name': 'Tesla T4', 'memory_bytes': 15 * GIB,
                        'compute_capability': '7.5'}]}
SUPPORT = {('vllm', MODEL): 'supported'}


def fake_launch(version='0.11.0', *extra):
    return [sys.executable, FAKE, '--port', '{port}', '--model', MODEL, '--flavor', 'vllm', '--version', version,
            '--usage', '--stream-tokens', '6', '--chunk-delay', '0.002', *extra]


@pytest.fixture
def repo(tmp_path):
    r = Repository(tmp_path / 'db', tmp_path / 'artifacts')
    r.put_record('model', 'qwen', {'source': 'huggingface', 'repo_id': MODEL, 'revision': REV,
                                   'tokenizer_revision': REV, 'architecture': 'qwen3_5'})
    r.put_record('deployment', 'dep', {'id': 'dep', 'engine': 'vllm', 'model': MODEL, 'base_url': 'http://x',
                                       'model_ref': 'qwen@1', 'engine_version': '0.11.0',
                                       'profile': {'dtype': 'bf16'}})
    r.put_record('workload', 'wl', {'name': 'wl', 'cases': [{'prompt': 'hi'}], 'concurrency': 2})
    return r


def make_manifest(repo, **over):
    kwargs = dict(deployment_id='dep', workload_id='wl', workload_fixture_sha256=sha256_canonical(WORKLOAD),
                  measurement={'evaluator_sha256': 'e' * 64, 'rounds': 2, 'warmup_requests': 1,
                               'gates': ['readiness', 'requests_succeeded', 'correctness']},
                  limits={'max_runtime_s': 600}, hardware={'gpu_count': 1, 'min_gpu_memory_bytes': 16 * GIB},
                  max_context_tokens=8192, max_output_tokens=256, concurrency=(1, 2))
    kwargs.update(over)
    return from_records(repo, **kwargs)


def write_inputs(tmp_path, manifest, workload=WORKLOAD):
    (tmp_path / 'm.json').write_text(manifest.to_json())
    (tmp_path / 'w.json').write_text(json.dumps(workload))
    return tmp_path / 'm.json', tmp_path / 'w.json'


def execute(tmp_path, manifest, environment=A100, launch=None, **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    m, w = write_inputs(tmp_path, manifest)
    return asyncio.run(run(m, w, tmp_path / 'out', environment=environment, provider='colab',
                           model_support=kw.pop('model_support', SUPPORT), launch=launch or fake_launch(),
                           ready_timeout=20, request_timeout=30, **kw))


def rewrite(bundle, out, mutate):
    with zipfile.ZipFile(bundle) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    mutate(members)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    out.write_bytes(buf.getvalue())
    return out


def test_round_trip_records_hardware_evidence_and_trials(repo, tmp_path):
    manifest = make_manifest(repo)
    bundle = execute(tmp_path, manifest)
    index, opened, members, _ = open_bundle(bundle, expected=manifest)
    assert index.status == 'completed' and opened.experiment_id == manifest.experiment_id
    # model size is not recorded, so fit is unknown: the runner probes before measuring
    assert index.preflight['status'] == 'probe_required'
    assert [s.name for s in index.steps] == ['discover_environment', 'workload_fixture', 'compatibility', 'probe',
                                             'round-0', 'round-1']
    assert index.gates == {'readiness': 'passed', 'requests_succeeded': 'passed', 'correctness': 'missing'}
    assert index.launch['engine_version_observed'] == '0.11.0'
    assert 'run/scorecard.json' in members and 'run/round-1/vllm/requests.jsonl' in members

    out = import_bundle(repo, bundle, expected=manifest)
    assert out['status'] == 'completed' and len(out['trials']) == 2 * 2 * 2  # rounds x cases x concurrency
    trial = repo.read('trial', out['trials'][0].split('@')[0])['body']
    assert trial['hardware_ref'] == out['hardware_ref'] and trial['deployment_ref'] == 'dep@1'
    assert trial['raw_artifact_sha256'] == out['bundle_sha256'] and trial['role'] == 'baseline'
    assert trial['metrics']['e2e_ms'] > 0 and trial['metrics']['failure_rate'] == 0.0
    assert trial['errors'] == ['gate correctness: missing']  # an unrun gate is never proof
    hardware = repo.read('hardware', out['hardware_ref'].split('@')[0])['body']
    assert hardware['gpus'][0]['name'] == 'NVIDIA A100-SXM4-40GB'
    assert repo.read('evidence', out['evidence_ref'].split('@')[0])['body']['verification'] == 'verified'
    assert repo.read_artifact(out['bundle_sha256'])  # the raw zip is kept
    assert import_bundle(repo, bundle)['trials'] == out['trials']  # idempotent


def test_tampered_or_padded_bundles_are_refused(repo, tmp_path):
    bundle = execute(tmp_path, make_manifest(repo))

    def edit(m):
        m['run/round-0/vllm/requests.jsonl'] = m['run/round-0/vllm/requests.jsonl'].replace(b'"success": true',
                                                                                             b'"success": false')
    with pytest.raises(BundleRejected, match='checksum'):
        open_bundle(rewrite(bundle, tmp_path / 't1.zip', edit))
    with pytest.raises(BundleRejected, match='unlisted'):
        open_bundle(rewrite(bundle, tmp_path / 't2.zip', lambda m: m.update({'extra.txt': b'x'})))
    with pytest.raises(BundleRejected, match='missing'):
        open_bundle(rewrite(bundle, tmp_path / 't3.zip', lambda m: m.pop('run/scorecard.json')))
    with pytest.raises(BundleRejected, match='unsafe'):
        open_bundle(rewrite(bundle, tmp_path / 't4.zip', lambda m: m.update({'../evil': b'x'})))

    def swap_dtype(m):
        index = json.loads(m['bundle.json'])
        index['manifest']['dtype'] = 'fp16'
        m['bundle.json'] = json.dumps(index).encode()
    with pytest.raises(BundleRejected, match='declared digest'):
        open_bundle(rewrite(bundle, tmp_path / 't5.zip', swap_dtype))

    def leak(m):
        m['run/round-0/vllm/server.log'] += b'\nAuthorization: Bearer abcdefghijklmnop1234\n'
        index = json.loads(m['bundle.json'])
        import hashlib
        index['files']['run/round-0/vllm/server.log'] = hashlib.sha256(m['run/round-0/vllm/server.log']).hexdigest()
        m['bundle.json'] = json.dumps(index).encode()
    with pytest.raises(BundleRejected, match='credential'):
        open_bundle(rewrite(bundle, tmp_path / 't6.zip', leak))


def test_manifest_mismatch_is_refused(repo, tmp_path):
    manifest = make_manifest(repo)
    bundle = execute(tmp_path, manifest)
    other = make_manifest(repo, dtype='fp16')
    with pytest.raises(BundleRejected, match='manifest mismatch'):
        import_bundle(repo, bundle, expected=other)


def test_unsupported_preflight_launches_nothing_and_records_no_trials(repo, tmp_path):
    manifest = make_manifest(repo, hardware={'gpu_count': 1})  # BF16 on a T4 (CC 7.5)
    marker = tmp_path / 'launched'
    bundle = execute(tmp_path, manifest, environment=T4, launch=[sys.executable, '-c',
                                                                 f'open({str(marker)!r}, "w")'])
    assert not marker.exists()
    index, _, members, _ = open_bundle(bundle)
    assert index.status == 'refused' and index.preflight['status'] == 'unsupported'
    assert any(a['change'] == {'dtype': 'fp16'} for a in index.preflight['alternatives'])
    assert not any(n.startswith('run/round-') for n in members)
    out = import_bundle(repo, bundle)
    assert out['trials'] == [] and 'refused' in out['trials_skipped']
    assert out['gates']['readiness'] == 'missing'


def test_probe_required_runs_the_probe_first_then_measures(repo, tmp_path):
    manifest = make_manifest(repo)
    bundle = execute(tmp_path, manifest, model_support={})  # engine x model unknown
    index, _, members, _ = open_bundle(bundle)
    assert index.preflight['status'] == 'probe_required' and index.status == 'completed'
    assert [s.name for s in index.steps][3] == 'probe' and 'run/probe/vllm/requests.jsonl' in members


def test_failed_probe_and_wrong_engine_version_are_recorded_as_failures(repo, tmp_path):
    manifest = make_manifest(repo)
    broken = execute(tmp_path / 'a', manifest, model_support={},
                     launch=[sys.executable, '-c', 'raise SystemExit(3)'])
    index, _, _, _ = open_bundle(broken)
    assert index.status == 'probe_failed' and index.gates['readiness'] == 'failed'
    wrong = execute(tmp_path / 'b', manifest, launch=fake_launch('0.12.0'))
    index, _, _, _ = open_bundle(wrong)
    assert index.status == 'failed' and '0.12.0' in index.failure_reason
    assert import_bundle(repo, wrong)['trials'] == []


def test_workload_must_match_the_manifest_fixture(repo, tmp_path):
    manifest = make_manifest(repo)
    m, _ = write_inputs(tmp_path, manifest)
    w = tmp_path / 'other.json'
    w.write_text(json.dumps({**WORKLOAD, 'seed': 4}))
    bundle = asyncio.run(run(m, w, tmp_path / 'out', environment=A100, model_support=SUPPORT,
                             launch=fake_launch(), ready_timeout=20))
    index, _, _, _ = open_bundle(bundle)
    assert index.status == 'refused' and 'fixture' in index.failure_reason


def test_failed_gate_stays_failed_after_import(repo, tmp_path):
    manifest = make_manifest(repo)
    bundle = execute(tmp_path, manifest)

    def fail_gate(m):
        index = json.loads(m['bundle.json'])
        index['gates']['correctness'] = 'failed'
        m['bundle.json'] = json.dumps(index).encode()
    out = import_bundle(repo, rewrite(bundle, tmp_path / 'g.zip', fail_gate))
    assert out['gates']['correctness'] == 'failed'
    trial = repo.read('trial', out['trials'][0].split('@')[0])['body']
    assert 'gate correctness: failed' in trial['errors']


def test_exported_notebook_pins_identity_and_order(repo, tmp_path):
    manifest = make_manifest(repo)
    nb = export_notebook(manifest, WORKLOAD, provider='colab', source_revision=SRC)
    meta = nb['metadata']['inferstack']
    assert meta['experiment_id'] == manifest.experiment_id and meta['manifest_digest'] == manifest.digest
    text = [''.join(c['source']) for c in nb['cells']]
    assert manifest.experiment_id in text[0] and SRC in text[1]
    assert "'vllm==0.11.0'" in text[3]
    assert '--preflight-only' in text[4] and 'SystemExit' in text[4]
    assert "'--out'" in text[5] and 'files.download' in text[6]
    # the embedded manifest is byte-for-byte the canonical one the digest covers
    namespace = {}
    exec(text[1].split('BASE =')[0].split('PLATFORM =', 1)[0] + 'MANIFEST_JSON = ' +
         text[1].split('MANIFEST_JSON = ', 1)[1].split('\n', 1)[0], namespace)
    assert ExecutionManifestV1.from_json(namespace['MANIFEST_JSON']).digest == manifest.digest
    compile('\n'.join(text[1:]), 'notebook', 'exec')  # every code cell is valid Python
    kaggle = export_notebook(manifest, WORKLOAD, provider='kaggle', source_revision=SRC)
    assert kaggle['metadata']['inferstack']['export_id'] != meta['export_id']


def test_export_refuses_mismatched_inputs(repo):
    manifest = make_manifest(repo)
    with pytest.raises(ValueError, match='fixture'):
        export_notebook(manifest, {**WORKLOAD, 'seed': 9}, source_revision=SRC)
    with pytest.raises(ValueError, match='40-hex'):
        export_notebook(manifest, WORKLOAD, source_revision='main')
    native = ExecutionManifestV1.model_validate({**json.loads(manifest.to_json()), 'engine': {
        'name': 'inferstack_native', 'version': '2.0.0rc1', 'source_revision': SRC}})
    with pytest.raises(ValueError, match='launch template'):
        export_notebook(native, WORKLOAD, source_revision=SRC)
    with pytest.raises(ValueError, match='source_revision'):
        export_notebook(native, WORKLOAD, source_revision='b' * 40, launch=['x', '{port}'])


def test_derived_vllm_launch_pins_revision_context_and_profile(repo):
    manifest = make_manifest(repo)
    argv = launch_argv(manifest, '/venv/bin/vllm')
    assert argv[:3] == ['/venv/bin/vllm', 'serve', MODEL]
    assert argv[argv.index('--revision') + 1] == REV and argv[argv.index('--max-model-len') + 1] == '8192'
    assert argv[argv.index('--dtype') + 1] == 'bfloat16'
