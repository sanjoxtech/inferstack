"""P2: immutable model artifact cache, read-only serving mounts, loader option."""
import errno
import hashlib
import json
import os
import subprocess
import sys
import threading

import pytest

from inferstack.execution.executor import BuildFailed, with_flags
from inferstack.infrastructure.model_cache import CacheError, ModelCache, cache_key, local_fetcher
from inferstack.infrastructure.remote import KubernetesProvider

IMAGE = 'registry.example/vllm@sha256:' + 'a' * 64


def model_from(source, **extra):
    checksums = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(source.rglob('*')) if p.is_file()}
    return {'source': 'huggingface', 'repo_id': 'org/m', 'revision': 'b' * 40, 'checksums': checksums, **extra}


@pytest.fixture
def source(tmp_path):
    src = tmp_path / 'src'
    (src / 'sub').mkdir(parents=True)
    (src / 'model.safetensors').write_bytes(os.urandom(4096))
    (src / 'config.json').write_text('{"a": 1}')
    (src / 'sub' / 'tokenizer.json').write_text('{}')
    return src


def test_publish_verifies_and_is_idempotent(tmp_path, source):
    cache, model = ModelCache(tmp_path / 'cache'), model_from(source)
    entry = cache.publish(model, local_fetcher(source))
    assert cache.verify(model)['path'] == str(entry)
    assert cache.publish(model, lambda *_: pytest.fail('must not refetch')) == entry
    assert not any((tmp_path / 'cache' / 'staging').iterdir())
    # Published read-only (mode bits; os.access would be True for root, e.g. on Colab).
    assert all(p.stat().st_mode & 0o222 == 0 for p in entry.rglob('*') if p.is_file())


def test_checksum_mismatch_never_publishes(tmp_path, source):
    model = model_from(source)
    (source / 'model.safetensors').write_bytes(b'tampered upstream')
    cache = ModelCache(tmp_path / 'cache')
    with pytest.raises(CacheError, match='checksum mismatch'):
        cache.publish(model, local_fetcher(source))
    assert not any(cache.entries.iterdir()) and not any(cache.staging.iterdir())
    with pytest.raises(CacheError, match='cache miss'):
        cache.verify(model)


def test_incomplete_and_corrupt_entries_fail_readiness(tmp_path, source):
    cache, model = ModelCache(tmp_path / 'cache'), model_from(source)
    key, _ = cache_key(model)
    (cache.root / 'entries' / key).mkdir(parents=True)  # e.g. a crashed non-atomic writer
    with pytest.raises(CacheError, match='incomplete'):
        cache.verify(model)
    os.rmdir(cache.root / 'entries' / key)
    entry = cache.publish(model, local_fetcher(source))
    target = entry / 'model.safetensors'
    target.chmod(0o644)
    data = bytearray(target.read_bytes())
    data[0] ^= 0xFF
    target.write_bytes(bytes(data))  # same size, different bytes (bit rot / tampering)
    cache.verify(model, deep=False)  # the cheap check cannot see it
    with pytest.raises(CacheError, match='corrupt'):
        cache.verify(model)


def test_revisions_never_share_an_entry(tmp_path, source):
    a, b = model_from(source), model_from(source, revision='c' * 40)
    assert cache_key(a)[0] != cache_key(b)[0]


def test_disk_full_cleans_staging(tmp_path, source):
    cache, model = ModelCache(tmp_path / 'cache'), model_from(source)

    def full(dest, files):
        (dest / 'partial').write_bytes(b'x')
        raise OSError(errno.ENOSPC, 'No space left on device')
    with pytest.raises(CacheError, match='full'):
        cache.publish(model, full)
    assert not any(cache.staging.iterdir())


def test_credential_expiry_surfaces_as_fetch_failure(tmp_path, source):
    cache, model = ModelCache(tmp_path / 'cache'), model_from(source)

    def expired(dest, files):
        raise PermissionError('401 Unauthorized: token expired')
    with pytest.raises(CacheError, match='fetch failed'):
        cache.publish(model, expired)


def test_eviction_is_lru_and_never_removes_leased_entries(tmp_path, source):
    cache = ModelCache(tmp_path / 'cache')
    models = [model_from(source, revision=c * 40) for c in 'bcd']
    for m in models:
        cache.publish(m, local_fetcher(source))
    size = cache.verify(models[0])['bytes']
    old, mid, new = (cache.path(m) for m in models)
    os.utime(old, (1, 1)), os.utime(mid, (2, 2))
    cache.lease(models[0], 'runtime-1')  # the oldest is in use
    os.utime(old, (1, 1))
    assert cache.evict(size * 2) == [mid.name]
    with pytest.raises(CacheError, match='leased'):
        cache.evict(0)
    cache.unlease(models[0], 'runtime-1')
    assert old.name in cache.evict(0)


def test_concurrent_publishers_produce_one_entry(tmp_path, source):
    model, errors = model_from(source), []

    def run():
        try:
            ModelCache(tmp_path / 'cache').publish(model, local_fetcher(source))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
    threads = [threading.Thread(target=run) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(list((tmp_path / 'cache' / 'entries').iterdir())) == 1
    ModelCache(tmp_path / 'cache').verify(model)


def test_cli_verify_fails_closed_on_miss(tmp_path, source):
    model = model_from(source)
    cmd = [sys.executable, '-m', 'inferstack.infrastructure.model_cache']
    miss = subprocess.run(cmd + ['verify', '--root', str(tmp_path / 'c'), '--model-json', json.dumps(model)],
                          capture_output=True, text=True)
    assert miss.returncode == 2 and json.loads(miss.stdout)['ok'] is False
    ok = subprocess.run(cmd + ['publish', '--root', str(tmp_path / 'c'), '--source', str(source),
                               '--model-json', json.dumps(model)], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    again = subprocess.run(cmd + ['verify', '--root', str(tmp_path / 'c'), '--model-json', json.dumps(model)],
                           capture_output=True, text=True)
    assert again.returncode == 0 and json.loads(again.stdout)['ok'] is True


def test_read_only_verify_does_not_create_directories(tmp_path, source):
    model = model_from(source)
    with pytest.raises(CacheError):
        ModelCache(tmp_path / 'absent').verify(model)
    assert not (tmp_path / 'absent').exists()


def test_serving_pod_mounts_cache_read_only_and_verifies_before_ready(source):
    model = model_from(source)
    provider = KubernetesProvider(namespace='inferstack')
    spec = {'image': IMAGE, 'container_port': 8000, 'compile_cache_size': '8Gi',
            'model_cache': {'claim': 'models', 'mount_path': '/models', 'model': model, 'verify_image': IMAGE}}
    deployment = provider.manifests('alloc-1', spec)[0]
    pod = deployment['spec']['template']['spec']
    volumes = {v['name']: v for v in pod['volumes']}
    assert volumes['model-cache']['persistentVolumeClaim'] == {'claimName': 'models', 'readOnly': True}
    assert volumes['scratch']['emptyDir'] == {'sizeLimit': '2Gi'}  # scratch stays separate and bounded
    assert not any('hostPath' in v for v in pod['volumes'])
    mounts = {m['name']: m for m in pod['containers'][0]['volumeMounts']}
    assert mounts['model-cache']['readOnly'] is True and mounts['compile-cache']['mountPath'] == '/var/cache/engine'
    [init] = pod['initContainers']
    assert init['args'][:4] == ['python', '-m', 'inferstack.infrastructure.model_cache', 'verify']
    assert all(m.get('readOnly') for m in init['volumeMounts'] if m['name'] == 'model-cache')
    assert 'envFrom' not in pod['containers'][0]  # serving never receives artifact credentials


def test_downloader_job_owns_credentials_and_write_access(source):
    job = KubernetesProvider(namespace='inferstack').cache_job_manifest(
        'publish-m', image=IMAGE, claim='models', model=model_from(source), credentials_secret='hf-token')
    container = job['spec']['template']['spec']['containers'][0]
    assert container['envFrom'] == [{'secretRef': {'name': 'hf-token'}}]
    assert container['securityContext']['readOnlyRootFilesystem'] is True
    assert job['spec']['template']['spec']['volumes'][0]['persistentVolumeClaim'] == {'claimName': 'models'}
    with pytest.raises(ValueError, match='pinned'):
        KubernetesProvider(namespace='inferstack').cache_job_manifest('x', image='python:3', claim='models',
                                                                      model=model_from(source))


def test_load_format_is_an_allowlisted_versioned_option():
    declared = {'profile': {'supported_load_formats': ['runai_streamer']}}
    argv = with_flags(['vllm', 'serve', 'm'], 'vllm', {'load_format': 'runai_streamer'}, deployment=declared)
    assert argv[-2:] == ['--load-format', 'runai_streamer']
    with pytest.raises(BuildFailed, match='supported_load_formats'):  # image/weights not declared capable
        with_flags(['vllm', 'serve', 'm'], 'vllm', {'load_format': 'runai_streamer'})
    with pytest.raises(BuildFailed, match='load_format'):
        with_flags(['vllm', 'serve'], 'vllm', {'load_format': '/etc/passwd'})


def test_startup_phases_from_pod_timestamps():
    pod = {'metadata': {'name': 'p', 'uid': 'u', 'creationTimestamp': '2026-11-09T00:00:00Z'},
           'spec': {'nodeName': 'gpu-1'},
           'status': {'phase': 'Running',
                      'conditions': [{'type': 'PodScheduled', 'status': 'True', 'lastTransitionTime': '2026-11-09T00:00:02Z'},
                                     {'type': 'Initialized', 'status': 'True', 'lastTransitionTime': '2026-11-09T00:00:30Z'},
                                     {'type': 'Ready', 'status': 'True', 'lastTransitionTime': '2026-11-09T00:03:00Z'}],
                      'containerStatuses': [{'state': {'running': {'startedAt': '2026-11-09T00:01:00Z'}}}]}}
    provider = KubernetesProvider(namespace='inferstack')
    provider.pods = lambda external: [pod]
    [phases] = provider.startup_phases({})
    assert phases == {'pod': 'p', 'uid': 'u', 'node': 'gpu-1', 'scheduled_s': 2.0, 'initialized_s': 30.0,
                      'started_s': 60.0, 'ready_s': 180.0}
