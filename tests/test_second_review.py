"""Regression tests for the second independent review (R01-R15) beyond its probes."""
import asyncio
import sys
import time

import pytest

from inferstack.control.store import Store
from inferstack.engines.managed import DeploymentSpec
from inferstack.infrastructure import AllocationManager, LocalProcessProvider, reconcile
from inferstack.infrastructure.allocations import ResourceConflict
from inferstack.integrations.outbox import Outbox
from inferstack.optimization.experiment import EndpointExecutor, EndpointVerifier
from inferstack.releases.attestation import _expected
from inferstack.releases.manager import ReleaseManager
from inferstack.storage.repository import Repository


@pytest.fixture
def repo(tmp_path):
    return Repository(tmp_path / 'db', tmp_path / 'artifacts')


def test_crashed_build_process_is_found_and_stopped_by_its_allocation_tag(repo, tmp_path):
    """R10: a build step's subprocess survives a worker crash; the reconciler stops it by tag."""
    from inferstack.execution.sandbox import Sandbox
    from inferstack.infrastructure.local import TAG
    import os
    import subprocess
    job = repo.submit('build-crash', {})
    worker = repo.register_worker(1)
    claim = repo.claim(1, worker_id=worker)
    provider = LocalProcessProvider()
    manager = AllocationManager(repo, worker)
    alloc = manager.intent(claim, 'experiment', purpose='build:x', provider='local', resource_type='process',
                           spec={'kind': 'build'})
    external = {'host': provider.host, 'kind': 'build', 'allocation_id': alloc['id']}
    manager.activate(alloc['id'], claim, external)
    # The build step outlives its worker (hard kill): a tagged process still running.
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],
                            env={**os.environ, TAG: alloc['id']}, start_new_session=True)
    try:
        deadline = time.time() + 10
        while proc.pid not in provider.find_tagged(alloc['id']) and time.time() < deadline:
            time.sleep(0.1)
        repo.clock = lambda: time.time() + 3600  # the worker's registration has expired
        [action] = reconcile(repo, provider)
        assert action['outcome'] == 'orphan_stopped', action
        assert proc.wait(timeout=10) is not None
        assert manager.get(alloc['id'])['state'] == 'released'
    finally:
        if proc.poll() is None:
            proc.kill()


def test_deadline_before_claim_enqueues_terminal_outcome(repo):
    """R14: every terminal transition writes the FactoryOS outcome, including deadline expiry."""
    now = time.time()
    repo.submit('late', {'outcome_callback': {'issue_id': 'i', 'idempotency_key': 'k'}}, deadline_at=now - 1)
    assert repo.claim(10) is None
    [row] = Outbox(repo).list()
    assert row['payload']['status'] == 'failed' and row['payload']['error'] == 'DeadlineExceeded'


def test_cancel_outcome_is_idempotent(repo):
    job = repo.submit('c', {'outcome_callback': {'issue_id': 'i', 'idempotency_key': 'k'}})
    assert repo.cancel(job) and not repo.cancel(job)
    [row] = Outbox(repo).list()
    assert row['payload']['status'] == 'cancelled'


def verifier(repo, tmp_path, **kw):
    return EndpointVerifier(repo, EndpointExecutor(repo, tmp_path / 'logs'), 'base',
                            [{'prompt': 'p', 'expected': 'four', 'max_tokens': 4}],
                            [{'metric': 'e2e_ms', 'direction': 'min'}], **kw)


@pytest.mark.asyncio
async def test_fragmented_sse_event_is_timed_when_it_completes(repo, tmp_path, monkeypatch):
    """R03: an SSE event split across reads counts once, when complete; usage events are not content."""
    v = verifier(repo, tmp_path)

    class Stream:
        async def stream(self, payload):
            yield b'data: {"choices":[{"delta":{"con'
            yield b'tent":"four"}}]}\n\n'
            yield b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
            yield b'data: [DONE]\n\n'

    ticks = iter([0.0, 0.050, 0.080])
    monkeypatch.setattr('inferstack.optimization.experiment.time.perf_counter', lambda: next(ticks))
    result = await v.phases(Stream())
    assert result['ttft_client_ms'] == pytest.approx(50.0)  # timed at the read completing the event
    assert result['chunk_gap_client_ms'] is None  # one content event: no content gap


@pytest.mark.asyncio
async def test_stream_without_terminal_event_has_no_ttft(repo, tmp_path):
    v = verifier(repo, tmp_path)

    class Stream:
        async def stream(self, payload):
            yield b'data: {"choices":[{"text":"four"}]}\n\n'

    assert (await v.phases(Stream()))['ttft_client_ms'] is None


def test_logprob_gate_distinguishes_missing_evidence_from_disagreement(repo, tmp_path):
    """R04: missing cases / malformed values are None; a different greedy sequence is False."""
    v = verifier(repo, tmp_path)
    good = [(['four'], [-0.1])]
    assert v.logit_agreement(good, good)[0] is True
    assert v.logit_agreement(good + good, good)[0] is None           # case coverage differs
    assert v.logit_agreement(good, [(['four'], [float('nan')])])[0] is None
    assert v.logit_agreement(good, [(['five'], [-0.1])])[0] is False
    assert v.logit_agreement(good, [(['four', 'x'], [-0.1, -0.2])])[0] is False


@pytest.mark.asyncio
async def test_gpu_launches_reserve_distinct_identified_devices(repo, tmp_path):
    """R12: each GPU launch gets its own identified device; when none is free it is refused."""
    spawned = []

    class Process:
        pid = 1

    class Provider:
        name, resource_type = 'local', 'process'

        async def spawn(self, alloc_id, argv, log, **kwargs):
            spawned.append(kwargs['env']['CUDA_VISIBLE_DEVICES'])
            return Process()

        def external_for(self, pid):
            return {'host': 'h', 'pid': pid}

    manager = AllocationManager(repo, 'w')
    executor = EndpointExecutor(repo, tmp_path / 'logs', allocations=manager, provider=Provider(),
                                gpu_devices=['GPU-a', 'GPU-b'])
    for name in ('a', 'b', 'c'):
        repo.submit(name, {})
        claim = repo.claim(100)
        spawn = executor._spawner({'job': claim, 'step_key': 's', 'namespace': name},
                                  DeploymentSpec(name, 'vllm', 'm', 'http://' + name, command=('vllm',)))
        if name == 'c':
            with pytest.raises(ResourceConflict):
                await spawn(['vllm'], None)
        else:
            await spawn(['vllm'], None)
    assert sorted(spawned) == ['GPU-a', 'GPU-b']


def test_release_refuses_unresolved_mutable_identity(repo):
    """R05: a declared identity reference must be resolved to a pinned revision."""
    repo.put_record('model', 'mdl', {'source': 'huggingface', 'repo_id': 'org/m', 'revision': 'b' * 40})
    manager = ReleaseManager(repo)
    with pytest.raises(ValueError, match='pinned revision'):
        manager.freeze_identity({'model_ref': 'mdl', 'model': 'm'}, {'identity': {}})
    frozen = manager.freeze_identity({'model_ref': 'mdl@1', 'model': 'm'}, {'identity': {}})
    assert frozen['model_revision'] == 'b' * 40 and frozen['model_ref'] == 'mdl@1'


def test_unpinned_reference_without_frozen_identity_is_not_attestable(repo):
    repo.put_record('model', 'mdl', {'source': 'huggingface', 'repo_id': 'org/m', 'revision': 'b' * 40})
    expected = _expected(repo, {'model': 'm', 'model_ref': 'mdl'}, {'body': {}})
    assert expected['model_revision'] is None  # -> attestation 'unverified', never the latest record
