"""H0 acceptance (INFERENCE_HARNESS_REVIEW.md R1-R5): evidence and executable contracts are trustworthy.

Each test is the corrected counterpart of a reproduction in
docs/verification/harness_gap_probes.py.
"""
import asyncio
import time

import pytest

from inferstack.control.worker import OptimizationWorker
from inferstack.execution.executor import BuildFailed, IsolatedExecutor, with_flags
from inferstack.optimization.dag import PlanRejected, validate_plan
from inferstack.optimization.experiment import EndpointExecutor
from inferstack.serving.evidence import LiveStream
from inferstack.storage.repository import Repository


def plan(params, action_type='config_change'):
    return {'hypothesis': 'H0 acceptance probe for a bounded candidate treatment.', 'baseline_ref': 'base@1',
            'evidence_ids': [],
            'actions': [{'id': 'change', 'type': action_type, 'depends_on': [], 'params': params},
                        {'id': 'measure', 'type': 'measure', 'depends_on': ['change'], 'params': {}}],
            'expected_measurements': ['e2e_ms'], 'gates': ['correctness', 'quality', 'health', 'comparable', 'complete'],
            'estimate': {'gpu_seconds': 1, 'wall_seconds': 1, 'provider_tokens': 0},
            'stop_rule': {'max_iterations': 1, 'stop_on': ['keep']}}


def validate(repo, raw):
    return validate_plan(raw, repository=repo, baseline_id='base', candidate_ids=[], allowed_evidence=[],
                         history_signatures=set())


@pytest.fixture
def repo(tmp_path):
    return Repository(tmp_path / 'db', tmp_path / 'artifacts')


def test_r3_loader_is_accepted_only_where_both_layers_can_execute_it(repo):
    repo.put_record('deployment', 'base', {'id': 'base', 'engine': 'vllm', 'model': 'm', 'base_url': 'http://b'})
    with pytest.raises(PlanRejected, match='supported_load_formats'):
        validate(repo, plan({'profile_patch': {'load_format': 'runai_streamer'}}))
    with pytest.raises(BuildFailed, match='supported_load_formats'):
        with_flags(['vllm', 'serve', 'm'], 'vllm', {'load_format': 'runai_streamer'},
                   deployment=repo.read('deployment', 'base')['body'])
    repo.put_record('deployment', 'base', {'id': 'base', 'engine': 'vllm', 'model': 'm', 'base_url': 'http://b',
                                           'profile': {'supported_load_formats': ['runai_streamer']}},
                    expected_revision=1)
    raw = plan({'profile_patch': {'load_format': 'runai_streamer'}})
    raw['baseline_ref'] = 'base@2'
    validate(repo, raw)  # accepted by the planner ...
    argv = with_flags(['vllm', 'serve', 'm'], 'vllm', {'load_format': 'runai_streamer'},
                      deployment=repo.read('deployment', 'base')['body'])
    assert argv[-2:] == ['--load-format', 'runai_streamer']  # ... and materialized by the executor


def test_r3_every_accepted_native_patch_is_materializable(repo, tmp_path):
    from pathlib import Path
    repo.put_record('deployment', 'base', {'id': 'base', 'engine': 'inferstack_native', 'model': 'qwen',
                                           'base_url': 'http://b'})
    patch = {'decode_priority': True, 'max_prefill_chunk_tokens': 256}
    validate(repo, plan({'profile_patch': patch}))
    out = with_flags(['python', '-m', 'inferstack', 'serve'], 'inferstack_native', patch, workspace=tmp_path,
                     source_root=Path(__file__).resolve().parents[1])
    assert out[-2] == '--profile' and Path(out[-1]).is_file()
    with pytest.raises(PlanRejected, match='must be bool'):
        validate(repo, plan({'profile_patch': {'decode_priority': 'yes'}}))


def test_r4_grouped_event_cannot_manufacture_zero_itl():
    stream = LiveStream(0)
    stream.feed(b'data: {"choices":[{"delta":{"content":"two tokens"}}],"usage":{"completion_tokens":2}}\n\n', 1)
    assert stream.phases(1.2)['itl_ms'] is None


@pytest.mark.asyncio
async def test_r5_slow_cleanup_keeps_leases_and_reaches_a_terminal_state(repo):
    class Planner:
        async def plan(self, request, history):
            return {'hypothesis': 'slow cleanup must not starve lease renewal'}

    stops = []

    class SlowProvider:
        def stop(self, external):
            time.sleep(0.65)  # a synchronous stop slower than two lease periods
            stops.append(external)
            return True

    class Allocations:
        def release(self, identifier, note):
            pass

        def mark_orphaned(self, identifier, note):
            pass

    class Executor(EndpointExecutor):
        def __init__(self):
            self.active, self.owned = {}, {}
            self.provider, self.allocations = SlowProvider(), Allocations()

        async def execute(self, plan, namespace, context=None):
            self.owned[namespace] = ('probe-allocation', {})
            return {'id': namespace}

    class Verifier:
        async def verify(self, plan, candidate):
            return {'decision': 'reject', 'reasons': ['probe only']}

    job = repo.submit('cleanup-probe', {'deployment_id': 'probe'})
    result = await OptimizationWorker(repo, Planner(), Executor(), Verifier(), lease_seconds=0.3,
                                      max_iterations=1).once()
    assert stops, 'cleanup ran'
    assert result['state'] in ('completed', 'failed'), (result['state'], [e['kind'] for e in result['events']])
    assert repo.get(job)['result'] is not None


@pytest.mark.asyncio
async def test_r5_event_loop_stays_responsive_during_build_teardown(tmp_path, repo):
    """_end_build (provider stop of tagged build processes) runs off the event loop."""
    ticks = []

    async def ticker():
        for _ in range(10):
            ticks.append(time.monotonic())
            await asyncio.sleep(0.05)

    class SlowLocal:
        name = 'local'
        host = 'h'

        def stop(self, external, grace_seconds=10.0):
            time.sleep(0.4)
            return True

    class Allocations:
        def get(self, alloc_id):
            return {'external': {'allocation_id': alloc_id}}

        def release(self, alloc_id, note):
            pass

    executor = IsolatedExecutor.__new__(IsolatedExecutor)
    executor.provider, executor.allocations = SlowLocal(), Allocations()
    await asyncio.gather(ticker(), executor._end_build('a', 'build_complete'))
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert max(gaps) < 0.3, gaps  # the loop kept running while the stop waited


def test_r2_prose_source_change_is_a_proposal_not_an_executable_plan(repo):
    repo.put_record('deployment', 'base', {'id': 'base', 'engine': 'vllm', 'model': 'm', 'base_url': 'http://b'})
    repo.put_record('technique', 'tech', {'name': 'tech', 'summary': 's', 'engines': [{'engine': 'vllm'}]})
    _, normalized = validate(repo, plan({'technique_id': 'tech', 'files': ['src/inferstack/kernels/x.py'],
                                         'description': 'implement a new kernel'}, 'source_patch'))
    assert normalized['executable'] is False
    assert normalized['needs_implementation'][0]['action_id'] == 'change'
    with pytest.raises(PlanRejected, match='stored patch rejected'):
        validate(repo, plan({'technique_id': 'tech', 'files': ['src/inferstack/kernels/x.py'],
                             'description': 'd', 'patch_sha256': 'f' * 64}, 'source_patch'))


@pytest.mark.asyncio
async def test_r2_worker_never_executes_a_proposal(repo):
    executed = []

    class Planner:
        async def plan(self, request, history):
            return {'hypothesis': 'needs code', 'executable': False,
                    'needs_implementation': [{'action_id': 'patch', 'technique_id': 't'}]}

    class Executor:
        async def execute(self, *args, **kwargs):
            executed.append(args)

    job = repo.submit('proposal', {'deployment_id': 'base'})
    result = await OptimizationWorker(repo, Planner(), Executor(), None, max_iterations=3).once()
    assert executed == [] and result['state'] == 'completed'
    assert result['result']['decision'] == 'needs_implementation'
    assert 'needs_implementation' in [e['kind'] for e in repo.get(job)['events']]


@pytest.mark.asyncio
async def test_r2_executor_refuses_plans_without_a_patch_artifact(tmp_path, repo):
    executor = IsolatedExecutor.__new__(IsolatedExecutor)
    plan_body = {'actions': [{'id': 'p', 'type': 'source_patch', 'params': {'technique_id': 't'}}],
                 'action_order': ['p']}
    with pytest.raises(BuildFailed, match='needs implementation'):
        await IsolatedExecutor.execute(executor, plan_body, 'ns', context={'job': {}})


def test_capability_evidence_references_are_current():
    """H0: a capability marked `tested` must cite a test file (and test) that exists.
    `tested` means an automated repository test on CPU, never GPU qualification."""
    import re
    from pathlib import Path
    from inferstack.engines.capabilities import _DEFAULTS
    root = Path(__file__).resolve().parents[1]
    stale = []
    for engine, ops in _DEFAULTS.items():
        for op, (state, basis, evidence) in ops.items():
            for path, test in re.findall(r'(tests/[\w/]+\.py)(?:::(\w+))?', evidence):
                target = root / path
                if not target.is_file() or (test and f'def {test}(' not in target.read_text()):
                    stale.append(f'{engine}.{op}: {path}{"::" + test if test else ""}')
            if basis == 'tested' and 'tests/' not in evidence and not re.search(r'\w+/\w+\.py', evidence):
                stale.append(f'{engine}.{op}: tested without a test reference ({evidence})')
    assert not stale, stale
