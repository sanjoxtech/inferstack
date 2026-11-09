"""H1 acceptance: versioned context snapshot and bounded cross-campaign experience.

Real worker jobs produce real verification evidence; a fresh job must then
(1) find a relevant earlier failure, (2) retrieve a matching success with exact
references, (3) treat a result from a different setup as unproven, (4) exclude
unrelated and unreadable evidence, and (5) refuse to repeat a treatment that
already failed on this exact setup.
"""
import json

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from inferstack.control.api import create_control_app
from inferstack.control.worker import OptimizationWorker
from inferstack.memory import context_snapshot, experience
from inferstack.optimization.dag import validate_plan
from inferstack.optimization.experiment import EndpointExecutor, EndpointVerifier
from inferstack.optimization.planner import ToolPlanner
from inferstack.providers.base import ProviderCapabilities, ProviderResponse
from inferstack.storage.repository import Repository

GPU = {'uuid': 'GPU-1', 'name': 'NVIDIA A100-SXM4-40GB', 'memory_bytes': 40 * 2**30, 'compute_capability': '8.0'}
T4 = {'uuid': 'GPU-2', 'name': 'Tesla T4', 'memory_bytes': 15 * 2**30, 'compute_capability': '7.5'}


class MemoryEngine:
    def __init__(self, spec):
        self.spec = spec

    async def start(self, *args, **kwargs):
        pass

    async def ready(self):
        return True

    async def close(self):
        pass


LATENCY = {'slow': 150.0, 'fast': 60.0, 't4-fast': 60.0, 'other-fast': 60.0}


class Verifier(EndpointVerifier):
    """Real verify() (evidence, identity pins, policy); deterministic per-deployment latency."""
    async def trial(self, engine):
        return {'e2e_ms': LATENCY.get(engine.spec.id, 100.0)}, True, ['four']


@pytest.fixture
def repo(tmp_path):
    r = Repository(tmp_path / 'db', tmp_path / 'artifacts')
    r.put_record('hardware', 'a100', {'provider': 'colab', 'observed_at': 1.0, 'gpus': [GPU], 'cuda_version': '12.8'})
    r.put_record('hardware', 't4', {'provider': 'kaggle', 'observed_at': 1.0, 'gpus': [T4], 'cuda_version': '12.8'})
    r.put_record('model', 'qwen', {'source': 'huggingface', 'repo_id': 'Qwen/Qwen3.5-2B-Base', 'revision': 'b' * 40})
    r.put_record('model', 'other', {'source': 'huggingface', 'repo_id': 'org/other', 'revision': 'c' * 40})
    common = {'engine': 'vllm', 'engine_version': '0.11.0', 'profile': {'dtype': 'bfloat16'}}
    for name, model, model_ref, hw in (('base', 'qwen', 'qwen@1', 'a100@1'), ('slow', 'qwen', 'qwen@1', 'a100@1'),
                                       ('fast', 'qwen', 'qwen@1', 'a100@1'), ('base-t4', 'qwen', 'qwen@1', 't4@1'),
                                       ('t4-fast', 'qwen', 'qwen@1', 't4@1'), ('base-other', 'other', 'other@1', 'a100@1'),
                                       ('other-fast', 'other', 'other@1', 'a100@1')):
        r.put_record('deployment', name, {'id': name, 'model': model, 'base_url': 'http://' + name,
                                          'model_ref': model_ref, 'hardware_ref': hw, **common})
    return r


def normalized_plan(repo, baseline, candidate):
    raw = {'hypothesis': f'try {candidate} for {baseline}', 'baseline_ref': f'{baseline}@1', 'evidence_ids': [],
           'actions': [{'id': 'swap', 'type': 'config_change', 'depends_on': [],
                        'params': {'candidate_deployment_id': candidate}},
                       {'id': 'measure', 'type': 'measure', 'depends_on': ['swap'], 'params': {}}],
           'expected_measurements': ['e2e_ms'], 'gates': ['correctness', 'quality', 'health', 'comparable', 'complete'],
           'estimate': {'gpu_seconds': 1, 'wall_seconds': 1}, 'stop_rule': {'max_iterations': 1, 'stop_on': ['keep']}}
    return validate_plan(raw, repository=repo, baseline_id=baseline, candidate_ids=[candidate], allowed_evidence=[],
                         history_signatures=set())[1]


async def campaign(repo, tmp_path, baseline, candidate):
    """One real worker job: plan -> execute -> real verifier evidence -> terminal result."""
    plan = normalized_plan(repo, baseline, candidate)

    class Planner:
        async def plan(self, request, history):
            return plan

    executor = EndpointExecutor(repo, tmp_path / 'logs', engine_factory=MemoryEngine)
    verifier = Verifier(repo, executor, baseline, [{'prompt': '2+2', 'expected': 'four', 'max_tokens': 4}],
                        [{'metric': 'e2e_ms', 'direction': 'min'}])
    job = repo.submit(f'campaign-{baseline}-{candidate}', {'deployment_id': baseline, 'goal': 'reduce latency'})
    result = await OptimizationWorker(repo, Planner(), executor, verifier, max_iterations=1).once()
    assert result['state'] == 'completed', result
    return job, plan


@pytest_asyncio.fixture
async def history(repo, tmp_path):
    jobs = {}
    jobs['failed'], plans_failed = await campaign(repo, tmp_path, 'base', 'slow')        # reject on target setup
    jobs['success'], plans_success = await campaign(repo, tmp_path, 'base', 'fast')      # keep on target setup
    jobs['t4'], _ = await campaign(repo, tmp_path, 'base-t4', 't4-fast')                 # keep on a T4 setup
    jobs['other'], _ = await campaign(repo, tmp_path, 'base-other', 'other-fast')        # unrelated model
    return jobs, plans_failed, plans_success


@pytest.mark.asyncio
async def test_fresh_job_retrieves_failure_success_and_treats_other_setup_as_unproven(repo, history):
    jobs, failed_plan, success_plan = history
    found = experience(repo, 'base')
    by_job = {i['job_id']: i for i in found['items']}
    failure, success, t4 = by_job[jobs['failed']], by_job[jobs['success']], by_job[jobs['t4']]
    assert failure['applicability'] == 'applicable' and failure['decision'] == 'reject'
    assert failure['meaning'] == 'known failure on this setup'
    assert success['applicability'] == 'applicable' and success['decision'] == 'keep'
    assert success['meaning'] == 'verified success on this setup'
    assert success['candidate_ref'] == 'fast@1' and success['treatment_signature'] == success_plan['treatment_signature']
    assert len(success['evidence_sha256']) == 64 and repo.read_artifact(success['evidence_sha256'])  # exact reference
    assert t4['applicability'] == 'transferable' and t4['decision'] == 'keep'
    assert any(d.startswith('hardware.') for d in t4['differences']) and 'not proof' in t4['meaning']
    assert jobs['other'] not in by_job and found['excluded']['unrelated'] == 1
    assert found['summary'] == {'applicable_successes': 1, 'applicable_failures': 1, 'transferable': 1}
    assert found['failed_here_signatures'] == [failed_plan['treatment_signature']]
    # From the T4 baseline's point of view the A100 results are the hypotheses instead.
    from_t4 = {i['job_id']: i['applicability'] for i in experience(repo, 'base-t4')['items']}
    assert from_t4[jobs['t4']] == 'applicable' and from_t4[jobs['success']] == 'transferable'


@pytest.mark.asyncio
async def test_tampered_evidence_is_reported_inaccessible_not_used(repo, history):
    jobs, _, _ = history
    sha = next(i['evidence_sha256'] for i in experience(repo, 'base')['items'] if i['job_id'] == jobs['success'])
    (repo.artifact_root / sha).write_bytes(b'{"forged": true}')
    found = experience(repo, 'base')
    assert jobs['success'] not in {i['job_id'] for i in found['items']}
    assert any(x['job_id'] == jobs['success'] and 'digest' in x['reason'] for x in found['excluded']['inaccessible'])
    assert found['summary']['applicable_successes'] == 0


@pytest.mark.asyncio
async def test_retrieval_is_bounded(repo, history):
    found = experience(repo, 'base', limit=1, scan_jobs=2)
    assert len(found['items']) == 1 and found['scanned_jobs'] <= 2
    with pytest.raises(ValueError):
        experience(repo, 'base', limit=0)


@pytest.mark.asyncio
async def test_context_snapshot_is_versioned_content_addressed_and_reports_missing(repo, history):
    first = context_snapshot(repo, 'base', workload=[{'prompt': '2+2', 'expected': 'four', 'max_tokens': 4}])
    snap = first['snapshot']
    assert snap['version'] == 1 and snap['deployment_ref'] == 'base@1'
    assert snap['identity']['model']['revision'] == 'b' * 40 and snap['identity']['hardware']['cuda_version'] == '12.8'
    assert snap['experience']['summary']['applicable_failures'] == 1
    assert json.loads(repo.read_artifact(first['sha256'])) == snap  # stored exactly, by digest
    assert any('live serving evidence' in m for m in snap['missing'])
    bare = context_snapshot(repo, 'base', store=False)['snapshot']
    assert any(m.startswith('workload') for m in bare['missing'])


class ScriptedProvider:
    """Duck-typed chat provider: proposes the treatment that already failed, then a new one."""
    name, model = 'scripted', 'scripted-1'
    capabilities = ProviderCapabilities(tools=False, json_mode=True)

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    async def chat(self, messages, tools=None):
        self.seen.append([dict(m) for m in messages])
        return ProviderResponse(content=json.dumps(self.replies.pop(0)), tool_calls=[], finish_reason='stop',
                                usage={'total_tokens': 10}, raw={})


@pytest.mark.asyncio
async def test_planner_sees_prior_campaigns_and_will_not_repeat_a_failed_treatment(repo, history):
    def raw(candidate):
        return {'hypothesis': f'try the {candidate} candidate deployment', 'baseline_ref': 'base@1', 'evidence_ids': [],
                'actions': [{'id': 'swap', 'type': 'config_change', 'depends_on': [],
                             'params': {'candidate_deployment_id': candidate}},
                            {'id': 'measure', 'type': 'measure', 'depends_on': ['swap'], 'params': {}}],
                'expected_measurements': ['e2e_ms'], 'gates': ['correctness', 'quality', 'health', 'comparable',
                                                               'complete'],
                'estimate': {'gpu_seconds': 1, 'wall_seconds': 1}, 'stop_rule': {'max_iterations': 1,
                                                                                  'stop_on': ['keep']}}
    provider = ScriptedProvider([raw('slow'), raw('fast')])
    planner = ToolPlanner(repo, provider, candidate_ids=['slow', 'fast'])
    plan = await planner.plan({'deployment_id': 'base', 'goal': 'reduce latency'}, [])
    assert plan['candidate_id'] == 'fast'
    context = json.loads(provider.seen[0][1]['content'])
    assert context['prior_campaigns']['summary']['applicable_failures'] == 1
    correction = provider.seen[1][-1]['content']
    assert 'already tried' in correction and 'earlier campaign' in correction


@pytest.mark.asyncio
async def test_control_api_and_cli_expose_experience_and_context(repo, history, tmp_path):
    client = TestClient(create_control_app(repo, 'tok'))
    h = {'Authorization': 'Bearer tok'}
    found = client.get('/control/experience/base', headers=h, params={'limit': 5}).json()
    assert found['summary']['applicable_failures'] == 1
    assert client.get('/control/experience/missing', headers=h).status_code == 404
    snap = client.post('/control/context/base', headers=h, json={'objectives': [{'metric': 'e2e_ms'}]}).json()
    assert len(snap['sha256']) == 64 and snap['snapshot']['workload']['objectives'] == [{'metric': 'e2e_ms'}]
    import subprocess
    import sys
    out = subprocess.run([sys.executable, '-m', 'inferstack.control', 'experience', '--deployment', 'base',
                          '--database', f"sqlite:///{tmp_path / 'db'}", '--state', str(tmp_path)],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)['summary']['applicable_successes'] == 1


def test_experience_query_runs_on_every_backend(database_url, tmp_path):
    backend = Repository(database_url, tmp_path / 'artifacts')
    backend.put_record('deployment', 'd', {'id': 'd', 'engine': 'vllm', 'model': 'm', 'base_url': 'http://d'})
    job = backend.submit('done', {'deployment_id': 'd'})
    claim = backend.claim(10)
    backend.finish(job, claim['lease_token'], {'decision': 'reject', 'experiments': [
        {'plan': {'treatment_signature': 'a' * 64}, 'verdict': {'decision': 'reject', 'evidence': {'sha256': 'f' * 64}}}]})
    found = experience(backend, 'd')
    assert found['scanned_jobs'] == 1 and found['items'] == []
    assert found['excluded']['inaccessible'][0]['reason'] == 'evidence artifact missing'
