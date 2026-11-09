"""H3 step 1 acceptance: a source idea becomes a validated, immutable stored patch before execution.

Covers the implementer handoff (human diff and coding-agent command), journaled attempts,
crash/cancel recovery, candidate identity and the worker's resume path.
"""
import asyncio
import subprocess
import sys

import pytest

from inferstack.control.worker import OptimizationWorker
from inferstack.execution import Sandbox
from inferstack.implementation import (ArtifactLocked, AttemptLimit, CommandImplementer, ImplementationDesk,
                                       ValidationFailed, candidate_identity)
from inferstack.optimization.dag import PlanRejected, validate_plan
from inferstack.storage.repository import Repository

KERNEL = 'def add(a, b):\n    return a + b\n'
FILE = 'src/inferstack/kernels/fused.py'
TESTS = ['tests/test_fused.py']


def diff(path, old, new):
    old_lines, new_lines = old.splitlines(), new.splitlines()
    body = ''.join(f'-{l}\n' for l in old_lines) + ''.join(f'+{l}\n' for l in new_lines)
    return f'--- a/{path}\n+++ b/{path}\n@@ -1,{len(old_lines)} +1,{len(new_lines)} @@\n' + body


GOOD = diff(FILE, KERNEL, 'def add(a, b):\n    # fused fast path\n    return b + a\n')
OTHER = diff(FILE, KERNEL, 'def add(a, b):\n    return sum((a, b))\n')
WRONG = diff(FILE, KERNEL, 'def add(a, b):\n    return a - b\n')
BROKEN = diff(FILE, KERNEL, 'def add(a, b)\n    return a + b\n')
CHEAT = diff('tests/test_fused.py', 'x', 'y')

needs_sandbox = pytest.mark.skipif(not Sandbox().filesystem_isolated,
                                   reason='needs a filesystem isolation boundary (sandbox-exec or bwrap)')


def proposal_plan():
    return {'hypothesis': 'A fused add kernel removes one launch on the decode path.', 'baseline_ref': 'base@1',
            'evidence_ids': [],
            'actions': [{'id': 'patch', 'type': 'source_patch', 'depends_on': [],
                         'params': {'technique_id': 'fuse', 'files': [FILE], 'description': 'fuse the add'}},
                        {'id': 'measure', 'type': 'measure', 'depends_on': ['patch'], 'params': {}}],
            'expected_measurements': ['e2e_ms'], 'gates': ['correctness', 'quality', 'health', 'comparable', 'complete'],
            'estimate': {'gpu_seconds': 1, 'wall_seconds': 1, 'provider_tokens': 0},
            'stop_rule': {'max_iterations': 1, 'stop_on': ['keep']}}


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'project'
    for path, text in {FILE: KERNEL, 'src/inferstack/__init__.py': '', 'src/inferstack/kernels/__init__.py': '',
                       'src/inferstack/evaluation/judge.py': 'THRESHOLD = 0.05\n',
                       'tests/test_fused.py': 'from inferstack.kernels.fused import add\n\n'
                                              'def test_add():\n    assert add(2, 3) == 5\n'}.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    git = lambda *a: subprocess.run(['git', '-C', str(root), *a], check=True, capture_output=True)
    git('init', '-q')
    git('add', '-A')
    git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'base')
    return root, subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True,
                                text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = Repository(tmp_path / 'db', tmp_path / 'artifacts')
    r.put_record('deployment', 'base', {'id': 'base', 'engine': 'inferstack_native', 'model': 'm',
                                        'base_url': 'http://b'})
    r.put_record('technique', 'fuse', {'name': 'fuse', 'summary': 's', 'engines': [{'engine': 'inferstack_native'}]})
    return r


@pytest.fixture
def desk(repo, project, tmp_path):
    root, _ = project
    return ImplementationDesk(repo, source_root=root, sandbox=Sandbox(), work_root=tmp_path / 'work', test_timeout=120,
                              default_tests={'src/inferstack/kernels/': TESTS})


def normalized(repo):
    _, plan = validate_plan(proposal_plan(), repository=repo, baseline_id='base', candidate_ids=[],
                            allowed_evidence=[], history_signatures=set())
    return plan


def open_handoff(desk, repo, project, **kw):
    plan = normalized(repo)
    assert plan['executable'] is False
    return desk.request(plan['needs_implementation'][0], plan, baseline_id='base', source_revision=project[1],
                        job_id='job-1', independent_tests=TESTS, **kw)['id']


# --- handoff contract ------------------------------------------------------------------

def test_handoff_is_controller_owned_and_idempotent(desk, repo, project):
    impl = open_handoff(desk, repo, project)
    body = desk.read(impl)['body']
    assert body['state'] == 'requested' and body['candidate'] is None
    assert body['handoff']['allowed_files'] == [FILE] and body['handoff']['acceptance']['independent_tests'] == TESTS
    assert body['handoff']['source_revision'] == project[1] and body['baseline_ref'] == 'base@1'
    assert open_handoff(desk, repo, project) == impl  # same proposal, same job: same handoff
    with pytest.raises(ValueError, match='outside the patchable roots'):
        desk.request({'action_id': 'x', 'files': ['tests/test_fused.py']}, normalized(repo), baseline_id='base',
                     source_revision=project[1], independent_tests=TESTS)
    with pytest.raises(ValueError, match='full commit hash'):
        desk.request({'action_id': 'x', 'files': [FILE]}, normalized(repo), baseline_id='base',
                     source_revision='main', independent_tests=TESTS)


@needs_sandbox
def test_human_diff_becomes_an_immutable_content_addressed_candidate(desk, repo, project):
    impl = open_handoff(desk, repo, project)
    body = desk.submit(impl, GOOD)['body']
    candidate = body['candidate']
    assert body['state'] == 'validated' and body['attempts'][0]['outcome'] == 'validated'
    assert candidate['candidate_id'] == candidate_identity(candidate)
    assert repo.read_artifact(candidate['patch_sha256']).decode() == GOOD
    assert repo.read_artifact(candidate['manifest_sha256'])  # manifest stored by hash
    assert desk.submit(impl, GOOD)['revision'] == desk.read(impl)['revision']  # idempotent resume
    with pytest.raises(ArtifactLocked, match='new candidate'):
        desk.submit(impl, OTHER)
    child = desk.revise(impl)['id']
    assert child != impl and desk.read(child)['body']['parent_ref'] == impl
    other = desk.submit(child, OTHER)['body']['candidate']
    assert other['candidate_id'] != candidate['candidate_id']  # a changed diff is a new identity
    assert desk.read(impl)['body']['candidate'] == candidate  # the parent is unchanged


@needs_sandbox
@pytest.mark.parametrize('patch,reason', [
    (WRONG, 'independent acceptance tests failed'), (BROKEN, 'does not compile'), (CHEAT, 'outside'),
    ('', 'empty diff')])
def test_invalid_submissions_are_rejected_and_journaled(desk, repo, project, patch, reason):
    impl = open_handoff(desk, repo, project)
    body = desk.submit(impl, patch)['body']
    assert body['state'] == 'requested' and body['candidate'] is None
    assert body['attempts'][0]['outcome'] == 'rejected' and reason in body['attempts'][0]['reason']


@needs_sandbox
def test_candidate_tests_supplement_but_never_replace_independent_tests(desk, repo, project):
    impl = open_handoff(desk, repo, project)
    passing = {'test_mine.py': 'def test_ok():\n    assert True\n'}
    body = desk.submit(impl, WRONG, candidate_tests=passing)['body']
    assert body['attempts'][0]['outcome'] == 'rejected' and 'independent' in body['attempts'][0]['reason']
    failing = {'test_mine.py': 'from inferstack.kernels.fused import add\n\ndef test_x():\n    assert add(1, 1) == 3\n'}
    body = desk.submit(impl, GOOD, candidate_tests=failing)['body']
    assert body['attempts'][1]['outcome'] == 'rejected' and 'candidate tests failed' in body['attempts'][1]['reason']
    body = desk.submit(impl, GOOD, candidate_tests=passing)['body']
    assert body['state'] == 'validated' and set(body['candidate']['candidate_tests']) == {'test_mine.py'}


@needs_sandbox
def test_attempts_are_bounded(desk, repo, project):
    impl = open_handoff(desk, repo, project, limits={'max_attempts': 2})
    desk.submit(impl, WRONG)
    assert desk.submit(impl, BROKEN)['body']['state'] == 'exhausted'
    with pytest.raises(AttemptLimit):
        desk.submit(impl, GOOD)


def test_crash_mid_attempt_is_abandoned_not_silently_resumed(desk, repo, project):
    impl = open_handoff(desk, repo, project)
    first = desk.begin_attempt(impl, {'kind': 'agent', 'name': 'a'})
    second = desk.begin_attempt(impl, {'kind': 'agent', 'name': 'a'})  # after a restart
    attempts = desk.read(impl)['body']['attempts']
    assert attempts[0]['outcome'] == 'abandoned' and attempts[1]['outcome'] == 'in_progress' and second == 2
    with pytest.raises(ValueError, match='not in progress'):
        desk.finish_attempt(impl, first, outcome='submitted', diff=GOOD)  # late output of the crashed run
    cancelled = desk.cancel(impl)['body']
    assert cancelled['state'] == 'cancelled' and cancelled['attempts'][1]['outcome'] == 'cancelled'
    with pytest.raises(AttemptLimit):
        desk.begin_attempt(impl, {'kind': 'human'})


def test_validation_fails_closed_without_isolation(repo, project, tmp_path):
    class NoBox:
        filesystem_isolated = False
    desk = ImplementationDesk(repo, source_root=project[0], sandbox=NoBox(), work_root=tmp_path / 'w')
    impl = open_handoff(desk, repo, project)
    body = desk.submit(impl, GOOD)['body']
    assert body['state'] == 'requested' and 'isolation' in body['attempts'][0]['reason']


# --- execution binding -------------------------------------------------------------------

@needs_sandbox
def test_only_the_validated_artifact_is_executable(desk, repo, project):
    impl = open_handoff(desk, repo, project)
    with pytest.raises(ValidationFailed, match='without a validated candidate'):
        desk.executable_plan([impl])
    desk.submit(impl, GOOD)
    raw = desk.executable_plan([impl])
    _, plan = validate_plan(raw, repository=repo, baseline_id='base', candidate_ids=[], allowed_evidence=[],
                            history_signatures=set())
    params = plan['actions'][0]['params']
    assert plan['executable'] is True and params['implementation_id'] == impl and params['tests'] == TESTS
    tampered = desk.executable_plan([impl])
    tampered['actions'][0]['params']['patch_sha256'] = repo.artifact(OTHER.encode())['sha256']
    with pytest.raises(PlanRejected, match='does not match validated implementation'):
        validate_plan(tampered, repository=repo, baseline_id='base', candidate_ids=[], allowed_evidence=[],
                      history_signatures=set())


class ProposalPlanner:
    candidate_ids, production_ids = (), ()

    def __init__(self, repo):
        self.repo = repo

    async def plan(self, request, history):
        return normalized(self.repo)


class RecordingExecutor:
    def __init__(self):
        self.plans = []

    async def execute(self, plan, namespace, context=None):
        self.plans.append(plan)
        raise RuntimeError('stop after capture')

    async def cleanup(self, candidate):
        pass


@needs_sandbox
@pytest.mark.asyncio
async def test_worker_opens_handoff_then_resumes_with_the_exact_artifact(desk, repo, project):
    executor = RecordingExecutor()
    worker = OptimizationWorker(repo, ProposalPlanner(repo), executor, None, implementations=desk,
                                source_revision=project[1])
    first = repo.submit('proposal', {'deployment_id': 'base', 'goal': 'faster'})
    result = (await worker.once())['result']
    assert result['decision'] == 'needs_implementation' and executor.plans == []
    impl = result['implementation_ids'][0]
    handoff = desk.read(impl)['body']
    assert handoff['job_id'] == first and handoff['handoff']['acceptance']['independent_tests'] == TESTS
    desk.submit(impl, GOOD)
    repo.submit('resume', {'deployment_id': 'base', 'implementation_ids': [impl]})
    resumed = await worker.once()
    assert executor.plans, resumed.get('result')
    plan = executor.plans[0]
    params = next(a for a in plan['actions'] if a['type'] == 'source_patch')['params']
    assert params['patch_sha256'] == desk.read(impl)['body']['candidate']['patch_sha256']
    assert params['implementation_id'] == impl and plan['executable'] is True


@pytest.mark.asyncio
async def test_worker_without_desk_still_never_executes_a_proposal(repo):
    executor = RecordingExecutor()
    repo.submit('proposal', {'deployment_id': 'base'})
    result = (await OptimizationWorker(repo, ProposalPlanner(repo), executor, None).once())['result']
    assert result['decision'] == 'needs_implementation' and executor.plans == []
    assert result['implementation_ids'] == [] and 'no implementation desk' in result['handoff_error']


# --- coding-agent path ------------------------------------------------------------------

AGENT = '''import json, pathlib, sys
handoff = json.loads(pathlib.Path(sys.argv[-1]).read_text())
mode = sys.argv[1]
target = pathlib.Path(handoff['allowed_files'][0])
if mode == 'good':
    target.write_text('def add(a, b):\\n    return b + a\\n')
    tests = pathlib.Path(sys.argv[-1]).parent / 'candidate_tests'
    tests.mkdir()
    (tests / 'test_agent.py').write_text('def test_agent():\\n    assert True\\n')
    (pathlib.Path(sys.argv[-1]).parent / 'COST.json').write_text(json.dumps({'provider_tokens': 1234}))
elif mode == 'cheat':
    pathlib.Path('tests/test_fused.py').write_text('def test_add():\\n    assert True\\n')
    target.write_text('def add(a, b):\\n    return a - b\\n')
elif mode == 'slow':
    import time
    time.sleep(30)
elif mode == 'escape':
    pathlib.Path(sys.argv[2]).write_text('owned')
'''


@pytest.fixture
def agent(tmp_path):
    path = tmp_path / 'tools' / 'agent.py'
    path.parent.mkdir()
    path.write_text(AGENT)
    return path


def implementer(project, agent, tmp_path, *args):
    return CommandImplementer([sys.executable, str(agent), *args], sandbox=Sandbox(), source_root=project[0],
                              work_root=tmp_path / 'agents', name='fixture-agent')


@needs_sandbox
@pytest.mark.asyncio
async def test_coding_agent_produces_a_validated_candidate(desk, repo, project, agent, tmp_path):
    impl = open_handoff(desk, repo, project)
    body = (await implementer(project, agent, tmp_path, 'good').run(desk, impl))['body']
    attempt = body['attempts'][0]
    assert body['state'] == 'validated' and attempt['implementer'] == {'kind': 'agent', 'name': 'fixture-agent'}
    assert attempt['cost']['provider_tokens'] == 1234 and attempt['cost']['wall_seconds'] >= 0
    assert set(body['candidate']['candidate_tests']) == {'test_agent.py'}
    assert '+    return b + a' in repo.read_artifact(body['candidate']['patch_sha256']).decode()


@needs_sandbox
@pytest.mark.asyncio
async def test_coding_agent_cannot_rewrite_the_tests_that_judge_it(desk, repo, project, agent, tmp_path):
    impl = open_handoff(desk, repo, project)
    body = (await implementer(project, agent, tmp_path, 'cheat').run(desk, impl))['body']
    assert body['state'] == 'requested' and body['attempts'][0]['outcome'] == 'rejected'
    assert 'tests/test_fused.py' in body['attempts'][0]['reason']


@needs_sandbox
@pytest.mark.asyncio
async def test_coding_agent_timeout_and_escape_are_recorded_failures(desk, repo, project, agent, tmp_path):
    impl = open_handoff(desk, repo, project, limits={'timeout_s': 2})
    body = (await implementer(project, agent, tmp_path, 'slow').run(desk, impl))['body']
    assert body['attempts'][0]['outcome'] == 'timeout' and body['state'] == 'requested'
    outside = tmp_path / 'outside.txt'
    body = (await implementer(project, agent, tmp_path, 'escape', str(outside)).run(desk, impl))['body']
    assert body['attempts'][1]['outcome'] == 'failed' and not outside.exists()


@needs_sandbox
@pytest.mark.asyncio
async def test_cancelling_an_agent_journals_the_attempt(desk, repo, project, agent, tmp_path):
    impl = open_handoff(desk, repo, project)
    task = asyncio.create_task(implementer(project, agent, tmp_path, 'slow').run(desk, impl))
    for _ in range(100):
        await asyncio.sleep(0.05)
        if desk.read(impl)['body']['attempts']:
            break
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert desk.read(impl)['body']['attempts'][0]['outcome'] == 'cancelled'


# --- control API ------------------------------------------------------------------------

@needs_sandbox
def test_api_patch_resume_and_conflicts(desk, repo, project):
    from fastapi.testclient import TestClient
    from inferstack.control.api import create_control_app
    headers = {'Authorization': 'Bearer tok'}
    impl = open_handoff(desk, repo, project)
    client = TestClient(create_control_app(repo, 'tok', implementations=desk))
    assert client.post(f'/control/implementations/{impl}/resume', headers=headers).status_code == 409  # not validated
    rejected = client.post(f'/control/implementations/{impl}/patch', headers=headers, json={'diff': WRONG}).json()
    assert rejected['body']['attempts'][0]['outcome'] == 'rejected'
    ok = client.post(f'/control/implementations/{impl}/patch', headers=headers, json={'diff': GOOD}).json()
    assert ok['body']['state'] == 'validated'
    assert client.post(f'/control/implementations/{impl}/patch', headers=headers,
                       json={'diff': OTHER}).status_code == 409  # immutable candidate
    job = client.post(f'/control/implementations/{impl}/resume', headers=headers).json()
    assert repo.get(job['job_id'])['payload']['implementation_ids'] == [impl]
    again = client.post(f'/control/implementations/{impl}/resume', headers=headers).json()
    assert again['job_id'] == job['job_id']  # idempotent per candidate identity
    assert client.get('/control/implementations/nope', headers=headers).status_code == 404
    assert client.post(f'/control/implementations/{impl}/patch', json={'diff': GOOD}).status_code == 401
    readonly = TestClient(create_control_app(repo, 'tok'))
    child = readonly.post(f'/control/implementations/{impl}/revise', headers=headers).json()['id']
    assert readonly.post(f'/control/implementations/{child}/patch', headers=headers,
                         json={'diff': GOOD}).status_code == 503  # no validator configured
