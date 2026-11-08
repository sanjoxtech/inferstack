"""Package F: patch confinement, sandbox controls, offline plugin installs and
isolated candidate launch with real processes."""
import base64
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import pytest

from inferstack.execution import BuildFailed, IsolatedExecutor, Limits, PatchRejected, Sandbox, check_patch
from inferstack.execution.executor import with_flags
from inferstack.infrastructure import AllocationManager, LocalProcessProvider
from inferstack.optimization.dag import PlanRejected, validate_plan
from inferstack.optimization.experiment import EndpointExecutor
from inferstack.storage.repository import Repository

FIXTURES = Path(__file__).parent / 'fixtures'
KERNEL = 'def add(a, b):\n    return a + b\n'


def diff(path, old, new):
    old_lines, new_lines = old.splitlines(), new.splitlines()
    body = ''.join(f'-{l}\n' for l in old_lines) + ''.join(f'+{l}\n' for l in new_lines)
    return (f'--- a/{path}\n+++ b/{path}\n@@ -1,{len(old_lines)} +1,{len(new_lines)} @@\n' + body)


GOOD = diff('src/inferstack/kernels/fused.py', KERNEL, 'def add(a, b):\n    # fused fast path\n    return b + a\n')
WRONG = diff('src/inferstack/kernels/fused.py', KERNEL, 'def add(a, b):\n    return a - b\n')
BROKEN = diff('src/inferstack/kernels/fused.py', KERNEL, 'def add(a, b)\n    return a + b\n')


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'project'
    for path, text in {'src/inferstack/kernels/fused.py': KERNEL, 'src/inferstack/__init__.py': '',
                       'src/inferstack/kernels/__init__.py': '',
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


# --- patch confinement -------------------------------------------------------------------

@pytest.mark.parametrize('patch,files,message', [
    (diff('tests/test_fused.py', 'x', 'y'), ['tests/test_fused.py'], 'outside'),
    (diff('src/inferstack/evaluation/judge.py', 'THRESHOLD = 0.05', 'THRESHOLD = 9'),
     ['src/inferstack/evaluation/judge.py'], 'outside'),
    (diff('src/inferstack/kernels/../../tests/x.py', 'a', 'b'), ['src/inferstack/kernels/../../tests/x.py'], 'unsafe'),
    ('--- /etc/passwd\n+++ /etc/passwd\n@@ -1 +1 @@\n-a\n+b\n', ['/etc/passwd'], 'unsafe'),
    (diff('src/inferstack/kernels/other.py', 'a', 'b'), ['src/inferstack/kernels/fused.py'], 'did not declare'),
    (diff('src/inferstack/kernels/data.bin', 'a', 'b'), ['src/inferstack/kernels/data.bin'], 'non-source'),
    ('diff --git a/src/inferstack/kernels/x.py b/src/inferstack/kernels/x.py\nnew file mode 120000\n'
     '--- /dev/null\n+++ b/src/inferstack/kernels/x.py\n@@ -0,0 +1 @@\n+/etc/passwd\n',
     ['src/inferstack/kernels/x.py'], 'symlink'),
    ('diff --git a/src/inferstack/kernels/x.py b/src/inferstack/kernels/x.py\nGIT binary patch\nliteral 1\n',
     ['src/inferstack/kernels/x.py'], 'binary'),
    ('diff --git a/src/inferstack/kernels/fused.py b/tests/test_fused.py\nrename from src/inferstack/kernels/fused.py\n'
     'rename to tests/test_fused.py\n', ['src/inferstack/kernels/fused.py'], 'outside'),
])
def test_patch_confinement(patch, files, message):
    with pytest.raises(PatchRejected, match=message):
        check_patch(patch, files)


def test_good_patch_passes_inspection():
    assert check_patch(GOOD, ['src/inferstack/kernels/fused.py']) == {'src/inferstack/kernels/fused.py'}


# --- sandbox --------------------------------------------------------------------------------

def test_sandbox_scrubs_credentials_and_isolates_network(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'sk-should-not-leak-000000')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'aws-should-not-leak')
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen()
    port = listener.getsockname()[1]
    threading.Thread(target=lambda: listener.accept(), daemon=True).start()
    sandbox = Sandbox()
    probe = (f'import os,socket,json;s=socket.socket();s.settimeout(2);'
             f'r=s.connect_ex(("127.0.0.1",{port}));print(json.dumps({{"connect":r,"env":sorted(os.environ)}}))')
    code, out = sandbox.run([sys.executable, '-c', probe], cwd=tmp_path, home=tmp_path)
    result = json.loads(out.strip().splitlines()[-1])
    assert code == 0 and not any('KEY' in k or 'SECRET' in k for k in result['env'])
    if sandbox.network_isolated:
        assert result['connect'] != 0  # even loopback is unreachable inside the build netns
    online_code, online = sandbox.run([sys.executable, '-c', probe], cwd=tmp_path, home=tmp_path, offline=False)
    assert json.loads(online.strip().splitlines()[-1])['connect'] == 0
    with pytest.raises(ValueError, match='credential-like'):
        sandbox.run(['true'], cwd=tmp_path, home=tmp_path, env={'GITHUB_TOKEN': 'x'})
    listener.close()


def test_sandbox_limits(tmp_path):
    sandbox = Sandbox(limits=Limits(cpu_seconds=1, wall_seconds=20))
    code, _ = sandbox.run([sys.executable, '-c', 'while True: pass'], cwd=tmp_path, home=tmp_path)
    assert code != 0  # SIGXCPU/SIGKILL from RLIMIT_CPU
    code, out = Sandbox(limits=Limits(wall_seconds=1)).run([sys.executable, '-c', 'import time; time.sleep(30)'],
                                                           cwd=tmp_path, home=tmp_path)
    assert code == 124 and 'killed after 1' in out


def test_required_network_isolation_is_enforced(monkeypatch):
    import inferstack.execution.sandbox as sb
    monkeypatch.setattr(sb, 'network_isolation_available', lambda: False)
    with pytest.raises(RuntimeError, match='required but unavailable'):
        sb.Sandbox(require_network_isolation=True)
    assert sb.Sandbox().describe()['network_isolation'] == 'unavailable'


# --- flags ------------------------------------------------------------------------------------

def test_profile_flags_replace_existing_values():
    argv = ['vllm', 'serve', 'm', '--max-num-seqs', '64', '--enable-prefix-caching']
    out = with_flags(argv, 'vllm', {'max_num_seqs': 256, 'enable_prefix_caching': False})
    assert out == ['vllm', 'serve', 'm', '--max-num-seqs', '256', '--no-enable-prefix-caching']
    with pytest.raises(BuildFailed, match='not supported'):
        with_flags([], 'inferstack_native', {'max_running_requests': 4})


# --- end to end -----------------------------------------------------------------------------

def make_wheel(directory, name, version):
    dist = f'{name}-{version}.dist-info'
    files = {f'{name}/__init__.py': f'VERSION = "{version}"\n',
             f'{dist}/METADATA': f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n',
             f'{dist}/WHEEL': 'Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n'}
    record = ''.join(f'{p},sha256={base64.urlsafe_b64encode(hashlib.sha256(c.encode()).digest()).rstrip(b"=").decode()},'
                     f'{len(c)}\n' for p, c in files.items()) + f'{dist}/RECORD,,\n'
    path = directory / f'{name}-{version}-py3-none-any.whl'
    with zipfile.ZipFile(path, 'w') as z:
        for p, c in files.items():
            z.writestr(p, c)
        z.writestr(f'{dist}/RECORD', record)
    return path


@pytest.fixture
def setup(tmp_path, project, monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'sk-should-not-leak-000000')
    root, revision = project
    repo = Repository(tmp_path / 'db', tmp_path / 'artifacts')
    port = 18000 + os.getpid() % 1000
    repo.put_record('deployment', 'base', {
        'id': 'base', 'engine': 'vllm', 'model': 'm', 'base_url': f'http://127.0.0.1:{port}',
        'command': ['python3', str(FIXTURES / 'fake_openai_server.py'), '--port', str(port), '--max-num-seqs', '64',
                    '--dump-env', str(tmp_path / 'env.json')], 'engine_version': '0.11.0',
        'profile': {'dtype': 'bfloat16'}})
    repo.put_record('technique', 'fused', {'name': 'fused add', 'summary': 's', 'dtypes': ['bfloat16'],
                                           'engines': [{'engine': 'vllm', 'versions': '>=0.10'}],
                                           'dependencies': [{'name': 'tinyplug', 'version': '==1.0.0',
                                                             'license': 'MIT'}]})
    wheels = tmp_path / 'index'
    wheels.mkdir()
    make_wheel(wheels, 'tinyplug', '1.0.0')
    job = repo.submit('j', {})
    claim = repo.claim(60, worker_id='w')
    allocations = AllocationManager(repo, 'w')
    executor = IsolatedExecutor(repo, tmp_path / 'logs', work_root=tmp_path / 'work', source_root=root,
                                source_revision=revision, allocations=allocations, provider=LocalProcessProvider(),
                                pip_index_args=('--no-index', '--find-links', str(wheels)))
    context = {'job': {**claim, 'id': job, 'step_attempt': 1}, 'step_key': 'iter0.experiment'}
    return repo, executor, context, tmp_path


def plan_for(repo, patch, tests=('tests/test_fused.py',)):
    raw = {'hypothesis': 'Fused add plus a larger batch improves throughput.', 'baseline_ref': 'base@1',
           'evidence_ids': [], 'actions': [
               {'id': 'cfg', 'type': 'config_change', 'depends_on': [], 'params': {'profile_patch': {'max_num_seqs': 256}}},
               {'id': 'plug', 'type': 'plugin_install', 'depends_on': [],
                'params': {'technique_id': 'fused', 'dependency': 'tinyplug==1.0.0'}},
               {'id': 'patch', 'type': 'source_patch', 'depends_on': ['plug'],
                'params': {'technique_id': 'fused', 'files': ['src/inferstack/kernels/fused.py'], 'description': 'x',
                           'patch_sha256': repo.artifact(patch.encode())['sha256'], 'tests': list(tests)}},
               {'id': 'measure', 'type': 'measure', 'depends_on': ['cfg', 'patch'], 'params': {}}],
           'expected_measurements': ['e2e_ms'], 'gates': ['correctness', 'quality', 'health', 'comparable', 'complete'],
           'estimate': {'gpu_seconds': 0, 'wall_seconds': 60}, 'stop_rule': {'max_iterations': 1, 'stop_on': ['keep']}}
    return validate_plan(raw, repository=repo, baseline_id='base', candidate_ids=[], allowed_evidence=[],
                         history_signatures=set())[1]


@pytest.mark.asyncio
async def test_isolated_candidate_end_to_end(setup):
    repo, executor, context, tmp = setup
    plan = plan_for(repo, GOOD)
    candidate = await executor.execute(plan, 'exp1', context)
    try:
        record = repo.read('deployment', candidate['deployment_id'])['body']
        assert candidate['derived'] and record['id'] == 'exp1-candidate' and record['plugin_refs'] == ['fused']
        assert record['command'][record['command'].index('--max-num-seqs') + 1] == '256'
        assert record['base_url'] != repo.read('deployment', 'base')['body']['base_url']
        dumped = json.loads((tmp / 'env.json').read_text())
        assert '/venv/bin/python' in dumped['executable'] and '--max-num-seqs' in dumped['unknown']
        assert not any('KEY' in k or 'SECRET' in k for k in dumped['env'])
        assert dumped['env']['PYTHONPATH'].endswith('/src/src') and 'INFERSTACK_ALLOCATION_ID' in dumped['env']
        ws_src = Path(dumped['env']['PYTHONPATH']).parent
        assert 'b + a' in (ws_src / 'src/inferstack/kernels/fused.py').read_text()
        assert (ws_src / 'src/inferstack/evaluation/judge.py').read_text() == 'THRESHOLD = 0.05\n'
        names = set(candidate['artifacts'])
        assert {'patch.diff', 'patch.tests.log', 'plug.wheels.json', 'plug.freeze.txt', 'build.log.json'} <= names
        wheels = json.loads(repo.read_artifact(candidate['artifacts']['plug.wheels.json']['sha256']))
        assert list(wheels) == ['tinyplug-1.0.0-py3-none-any.whl']
        assert 'tinyplug==1.0.0' in repo.read_artifact(candidate['artifacts']['plug.freeze.txt']['sha256']).decode()
        assert candidate['isolation']['env'] == 'allowlisted'
    finally:
        await executor.cleanup(candidate)
    assert not Path(dumped['env']['HOME']).exists()  # workspace removed
    allocs = AllocationManager(repo, 't').list(job_id=context['job']['id'])
    assert allocs and all(a['state'] == 'released' for a in allocs)


@pytest.mark.asyncio
@pytest.mark.parametrize('patch,message', [(WRONG, 'declared tests failed'), (BROKEN, 'does not compile')])
async def test_failed_builds_launch_nothing_and_clean_up(setup, patch, message):
    repo, executor, context, tmp = setup
    plan = plan_for(repo, patch)
    with pytest.raises(BuildFailed, match=message):
        await executor.execute(plan, 'exp2', context)
    assert list((tmp / 'work').iterdir()) == [] and not (tmp / 'env.json').exists()
    assert AllocationManager(repo, 't').list(job_id=context['job']['id']) == []
    with pytest.raises(KeyError):
        repo.read('deployment', 'exp2-candidate')


def test_unsafe_stored_patch_is_rejected_at_planning(setup):
    repo = setup[0]
    with pytest.raises(PlanRejected, match='stored patch rejected'):
        plan_for(repo, diff('src/inferstack/evaluation/judge.py', 'THRESHOLD = 0.05', 'THRESHOLD = 9'))


@pytest.mark.asyncio
async def test_registered_candidate_executor_refuses_dag_actions(setup):
    repo, _, context, tmp = setup
    with pytest.raises(ValueError, match='use IsolatedExecutor'):
        await EndpointExecutor(repo, tmp / 'logs').execute(plan_for(repo, GOOD), 'exp3')
