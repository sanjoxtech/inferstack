"""V16: the supported worker path selects the isolated executor, retrieval and live telemetry."""
import subprocess

import httpx
import pytest

from inferstack.control.wiring import build_executor, build_fetcher, pinned_source_revision, telemetry_source
from inferstack.execution.executor import BuildFailed, IsolatedExecutor
from inferstack.infrastructure import AllocationManager, LocalProcessProvider
from inferstack.optimization.experiment import EndpointExecutor
from inferstack.research.fetch import SafeFetcher
from inferstack.storage.repository import Repository


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / 'src'
    root.mkdir()
    (root / 'a.py').write_text('x = 1\n')
    git = lambda *a: subprocess.run(['git', '-C', str(root), *a], check=True, capture_output=True)
    git('init', '-q')
    git('add', '-A')
    git('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'base')
    return root


def parts(tmp_path):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    return repo, AllocationManager(repo, 'w'), LocalProcessProvider()


def test_executor_selection(tmp_path, checkout):
    repo, allocations, provider = parts(tmp_path)
    assert type(build_executor(repo, {}, state=tmp_path, allocations=allocations, provider=provider)) is EndpointExecutor
    config = {'executor': {'kind': 'isolated', 'source_root': str(checkout), 'require_filesystem_isolation': False}}
    executor = build_executor(repo, config, state=tmp_path, allocations=allocations, provider=provider)
    assert isinstance(executor, IsolatedExecutor) and executor.source_revision == pinned_source_revision(checkout)
    with pytest.raises(ValueError, match='unknown executor'):
        build_executor(repo, {'executor': {'kind': 'magic'}}, state=tmp_path, allocations=allocations,
                       provider=provider)


def test_dirty_source_is_not_a_pinned_identity(checkout):
    (checkout / 'a.py').write_text('x = 2\n')
    with pytest.raises(ValueError, match='uncommitted'):
        pinned_source_revision(checkout)


@pytest.mark.asyncio
async def test_remote_provider_refuses_locally_built_candidates(tmp_path, checkout):
    repo, allocations, _ = parts(tmp_path)

    class Remote(LocalProcessProvider):
        name, launches_remotely = 'kubernetes', True
    config = {'executor': {'kind': 'isolated', 'source_root': str(checkout), 'require_filesystem_isolation': False}}
    executor = build_executor(repo, config, state=tmp_path, allocations=allocations, provider=Remote())
    repo.put_record('deployment', 'base', {'id': 'base', 'engine': 'vllm', 'model': 'm', 'base_url': 'http://b'})
    plan = {'baseline_id': 'base', 'baseline_revision': 1, 'action_order': ['p'],
            'actions': [{'id': 'p', 'type': 'plugin_install', 'params': {'dependency': 'x==1'}}]}
    with pytest.raises(BuildFailed, match='image build'):
        await executor.execute(plan, 'n', context={'job': {}, 'step_key': 's'})


def test_retrieval_is_configured_by_default():
    assert isinstance(build_fetcher({}), SafeFetcher)
    assert build_fetcher({'research': {'retrieval': False}}) is None


@pytest.mark.asyncio
async def test_planner_telemetry_reads_the_engine_adapter(tmp_path):
    repo = Repository(tmp_path / 'db', tmp_path / 'a')
    repo.put_record('deployment', 'd', {'id': 'd', 'engine': 'vllm', 'model': 'm', 'base_url': 'http://engine'})
    text = '# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running 3\n'
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text=text) if r.url.path == '/metrics'
                                    else httpx.Response(404))
    snapshot = await telemetry_source(repo, transport=transport)('d')
    assert snapshot['engine'] == 'vllm' and 'raw_text' not in snapshot
