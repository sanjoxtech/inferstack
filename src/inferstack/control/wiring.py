"""Worker composition from configuration: the supported path for all three entry points.

`build_worker_parts(repo, config, ...)` selects the executor, retrieval and the
planner's live telemetry from one explicit configuration so that optimize,
technique-investigation and feedback-diagnosis jobs run through the same worker:

- executor: `{"executor": {"kind": "isolated", "source_root": ".", "work_root": ...}}`
  selects IsolatedExecutor (config/plugin/source actions). The source revision is
  resolved once to a full commit hash; a dirty tree is refused because it is not a
  reproducible identity. Default `endpoint` runs registered candidates only.
- retrieval: SafeFetcher (SSRF-guarded) unless `{"research": {"retrieval": false}}`.
- telemetry: adapter-backed snapshots of registered deployments for the planner's
  `query_telemetry` tool, unless `{"planner_telemetry": false}`.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def pinned_source_revision(source_root):
    root = Path(source_root)
    head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    if head.returncode:
        raise ValueError(f'{root} is not a git checkout; isolated execution needs a pinned source revision')
    dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ValueError('source tree has uncommitted changes; commit them so the candidate source is pinned')
    return head.stdout.strip()


def telemetry_source(repository, transport=None):
    """async (deployment_id) -> normalized telemetry snapshot from the deployment's engine adapter."""
    import httpx

    from inferstack.engines.adapters import adapter_for

    async def snapshot(deployment_id):
        body = repository.read('deployment', deployment_id)['body']
        async with httpx.AsyncClient(base_url=body['base_url'].rstrip('/') + '/', timeout=10,
                                     transport=transport) as client:
            result = await adapter_for(body['engine'], client).telemetry(engine_version=body.get('engine_version'))
        result.pop('raw_text', None)
        return result
    return snapshot


def host_gpu_devices(config, provider_name):
    """Identified GPU devices for exclusive placement: explicit config, else the local inventory (UUIDs)."""
    infra = config.get('infrastructure') or {}
    if infra.get('gpu_devices') is not None:
        return list(infra['gpu_devices'])
    if provider_name != 'local':
        return []
    from inferstack.infrastructure.inventory import collect
    try:
        hardware, _ = collect(provider='attached')
    except Exception:  # noqa: BLE001 - no inventory means no identified devices (GPU launches refused)
        return []
    return [g['uuid'] for g in hardware.get('gpus') or [] if g.get('uuid')]


def build_executor(repository, config, *, state, allocations, provider, pricing=None, gpu_name=None,
                   gpu_devices=None):
    from inferstack.optimization.experiment import EndpointExecutor
    spec = dict(config.get('executor') or {'kind': 'endpoint'})
    kind = spec.pop('kind', 'endpoint')
    common = dict(allocations=allocations, provider=provider, pricing=pricing, gpu_name=gpu_name,
                  gpu_devices=gpu_devices)
    if kind == 'endpoint':
        return EndpointExecutor(repository, Path(state) / 'logs', **common)
    if kind != 'isolated':
        raise ValueError(f'unknown executor kind {kind!r}')
    from inferstack.execution.executor import IsolatedExecutor
    from inferstack.execution.sandbox import Sandbox
    source_root = Path(spec.pop('source_root', '.')).resolve()
    revision = spec.pop('source_revision', None) or pinned_source_revision(source_root)
    sandbox = Sandbox(require_filesystem_isolation=spec.pop('require_filesystem_isolation', True),
                      require_network_isolation=spec.pop('require_network_isolation', False))
    work_root = Path(spec.pop('work_root', Path(state) / 'work'))
    builder = spec.pop('image_builder', None)
    if builder is not None:
        from inferstack.infrastructure.images import ImageBuilder
        spec['image_builder'] = ImageBuilder(**builder)
    return IsolatedExecutor(repository, Path(state) / 'logs', work_root=work_root, source_root=source_root,
                            source_revision=revision, sandbox=sandbox, **spec, **common)


def build_fetcher(config):
    if not (config.get('research') or {}).get('retrieval', True):
        return None
    from inferstack.research.fetch import SafeFetcher
    return SafeFetcher(**{k: v for k, v in (config.get('research') or {}).items() if k != 'retrieval'})


def implementation_desk(repository, executor=None, *, source_root=None, state='.inferstack'):
    """H3 implementer handoff wired to the isolated executor's pinned source and sandbox.

    Returns None when there is no pinned source tree (endpoint-only executor): proposals
    are then recorded as `needs_implementation` without a handoff."""
    from inferstack.implementation import ImplementationDesk
    if executor is not None and getattr(executor, 'source_revision', None):
        return ImplementationDesk(repository, source_root=executor.source_root, sandbox=executor.sandbox,
                                  work_root=Path(executor.work_root) / 'implementations',
                                  test_timeout=executor.test_timeout)
    if source_root:
        from inferstack.execution.sandbox import Sandbox
        return ImplementationDesk(repository, source_root=Path(source_root).resolve(),
                                  sandbox=Sandbox(require_filesystem_isolation=True),
                                  work_root=Path(state) / 'implementations')
    return None
