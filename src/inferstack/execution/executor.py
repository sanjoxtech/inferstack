"""Isolated execution of experiment DAG actions.

For one experiment:
1. `config_change` selects a registered candidate or derives one from the
   baseline with an allowlisted profile patch mapped to engine flags.
2. `plugin_install` downloads the pinned dependency into a wheelhouse (network),
   records each wheel's SHA-256, then installs offline into a per-experiment venv.
3. `source_patch` applies a stored, inspected diff to a pristine copy of the
   pinned source revision, restores tests/evaluator from that revision, compiles,
   and runs the declared tests offline in the sandbox.
4. The candidate is recorded as a new deployment snapshot and launched with a
   scrubbed environment, new port, allocation identity and resource limits.
Artifacts (diff, wheel hashes, freeze, build/test logs, isolation report) are
stored by hash and returned with the candidate. Workspaces are removed on
cleanup and on every failure path.

Untrusted code (plugins, patched source, their tests and the derived candidate
server) only runs inside the sandbox's filesystem boundary; without one, those
experiments are refused. Builds run off the worker's event loop so lease
heartbeats, cancellation and deadlines keep running; cancellation kills the
build's process group and waits for it. Build steps carry a build allocation
(intent before the first process, tag on every process) so a crash mid-build is
reconciled like any other launch.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from inferstack.execution.patches import check_patch
from inferstack.execution.sandbox import Sandbox
from inferstack.optimization.experiment import EndpointExecutor

ENGINE_FLAGS = {
    'vllm': {'max_num_seqs': '--max-num-seqs', 'max_num_batched_tokens': '--max-num-batched-tokens',
             'gpu_memory_utilization': '--gpu-memory-utilization', 'block_size': '--block-size',
             'kv_cache_dtype': '--kv-cache-dtype',
             'enable_prefix_caching': ('--enable-prefix-caching', '--no-enable-prefix-caching'),
             'enable_chunked_prefill': ('--enable-chunked-prefill', '--no-enable-chunked-prefill'),
             # Versioned loader option; runai_streamer needs the optional dependency in the image
             # and Safetensors weights, and must be qualified per vLLM/image combination.
             'load_format': '--load-format'},
    'sglang': {'max_running_requests': '--max-running-requests', 'chunked_prefill_size': '--chunked-prefill-size',
               'mem_fraction_static': '--mem-fraction-static', 'schedule_policy': '--schedule-policy',
               'kv_cache_dtype': '--kv-cache-dtype', 'disable_radix_cache': ('--disable-radix-cache', None)},
}
FLAG_VALUES = {'load_format': {'auto', 'safetensors', 'runai_streamer'}}
_PIN = re.compile(r'^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.+!_-]*)$')


class BuildFailed(RuntimeError):
    pass


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def with_flags(argv, engine, patch):
    flags = ENGINE_FLAGS.get(engine)
    if flags is None:
        raise BuildFailed(f'profile_patch is not supported for engine {engine}')
    argv = list(argv)
    for key, value in patch.items():
        if key in FLAG_VALUES and value not in FLAG_VALUES[key]:
            raise BuildFailed(f'{key}={value!r} is not one of {sorted(FLAG_VALUES[key])}')
        flag = flags[key]
        names = [f for f in (flag if isinstance(flag, tuple) else (flag,)) if f]
        for name in names:  # remove any existing occurrence (with its value for valued flags)
            while name in argv:
                i = argv.index(name)
                del argv[i:i + (1 if isinstance(flag, tuple) else 2)]
        if isinstance(flag, tuple):
            on, off = flag
            chosen = on if value else off
            if chosen:
                argv.append(chosen)
        else:
            argv += [flag, str(value)]
    return argv


def relocate(body, port):
    argv = list(body.get('command') or [])
    if '--port' not in argv:
        raise BuildFailed('derived candidates need a launch command with --port')
    argv[argv.index('--port') + 1] = str(port)
    url = urlparse(body['base_url'])
    return argv, urlunparse(url._replace(netloc=f'{url.hostname}:{port}'))


class IsolatedExecutor(EndpointExecutor):
    def __init__(self, repository, log_root, *, work_root, source_root, source_revision, allocations, provider,
                 sandbox=None, pip_index_args=(), test_timeout=900, keep_workspaces=False, image_builder=None, **kw):
        super().__init__(repository, log_root, allocations=allocations, provider=provider, **kw)
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.source_root, self.source_revision = Path(source_root), source_revision
        if not re.fullmatch(r'[0-9a-f]{40}', source_revision or ''):
            raise ValueError('source_revision must be a full commit hash')
        self.sandbox = sandbox or Sandbox()
        self.pip_index_args, self.test_timeout = tuple(pip_index_args), test_timeout
        # A local package index named in pip_index_args must be readable inside the boundary.
        self.sandbox.readable = [*self.sandbox.readable,
                                 *(Path(a) for a in self.pip_index_args if not a.startswith('-') and Path(a).is_dir())]
        self.keep_workspaces = keep_workspaces
        self.workspaces = {}
        self.image_builder = image_builder  # required for plugin/source candidates on container providers

    # --- materialization -----------------------------------------------------------

    def _artifact(self, store, name, content):
        store[name] = self.repository.artifact(content if isinstance(content, bytes) else content.encode())

    def build_source(self, ws, action, artifacts, log):
        params = action['params']
        if 'patch_sha256' not in params:
            raise BuildFailed(f"{action['id']}: source_patch has no stored patch (patch_sha256)")
        diff = self.repository.read_artifact(params['patch_sha256']).decode()
        touched = check_patch(diff, params['files'])
        src = ws / 'src'
        if not src.exists():
            archive = subprocess.run(['git', '-C', str(self.source_root), 'archive', '--format=tar',
                                      self.source_revision], capture_output=True)
            if archive.returncode:
                raise BuildFailed(f'cannot export source revision {self.source_revision}')
            src.mkdir(parents=True)
            subprocess.run(['tar', '-x', '-C', str(src)], input=archive.stdout, check=True)
            subprocess.run(['git', 'init', '-q', str(src)], check=True)
        apply = subprocess.run(['git', 'apply', '--whitespace=nowarn', '-'], cwd=src, input=diff.encode(),
                               capture_output=True)
        if apply.returncode:
            raise BuildFailed(f"{action['id']}: patch does not apply: {apply.stderr.decode()[-500:]}")
        # Tests and evaluator always come from the pinned revision, never the patch.
        for protected in ('tests', 'src/inferstack/evaluation'):
            shutil.rmtree(src / protected, ignore_errors=True)
            archive = subprocess.run(['git', '-C', str(self.source_root), 'archive', '--format=tar',
                                      self.source_revision, protected], capture_output=True)
            if archive.returncode:
                raise BuildFailed(f'cannot restore {protected} from {self.source_revision}')
            subprocess.run(['tar', '-x', '-C', str(src)], input=archive.stdout, check=True)
        self._artifact(artifacts, f"{action['id']}.diff", diff)
        code, out = self.sandbox.run([sys.executable, '-m', 'compileall', '-q', *sorted(touched)], cwd=src, home=ws,
                                     log=log, timeout=300)
        if code:
            raise BuildFailed(f"{action['id']}: patched source does not compile:\n{out[-1500:]}")
        tests = params.get('tests') or []
        for test in tests:
            if not re.fullmatch(r'tests/test_[A-Za-z0-9_]+\.py(::[A-Za-z0-9_\[\]-]+)?', test):
                raise BuildFailed(f"{action['id']}: invalid test selector {test!r}")
        if tests:
            code, out = self.sandbox.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *tests],
                                         cwd=src, home=ws, env={'PYTHONPATH': str(src / 'src')},
                                         timeout=self.test_timeout, log=log)
            self._artifact(artifacts, f"{action['id']}.tests.log", out)
            if code:
                raise BuildFailed(f"{action['id']}: declared tests failed on the patched source")
        return src

    def build_plugin(self, ws, action, artifacts, log):
        dependency = action['params']['dependency']
        if not _PIN.match(dependency):
            raise BuildFailed(f'{dependency!r} is not an exact pin')
        venv, wheels = ws / 'venv', ws / 'wheelhouse'
        if not venv.exists():
            code, out = self.sandbox.run([sys.executable, '-m', 'venv', '--system-site-packages', str(venv)],
                                         cwd=ws, home=ws, log=log, timeout=300)
            if code:
                raise BuildFailed(f'venv creation failed: {out[-500:]}')
        wheels.mkdir(exist_ok=True)
        # Download is the only networked step; install happens offline from recorded wheels.
        code, out = self.sandbox.run([str(venv / 'bin' / 'python'), '-m', 'pip', 'download', '--no-deps',
                                      '--only-binary=:all:', '-d', str(wheels), *self.pip_index_args, dependency],
                                     cwd=ws, home=ws, offline=False, log=log, timeout=600)
        if code:
            raise BuildFailed(f'cannot download {dependency}: {out[-800:]}')
        hashes = {w.name: hashlib.sha256(w.read_bytes()).hexdigest() for w in sorted(wheels.glob('*.whl'))}
        code, out = self.sandbox.run([str(venv / 'bin' / 'python'), '-m', 'pip', 'install', '--no-index', '--no-deps',
                                      '--find-links', str(wheels), dependency], cwd=ws, home=ws, log=log, timeout=600)
        if code:
            raise BuildFailed(f'offline install of {dependency} failed: {out[-800:]}')
        _, freeze = self.sandbox.run([str(venv / 'bin' / 'python'), '-m', 'pip', 'freeze', '--local'], cwd=ws,
                                     home=ws, log=log, timeout=120)
        self._artifact(artifacts, f"{action['id']}.wheels.json", json.dumps(hashes, sort_keys=True))
        self._artifact(artifacts, f"{action['id']}.freeze.txt", freeze)
        return venv

    # --- execution ------------------------------------------------------------------

    async def _offload(self, fn, *args):
        """Run a blocking build step in a thread; on cancellation kill it and wait for cleanup."""
        task = asyncio.ensure_future(asyncio.to_thread(fn, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self.sandbox.terminate_all()
            await asyncio.gather(task, return_exceptions=True)
            raise

    def _build_allocation(self, context, namespace):
        """Persisted, active build allocation identified by its tag (no single leader process).

        Every build/test/image step runs inside `Sandbox.owned_by(id)`, so each subprocess
        carries INFERSTACK_ALLOCATION_ID; the reconciler finds and stops them after a crash."""
        if self.allocations is None or not context or 'job' not in context:
            return None
        from inferstack.infrastructure.local import LocalProcessProvider
        alloc = self.allocations.intent(context['job'], context['step_key'], purpose='build:' + namespace,
                                        provider='local', resource_type='process',
                                        spec={'kind': 'build', 'namespace': namespace}, gpu_count=0)
        if alloc['state'] != 'intent':
            raise RuntimeError(f"build allocation {alloc['id']} already {alloc['state']}")
        host = self.provider.host if getattr(self.provider, 'name', None) == 'local' else LocalProcessProvider().host
        external = {'host': host, 'kind': 'build', 'allocation_id': alloc['id']}
        self.allocations.record_external(alloc['id'], external)
        self.allocations.activate(alloc['id'], context['job'], external)
        return alloc['id']

    def _end_build(self, alloc_id, note):
        """Release the build allocation only after no tagged build process remains."""
        from inferstack.infrastructure.local import LocalProcessProvider
        external = self.allocations.get(alloc_id)['external'] or {'allocation_id': alloc_id}
        provider = self.provider if getattr(self.provider, 'name', None) == 'local' else LocalProcessProvider()
        if provider.stop(external, grace_seconds=5.0):
            self.allocations.release(alloc_id, note)
        else:
            self.allocations.mark_orphaned(alloc_id, note + '_stop_failed')

    async def execute(self, plan, namespace, context=None):
        if not plan.get('actions'):
            return await super().execute(plan, namespace, context)
        if context is None:
            raise ValueError('isolated execution requires a job context (allocation ownership)')
        untrusted = [a['id'] for a in plan['actions'] if a.get('type') in ('plugin_install', 'source_patch')]
        remote = getattr(self.provider, 'launches_remotely', False)
        if untrusted and remote and self.image_builder is None:
            # A local build cannot change the pinned image a container provider launches.
            raise BuildFailed(f'actions {untrusted} need a candidate image build for {self.provider.name}; '
                              'configure an image builder (executor.image_builder) or use the local provider')
        if untrusted and not self.sandbox.filesystem_isolated:
            raise BuildFailed(f'actions {untrusted} run untrusted code but this host has no filesystem isolation '
                              'boundary (sandbox-exec or bwrap); refusing')
        actions = {a['id']: a for a in plan['actions']}
        baseline = self.repository.read('deployment', plan['baseline_id'])
        if baseline['revision'] != plan['baseline_revision']:
            raise ValueError('baseline changed after planning')
        ws = Path(tempfile.mkdtemp(prefix=f'{namespace[:40]}-', dir=self.work_root))
        self.workspaces[namespace] = ws
        artifacts, log, src, venv = {}, [], None, None
        source = baseline
        body = dict(baseline['body'])
        build_alloc = owner = None
        try:
            if untrusted:
                build_alloc = self._build_allocation(context, namespace)
            owner = Sandbox.owned_by(build_alloc)
            owner.__enter__()  # every build/test/image subprocess below carries the build allocation tag
            for action_id in plan['action_order']:
                action = actions[action_id]
                params = action['params']
                if action['type'] == 'config_change' and 'candidate_deployment_id' in params:
                    source = self.repository.read('deployment', params['candidate_deployment_id'])
                    if source['revision'] != plan.get('candidate_revision', source['revision']):
                        raise ValueError('candidate changed after planning')
                    body = dict(source['body'])
                elif action['type'] == 'config_change':
                    body['command'] = with_flags(body.get('command') or [], body['engine'], params['profile_patch'])
                    body['profile'] = {**(body.get('profile') or {}), **params['profile_patch']}
                elif action['type'] == 'plugin_install':
                    venv = await self._offload(self.build_plugin, ws, action, artifacts, log)
                    body['plugin_refs'] = sorted({*body.get('plugin_refs', []), params['technique_id']})
                elif action['type'] == 'source_patch':
                    src = await self._offload(self.build_source, ws, action, artifacts, log)
            derived = source is baseline or venv is not None or src is not None or body != source['body']
            env = {}
            if venv is not None:
                env['PATH'] = f"{venv / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
                env['VIRTUAL_ENV'] = str(venv)
                if body.get('command') and Path(body['command'][0]).name.startswith('python'):
                    body['command'] = [str(venv / 'bin' / 'python'), *body['command'][1:]]
            if src is not None:
                env['PYTHONPATH'] = str(src / 'src')
            if derived and remote:
                body = await self._containerize(body, ws, namespace, plan, actions, src, artifacts,
                                                allocation_id=build_alloc)
            if owner is not None:
                owner.__exit__(None, None, None)
                owner = None
            if build_alloc is not None:
                self._end_build(build_alloc, 'build_complete')  # confirmed no tagged build process remains
                build_alloc = None
            if derived and not remote:
                body['command'], body['base_url'] = relocate(body, free_port())
                if venv is not None or src is not None:
                    # The candidate server runs untrusted code: launch it inside the boundary.
                    body['command'] = self.sandbox.launcher(body['command'], workspace=ws)
            if derived:
                body['id'] = f'{namespace}-candidate'
                self.repository.put_record('deployment', body['id'], body)
                record = self.repository.read('deployment', body['id'])
            else:
                record = source
            self._artifact(artifacts, 'build.log.json', json.dumps(log, default=str))
            isolation = {**self.sandbox.describe(), 'workspace': ws.name, 'source_revision': self.source_revision}
            from inferstack.execution.sandbox import scrubbed_env
            return await self.launch(record, plan, namespace, context, env=scrubbed_env(env, home=ws),
                                     extra={'artifacts': artifacts, 'isolation': isolation, 'derived': derived})
        except BaseException:
            if owner is not None:
                owner.__exit__(None, None, None)
            if build_alloc is not None:
                self.sandbox.terminate_all()
                self._end_build(build_alloc, 'build_aborted')
            self._discard(namespace)
            raise

    async def _containerize(self, body, ws, namespace, plan, actions, src, artifacts, allocation_id=None):
        """Derived candidate for a container provider: build its image (if code changed) and relocate its port."""
        container = dict((body.get('profile') or {}).get('container') or {})
        if not container.get('image'):
            raise BuildFailed('container candidates need profile.container.image on the baseline deployment')
        deps = [actions[a]['params']['dependency'] for a in plan['action_order']
                if actions[a]['type'] == 'plugin_install']
        if deps or src is not None:
            identity = await self._offload(lambda: self.image_builder.build(
                ws, container['image'], tag=f'inferstack-candidate:{namespace[:60].lower()}', dependencies=deps,
                with_src=src is not None, **({'allocation_id': allocation_id} if allocation_id else {})))
            self._artifact(artifacts, 'image.Dockerfile', identity['dockerfile'])
            self._artifact(artifacts, 'image.json', json.dumps({k: v for k, v in identity.items()
                                                                if k not in ('dockerfile', 'log_tail')}, sort_keys=True))
            container['image'] = identity['image']
            body['image_digest'] = identity['image'].split('@', 1)[1] if '@' in identity['image'] else identity['image']
        if self.provider.name == 'docker':
            container['host_port'] = free_port()
            url = urlparse(body['base_url'])
            body['base_url'] = urlunparse(url._replace(netloc=f"{url.hostname}:{container['host_port']}"))
        body['profile'] = {**(body.get('profile') or {}), 'container': container}
        body['command'] = []
        return body

    @staticmethod
    def tree_sha256(root):
        """Content digest of a runnable workspace (paths, modes and bytes), excluding logs/tmp."""
        digest = hashlib.sha256()
        root = Path(root)
        for path in sorted(p for p in root.rglob('*') if p.is_file() and not p.is_symlink()):
            rel = path.relative_to(root).as_posix()
            if rel.startswith('.tmp/') or '__pycache__' in rel or rel == 'env.json':
                continue
            digest.update(rel.encode() + b'\0' + oct(path.stat().st_mode & 0o777).encode() + b'\0')
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()

    async def retain(self, candidate, context):
        """Keep the candidate serving and its workspace as the release's runnable artifact."""
        serving = await super().retain(candidate, context)
        ws = self.workspaces.pop(candidate['id'], None)
        if ws is None:
            return serving
        runnable = {'path': str(ws), 'tree_sha256': await asyncio.to_thread(self.tree_sha256, ws)}
        return {**(serving or {}), 'runnable': runnable}

    def _discard(self, namespace):
        ws = self.workspaces.pop(namespace, None)
        if ws is not None and not self.keep_workspaces:
            shutil.rmtree(ws, ignore_errors=True)

    async def cleanup(self, candidate):
        try:
            await super().cleanup(candidate)
        finally:
            self._discard(candidate['id'])
