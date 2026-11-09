"""Bounded implementer handoff (harness H3, step 1).

A planner proposal that needs source code (`source_patch` without a stored patch) is
turned into an `implementation` record whose handoff the controller fixes:

- pinned source revision and baseline deployment revision;
- the scoped hypothesis, technique and evidence references;
- the files the implementer may touch (inside the patchable roots only);
- limits: attempts, wall time, patch size;
- independent acceptance tests chosen by the controller, never by the implementer.

An implementer (a human-supplied diff or a coding agent run in the sandbox) can only
submit attempts. Every attempt is journaled before work starts, so a crash leaves an
`in_progress` attempt that recovery marks `abandoned` (counted against the limit)
instead of silently regenerating a different treatment. A submitted diff is inspected,
applied to a pristine export of the pinned revision with tests/evaluator restored,
compiled and tested in an isolated workspace. Only then does the record become
`validated`, holding an immutable content-addressed candidate manifest. A different
diff is a different candidate: it needs a revision (a child record), never an
overwrite. Candidate-owned tests run in addition to the independent tests.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from inferstack.execution.patches import ALLOWED_ROOTS, FORBIDDEN, MAX_PATCH_BYTES, PatchRejected, _export, \
    check_patch, materialize_source

HANDOFF_VERSION = 1
DEFAULT_LIMITS = {'max_attempts': 3, 'timeout_s': 1800, 'max_patch_bytes': MAX_PATCH_BYTES}
# Controller-owned acceptance tests per patchable root (used when the request names none).
DEFAULT_TESTS = {'src/inferstack/kernels/': ('tests/test_kernels_reference.py',),
                 'src/inferstack/runtime/': ('tests/test_kernels_reference.py',)}
TEST_SELECTOR = re.compile(r'tests/test_[A-Za-z0-9_]+\.py(::[A-Za-z0-9_\[\]-]+)?')
CANDIDATE_TEST = re.compile(r'test_[a-z0-9_]{1,60}\.py')
MAX_CANDIDATE_TESTS = 10


class ValidationFailed(RuntimeError):
    pass


class ArtifactLocked(ValueError):
    """A validated candidate is immutable; a changed diff needs a revision."""


class AttemptLimit(RuntimeError):
    pass


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def candidate_identity(manifest):
    """Identity of the treatment: pinned source + exact diff + dependencies + candidate tests.
    Logs, timing and the implementer's name are evidence, not identity."""
    return 'cand-' + _sha({k: manifest[k] for k in ('source_revision', 'patch_sha256', 'dependencies',
                                                     'candidate_tests')})[:24]


class ImplementationDesk:
    def __init__(self, repository, *, source_root=None, sandbox=None, work_root=None, test_timeout=900,
                 default_tests=None):
        self.repository = repository
        self.default_tests = DEFAULT_TESTS if default_tests is None else dict(default_tests)
        self.source_root = Path(source_root) if source_root else None
        self.sandbox, self.test_timeout = sandbox, test_timeout
        self.work_root = Path(work_root) if work_root else None

    # --- records ------------------------------------------------------------------

    def read(self, impl_id):
        return self.repository.read('implementation', impl_id)

    def _update(self, impl_id, mutate):
        record = self.read(impl_id)
        body = json.loads(json.dumps(record['body']))
        result = mutate(body)
        self.repository.put_record('implementation', impl_id, body, expected_revision=record['revision'])
        return result

    def request(self, proposal, plan, *, baseline_id, source_revision, job_id=None, evidence_refs=(), limits=None,
                independent_tests=None, parent_id=None):
        """Create (idempotently) the handoff for one `needs_implementation` proposal."""
        files = sorted(set(proposal.get('files') or []))
        if not files:
            raise ValueError('a proposal needs the files it may change')
        for path in files:
            if not isinstance(path, str) or '..' in path.split('/') or path.startswith(FORBIDDEN) \
                    or not path.startswith(ALLOWED_ROOTS):
                raise ValueError(f'{path!r} is outside the patchable roots {ALLOWED_ROOTS}')
        if not re.fullmatch(r'[0-9a-f]{40}', source_revision or ''):
            raise ValueError('source_revision must be a full commit hash')
        tests = set(independent_tests or ())
        if not independent_tests:
            action = next((a for a in plan.get('actions', []) if a.get('id') == proposal.get('action_id')), {})
            tests |= set((action.get('params') or {}).get('tests') or ())
            for path in files:
                tests |= {t for root, ts in self.default_tests.items() if path.startswith(root) for t in ts}
        if not tests:
            raise ValueError('no independent acceptance tests for these files; the controller must name them')
        bad = [t for t in tests if not TEST_SELECTOR.fullmatch(t)]
        if bad:
            raise ValueError(f'invalid test selectors: {bad}')
        baseline = self.repository.read('deployment', baseline_id)
        handoff = {'version': HANDOFF_VERSION, 'source_revision': source_revision,
                   'baseline_ref': f"{baseline_id}@{baseline['revision']}",
                   'hypothesis': str(plan.get('hypothesis', ''))[:4000],
                   'action_id': proposal.get('action_id'), 'technique_id': proposal.get('technique_id'),
                   'description': str(proposal.get('description') or '')[:4000], 'allowed_files': files,
                   'evidence_refs': sorted(evidence_refs), 'limits': {**DEFAULT_LIMITS, **(limits or {})},
                   'acceptance': {'independent_tests': sorted(tests)}}
        handoff_sha = _sha(handoff)
        impl_id = 'impl-' + _sha({'handoff': handoff_sha, 'job': job_id, 'parent': parent_id})[:20]
        try:
            return self.read(impl_id)
        except KeyError:
            pass
        raw_plan = {k: plan[k] for k in ('hypothesis', 'baseline_ref', 'evidence_ids', 'actions',
                                         'expected_measurements', 'gates', 'estimate', 'stop_rule') if k in plan}
        self.repository.put_record('implementation', impl_id, {
            'job_id': job_id, 'baseline_ref': handoff['baseline_ref'], 'parent_ref': parent_id,
            'proposal': {k: proposal.get(k) for k in ('action_id', 'technique_id', 'files', 'description')},
            'plan': raw_plan, 'handoff': handoff, 'handoff_sha256': handoff_sha, 'state': 'requested',
            'created_at': self.repository.clock()}, expected_revision=0)
        return self.read(impl_id)

    def revise(self, impl_id):
        """A child handoff for a changed treatment; the validated parent stays unchanged."""
        body = self.read(impl_id)['body']
        handoff = body['handoff']
        return self.request(body['proposal'], body['plan'], baseline_id=body['baseline_ref'].split('@')[0],
                            source_revision=handoff['source_revision'], job_id=body['job_id'],
                            evidence_refs=handoff['evidence_refs'], limits=handoff['limits'],
                            independent_tests=handoff['acceptance']['independent_tests'], parent_id=impl_id)

    # --- attempts -----------------------------------------------------------------

    def begin_attempt(self, impl_id, implementer):
        """Journal an attempt before any work starts. Returns its number."""
        now = self.repository.clock()

        def mutate(body):
            if body['state'] == 'validated':
                raise ArtifactLocked(f'{impl_id} is validated; a changed diff needs a revision (revise())')
            if body['state'] in ('exhausted', 'cancelled'):
                raise AttemptLimit(f"{impl_id} is {body['state']}")
            for attempt in body['attempts']:
                if attempt['outcome'] == 'in_progress':  # interrupted earlier (crash): never resumed silently
                    attempt.update(outcome='abandoned', finished_at=now,
                                   reason='attempt was interrupted; its output is not reused')
            if len(body['attempts']) >= body['handoff']['limits']['max_attempts']:
                body['state'] = 'exhausted'
                return None
            number = len(body['attempts']) + 1
            body['attempts'].append({'n': number, 'implementer': implementer, 'started_at': now,
                                     'outcome': 'in_progress'})
            body['state'] = 'implementing'
            return number
        number = self._update(impl_id, mutate)
        if number is None:
            raise AttemptLimit(f'{impl_id}: attempt limit reached')
        return number

    def finish_attempt(self, impl_id, number, *, outcome, diff=None, candidate_tests=None, dependencies=(),
                       logs='', cost=None):
        """Record an attempt's outcome. A submitted diff is validated before it can become the candidate."""
        current = self.read(impl_id)['body']
        attempt = next((a for a in current['attempts'] if a['n'] == number), None)
        if attempt is None or attempt['outcome'] != 'in_progress':
            raise ValueError(f'attempt {number} of {impl_id} is not in progress (it may have been abandoned)')
        record = {'outcome': outcome, 'cost': dict(cost or {})}
        manifest = None
        if outcome == 'submitted':
            if not isinstance(diff, str) or not diff.strip():
                record.update(outcome='rejected', reason='empty diff')
            else:
                record['patch_sha256'] = self.repository.artifact(diff.encode())['sha256']
                try:
                    manifest, validation_log = self.validate(current['handoff'], diff, candidate_tests or {},
                                                             dependencies)
                    record['validation_log_sha256'] = self.repository.artifact(validation_log.encode())['sha256']
                    record.update(outcome='validated', candidate_id=manifest['candidate_id'])
                except (PatchRejected, ValidationFailed) as exc:
                    record.update(outcome='rejected', reason=str(exc)[:2000])
                    log = getattr(exc, 'log', None)
                    if log:
                        record['validation_log_sha256'] = self.repository.artifact(log.encode())['sha256']
        if logs:
            record['logs_sha256'] = self.repository.artifact(logs.encode() if isinstance(logs, str) else logs)['sha256']
        record['finished_at'] = self.repository.clock()

        def mutate(body):
            target = next(a for a in body['attempts'] if a['n'] == number)
            if target['outcome'] != 'in_progress':
                raise ValueError(f'attempt {number} of {impl_id} was abandoned while it ran')
            target.update(record)
            if manifest is not None:
                body['state'], body['candidate'] = 'validated', manifest
            elif body['state'] != 'cancelled':
                limit = body['handoff']['limits']['max_attempts']
                body['state'] = 'exhausted' if len(body['attempts']) >= limit else 'requested'
        self._update(impl_id, mutate)
        return self.read(impl_id)

    def submit(self, impl_id, diff, *, implementer=None, candidate_tests=None, dependencies=(), cost=None):
        """Human/operator path: one journaled attempt with a supplied diff. Idempotent for the same artifact."""
        body = self.read(impl_id)['body']
        if body['state'] == 'validated':
            same = (hashlib.sha256(diff.encode()).hexdigest() == body['candidate']['patch_sha256']
                    and _candidate_tests(candidate_tests or {}) == body['candidate']['candidate_tests'])
            if same:
                return self.read(impl_id)
            raise ArtifactLocked(f'{impl_id} already has a validated candidate; a changed diff is a new candidate '
                                 '(revise() creates a child handoff)')
        number = self.begin_attempt(impl_id, implementer or {'kind': 'human'})
        return self.finish_attempt(impl_id, number, outcome='submitted', diff=diff, candidate_tests=candidate_tests,
                                   dependencies=dependencies, cost=cost)

    def cancel(self, impl_id, reason='cancelled'):
        now = self.repository.clock()

        def mutate(body):
            if body['state'] == 'validated':
                raise ArtifactLocked('a validated candidate is kept; it simply is not executed')
            for attempt in body['attempts']:
                if attempt['outcome'] == 'in_progress':
                    attempt.update(outcome='cancelled', finished_at=now, reason=reason)
            body['state'] = 'cancelled'
        self._update(impl_id, mutate)
        if self.sandbox is not None:
            self.sandbox.terminate_all()
        return self.read(impl_id)

    # --- validation ---------------------------------------------------------------

    def validate(self, handoff, diff, candidate_tests, dependencies=()):
        """Inspect, apply, compile and test a diff in an isolated workspace. Returns (manifest, log)."""
        if len(diff.encode()) > handoff['limits']['max_patch_bytes']:
            raise PatchRejected('patch exceeds the handoff size limit')
        touched = check_patch(diff, handoff['allowed_files'])
        if dependencies:
            raise ValidationFailed('new dependencies need a pinned plugin_install action, not a source candidate')
        tests = _candidate_tests(candidate_tests)
        if self.source_root is None or self.sandbox is None or not self.sandbox.filesystem_isolated:
            raise ValidationFailed('validation needs the pinned source tree and a filesystem isolation boundary')
        root = self.work_root or Path(tempfile.gettempdir())
        root.mkdir(parents=True, exist_ok=True)
        ws = Path(tempfile.mkdtemp(prefix='impl-', dir=root))
        log, out_text = [], []
        try:
            try:
                src = materialize_source(self.source_root, handoff['source_revision'], diff, ws / 'src')
            except RuntimeError as exc:
                raise ValidationFailed(str(exc)) from None
            env = {'PYTHONPATH': str(src / 'src')}
            code, out = self.sandbox.run([sys.executable, '-m', 'compileall', '-q', *sorted(touched)], cwd=src,
                                         home=ws, log=log, timeout=300)
            out_text.append(out)
            if code:
                raise _failed('patched source does not compile', out_text)
            independent = handoff['acceptance']['independent_tests']
            code, out = self.sandbox.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                                          *independent], cwd=src, home=ws, env=env, timeout=self.test_timeout,
                                         log=log)
            out_text.append(out)
            if code:
                raise _failed('independent acceptance tests failed on the patched source', out_text)
            if candidate_tests:
                folder = src / 'candidate_tests'
                folder.mkdir()
                for name, text in candidate_tests.items():
                    (folder / name).write_text(text)
                code, out = self.sandbox.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                                              *sorted(f'candidate_tests/{n}' for n in candidate_tests)], cwd=src,
                                             home=ws, env=env, timeout=self.test_timeout, log=log)
                out_text.append(out)
                if code:
                    raise _failed('candidate tests failed on the patched source', out_text)
        finally:
            shutil.rmtree(ws, ignore_errors=True)
        manifest = {'version': 1, 'source_revision': handoff['source_revision'],
                    'patch_sha256': hashlib.sha256(diff.encode()).hexdigest(), 'touched_files': sorted(touched),
                    'dependencies': [], 'candidate_tests': tests,
                    'independent_tests': handoff['acceptance']['independent_tests'],
                    'handoff_sha256': _sha(handoff), 'sandbox': self.sandbox.describe()['filesystem_isolation']}
        manifest['candidate_id'] = candidate_identity(manifest)
        manifest['manifest_sha256'] = self.repository.artifact(
            json.dumps(manifest, sort_keys=True).encode())['sha256']
        return manifest, '\n'.join(out_text)

    # --- execution handoff -----------------------------------------------------------

    def executable_plan(self, impl_ids):
        """Raw plan with each proposal's validated patch, the controller's tests and the candidate identity."""
        records = [self.read(i)['body'] for i in impl_ids]
        if not records:
            raise ValueError('no implementations')
        plans = {_sha(r['plan']) for r in records}
        if len(plans) != 1:
            raise ValueError('implementations belong to different plans')
        pending = [i for i, r in zip(impl_ids, records) if r['state'] != 'validated']
        if pending:
            raise ValidationFailed(f'implementations without a validated candidate: {pending}')
        raw = json.loads(json.dumps(records[0]['plan']))
        by_action = {r['proposal']['action_id']: (i, r) for i, r in zip(impl_ids, records)}
        for action in raw['actions']:
            if action['type'] == 'source_patch' and action['id'] in by_action:
                impl_id, record = by_action[action['id']]
                action['params'] = {'technique_id': record['proposal']['technique_id'],
                                    'files': record['proposal']['files'],
                                    'description': record['proposal'].get('description') or '',
                                    'patch_sha256': record['candidate']['patch_sha256'],
                                    'tests': record['candidate']['independent_tests'],
                                    'implementation_id': impl_id}
        return raw


def _candidate_tests(tests):
    if not isinstance(tests, dict) or len(tests) > MAX_CANDIDATE_TESTS:
        raise ValidationFailed(f'candidate tests must be a mapping of at most {MAX_CANDIDATE_TESTS} files')
    for name, text in tests.items():
        if not CANDIDATE_TEST.fullmatch(str(name)) or not isinstance(text, str) or len(text) > 200_000:
            raise ValidationFailed(f'invalid candidate test {name!r}')
    return {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sorted(tests.items())}


def _failed(message, outputs):
    exc = ValidationFailed(f"{message}:\n{(outputs[-1] if outputs else '')[-1500:]}")
    exc.log = '\n'.join(outputs)
    return exc


class CommandImplementer:
    """Coding-agent path: run an implementer command on a pinned export inside the sandbox.

    The command gets the handoff file path as its last argument and may edit files in
    its working directory (the exported source). Its git metadata lives outside the
    writable workspace, so it cannot alter how its own diff is computed. Optional
    outputs: `candidate_tests/test_*.py` beside the handoff and `COST.json` (accounting
    only). The harvested diff goes through the same validation as a human diff."""

    def __init__(self, argv, *, sandbox, source_root, work_root, name='command', offline=True, env=None):
        self.argv, self.sandbox, self.name = list(argv), sandbox, name
        self.source_root, self.work_root = Path(source_root), Path(work_root)
        self.offline, self.env = offline, dict(env or {})

    async def run(self, desk, impl_id):
        handoff = desk.read(impl_id)['body']['handoff']
        number = desk.begin_attempt(impl_id, {'kind': 'agent', 'name': self.name})
        self.work_root.mkdir(parents=True, exist_ok=True)
        holder = Path(tempfile.mkdtemp(prefix='agent-', dir=self.work_root))
        task = asyncio.ensure_future(asyncio.to_thread(self._work, holder, handoff))
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            self.sandbox.terminate_all()
            await asyncio.gather(task, return_exceptions=True)
            shutil.rmtree(holder, ignore_errors=True)
            try:
                desk.finish_attempt(impl_id, number, outcome='cancelled')
            except ValueError:
                pass  # already closed by desk.cancel()
            raise
        except Exception as exc:  # noqa: BLE001 - recorded as the attempt's failure
            result = {'outcome': 'failed', 'logs': f'{type(exc).__name__}: {exc}'}
        finally:
            if task.done():
                shutil.rmtree(holder, ignore_errors=True)
        return desk.finish_attempt(impl_id, number, **result)

    def _work(self, holder, handoff):
        ws, git_dir = holder / 'ws', holder / 'git'
        src = ws / 'src'
        src.mkdir(parents=True)
        _export(self.source_root, handoff['source_revision'], src)
        git = ['git', f'--git-dir={git_dir}', f'--work-tree={src}', '-c', 'core.hooksPath=/dev/null',
               '-c', 'user.name=inferstack', '-c', 'user.email=inferstack@localhost']
        subprocess.run(['git', 'init', '-q', '--bare', str(git_dir)], check=True, capture_output=True)
        subprocess.run([*git, 'add', '-A'], check=True, capture_output=True)
        subprocess.run([*git, 'commit', '-qm', 'pinned'], check=True, capture_output=True)
        handoff_file = ws / 'HANDOFF.json'
        handoff_file.write_text(json.dumps(handoff, indent=2, sort_keys=True))
        started = time.monotonic()
        # Only the declared program files are readable, never their directories.
        programs = [Path(a) for a in self.argv if a.startswith('/') and Path(a).is_file()]
        code, out = self.sandbox.run([*self.argv, str(handoff_file)], cwd=src, home=ws, env=self.env,
                                     offline=self.offline, timeout=handoff['limits']['timeout_s'],
                                     readable_files=programs)
        cost = {'wall_seconds': round(time.monotonic() - started, 3)}
        try:
            reported = json.loads((ws / 'COST.json').read_text())
            cost.update({k: v for k, v in reported.items() if k in ('provider_tokens', 'provider_usd')
                         and isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0})
        except (OSError, ValueError, AttributeError):
            pass
        if code == 124:
            return {'outcome': 'timeout', 'logs': out, 'cost': cost}
        if code:
            return {'outcome': 'failed', 'logs': out, 'cost': cost}
        subprocess.run([*git, 'add', '-A'], check=True, capture_output=True)
        diff = subprocess.run([*git, 'diff', '--cached', '--no-ext-diff', '--no-textconv'], capture_output=True,
                              text=True, check=True).stdout
        tests = {}
        folder = ws / 'candidate_tests'
        if folder.is_dir():
            for path in sorted(folder.iterdir())[:MAX_CANDIDATE_TESTS + 1]:
                if path.is_file() and not path.is_symlink():
                    tests[path.name] = path.read_text(errors='replace')
        return {'outcome': 'submitted', 'diff': diff, 'candidate_tests': tests, 'logs': out, 'cost': cost}
