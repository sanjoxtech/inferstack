"""Isolation boundary for experiment builds, tests and candidate processes.

What is enforced, and how it is reported:
- Filesystem: untrusted code may write only inside its experiment workspace (plus
  device nodes). Everything else (trusted evaluator, fixtures, baseline artifacts,
  the control database, the active deployment) is read-only to it, and common
  credential locations are unreadable. Backends: macOS Seatbelt (`sandbox-exec`)
  or Linux bubblewrap (`bwrap`, read-only root bind with a writable workspace).
  Without a backend `filesystem_isolated` is False and the isolated executor
  refuses plugin/source experiments (see IsolatedExecutor).
- Environment: children get an allowlisted environment only. Credentials
  (anything *_KEY, *_TOKEN, *_SECRET, *_PASSWORD, provider and cloud variables)
  never reach build steps or candidates.
- Network: offline steps have no network (Seatbelt deny rule, bwrap
  `--unshare-net`, or `unshare -n`). Availability is probed and reported, and
  `require_network_isolation=True` turns its absence into a refusal.
- Resources: CPU time, open files, file size and (optionally) address space are
  capped with rlimits (also for candidate servers, via `wrap`); wall time with a
  timeout that kills the process group; GPUs via the allocation's device list.
Every running step is tracked so cancellation can kill its process group.
"""
from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import sys
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path

_ALLOCATION: ContextVar[str | None] = ContextVar('inferstack_build_allocation', default=None)

ENV_ALLOW = ('PATH', 'LANG', 'LC_ALL', 'TZ', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
             'CUDA_HOME', 'LD_LIBRARY_PATH', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'PYTHONHASHSEED')
SECRET_MARKERS = ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'PASSWD', 'CREDENTIAL', 'AUTH', 'COOKIE', 'SESSION')
# Credential locations untrusted code must not read (relative to the real home).
SECRET_PATHS = ('.ssh', '.aws', '.config/gcloud', '.kube', '.docker', '.netrc', '.claude/.env', '.git-credentials',
                '.gnupg', '.azure', '.huggingface', '.cache/huggingface/token')


def scrubbed_env(extra=None, *, home):
    env = {k: os.environ[k] for k in ENV_ALLOW if k in os.environ}
    tmp = Path(home) / '.tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    env.update({'HOME': str(home), 'TMPDIR': str(tmp), 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PIP_NO_INPUT': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1'})
    for key, value in (extra or {}).items():
        if any(m in key.upper() for m in SECRET_MARKERS) and key != 'INFERSTACK_ALLOCATION_ID':
            raise ValueError(f'refusing to pass credential-like variable {key} into a sandbox')
        env[key] = value
    return env


def _works(argv):
    try:
        return subprocess.run(argv, capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def filesystem_backend():
    """'seatbelt', 'bwrap' or None, verified by actually running a confined no-op."""
    if sys.platform == 'darwin' and shutil.which('sandbox-exec'):
        if _works(['sandbox-exec', '-p', '(version 1)(allow default)(deny file-write*)', '/usr/bin/true']):
            return 'seatbelt'
    if shutil.which('bwrap') and _works(['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc',
                                          '--unshare-net', 'true']):
        return 'bwrap'
    return None


def _sb_quote(path):
    return '"' + str(path).replace('\\', '\\\\').replace('"', '\\"') + '"'


# System locations untrusted code may read (contents). Everything else is denied:
# reads are an allowlist, not a blacklist of known credential directories.
SYSTEM_READ_ROOTS = ('/System', '/usr', '/bin', '/sbin', '/Library/Frameworks', '/Library/Developer/CommandLineTools',
                     '/private/etc', '/etc', '/private/var/db/timezone', '/private/var/db/dyld', '/opt/homebrew',
                     '/usr/local', '/lib', '/lib64', '/lib32', '/opt/conda', '/opt/nvidia', '/var/lib/dpkg')


def _python_roots():
    roots = set()
    for value in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        roots.add(Path(value))
        roots.add(Path(value).resolve())
    executable = Path(sys.executable)
    roots.add(executable.parent)
    roots.add(executable.resolve().parents[1])
    return roots


def read_roots(readable=()):
    """Directories whose contents a confined process may read.

    System libraries, this Python installation and its site-packages, the installed
    InferStack package directory (trusted launcher, not the surrounding project), the
    Hugging Face model cache (weights; the token file is denied separately) and the
    caller's explicit `readable` paths. Project directories, home config and any other
    host files are unreadable, so a host project's `.env` is not reachable."""
    roots = {Path(p) for p in SYSTEM_READ_ROOTS}
    roots |= _python_roots()
    roots.add(Path(__file__).resolve().parents[1])  # the inferstack package itself
    hub = Path(os.environ.get('HF_HUB_CACHE') or Path(os.environ.get('HF_HOME') or
               Path(os.path.expanduser('~')) / '.cache' / 'huggingface') / 'hub')
    roots.add(hub)
    roots |= {Path(p) for p in readable}
    return sorted({r for r in roots if r.exists()} | {r.resolve() for r in roots if r.exists()})


def seatbelt_profile(writable, *, offline, home=None, readable=()):
    home = Path(home or os.path.expanduser('~'))
    roots = read_roots(readable) + [Path(w).resolve() for w in writable]
    rules = ['(version 1)', '(allow default)', '(deny file-write*)',
             '(allow file-write* ' + ' '.join(f'(subpath {_sb_quote(Path(w).resolve())})' for w in writable) +
             ' (regex #"^/dev/"))',
             # Contents are an allowlist; metadata (stat/path traversal) stays allowed so
             # interpreters can resolve paths, but no file outside the roots can be read.
             '(deny file-read-data)',
             '(allow file-read-data (literal "/") (regex #"^/dev/") ' +
             ' '.join(f'(subpath {_sb_quote(r)})' for r in roots) +
             # directory listing (names only) of the package's parent so imports resolve
             f' (literal {_sb_quote(Path(__file__).resolve().parents[2])}))',
             # Process credentials: no keychain access and no other process's details.
             '(deny mach-lookup (global-name "com.apple.SecurityServer") (global-name "com.apple.securityd"))']
    secrets = [home / p for p in SECRET_PATHS]
    rules.append('(deny file-read* ' + ' '.join(f'(subpath {_sb_quote(p)})' for p in secrets) + ')')
    if offline:
        rules.append('(deny network-outbound (remote ip))')
    return ''.join(rules)


def bwrap_prefix(writable, *, offline, readable=(), home=None):
    """bubblewrap mount namespace with only the read roots visible (read-only), the
    writable paths bound read-write, fresh /dev, /proc (own PID namespace, so other
    processes' environments are not visible), /tmp, and credential paths masked."""
    home = Path(home or os.path.expanduser('~'))
    prefix = ['bwrap', '--die-with-parent', '--new-session', '--unshare-pid', '--unshare-ipc',
              '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp']
    seen = set()
    for root in read_roots(readable):
        source = Path(root)
        if source in seen:
            continue
        seen.add(source)
        if source.is_symlink():  # e.g. /bin -> usr/bin on merged-/usr systems
            prefix += ['--symlink', os.readlink(source), str(source)]
        else:
            prefix += ['--ro-bind-try', str(source), str(source)]
    for secret in SECRET_PATHS:
        path = home / secret
        if path.is_dir():
            prefix += ['--tmpfs', str(path)]
        elif path.exists():
            prefix += ['--ro-bind', '/dev/null', str(path)]
    for w in writable:
        prefix += ['--bind', str(Path(w).resolve()), str(Path(w).resolve())]
    if offline:
        prefix.append('--unshare-net')
    return prefix


def network_isolation_available():
    if os.name != 'posix' or shutil.which('unshare') is None:
        return False
    try:
        return subprocess.run(['unshare', '-n', 'true'], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@dataclass(frozen=True)
class Limits:
    cpu_seconds: int = 1800
    open_files: int = 4096
    file_bytes: int = 4 * 2**30
    address_space_bytes: int | None = None
    wall_seconds: float = 1800

    def apply(self):
        resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_seconds, self.cpu_seconds))
        resource.setrlimit(resource.RLIMIT_NOFILE, (self.open_files, self.open_files))
        resource.setrlimit(resource.RLIMIT_FSIZE, (self.file_bytes, self.file_bytes))
        if self.address_space_bytes:
            resource.setrlimit(resource.RLIMIT_AS, (self.address_space_bytes, self.address_space_bytes))


class Sandbox:
    def __init__(self, *, require_network_isolation=False, require_filesystem_isolation=False, limits=Limits()):
        self.backend = filesystem_backend()
        self.filesystem_isolated = self.backend is not None
        self.netns = self.backend is None and network_isolation_available()
        self.network_isolated = self.backend is not None or self.netns
        if require_filesystem_isolation and not self.filesystem_isolated:
            raise RuntimeError('filesystem isolation is required but unavailable (need sandbox-exec or bwrap)')
        if require_network_isolation and not self.network_isolated:
            raise RuntimeError('network isolation is required but unavailable on this host')
        self.limits = limits
        self.readable = []  # operator-declared read roots for every step (e.g. a local wheel index)
        self._live, self._lock = set(), threading.Lock()

    def describe(self):
        return {'filesystem_isolation': self.backend or 'unavailable',
                'network_isolation': self.backend or ('netns' if self.netns else 'unavailable'),
                'env': 'allowlisted', 'limits': asdict(self.limits)}

    def wrap(self, argv, *, workspace, offline, extra_writable=(), readable=()):
        """Prefix argv with the isolation boundary: writes confined to the workspace, reads to
        `read_roots(readable)` plus the writable paths."""
        writable = [Path(workspace), *map(Path, extra_writable)]
        argv = list(argv)
        if self.backend == 'seatbelt':
            return ['sandbox-exec', '-p', seatbelt_profile(writable, offline=offline, readable=readable), *argv]
        if self.backend == 'bwrap':
            return [*bwrap_prefix(writable, offline=offline, readable=readable), '--', *argv]
        if offline and self.netns:
            return ['unshare', '-n', '--', *argv]
        return argv

    def launcher(self, argv, *, workspace, offline=False, readable=()):
        """argv for a long-running candidate server: isolation boundary plus rlimits."""
        limits = self.limits
        # The trusted launcher runs isolated (-I) from this installation, never from the
        # candidate's PYTHONPATH, so patched source cannot replace it.
        applied = [sys.executable, '-I', str(Path(__file__).with_name('limit_exec.py')), str(limits.cpu_seconds),
                   str(limits.open_files), str(limits.file_bytes), str(limits.address_space_bytes or 0), '--', *argv]
        # The declared program files of the command are readable (their directories only).
        programs = [Path(a).parent for a in argv if isinstance(a, str) and a.startswith('/') and Path(a).is_file()]
        return self.wrap(applied, workspace=workspace, offline=offline, readable=[*readable, *programs])

    def terminate_all(self):
        """Kill every running step's process group (cancellation/deadline)."""
        with self._lock:
            procs = list(self._live)
        for proc in procs:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    @staticmethod
    @contextmanager
    def owned_by(allocation_id):
        """Tag every step run in this context (including threads started from it, which copy
        the context) with its persisted allocation identity, so a crashed worker's build
        processes are found by the reconciler through the same tag as launched servers."""
        token = _ALLOCATION.set(allocation_id)
        try:
            yield
        finally:
            _ALLOCATION.reset(token)

    def run(self, argv, *, cwd, home, env=None, offline=True, timeout=None, log=None, extra_writable=(), readable=()):
        """Run a build/test step inside the boundary. Returns (returncode, output). Never raises on non-zero."""
        argv = self.wrap(argv, workspace=home, offline=offline,
                         extra_writable=[cwd, *extra_writable] if Path(cwd).resolve() != Path(home).resolve()
                         else extra_writable, readable=[*self.readable, *readable])
        limits = self.limits

        def preexec():
            os.setsid()
            limits.apply()
        child_env = scrubbed_env(env, home=home)
        if _ALLOCATION.get() is not None:
            from inferstack.infrastructure.local import TAG
            child_env[TAG] = _ALLOCATION.get()
        proc = subprocess.Popen(argv, cwd=cwd, env=child_env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, preexec_fn=preexec, text=True)
        with self._lock:
            self._live.add(proc)
        try:
            try:
                output, _ = proc.communicate(timeout=timeout or limits.wall_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                output, _ = proc.communicate()
                output = (output or '') + f'\n[sandbox] killed after {timeout or limits.wall_seconds}s'
                return 124, output
        finally:
            with self._lock:
                self._live.discard(proc)
        if log is not None:
            log.append({'argv': argv[:64], 'returncode': proc.returncode,
                        'output_tail': (output or '')[-4000:]})
        return proc.returncode, output or ''
