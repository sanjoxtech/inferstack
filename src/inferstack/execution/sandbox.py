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
from dataclasses import asdict, dataclass
from pathlib import Path

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


def seatbelt_profile(writable, *, offline, home=None):
    home = Path(home or os.path.expanduser('~'))
    rules = ['(version 1)', '(allow default)', '(deny file-write*)',
             '(allow file-write* ' + ' '.join(f'(subpath {_sb_quote(Path(w).resolve())})' for w in writable) +
             ' (regex #"^/dev/"))']
    secrets = [home / p for p in SECRET_PATHS]
    rules.append('(deny file-read* ' + ' '.join(f'(subpath {_sb_quote(p)})' for p in secrets) + ')')
    if offline:
        rules.append('(deny network-outbound (remote ip))')
    return ''.join(rules)


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
        self._live, self._lock = set(), threading.Lock()

    def describe(self):
        return {'filesystem_isolation': self.backend or 'unavailable',
                'network_isolation': self.backend or ('netns' if self.netns else 'unavailable'),
                'env': 'allowlisted', 'limits': asdict(self.limits)}

    def wrap(self, argv, *, workspace, offline, extra_writable=()):
        """Prefix argv with the isolation boundary: writes confined to the workspace."""
        writable = [Path(workspace), *map(Path, extra_writable)]
        argv = list(argv)
        if self.backend == 'seatbelt':
            return ['sandbox-exec', '-p', seatbelt_profile(writable, offline=offline), *argv]
        if self.backend == 'bwrap':
            prefix = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp',
                      '--die-with-parent', '--new-session']
            home = Path(os.path.expanduser('~'))
            for secret in SECRET_PATHS:
                if (home / secret).is_dir():
                    prefix += ['--tmpfs', str(home / secret)]
                elif (home / secret).exists():
                    prefix += ['--ro-bind', '/dev/null', str(home / secret)]
            for w in writable:
                prefix += ['--bind', str(w.resolve()), str(w.resolve())]
            if offline:
                prefix.append('--unshare-net')
            return [*prefix, '--', *argv]
        if offline and self.netns:
            return ['unshare', '-n', '--', *argv]
        return argv

    def launcher(self, argv, *, workspace, offline=False):
        """argv for a long-running candidate server: isolation boundary plus rlimits."""
        limits = self.limits
        # The trusted launcher runs isolated (-I) from this installation, never from the
        # candidate's PYTHONPATH, so patched source cannot replace it.
        applied = [sys.executable, '-I', str(Path(__file__).with_name('limit_exec.py')), str(limits.cpu_seconds),
                   str(limits.open_files), str(limits.file_bytes), str(limits.address_space_bytes or 0), '--', *argv]
        return self.wrap(applied, workspace=workspace, offline=offline)

    def terminate_all(self):
        """Kill every running step's process group (cancellation/deadline)."""
        with self._lock:
            procs = list(self._live)
        for proc in procs:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    def run(self, argv, *, cwd, home, env=None, offline=True, timeout=None, log=None, extra_writable=()):
        """Run a build/test step inside the boundary. Returns (returncode, output). Never raises on non-zero."""
        argv = self.wrap(argv, workspace=home, offline=offline,
                         extra_writable=[cwd, *extra_writable] if Path(cwd).resolve() != Path(home).resolve()
                         else extra_writable)
        limits = self.limits

        def preexec():
            os.setsid()
            limits.apply()
        proc = subprocess.Popen(argv, cwd=cwd, env=scrubbed_env(env, home=home), stdout=subprocess.PIPE,
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
