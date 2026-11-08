"""Process isolation primitives for experiment builds and candidates.

What is enforced, and how it is reported:
- Environment: children get an allowlisted environment only. Credentials
  (anything *_KEY, *_TOKEN, *_SECRET, *_PASSWORD, provider and cloud variables)
  never reach build steps or candidates.
- Network: build/test steps run inside a new network namespace (`unshare -n`)
  when the host allows it. Availability is probed, recorded with every run, and
  `require_network_isolation=True` turns its absence into a refusal.
- Resources: CPU time, open files, file size and (optionally) address space are
  capped with rlimits; wall time with a timeout that kills the process group.
These are process-level controls on an attached host, not a VM or container
boundary; Package H runs the same steps inside containers on clusters.
"""
from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
from dataclasses import dataclass

ENV_ALLOW = ('PATH', 'LANG', 'LC_ALL', 'TZ', 'TMPDIR', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
             'CUDA_HOME', 'LD_LIBRARY_PATH', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'PYTHONHASHSEED')
SECRET_MARKERS = ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'PASSWD', 'CREDENTIAL', 'AUTH', 'COOKIE', 'SESSION')


def scrubbed_env(extra=None, *, home):
    env = {k: os.environ[k] for k in ENV_ALLOW if k in os.environ}
    env.update({'HOME': str(home), 'PYTHONNOUSERSITE': '1', 'PIP_NO_INPUT': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1'})
    for key, value in (extra or {}).items():
        if any(m in key.upper() for m in SECRET_MARKERS) and key != 'INFERSTACK_ALLOCATION_ID':
            raise ValueError(f'refusing to pass credential-like variable {key} into a sandbox')
        env[key] = value
    return env


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
    def __init__(self, *, require_network_isolation=False, limits=Limits()):
        self.network_isolated = network_isolation_available()
        if require_network_isolation and not self.network_isolated:
            raise RuntimeError('network isolation is required but unavailable on this host (unshare -n)')
        self.limits = limits

    def describe(self):
        return {'network_isolation': 'netns' if self.network_isolated else 'unavailable',
                'env': 'allowlisted', 'limits': self.limits.__dict__}

    def run(self, argv, *, cwd, home, env=None, offline=True, timeout=None, log=None):
        """Run a build/test step. Returns (returncode, output). Never raises on non-zero."""
        if offline and self.network_isolated:
            argv = ['unshare', '-n', '--', *argv]
        limits = self.limits

        def preexec():
            os.setsid()
            limits.apply()
        proc = subprocess.Popen(argv, cwd=cwd, env=scrubbed_env(env, home=home), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, preexec_fn=preexec, text=True)
        try:
            output, _ = proc.communicate(timeout=timeout or limits.wall_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            output, _ = proc.communicate()
            output = (output or '') + f'\n[sandbox] killed after {timeout or limits.wall_seconds}s'
            return 124, output
        if log is not None:
            log.append({'argv': argv, 'returncode': proc.returncode, 'output_tail': (output or '')[-4000:]})
        return proc.returncode, output or ''
