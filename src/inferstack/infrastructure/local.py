"""Attached-host process provider with verifiable process identity.

Each launched process gets INFERSTACK_ALLOCATION_ID in its environment and runs
in its own session (process group). Identity is (host, pid, pgid, start token,
allocation id);
the start token is the kernel start time from /proc on Linux or `ps lstart` on
macOS, so a reused PID is never mistaken for our process. A launch whose identity
was not recorded before a crash can still be found by scanning for the tag.

An allocation is the whole process group plus every process carrying its tag,
not just the leader: engines fork workers, and a launcher may exit while its
children keep the GPU. `stop` is confirmed only when no member remains. A process
group id cannot be reused while the group has members, so killing the group of a
recorded pgid cannot hit an unrelated process.
"""
from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

TAG = 'INFERSTACK_ALLOCATION_ID'


def _start_token(pid):
    stat = Path(f'/proc/{pid}/stat')
    try:
        if stat.exists():
            # Field 22 (starttime) follows the parenthesized comm, which may contain spaces.
            return 'proc:' + stat.read_text().rsplit(')', 1)[1].split()[19]
        out = subprocess.run(['ps', '-o', 'lstart=', '-p', str(pid)], capture_output=True, text=True, timeout=5)
        return 'ps:' + out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, IndexError, subprocess.SubprocessError):
        return None


def _zombie(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z'
    except (OSError, IndexError):
        return False


class LocalProcessProvider:
    name = 'local'
    resource_type = 'process'

    def __init__(self, host=None):
        self.host = host or socket.gethostname()
        self._allocation_of = {}

    def external_for(self, pid):
        external = {'host': self.host, 'pid': pid, 'pgid': os.getpgid(pid), 'start': _start_token(pid)}
        if pid in self._allocation_of:
            external['allocation_id'] = self._allocation_of[pid]
        return external

    async def spawn(self, alloc_id, argv, log, env=None, inherit_env=True):
        """Launch tagged in a new session. With inherit_env=False the child gets exactly
        `env` (plus the tag): no worker credentials leak into candidates."""
        child_env = {**(os.environ if inherit_env else {}), **(env or {}), TAG: alloc_id}
        process = await asyncio.create_subprocess_exec(*argv, stdout=log, stderr=log, env=child_env,
                                                       start_new_session=True)
        self._allocation_of[process.pid] = alloc_id
        return process

    def owns(self, external):
        return isinstance(external, dict) and external.get('host') == self.host

    def alive(self, external):
        pid = external['pid']
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass
        if _zombie(pid):
            return False
        token = _start_token(pid)
        return token is not None and token == external.get('start')

    def find_tagged(self, alloc_id):
        """PIDs whose environment carries this allocation's tag (Linux /proc; macOS ps eww)."""
        needle = f'{TAG}={alloc_id}'.encode()
        found = []
        proc = Path('/proc')
        if proc.exists():
            for entry in proc.iterdir():
                if entry.name.isdigit():
                    try:
                        if needle in (entry / 'environ').read_bytes().split(b'\0') and not _zombie(int(entry.name)):
                            found.append(int(entry.name))
                    except OSError:
                        continue
            return found
        out = subprocess.run(['ps', 'eww', '-ax', '-o', 'pid=,command='], capture_output=True, timeout=10)
        for line in out.stdout.splitlines():
            if needle in line.split():
                found.append(int(line.split()[0]))
        return found

    def observed_argv(self, external):
        """Command line of the identified running process (attestation evidence), or None."""
        if not self.alive(external):
            return None
        cmdline = Path(f"/proc/{external['pid']}/cmdline")
        try:
            if cmdline.exists():
                return [a.decode() for a in cmdline.read_bytes().split(b'\0') if a]
            out = subprocess.run(['ps', '-ww', '-o', 'args=', '-p', str(external['pid'])], capture_output=True,
                                 text=True, timeout=5)
            return out.stdout.strip().split() if out.returncode == 0 and out.stdout.strip() else None
        except (OSError, subprocess.SubprocessError):
            return None

    @staticmethod
    def group_members(pgid):
        """Live, non-zombie PIDs in a process group."""
        if not pgid:
            return []
        try:
            out = subprocess.run(['ps', '-ax', '-o', 'pid=,pgid=,stat='], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None  # unknown: never reported as empty
        if out.returncode != 0:
            return None
        members = []
        for line in out.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] == str(pgid) and not parts[2].startswith('Z'):
                members.append(int(parts[0]))
        return members

    def members(self, external):
        """Every live process belonging to the allocation, or None when membership is unknown."""
        group = self.group_members(external.get('pgid'))
        if group is None:
            return None
        tagged = self.find_tagged(external['allocation_id']) if external.get('allocation_id') else []
        found = set(group) | set(tagged)
        if self.alive(external):
            found.add(external['pid'])
        found.discard(os.getpid())
        return sorted(found)

    def residual(self, external):
        """True when any member of the allocation is still running (or membership is unknown)."""
        members = self.members(external)
        return members is None or bool(members)

    def stop(self, external, grace_seconds=10.0):
        """Terminate the whole allocation; True only when no member process remains."""
        if not self.residual(external):
            return True
        for sig, wait in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, 5.0)):
            members = self.members(external) or []
            try:
                os.killpg(external['pgid'], sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                return False
            for pid in members:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    return False
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                if not self.residual(external):
                    return True
                time.sleep(0.05)
        return not self.residual(external)
