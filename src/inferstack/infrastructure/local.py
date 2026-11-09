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
    stat = Path(f'/proc/{pid}/stat')
    try:
        if stat.exists():
            return stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
        # No /proc (macOS): ask ps for the process state.
        out = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True, timeout=5)
        return out.returncode == 0 and out.stdout.strip().startswith('Z')
    except (OSError, IndexError, subprocess.SubprocessError):
        return False


_CHILDREN = {}  # pid -> Popen for children this process spawned (reaped so they never linger as zombies)


def _reap():
    for pid, popen in list(_CHILDREN.items()):
        if popen.poll() is not None:
            _CHILDREN.pop(pid, None)


class DetachedProcess:
    """Awaitable view of a Popen child for code written against asyncio processes."""

    def __init__(self, popen):
        self.popen, self.pid = popen, popen.pid

    @property
    def returncode(self):
        return self.popen.poll()

    async def wait(self):
        while self.popen.poll() is None:
            await asyncio.sleep(0.05)
        return self.popen.returncode

    def terminate(self):
        self.popen.terminate()

    def kill(self):
        self.popen.kill()


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
        `env` (plus the tag): no worker credentials leak into candidates.

        Uses subprocess.Popen, not asyncio subprocesses: an asyncio subprocess transport
        that is garbage-collected (or whose event loop goes away) kills its child, which
        would stop retained candidates and serving runtimes whose process handle was
        dropped. A dropped Popen never kills its child."""
        child_env = {**(os.environ if inherit_env else {}), **(env or {}), TAG: alloc_id}
        popen = await asyncio.to_thread(subprocess.Popen, list(argv), stdout=log, stderr=log, env=child_env,
                                        start_new_session=True)
        self._allocation_of[popen.pid] = alloc_id
        _CHILDREN[popen.pid] = popen
        return DetachedProcess(popen)

    def owns(self, external):
        return isinstance(external, dict) and external.get('host') == self.host

    def alive(self, external):
        _reap()  # our own exited children must not count as alive zombies
        pid = external.get('pid')
        if pid is None:
            return False  # tag-identified allocation (build steps): no single leader process
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
        proc = Path('/proc')
        if proc.is_dir() and (proc / 'self' / 'stat').exists():
            # Linux: read /proc directly (minimal container images have no `ps`).
            members = []
            for entry in proc.iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    stat = (entry / 'stat').read_text()
                except OSError:
                    continue
                fields = stat[stat.rfind(')') + 2:].split()  # after "(comm)": state ppid pgrp ...
                if len(fields) >= 3 and fields[2] == str(pgid) and fields[0] != 'Z':
                    members.append(int(entry.name))
            return members
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

    def ownership(self, external):
        _reap()
        return self._ownership(external)

    def _ownership(self, external):
        """(owned_pids, uncertain_pids, group_verified) or None when membership is unknown.

        Ownership is proven only by durable identity: the leader with a matching start
        token (which also pins its process group, since a live leader holds its PGID),
        or a process carrying this allocation's tag. A historical numeric PGID alone
        proves nothing once its leader is gone: the ID may belong to an unrelated,
        reused group, so its untagged members are reported as uncertain, never signalled.
        """
        group = self.group_members(external.get('pgid'))
        if group is None:
            return None
        tagged = set(self.find_tagged(external['allocation_id'])) if external.get('allocation_id') else set()
        leader = self.alive(external)
        owned = set(tagged)
        if leader:
            owned |= set(group) | {external['pid']}
        uncertain = set() if leader else set(group) - tagged
        owned.discard(os.getpid())
        uncertain.discard(os.getpid())
        return sorted(owned), sorted(uncertain), leader

    def members(self, external):
        """Every live process provably belonging to the allocation, or None when unknown/uncertain."""
        found = self.ownership(external)
        if found is None or found[1]:
            return None
        return found[0]

    def residual(self, external):
        """True when any member of the allocation is still running (or membership is unknown)."""
        members = self.members(external)
        return members is None or bool(members)

    def stop(self, external, grace_seconds=10.0):
        """Terminate the whole allocation; True only when no member process remains.

        Signals go only to provably owned processes (and to the process group only while
        its verified leader holds it). Uncertain ownership returns False so the caller
        records an operator-visible orphaned/unresolved allocation."""
        if not self.residual(external):
            return True
        for sig, wait in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, 5.0)):
            found = self.ownership(external)
            if found is None:
                return False
            owned, uncertain, group_verified = found
            if not owned:
                return not uncertain  # nothing of ours left; uncertain processes are never signalled
            if group_verified:
                try:
                    os.killpg(external['pgid'], sig)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    return False
            for pid in owned:
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
