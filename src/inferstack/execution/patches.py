"""Unified-diff inspection and confinement before a patch is applied.

The diff is parsed (not trusted): every touched path must be a regular text file
under an allowed root, declared by the plan, with no traversal, absolute paths,
symlink/mode changes, binary hunks or renames out of bounds. Tests, the
evaluator and the verifier are never patchable, whatever the plan says.
"""
from __future__ import annotations

import re

ALLOWED_ROOTS = ('src/inferstack/kernels/', 'src/inferstack/runtime/')
FORBIDDEN = ('tests/', 'src/inferstack/evaluation/', 'src/inferstack/optimization/', 'src/inferstack/control/',
             'src/inferstack/execution/', 'scripts/', 'baselines/', 'results/', '.github/', 'pyproject.toml')
MAX_PATCH_BYTES = 512 * 1024


class PatchRejected(ValueError):
    pass


def _clean(path):
    path = path.strip()
    if path == '/dev/null':
        return None
    if path.startswith(('a/', 'b/')):
        path = path[2:]
    parts = path.split('/')
    if path.startswith('/') or '..' in parts or '' in parts or '\\' in path or '\0' in path:
        raise PatchRejected(f'unsafe path in patch: {path!r}')
    return path


def touched_paths(diff: str) -> set[str]:
    if len(diff.encode()) > MAX_PATCH_BYTES:
        raise PatchRejected('patch too large')
    paths = set()
    for line in diff.splitlines():
        if line.startswith(('GIT binary patch', 'Binary files ')):
            raise PatchRejected('binary patches are not allowed')
        if re.match(r'^(old|new) mode |^new file mode 120000|^(deleted|new) file mode 1[0-9]{5}$', line) and \
                not re.match(r'^new file mode 100644$|^deleted file mode 100644$', line):
            raise PatchRejected(f'mode or symlink change not allowed: {line}')
        if line.startswith(('rename from ', 'rename to ', 'copy from ', 'copy to ')):
            paths.add(_clean(line.split(' ', 2)[2]))
        if line.startswith(('--- ', '+++ ')):
            cleaned = _clean(line[4:].split('\t')[0])
            if cleaned:
                paths.add(cleaned)
    if not paths:
        raise PatchRejected('patch touches no files')
    return paths


def check_patch(diff: str, declared_files) -> set[str]:
    paths = touched_paths(diff)
    for path in paths:
        if path.startswith(FORBIDDEN) or not path.startswith(ALLOWED_ROOTS):
            raise PatchRejected(f'patch touches {path}, outside {ALLOWED_ROOTS}')
        if not path.endswith(('.py', '.cu', '.cuh', '.cpp', '.h', '.triton', '.json')):
            raise PatchRejected(f'patch touches non-source file {path}')
    undeclared = paths - set(declared_files)
    if undeclared:
        raise PatchRejected(f'patch touches files the plan did not declare: {sorted(undeclared)}')
    return paths
