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


PROTECTED_TREES = ('tests', 'src/inferstack/evaluation')


def _export(source_root, revision, dest, *paths):
    import subprocess
    archive = subprocess.run(['git', '-C', str(source_root), 'archive', '--format=tar', revision, *paths],
                             capture_output=True)
    if archive.returncode:
        raise RuntimeError(f"cannot export {' '.join(paths) or 'source'} at revision {revision}")
    subprocess.run(['tar', '-x', '-C', str(dest)], input=archive.stdout, check=True)


def materialize_source(source_root, revision, diff, src):
    """Apply an inspected diff to a pristine export of `revision` in `src`.

    Exports the revision when `src` does not exist yet (several patches can stack), then
    restores tests and the evaluator from the pinned revision: a patch never changes the
    code that judges it. Raises RuntimeError with the reason on failure."""
    import shutil
    import subprocess
    from pathlib import Path
    src = Path(src)
    if not src.exists():
        src.mkdir(parents=True)
        _export(source_root, revision, src)
        subprocess.run(['git', 'init', '-q', str(src)], check=True)
    apply = subprocess.run(['git', 'apply', '--whitespace=nowarn', '-'], cwd=src, input=diff.encode(),
                           capture_output=True)
    if apply.returncode:
        raise RuntimeError(f'patch does not apply: {apply.stderr.decode()[-500:]}')
    for protected in PROTECTED_TREES:
        shutil.rmtree(src / protected, ignore_errors=True)
        try:
            _export(source_root, revision, src, protected)
        except RuntimeError:
            if (Path(source_root) / protected).exists():
                raise
    return src
