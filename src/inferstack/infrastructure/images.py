"""Candidate container images built from an isolated experiment workspace.

The build runs inside Docker (`docker build --network none`) on a digest-pinned base
image, so a candidate's plugins and patched source are installed for the
container's own platform, offline, from the experiment's recorded wheelhouse:

    FROM <base@sha256>            # pinned
    COPY wheelhouse /opt/inferstack/wheelhouse
    COPY src /opt/inferstack/src  # patched source (optional)
    RUN pip install --no-index --find-links /opt/inferstack/wheelhouse <pins>   # offline
    ENV PYTHONPATH=/opt/inferstack/src/src

The result is content-addressed: a local image ID (`sha256:...`) for the Docker
provider, or, after `push`, a registry digest reference (`repo@sha256:...`) that
Kubernetes can pull. Build context, Dockerfile and log are hashed into artifacts.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

_PIN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9][A-Za-z0-9.+!_-]*$')


class ImageBuildFailed(RuntimeError):
    pass


class ImageBuilder:
    def __init__(self, *, docker='docker', registry=None, timeout=1800, runner=None):
        """registry: 'host:port/repo' to push to (needed for Kubernetes); None keeps images local."""
        self.docker, self.registry, self.timeout = docker, registry, timeout
        self.runner = runner or self._run

    def _run(self, argv, cwd=None, allocation_id=None):
        env = None
        if allocation_id:
            import os
            from inferstack.infrastructure.local import TAG
            env = {**os.environ, TAG: allocation_id}  # the docker CLI process carries the build's identity
        try:
            proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=self.timeout, env=env)
        except subprocess.TimeoutExpired:
            return 124, 'timeout'
        return proc.returncode, proc.stdout + proc.stderr

    def dockerfile(self, base_image, *, dependencies=(), with_src=False, python='python'):
        if not re.search(r'@sha256:[0-9a-f]{64}$', base_image):
            raise ImageBuildFailed('base image must be pinned by digest')
        for dep in dependencies:
            if not _PIN.match(dep):
                raise ImageBuildFailed(f'{dep!r} is not an exact pin')
        lines = [f'FROM {base_image}', 'COPY wheelhouse /opt/inferstack/wheelhouse']
        if with_src:
            lines.append('COPY src /opt/inferstack/src')
        if dependencies:
            lines.append(f'RUN {python} -m pip install --no-index --no-deps --find-links /opt/inferstack/wheelhouse '
                         + ' '.join(dependencies))
        if with_src:
            lines.append('ENV PYTHONPATH=/opt/inferstack/src/src')
        lines.append('LABEL inferstack.candidate=true')
        return '\n'.join(lines) + '\n'

    def build(self, workspace, base_image, *, tag, dependencies=(), with_src=False, allocation_id=None):
        """Build from `workspace/{wheelhouse,src}`. Returns identity and hashed build inputs."""
        workspace = Path(workspace)
        context = workspace / 'image'
        shutil.rmtree(context, ignore_errors=True)
        context.mkdir()
        wheels = workspace / 'wheelhouse'
        shutil.copytree(wheels, context / 'wheelhouse') if wheels.is_dir() else (context / 'wheelhouse').mkdir()
        if with_src:
            shutil.copytree(workspace / 'src', context / 'src', ignore=shutil.ignore_patterns('.git', '__pycache__'))
        dockerfile = self.dockerfile(base_image, dependencies=dependencies, with_src=with_src)
        (context / 'Dockerfile').write_text(dockerfile)
        inputs = hashlib.sha256()
        for path in sorted(p for p in context.rglob('*') if p.is_file()):
            inputs.update(path.relative_to(context).as_posix().encode() + b'\0' + path.read_bytes())
        argv = [self.docker, 'build', '--network', 'none', '--pull=false', '-t', tag]
        if allocation_id:
            argv += ['--label', f'inferstack.allocation={allocation_id}']
        argv.append(str(context))
        code, log = (self.runner(argv, allocation_id=allocation_id) if allocation_id and self.runner == self._run
                     else self.runner(argv))
        if code:
            raise ImageBuildFailed(f'image build failed:\n{log[-1500:]}')
        code, image_id = self.runner([self.docker, 'image', 'inspect', '--format', '{{.Id}}', tag])
        if code or not re.fullmatch(r'sha256:[0-9a-f]{64}', image_id.strip()):
            raise ImageBuildFailed('built image has no content identity')
        identity = {'image': image_id.strip(), 'image_id': image_id.strip(), 'tag': tag, 'base_image': base_image,
                    'context_sha256': inputs.hexdigest(), 'dockerfile': dockerfile, 'log_tail': log[-4000:]}
        if self.registry:
            identity.update(self.push(tag, image_id.strip()))
        return identity

    def push(self, tag, image_id):
        target = f'{self.registry}:{image_id.split(":", 1)[1][:20]}'
        for argv in ([self.docker, 'tag', tag, target], [self.docker, 'push', target]):
            code, log = self.runner(argv)
            if code:
                raise ImageBuildFailed(f'{argv[1]} failed: {log[-800:]}')
        code, out = self.runner([self.docker, 'image', 'inspect', '--format', '{{json .RepoDigests}}', target])
        repo = self.registry.rsplit(':', 1)[0] if self.registry.count(':') > 1 else self.registry
        digests = [d for d in (json.loads(out) if code == 0 else []) if d.startswith(repo + '@sha256:')]
        if len(digests) != 1:
            raise ImageBuildFailed('pushed image has no unique registry digest')
        return {'image': digests[0], 'pushed': target}
