from __future__ import annotations
import asyncio
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
import httpx


@dataclass(frozen=True)
class DeploymentSpec:
    id: str
    engine: str
    model: str
    base_url: str
    api_key_env: str | None = None
    command: tuple[str,...] = ()

    def __post_init__(self):
        parsed=urlparse(self.base_url)
        if parsed.scheme not in {'http','https'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('base_url must be HTTP(S) without embedded credentials')
        if self.engine not in {'inferstack_native','vllm','sglang','generic_openai'}:
            raise ValueError('unsupported engine')
        if not self.id or not self.model:raise ValueError('deployment id and model required')
        if not isinstance(self.command,(list,tuple)) or any(not isinstance(x,str) or not x for x in self.command):
            raise ValueError('command must be an argument list')


class ManagedEngine:
    """Attach to an endpoint, or supervise an explicitly configured local process."""
    def __init__(self,spec:DeploymentSpec, transport=None):
        self.spec=spec
        headers={}
        if spec.api_key_env:
            key=os.environ.get(spec.api_key_env)
            if not key:raise ValueError('missing engine credential environment variable')
            headers['Authorization']='Bearer '+key
        self.client=httpx.AsyncClient(base_url=spec.base_url.rstrip('/')+'/',headers=headers,
                                      timeout=httpx.Timeout(120,connect=10),transport=transport)
        self.process=None
        self.log=None

    async def ready(self):
        try:
            response=await self.client.get('v1/models')
            response.raise_for_status()
            return any(m.get('id')==self.spec.model for m in response.json().get('data',[]))
        except (httpx.HTTPError,ValueError,TypeError,AttributeError):return False

    async def start(self,log_path,timeout=300):
        if self.process is not None:raise RuntimeError('worker already started')
        if not self.spec.command:
            if not await self.ready():raise RuntimeError('attached endpoint is not ready')
            return
        path=Path(log_path);path.parent.mkdir(parents=True,exist_ok=True)
        self.log=path.open('ab')
        try:
            self.process=await asyncio.create_subprocess_exec(*self.spec.command,stdout=self.log,stderr=self.log,start_new_session=True)
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                if self.process.returncode is not None:raise RuntimeError('engine exited before readiness')
                if await asyncio.wait_for(self.ready(),max(.01,deadline-time.monotonic())):return
                await asyncio.sleep(.25)
            raise TimeoutError('engine readiness timed out')
        except BaseException:
            await self.stop()
            raise

    async def stop(self):
        if self.process is not None and self.process.returncode is None:
            try:os.killpg(self.process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:await asyncio.wait_for(self.process.wait(),10)
            except asyncio.TimeoutError:
                try:os.killpg(self.process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                await self.process.wait()
        self.process=None
        if self.log:self.log.close();self.log=None

    async def complete(self,payload,chat=False):
        body=dict(payload);body['model']=self.spec.model;body['stream']=False
        response=await self.client.post('v1/chat/completions' if chat else 'v1/completions',json=body)
        response.raise_for_status()
        return response.json()

    async def stream(self,payload,chat=False):
        body=dict(payload);body['model']=self.spec.model;body['stream']=True
        async with self.client.stream('POST','v1/chat/completions' if chat else 'v1/completions',json=body) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():yield chunk

    async def metrics(self):
        response=await self.client.get('metrics')
        response.raise_for_status()
        return response.text

    async def close(self):
        await self.stop()
        await self.client.aclose()
