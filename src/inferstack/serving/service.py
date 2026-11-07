from __future__ import annotations
import uuid
from inferstack.core.request import Request,SamplingParams

class ModelService:
    def __init__(self,model_id,tokenizer,engine):self.model_id=model_id;self.tokenizer=tokenizer;self.engine=engine;self.started=False
    async def start(self):await self.engine.start();self.started=True
    async def close(self):await self.engine.close();self.started=False
    def ready(self):return bool(self.started and getattr(self.engine,'healthy',True))
    def diagnostics(self):
        fn=getattr(self.engine,'diagnostics',None);return fn() if callable(fn) else {'started':self.started,'engine':type(self.engine).__name__}
    async def submit(self,prompt:str,sampling:SamplingParams):
        ids=tuple(self.tokenizer.encode(prompt,add_special_tokens=False));r=Request(uuid.uuid4().hex,ids,sampling);q=await self.engine.submit(r);return r,q
    def cancel(self,rid):return self.engine.cancel(rid)
    def decode_token(self,tok:int):return self.tokenizer.decode([int(tok)],skip_special_tokens=True)

class ModelServiceRegistry:
    def __init__(self):self._services={};self._aliases={};self.started=False
    def register(self,service,*aliases):
        self._services[service.model_id]=service
        for a in aliases:self._aliases[a]=service.model_id
    def get(self,name):
        key=self._aliases.get(name,name)
        if key not in self._services:raise KeyError(name)
        return self._services[key]
    def list(self):return list(self._services.values())
    async def start_all(self):
        for s in self.list():await s.start()
        self.started=True
    async def close_all(self):
        for s in self.list():await s.close()
        self.started=False
    def ready(self):return bool(self.started and self._services and all(s.ready() for s in self.list()))
