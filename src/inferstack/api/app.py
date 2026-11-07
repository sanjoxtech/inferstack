from __future__ import annotations
import asyncio,json
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException,Request as HTTPRequest
from fastapi.responses import StreamingResponse
from pydantic import BaseModel,Field
from inferstack import __version__
from inferstack.core.request import SamplingParams

class CompletionBody(BaseModel):
    model:str;prompt:str;max_tokens:int=64;temperature:float=0.0;top_p:float=1.0;top_k:int|None=None;min_p:float=0.0;repetition_penalty:float=1.0;seed:int|None=None;stream:bool=False
class ChatMessage(BaseModel):role:str;content:str
class ChatBody(BaseModel):
    model:str;messages:list[ChatMessage];max_tokens:int=64;temperature:float=0.0;top_p:float=1.0;top_k:int|None=None;min_p:float=0.0;repetition_penalty:float=1.0;seed:int|None=None;stream:bool=False

def create_app(registry):
    @asynccontextmanager
    async def life(app):
        await registry.start_all()
        try:yield
        finally:await registry.close_all()
    app=FastAPI(title='InferStack',version=__version__,lifespan=life)
    @app.get('/health')
    def health():return {'status':'ok','version':__version__}
    @app.get('/ready')
    def ready():
        if not registry.ready():raise HTTPException(503,{'message':'model services are not ready','models':{s.model_id:s.diagnostics() for s in registry.list()}})
        return {'status':'ready','version':__version__,'models':[s.model_id for s in registry.list()]}
    @app.get('/v1/models')
    def models():return {'object':'list','data':[{'id':s.model_id,'object':'model','owned_by':'inferstack'} for s in registry.list()]}
    @app.get('/v1/native/status')
    def native_status():return {'version':__version__,'ready':registry.ready(),'models':{s.model_id:s.diagnostics() for s in registry.list()}}
    def sampling(b):return SamplingParams(max_new_tokens=b.max_tokens,temperature=b.temperature,top_p=b.top_p,top_k=b.top_k,min_p=b.min_p,repetition_penalty=b.repetition_penalty,seed=b.seed)
    async def completion(service,prompt,b,chat=False):
        r,q=await service.submit(prompt,sampling(b))
        if not b.stream:
            toks=[]
            while True:
                x=await q.get()
                if x is None:break
                if isinstance(x,BaseException):raise HTTPException(500,str(x))
                toks.append(int(x))
            text=service.tokenizer.decode(toks,skip_special_tokens=True)
            if chat:return {'id':r.request_id,'object':'chat.completion','model':service.model_id,'choices':[{'index':0,'message':{'role':'assistant','content':text},'finish_reason':'stop'}]}
            return {'id':r.request_id,'object':'text_completion','model':service.model_id,'choices':[{'index':0,'text':text,'finish_reason':'stop'}]}
        async def gen():
            try:
                while True:
                    x=await q.get()
                    if x is None:break
                    if isinstance(x,BaseException):
                        yield 'data: '+json.dumps({'error':str(x)})+'\n\n';break
                    piece=service.decode_token(int(x));payload={'choices':[{'index':0,'delta':{'content':piece}}]} if chat else {'choices':[{'index':0,'text':piece}]}
                    yield 'data: '+json.dumps(payload,separators=(',',':'))+'\n\n'
                yield 'data: [DONE]\n\n'
            except (asyncio.CancelledError,GeneratorExit):
                service.cancel(r.request_id);raise
            finally:
                if not r.done:service.cancel(r.request_id)
        return StreamingResponse(gen(),media_type='text/event-stream')
    @app.post('/v1/completions')
    async def completions(b:CompletionBody):
        if not registry.ready():raise HTTPException(503,'model services are not ready')
        try:s=registry.get(b.model)
        except KeyError:raise HTTPException(404,f'unknown model {b.model}')
        return await completion(s,b.prompt,b,False)
    @app.post('/v1/chat/completions')
    async def chat(b:ChatBody):
        if not registry.ready():raise HTTPException(503,'model services are not ready')
        try:s=registry.get(b.model)
        except KeyError:raise HTTPException(404,f'unknown model {b.model}')
        if hasattr(s.tokenizer,'apply_chat_template'):
            prompt=s.tokenizer.apply_chat_template([m.model_dump() for m in b.messages],tokenize=False,add_generation_prompt=True)
        else:prompt='\n'.join(f'{m.role}: {m.content}' for m in b.messages)+'\nassistant:'
        return await completion(s,prompt,b,True)
    return app
