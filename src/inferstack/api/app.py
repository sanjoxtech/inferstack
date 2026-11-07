from __future__ import annotations
import asyncio,json
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException,Request as HTTPRequest
from fastapi.responses import StreamingResponse
from pydantic import BaseModel,Field
from inferstack import __version__
from inferstack.core.request import SamplingParams

class StreamOptions(BaseModel):
    include_usage:bool=False

class CompletionBody(BaseModel):
    model:str;prompt:str;max_tokens:int=Field(default=64,ge=1,le=32768);temperature:float=Field(default=0.0,ge=0,allow_inf_nan=False);top_p:float=Field(default=1.0,gt=0,le=1);top_k:int|None=Field(default=None,ge=1);min_p:float=Field(default=0.0,ge=0,le=1);repetition_penalty:float=Field(default=1.0,gt=0,allow_inf_nan=False);seed:int|None=None;stream:bool=False;stream_options:StreamOptions|None=None
class ChatMessage(BaseModel):role:str;content:str
class ChatBody(BaseModel):
    model:str;messages:list[ChatMessage];max_tokens:int=Field(default=64,ge=1,le=32768);temperature:float=Field(default=0.0,ge=0,allow_inf_nan=False);top_p:float=Field(default=1.0,gt=0,le=1);top_k:int|None=Field(default=None,ge=1);min_p:float=Field(default=0.0,ge=0,le=1);repetition_penalty:float=Field(default=1.0,gt=0,allow_inf_nan=False);seed:int|None=None;stream:bool=False;stream_options:StreamOptions|None=None

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
        def usage():
            count=len(r.generated_token_ids)
            return {'prompt_tokens':len(r.prompt_token_ids),'completion_tokens':count,'total_tokens':len(r.prompt_token_ids)+count}
        if not b.stream:
            toks=[]
            try:
                while True:
                    x=await q.get()
                    if x is None:break
                    if isinstance(x,BaseException):raise HTTPException(500,'engine generation failed')
                    toks.append(int(x))
            finally:
                if not r.done:service.cancel(r.request_id)
            text=service.tokenizer.decode(toks,skip_special_tokens=True)
            if chat:return {'id':r.request_id,'object':'chat.completion','model':service.model_id,'usage':usage(),'choices':[{'index':0,'message':{'role':'assistant','content':text},'finish_reason':r.finish_reason or 'stop'}]}
            return {'id':r.request_id,'object':'text_completion','model':service.model_id,'usage':usage(),'choices':[{'index':0,'text':text,'finish_reason':r.finish_reason or 'stop'}]}
        async def gen():
            tokens=[];emitted=''
            try:
                while True:
                    x=await q.get()
                    if x is None:break
                    if isinstance(x,BaseException):
                        yield 'data: '+json.dumps({'error':'engine generation failed'})+'\n\n';return
                    tokens.append(int(x))
                    decoded=service.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    if decoded.endswith('\ufffd'):continue
                    if not decoded.startswith(emitted):
                        yield 'data: '+json.dumps({'error':'tokenizer streaming prefix changed'})+'\n\n';return
                    piece=decoded[len(emitted):];emitted=decoded
                    payload={'choices':[{'index':0,'delta':{'content':piece}}]} if chat else {'choices':[{'index':0,'text':piece}]}
                    yield 'data: '+json.dumps(payload,separators=(',',':'))+'\n\n'
                decoded=service.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                if decoded.startswith(emitted) and len(decoded)>len(emitted):
                    piece=decoded[len(emitted):]
                    tail={'choices':[{'index':0,'delta':{'content':piece}}]} if chat else {'choices':[{'index':0,'text':piece}]}
                    yield 'data: '+json.dumps(tail)+'\n\n'
                final={'choices':[{'index':0,'finish_reason':r.finish_reason or 'stop'}]}
                yield 'data: '+json.dumps(final)+'\n\n'
                if b.stream_options and b.stream_options.include_usage:
                    times=r.token_emission_times
                    timing={'token_count':len(times),'decode_elapsed_ms':(times[-1]-times[0])*1000 if len(times)>1 else None}
                    yield 'data: '+json.dumps({'choices':[],'usage':usage(),'inferstack_timing':timing})+'\n\n'
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
