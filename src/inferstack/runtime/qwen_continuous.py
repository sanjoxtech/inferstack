from __future__ import annotations
import asyncio, functools, inspect, math, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from typing import Any

from inferstack.core.request import Request, RequestState
from inferstack.runtime.hybrid_prefix_cache import HybridPrefixStateCache, fork_cache_cow, merge_cache_rows, repeat_cache_rows, select_cache_rows
from inferstack.runtime.sampler import FastSampler


def _deliver(items):
    for q,x in items:q.put_nowait(x)

class NativeQwenContinuousError(RuntimeError): pass

@dataclass
class ActiveBatch:
    requests:list[Request];queues:list[asyncio.Queue];past:Any;attention_mask:Any;previous_gpu:Any;decode_positions:Any;generated:list[int];stop_sets:list[set[int]];base_width:int;decode_step:int=0;prompt_hidden:Any=None
    @property
    def done(self):return not self.requests
    @property
    def physical_width(self):return int(self.base_width+self.decode_step)

@dataclass
class PrefillJob:
    requests:list[Request];queues:list[asyncio.Queue];input_ids:Any;attention_mask:Any;position_ids:Any;lengths:list[int];base_width:int;max_new:int;prefix_len:int=0;cursor:int=0;past:Any=None;terminal_logits:Any=None;hidden_tail:Any=None
    @property
    def input_width(self):return int(self.input_ids.shape[1])
    @property
    def done(self):return self.cursor>=self.input_width

class QwenHybridContinuousEngine:
    def __init__(self,model,tokenizer,cfg,device='cuda:0'):
        self.model=model;self.tokenizer=tokenizer;self.cfg=cfg;self.device=device;self.started=False;self._stop=False;self._task=None;self._fatal_error=None
        self._pending=asyncio.Queue(maxsize=max(1,int(cfg.native.max_pending_requests)));self._active=[];self._prefill_jobs=[];self._requests={};self._queues={}
        self._loop_ref=None;self._gpu_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='inferstack-gpu')
        self._sampler=FastSampler(device, bool(cfg.native.fused_sampler));self._prefix_cache=HybridPrefixStateCache(cfg.native.prefix_state_cache_max_entries if cfg.native.prefix_state_cache else 0,cfg.native.prefix_cache_min_tokens,cfg.native.prefix_cache_block_tokens,cfg.native.prefix_state_cache_max_mb*1024*1024)
        self._supports_logits_to_keep=self._detect_logits_to_keep(model);self._compiled=False;self._compiled_model=None;self._compile_error=None;self._graph_shapes=set();self._graph_first=0;self._graph_reuse=0;self._graph_bucket_rows=0;self._graph_static_cache_uses=0;self._graph_mark_steps=0
        self._stats={k:0 for k in ('cohorts','requests_batched','max_observed_batch','max_active_batches','mid_generation_admissions','live_rebatches','rows_refilled','compactions','rows_compacted','prefill_forward_calls','prefill_chunks','decode_forward_calls','decode_priority_turns','tokens_emitted','cancellations','cancelled_before_gpu','threadsafe_queue_emits','prefix_block_snapshots','prefill_tokens')}
        self._ttft={k:0.0 for k in ('requests','queue_ms','wait_ms','prefill_ms','total_ms','decode_steps','other_prefill_forwards')}
        self._emit_batched=bool(getattr(cfg.native,'emit_batched',False));self._lane=None;self._static_diag={'enabled':bool(getattr(cfg.native,'static_decode',False)),'active':False};self._gdn_diag={};self._forward_fusion_diag={};self._profile={'enabled':bool(getattr(cfg.native,'profile_decode',False)),'decode_calls':0,'decode_wall_ms':0.0,'model_gpu_ms':0.0,'sampler_gpu_ms':0.0,'accept_wall_ms':0.0,'compact_wall_ms':0.0,'syncs':0}
    @staticmethod
    def _detect_logits_to_keep(model):
        try:
            return any(p.name=='logits_to_keep' or p.kind is inspect.Parameter.VAR_KEYWORD for p in inspect.signature(model.forward).parameters.values())
        except Exception:return False
    @property
    def healthy(self):return bool(self.started and self._fatal_error is None and (self._task is None or not self._task.done()))
    @property
    def fatal_error(self):return self._fatal_error
    def _emit(self,q,item):
        self._stats['threadsafe_queue_emits']+=1
        loop=self._loop_ref
        if loop is None or loop.is_closed():return
        loop.call_soon_threadsafe(q.put_nowait,item)
    def _emit_many(self,items):
        """P6: deliver a step's (queue, item) pairs with one event-loop wakeup instead of one per
        token. Each call_soon_threadsafe wakes the loop thread, which then contends for the GIL with
        the GPU thread; at 16 rows that cost ~9 ms of host time per spec step (L4, prefill_heavy c16).
        Per-queue order is kept."""
        if not items:return
        if not self._emit_batched:
            for q,x in items:self._emit(q,x)
            return
        self._stats['threadsafe_queue_emits']+=len(items);self._stats['threadsafe_batches']=self._stats.get('threadsafe_batches',0)+1
        loop=self._loop_ref
        if loop is None or loop.is_closed():return
        loop.call_soon_threadsafe(_deliver,items)
    async def _gpu(self,fn,*args):
        loop=asyncio.get_running_loop()
        if self._lane is None:return await loop.run_in_executor(self._gpu_executor,functools.partial(fn,*args))
        return await loop.run_in_executor(self._gpu_executor,functools.partial(self._timed_gpu,fn,*args))
    def _timed_gpu(self,fn,*args):
        # P6 diagnosis: time the GPU thread spends inside work items (launch + host code + syncs).
        # Cell wall minus this is time the GPU thread sat idle (event-loop hops, HTTP, scheduling).
        t=time.perf_counter()
        try:return fn(*args)
        finally:
            st=self._lane.stats;st['gpu_thread_ms']=st.get('gpu_thread_ms',0.0)+(time.perf_counter()-t)*1000;st['gpu_thread_calls']=st.get('gpu_thread_calls',0)+1
    def _model_call(self,**kw):
        if self._supports_logits_to_keep:kw['logits_to_keep']=1
        # Compile only the single-token decode graph. Prefill remains eager so the
        # first long prompt does not poison/overspecialize the decode cudagraph.
        seq=int(kw.get('input_ids').shape[1]) if kw.get('input_ids') is not None else 0
        model=self._compiled_model if (self._compiled_model is not None and (not self.cfg.native.explicit_cuda_graph or seq==1)) else self.model
        if self.cfg.native.explicit_cuda_graph and seq==1:
            try:
                import torch
                torch.compiler.cudagraph_mark_step_begin();self._graph_mark_steps+=1
            except Exception:
                pass
        return model(**kw)
    def _maybe_patch(self):
        if self.cfg.native.custom_gdn_triton:
            try:
                from inferstack.kernels.gdn_triton import install_qwen35_gdn_kernel
                from inferstack.kernels.qwen_conv_triton import install_qwen35_conv_kernel
                self._gdn_diag=install_qwen35_gdn_kernel(True,True);self._gdn_diag['conv']=install_qwen35_conv_kernel(True)
            except Exception as exc:
                self._gdn_diag={'patched':False,'patch_error':f'{type(exc).__name__}: {exc}'}
                if self.cfg.native.strict_custom_kernels: raise
        if bool(getattr(self.cfg.native,'forward_fusion',False)):
            try:
                from inferstack.kernels.qwen_forward_fusion import install_qwen35_forward_fusion
                self._forward_fusion_diag=install_qwen35_forward_fusion(self.model,self.cfg)
            except Exception as exc:
                self._forward_fusion_diag={'installed':False,'install_error':f'{type(exc).__name__}: {exc}'}
                if bool(getattr(self.cfg.native,'forward_fusion_strict',False)) or type(exc).__name__=='FusionBackendUnavailable': raise
    def _maybe_compile(self):
        if not (self.cfg.kernels.torch_compile and self.cfg.native.compile_model):return
        try:
            import torch
            if self.cfg.native.explicit_cuda_graph:
                # CUDAGraph Trees may otherwise overwrite outputs that remain live into
                # the next decode iteration (notably cache state). Ask Inductor to clone
                # user-visible persistent outputs and explicitly mark iteration boundaries.
                if bool(getattr(self.cfg.native,'graph_generation_cloning',True)):
                    try:torch._inductor.config.triton.cudagraph_trees_generation_cloning='user_visible'
                    except Exception:pass
                self._compiled_model=torch.compile(self.model,backend='cudagraphs',dynamic=False,fullgraph=False)
            else:
                self._compiled_model=torch.compile(self.model,mode=self.cfg.kernels.compile_mode,dynamic=self.cfg.native.compile_dynamic)
            self._compiled=True
        except Exception as exc:
            self._compile_error=f'{type(exc).__name__}: {exc}'
            if self.cfg.native.strict_compile:raise
    def _maybe_static_lane(self):
        """Static decode lane (runtime/qwen_static_lane.py): decode by CUDA-graph replay over a
        fixed slot pool. Any setup failure falls back to the Transformers decode path."""
        n=self.cfg.native
        if not bool(getattr(n,'static_decode',False)) or self._lane is not None:return
        try:
            import torch
            from inferstack.runtime.qwen_static_decode import QwenStaticDecoder
            from inferstack.runtime.qwen_static_lane import StaticDecodeLane
            slots=int(getattr(n,'static_decode_slots',0) or n.max_requests_per_batch);max_len=int(n.static_decode_max_tokens)
            gb=str(n.static_decode_gdn_backend);ab=str(getattr(n,'static_decode_attention','sdpa') or 'sdpa');sp=None
            if bool(getattr(n,'static_decode_spec',False)):
                from inferstack.runtime.qwen_spec_decode import QwenSpecDecoder,resolve_mtp
                tensors,seed=resolve_mtp(str(getattr(n,'static_decode_spec_mtp','') or ''))
                dec=QwenSpecDecoder(self.model,self.device,gdn_backend=gb,attn_backend=ab,mtp_tensors=tensors,mtp_seed=seed,draft_vocab=int(getattr(n,'static_decode_spec_draft_vocab',0) or 0) or None)
            else:dec=QwenStaticDecoder(self.model,self.device,gdn_backend=gb,attn_backend=ab)
            ov=int(getattr(n,'static_decode_overlap_slots',0) or 0) if bool(getattr(n,'static_decode_overlap_prefill',False)) else 0
            # P7: ov spare slots above the decode buckets for overlapped prefills (decode never reaches them)
            st=dec.new_state(slots+ov,max_len,[b for b in n.static_decode_batch_buckets if int(b)<=slots]+[slots],list(n.static_decode_len_buckets))
            if bool(getattr(n,'static_decode_spec',False)):sp=dec.new_spec_state(st)
            t=time.perf_counter();graphs=(dec.capture_spec(st,sp) if sp is not None else dec.capture(st)) if bool(n.static_decode_cuda_graph) else 0
            if bool(getattr(n,'static_decode_packed_prefill',False)) and str(self.device).startswith('cuda'):
                # Compile/autotune the varlen FLA kernels now; warm-up writes slots, so zero after.
                t0=time.perf_counter();k=min(2,st.batch);ls=[min(max_len-1,700),min(max_len-1,37)][:k]
                with torch.inference_mode():dec.prefill_packed(st,range(k),[[1]*x for x in ls])
                dec.reset(st);torch.cuda.synchronize();self._static_diag['packed_warmup_s']=round(time.perf_counter()-t0,3)
                if bool(getattr(n,'static_decode_packed_graphs',False)):
                    # P4: capture small packed prefills (k rows x L tokens) as CUDA graphs before serving.
                    t0=time.perf_counter();pg=0
                    ll=tuple(int(x) for x in (getattr(n,'static_decode_prefill_graph_long_lens',()) or ()));lr=int(getattr(n,'static_decode_prefill_graph_long_rows',0) or 0)
                    if ll and lr:dec.PREFILL_GRAPH_LONG=(ll,lr)   # P8: longer graph buckets for a few rows
                    for kk in range(1,min(dec.PREFILL_GRAPH_MAX_ROWS,st.batch)+1):
                        for L in dec.prefill_graph_lens(kk):
                            if L<max_len:dec.capture_prefill_graph(st,kk,L);pg+=1
                    dec.reset(st);torch.cuda.synchronize();self._static_diag.update(prefill_graphs=pg,prefill_graph_capture_s=round(time.perf_counter()-t0,3))
            self._lane=StaticDecodeLane(self,dec,st,spec=sp);self._static_diag.update(active=True,spec=sp is not None,gdn_backend=dec.gdn_backend,graphs_captured=graphs,capture_s=round(time.perf_counter()-t,3),pool_bytes=sum(x.numel()*x.element_size() for x in [st.mask,st.logits]+[y for kv in st.kv.values() for y in kv]+list(st.conv.values())+list(st.rec.values())))
        except Exception as exc:
            self._lane=None;self._static_diag.update(active=False,error=f'{type(exc).__name__}: {exc}')
            if bool(getattr(n,'static_decode_strict',False)):raise
    def _maybe_prefill_warmup(self):
        """Compile and autotune the prefill kernels (Triton/FLA) before serving. Without it the
        first large prefill after startup paid a one-off ~15 s stall on the L4 (n43, moderate c16).
        Shapes cover B=1 and B>1 with left padding, a length not divisible by 16, and a chunk that
        continues from a cache (conv/recurrent initial state)."""
        n=self.cfg.native
        if not bool(getattr(n,'prefill_warmup',False)) or not str(self.device).startswith('cuda'):return
        import torch
        t=time.perf_counter();cfgs=[];vocab=int(getattr(getattr(self.model,'config',None),'vocab_size',0) or 1000)
        try:
            g=torch.Generator(device='cpu').manual_seed(0)
            for B,T,pad in ((1,2048,0),(1,1999,0),(4,520,37)):
                ids=torch.randint(0,min(vocab,30000),(B,T),generator=g).to(self.device);mask=torch.ones((B,T),dtype=torch.long,device=self.device)
                if pad:mask[1:,:pad]=0
                pos=(mask.cumsum(-1)-1).clamp_min(0)
                with torch.inference_mode():
                    out=self._model_call(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=True)
                    more=torch.randint(0,min(vocab,30000),(B,301),generator=g).to(self.device);m2=torch.cat([mask,torch.ones((B,301),dtype=torch.long,device=self.device)],1)
                    p2=pos[:,-1:]+1+torch.arange(301,device=self.device)
                    self._model_call(input_ids=more,attention_mask=m2,position_ids=p2,past_key_values=out.past_key_values,use_cache=True)
                cfgs.append([B,T,pad])
            torch.cuda.synchronize();self._static_diag['prefill_warmup']={'shapes':cfgs,'s':round(time.perf_counter()-t,3)}
        except Exception as exc:
            self._static_diag['prefill_warmup']={'shapes':cfgs,'error':f'{type(exc).__name__}: {exc}'[:300]}
    def _decode_work(self):return bool(self._active or (self._lane is not None and (self._lane.rows or self._lane.overlap_busy())))
    async def start(self):
        if self.started:return
        self._loop_ref=asyncio.get_running_loop();self._maybe_patch();self._maybe_compile();await self._gpu(self._maybe_prefill_warmup);await self._gpu(self._maybe_static_lane);self._stop=False;self._fatal_error=None;self.started=True;self._task=asyncio.create_task(self._loop_guarded(),name='inferstack-qwen-native')
    async def close(self):
        self._stop=True
        if self._task:
            try:await self._task
            except Exception:pass
            self._task=None
        self.started=False;self._fail_all(NativeQwenContinuousError('native engine closed'))
        self._active.clear();self._prefill_jobs.clear()
        if self._lane is not None:self._lane.clear()
        self._gpu_executor.shutdown(wait=False,cancel_futures=True)
    async def _loop_guarded(self):
        try:await self._loop()
        except asyncio.CancelledError:raise
        except Exception as exc:
            import sys,traceback;traceback.print_exc(file=sys.stderr)   # the only place a scheduler crash is visible in server.log
            self._fatal_error=f'{type(exc).__name__}: {exc}';self._fail_all(NativeQwenContinuousError(f'native scheduler failed: {self._fatal_error}'));self._stop=True;raise
    def _fail_all(self,exc):
        for rid,q in list(self._queues.items()):
            r=self._requests.get(rid)
            if r and not r.done:r.state=RequestState.FAILED;r.error=str(exc)
            self._emit(q,exc);self._emit(q,None)
        self._queues.clear();self._requests.clear()
    def _context_limit(self):
        cfg=getattr(self.model,'config',None);text=getattr(cfg,'text_config',cfg)
        for n in ('max_position_embeddings','max_sequence_length','seq_length'):
            v=getattr(text,n,None)
            if isinstance(v,int) and v>0:return v
        return None
    async def submit(self,r):
        if not self.healthy:raise RuntimeError(f'native engine unavailable: {self._fatal_error or "not started"}')
        if not r.prompt_token_ids:raise ValueError('empty prompt')
        limit=self._context_limit()
        if limit and len(r.prompt_token_ids)+r.sampling.max_new_tokens>limit:raise ValueError(f'request exceeds context {limit}')
        q=asyncio.Queue();self._requests[r.request_id]=r;self._queues[r.request_id]=q;r._t_submit=time.perf_counter()
        try:self._pending.put_nowait((r,q))
        except asyncio.QueueFull:
            self._requests.pop(r.request_id,None);self._queues.pop(r.request_id,None);raise RuntimeError('native admission queue full')
        return q
    def cancel(self,rid):
        r=self._requests.get(rid)
        if not r or r.done:return False
        r.state=RequestState.CANCELLED;self._stats['cancellations']+=1;return True
    def _pad_id(self):
        x=getattr(self.tokenizer,'pad_token_id',None);x=getattr(self.tokenizer,'eos_token_id',0) if x is None else x;return int(x or 0)
    def _stop_ids(self,r):
        ids=set(map(int,r.sampling.stop_token_ids));e=getattr(self.tokenizer,'eos_token_id',None)
        if isinstance(e,(list,tuple)):ids.update(map(int,e))
        elif e is not None:ids.add(int(e))
        return ids
    def _common_prefix(self,rs):
        if not rs:return ()
        a=rs[0].prompt_token_ids;n=len(a)
        for r in rs[1:]:
            n=min(n,len(r.prompt_token_ids));i=0
            while i<n and a[i]==r.prompt_token_ids[i]:i+=1
            n=i
        return tuple(a[:n])
    def _fits(self,items,r):
        lens=[len(x[0].prompt_token_ids) for x in items]+[len(r.prompt_token_ids)];return len(lens)*max(lens)<=int(self.cfg.native.max_batch_tokens)
    async def _gather(self,wait):
        while True:
            try:first=await self._pending.get() if wait else self._pending.get_nowait()
            except asyncio.QueueEmpty:return []
            if first[0].state!=RequestState.CANCELLED:break
            self._stats['cancelled_before_gpu']+=1;self._finish_cancelled(*first)
        out=[first]
        if wait and self.cfg.native.batch_wait_ms>0:await asyncio.sleep(self.cfg.native.batch_wait_ms/1000)
        defer=[]
        while len(out)<self.cfg.native.max_requests_per_batch:
            try:item=self._pending.get_nowait()
            except asyncio.QueueEmpty:break
            if item[0].state==RequestState.CANCELLED:self._stats['cancelled_before_gpu']+=1;self._finish_cancelled(*item);continue
            if self._fits(out,item[0]):out.append(item)
            else:defer.append(item);break
        for x in defer:await self._pending.put(x)
        return out
    def _finish_cancelled(self,r,q):
        r.state=RequestState.CANCELLED;self._requests.pop(r.request_id,None);self._queues.pop(r.request_id,None);self._emit(q,None)
    def _make_static_cache(self,total_tokens:int):
        if not (self.cfg.native.explicit_cuda_graph and getattr(self.cfg.native,'graph_static_cache',False)):
            return None
        try:
            from transformers import StaticCache
            text=getattr(self.model.config,'get_text_config',lambda **_:getattr(self.model.config,'text_config',self.model.config))(decoder=True)
            interval=max(1,int(getattr(self.cfg.native,'kv_padding_interval_size',128)))
            max_len=max(int(getattr(self.cfg.native,'graph_cache_max_tokens',256)),int(math.ceil(max(1,total_tokens)/interval)*interval))
            self._graph_static_cache_uses+=1
            return StaticCache(config=text,max_cache_len=max_len)
        except Exception as exc:
            if self.cfg.native.strict_compile:raise
            self._compile_error=(self._compile_error+'; ' if self._compile_error else '')+f'StaticCache: {type(exc).__name__}: {exc}'
            return None
    def _cuda_ms(self,fn):
        if not self._profile['enabled'] or not str(self.device).startswith('cuda'):
            return fn(),0.0
        try:
            import torch
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();out=fn();b.record();b.synchronize();self._profile['syncs']+=1;return out,float(a.elapsed_time(b))
        except Exception:
            t=time.perf_counter();out=fn();return out,(time.perf_counter()-t)*1000.0
    def _prepare_prefill(self,cohort):
        import torch
        rs=[x[0] for x in cohort if x[0].state!=RequestState.CANCELLED];qs=[x[1] for x in cohort if x[0].state!=RequestState.CANCELLED]
        if not rs:return None
        now=time.perf_counter()
        for r in rs:r.state=RequestState.RUNNING;r._t_prep=now;r._ds0=self._stats['decode_forward_calls'];r._pf0=self._stats['prefill_forward_calls']
        B=len(rs);lens=[len(r.prompt_token_ids) for r in rs];max_new=max(r.sampling.max_new_tokens for r in rs);pad=self._pad_id();common=self._common_prefix(rs);entry=self._prefix_cache.longest(common) if self._prefix_cache.enabled else None
        if entry:
            past=self._prefix_cache.fork(entry);prefix=len(entry.tokens);past=repeat_cache_rows(past,B,self.device) if B>1 else past;suffix=[r.prompt_token_ids[prefix:] for r in rs];w=max(map(len,suffix),default=0);base=prefix+w;mask=torch.zeros((B,base+max_new),dtype=torch.long,device=self.device);mask[:,:prefix]=1
            inp=torch.full((B,w),pad,dtype=torch.long,device=self.device) if w else torch.empty((B,0),dtype=torch.long,device=self.device);pos=torch.zeros((B,w),dtype=torch.long,device=self.device)
            for i,toks in enumerate(suffix):
                if toks:
                    t=torch.tensor(toks,dtype=torch.long,device=self.device);inp[i,-len(t):]=t;mask[i,prefix+w-len(t):prefix+w]=1;pos[i,-len(t):]=torch.arange(prefix,prefix+len(t),device=self.device)
            logits=entry.last_logits.expand(B,-1) if entry.last_logits is not None else None
            return PrefillJob(rs,qs,inp,mask,pos,lens,base,max_new,prefix,0,past,logits)
        w=max(lens);inp=torch.full((B,w),pad,dtype=torch.long,device=self.device)
        past=self._make_static_cache(w+max_new)
        # CUDA-graph lane: attention-mask *shape* must remain static across decode.
        # Match the StaticCache capacity up-front; future positions stay 0 until each
        # decode step marks the current token visible. Normal dynamic-cache serving keeps
        # the smaller exact-capacity mask.
        mask_width=w+max_new
        if past is not None and hasattr(past,'get_max_length'):
            try:mask_width=max(mask_width,int(past.get_max_length()))
            except Exception:pass
        mask=torch.zeros((B,mask_width),dtype=torch.long,device=self.device)
        for i,r in enumerate(rs):
            t=torch.tensor(r.prompt_token_ids,dtype=torch.long,device=self.device);inp[i,-len(t):]=t;mask[i,w-len(t):w]=1
        pm=mask[:,:w];pos=pm.cumsum(-1)-1;pos.masked_fill_(pm==0,0)
        return PrefillJob(rs,qs,inp,mask,pos,lens,w,max_new,0,0,past,None)
    def _packed_admit(self,cohort):
        """P1: send what fits through the static lane's packed prefill; return the rest."""
        live=[]
        for r,q in cohort:
            if r.state==RequestState.CANCELLED:self._stats['cancelled_before_gpu']+=1;self._finish_cancelled(r,q)
            else:live.append((r,q))
        return self._lane.admit_packed(live) if live else []
    def _maybe_store_prefix(self,job):
        if len(job.requests)!=1 or not self._prefix_cache.enabled or job.past is None or job.terminal_logits is None:return
        logical=job.prefix_len+job.cursor;block=self._prefix_cache.block_tokens
        if logical>=self._prefix_cache.min_tokens and logical%block==0 and logical<=len(job.requests[0].prompt_token_ids):
            toks=tuple(job.requests[0].prompt_token_ids[:logical])
            if self._prefix_cache.put(toks,job.past,logical,job.terminal_logits.detach()):self._stats['prefix_block_snapshots']+=1
    def _prefill_budget(self):
        """Token budget for the next prefill forward (0 = fixed-chunk policy)."""
        n=self.cfg.native;idle=int(getattr(n,'prefill_token_budget_idle',0) or 0);busy=int(getattr(n,'prefill_token_budget_busy',0) or 0)
        if idle<=0 and busy<=0:return 0
        if not self._decode_work():return idle or busy
        b=busy or idle;steps=int(getattr(n,'prefill_backlog_drain_steps',0) or 0)
        if steps>0:b=max(b,min(idle or b,-(-self._prefill_backlog()//steps)))
        return b
    def _prefill_backlog(self):
        """Prompt tokens (rows x remaining columns) still to prefill in queued jobs."""
        return sum(len(j.requests)*max(0,j.input_width-j.cursor) for j in self._prefill_jobs)
    def _prefill_end(self,job,budget=0):
        """End column of the next prefill forward.

        Fixed policy: one prefill_chunk_tokens chunk. Budget policy: as many columns as
        budget/rows allows (a multiple of the chunk). For a single-row job with the prefix
        cache on, forwards end on prefix-cache block boundaries and the last full block
        before the prompt end is always a boundary, so the reusable snapshot is kept."""
        chunk=max(1,int(self.cfg.native.prefill_chunk_tokens));s=job.cursor;W=job.input_width
        if budget<=0:return min(W,s+chunk)
        rows=max(1,len(job.requests));span=max(chunk,(budget//rows)//chunk*chunk);e=min(W,s+span)
        if rows==1 and self._prefix_cache.enabled:
            blk=self._prefix_cache.block_tokens;p=job.prefix_len
            last=(p+W)//blk*blk-p
            if s<last<e:e=last
            elif e<W:
                al=(p+e)//blk*blk-p
                if al>s:e=al
        return e
    def _advance_prefill(self,job,chunks=1,budget=0):
        import torch
        for _ in range(max(1,int(chunks))):
            if job.done:break
            s=job.cursor;e=self._prefill_end(job,budget)
            self._stats['prefill_tokens']+=int(e-s)*len(job.requests)
            if s==0:
                now=time.perf_counter()
                for r in job.requests:
                    if getattr(r,'_t_pf',None) is None:r._t_pf=now;r._pf1=self._stats['prefill_forward_calls']
            hook=self._spec_hidden_hook(job)
            try:
                with torch.inference_mode():out=self._model_call(input_ids=job.input_ids[:,s:e],attention_mask=job.attention_mask[:,:job.prefix_len+e],position_ids=job.position_ids[:,s:e],past_key_values=job.past,use_cache=True)
            finally:
                if hook is not None:hook.remove()
            self._stats['prefill_forward_calls']+=1;self._stats['prefill_chunks']+=1;job.past=out.past_key_values;job.terminal_logits=out.logits[:,-1,:];job.cursor=e;self._maybe_store_prefix(job)
        if not job.done:return None
        if job.terminal_logits is None:raise NativeQwenContinuousError('prefill ended without logits')
        for r in job.requests:r.prompt_cursor=len(r.prompt_token_ids)
        nxt=self._sampler.sample(job.terminal_logits,job.requests,0)
        state=ActiveBatch(job.requests,job.queues,job.past,job.attention_mask,nxt,torch.tensor(job.lengths,dtype=torch.long,device=self.device),[0]*len(job.requests),[self._stop_ids(r) for r in job.requests],job.base_width,0,job.hidden_tail)
        self._accept(state,nxt);self._compact(state)
        if state.done:return None
        if self._lane is not None and self._lane.try_admit(state):return None
        return state
    def _ttft_record(self,r):
        """First token of r: split TTFT into queue (submit -> cohort prepared), wait (prepared ->
        its first prefill forward), prefill (first forward -> first token). Also counts decode
        steps and other jobs' prefill forwards that ran between prepare and the first token."""
        ts,tp,tf=getattr(r,'_t_submit',None),getattr(r,'_t_prep',None),getattr(r,'_t_pf',None)
        if ts is None or tp is None or tf is None:return
        te=r.token_emission_times[-1];a=self._ttft
        a['requests']+=1;a['queue_ms']+=(tp-ts)*1e3;a['wait_ms']+=(tf-tp)*1e3;a['prefill_ms']+=(te-tf)*1e3;a['total_ms']+=(te-ts)*1e3
        a['decode_steps']+=self._stats['decode_forward_calls']-r._ds0;a['other_prefill_forwards']+=r._pf1-r._pf0
    def _spec_hidden_hook(self,job):
        """N6.2: keep the post-final-norm hidden states of the last prefill columns (the spec
        lane's MTP history). Only when the static lane runs speculative decoding."""
        lane=self._lane
        if lane is None or getattr(lane,'sp',None) is None:return None
        import torch
        w=int(getattr(self.cfg.native,'static_decode_spec_window',256) or 256)
        def keep(m,i,o):
            h=o[:,-w:].detach().clone()
            job.hidden_tail=h if job.hidden_tail is None else torch.cat([job.hidden_tail,h],1)[:,-w:]
        return lane.dec.text.norm.register_forward_hook(keep)
    def _accept(self,state,tokens_gpu):
        self._accept_host(state,tokens_gpu.detach().cpu().tolist())
    def _accept_host(self,state,toks,buf=None):
        """Emit one token per row from a host list; None skips the row (spec lane second tokens).
        buf (P6): collect (queue, item) pairs for the caller to deliver with _emit_many; without
        it the pairs are delivered at the end of this call in one wakeup."""
        out=[] if buf is None else buf;now=time.perf_counter()
        for i,t in enumerate(toks):
            r=state.requests[i]
            if t is None or r.state!=RequestState.RUNNING:continue
            t=int(t);r.token_emission_times.append(now);r.generated_token_ids.append(t);state.generated[i]+=1;out.append((state.queues[i],t));self._stats['tokens_emitted']+=1
            if state.generated[i]==1:self._ttft_record(r)
            if t in state.stop_sets[i] or state.generated[i]>=r.sampling.max_new_tokens:
                r.finish_reason='stop' if t in state.stop_sets[i] else 'length';r.state=RequestState.FINISHED;out.append((state.queues[i],None));self._requests.pop(r.request_id,None);self._queues.pop(r.request_id,None)
        if buf is None:self._emit_many(out)
    def _compact(self,state):
        import torch
        keep=[i for i,r in enumerate(state.requests) if r.state==RequestState.RUNNING]
        if len(keep)==len(state.requests):return
        static_graph=bool(self.cfg.native.explicit_cuda_graph and getattr(self.cfg.native,'graph_static_cache',False))
        # Static-cache graph lane must keep a stable batch dimension/address. Rows that
        # finish early become inert dummies until the whole batch is complete.
        if static_graph and keep:
            for i,r in enumerate(state.requests):
                if i not in keep and r.state==RequestState.CANCELLED and r.request_id in self._queues:
                    self._emit(state.queues[i],None);self._requests.pop(r.request_id,None);self._queues.pop(r.request_id,None)
            return
        self._stats['compactions']+=1;self._stats['rows_compacted']+=len(state.requests)-len(keep)
        for i,r in enumerate(state.requests):
            if i not in keep and r.state==RequestState.CANCELLED:self._emit(state.queues[i],None);self._requests.pop(r.request_id,None);self._queues.pop(r.request_id,None)
        if not keep:state.requests.clear();state.queues.clear();return
        idx=torch.tensor(keep,dtype=torch.long,device=self.device);state.past=select_cache_rows(state.past,idx);state.attention_mask=state.attention_mask.index_select(0,idx);state.previous_gpu=state.previous_gpu.index_select(0,idx);state.decode_positions=state.decode_positions.index_select(0,idx);state.requests=[state.requests[i] for i in keep];state.queues=[state.queues[i] for i in keep];state.generated=[state.generated[i] for i in keep];state.stop_sets=[state.stop_sets[i] for i in keep]
        if state.prompt_hidden is not None:state.prompt_hidden=state.prompt_hidden.index_select(0,idx)
    def _remaining(self,state):return max((r.sampling.max_new_tokens-g for r,g in zip(state.requests,state.generated) if r.state==RequestState.RUNNING),default=0)
    def _merge_states(self,states):
        import torch
        if len(states)==1:return states[0]
        target=max(s.physical_width for s in states);rs=[r for s in states for r in s.requests];qs=[q for s in states for q in s.queues];gen=[g for s in states for g in s.generated];stops=[x for s in states for x in s.stop_sets];remain=max(map(self._remaining,states),default=1);masks=[]
        for s in states:
            m=torch.zeros((len(s.requests),target+remain),dtype=s.attention_mask.dtype,device=s.attention_mask.device);m[:,:s.physical_width]=s.attention_mask[:,:s.physical_width];masks.append(m)
        past=merge_cache_rows([s.past for s in states],target);merged=ActiveBatch(rs,qs,past,torch.cat(masks),torch.cat([s.previous_gpu for s in states]),torch.cat([s.decode_positions for s in states]),gen,stops,target,0);self._stats['live_rebatches']+=1;self._stats['rows_refilled']+=sum(len(s.requests) for s in states[1:]);self._stats['max_observed_batch']=max(self._stats['max_observed_batch'],len(rs));return merged
    def _rebatch(self):
        if not self.cfg.native.live_batch_refill or len(self._active)<2:return
        maxb=int(self.cfg.native.max_requests_per_batch);out=[];bucket=[];rows=0
        for s in self._active:
            n=len(s.requests)
            if bucket and rows+n>maxb:out.append(self._merge_states(bucket));bucket=[];rows=0
            bucket.append(s);rows+=n
        if bucket:out.append(self._merge_states(bucket))
        self._active=out
    def _graph_bucket(self,b):
        for x in sorted(set(map(int,self.cfg.native.cuda_graph_batch_sizes))):
            if b<=x:return x
        return b
    def _record_graph(self,b,w):
        if not self.cfg.native.explicit_cuda_graph:return
        bb=self._graph_bucket(b);interval=max(1,int(self.cfg.native.kv_padding_interval_size));ww=math.ceil(max(1,w)/interval)*interval;key=(bb,ww)
        if key in self._graph_shapes:self._graph_reuse+=1
        else:self._graph_shapes.add(key);self._graph_first+=1
        self._graph_bucket_rows+=max(0,bb-b)
    def _decode(self,state):
        import torch
        prof=bool(self._profile['enabled']);wall0=time.perf_counter() if prof else 0.0
        if prof:
            c0=time.perf_counter();self._compact(state);self._profile['compact_wall_ms']+=(time.perf_counter()-c0)*1000.0
        else:self._compact(state)
        if state.done:return
        B=len(state.requests);col=state.physical_width;self._record_graph(B,col)
        if col>=state.attention_mask.shape[1]:
            grow=max(16,self._remaining(state),1)
            state.attention_mask=torch.cat([state.attention_mask,torch.zeros((B,grow),dtype=state.attention_mask.dtype,device=self.device)],dim=1)
        state.attention_mask[:,col]=1;pos=state.decode_positions.unsqueeze(1);call_ids=state.previous_gpu.unsqueeze(1);call_pos=pos;call_past=state.past;bucket=B
        static_graph=bool(self.cfg.native.explicit_cuda_graph and getattr(self.cfg.native,'graph_static_cache',False))
        call_mask=state.attention_mask if static_graph else state.attention_mask[:,:col+1]
        if self.cfg.native.explicit_cuda_graph and not static_graph:
            bucket=self._graph_bucket(B)
            if bucket>B:
                idx=torch.cat([torch.arange(B,device=self.device),torch.zeros(bucket-B,dtype=torch.long,device=self.device)]);call_past=select_cache_rows(fork_cache_cow(state.past),idx);call_ids=call_ids.index_select(0,idx);call_mask=call_mask.index_select(0,idx);call_pos=call_pos.index_select(0,idx)
        def model_forward():
            with torch.inference_mode():
                return self._model_call(input_ids=call_ids,attention_mask=call_mask,position_ids=call_pos,past_key_values=call_past,use_cache=True)
        out,model_ms=self._cuda_ms(model_forward);self._profile['model_gpu_ms']+=model_ms
        self._stats['decode_forward_calls']+=1;logits=out.logits[:B,-1,:]
        state.past=select_cache_rows(out.past_key_values,torch.arange(B,device=self.device)) if (bucket>B and not static_graph) else out.past_key_values
        nxt,sampler_ms=self._cuda_ms(lambda:self._sampler.sample(logits,state.requests,state.decode_step+1));self._profile['sampler_gpu_ms']+=sampler_ms
        state.decode_positions+=1;state.decode_step+=1;state.previous_gpu=nxt
        if prof:
            a0=time.perf_counter();self._accept(state,nxt);self._profile['accept_wall_ms']+=(time.perf_counter()-a0)*1000.0
            c0=time.perf_counter();self._compact(state);self._profile['compact_wall_ms']+=(time.perf_counter()-c0)*1000.0
            self._profile['decode_calls']+=1;self._profile['decode_wall_ms']+=(time.perf_counter()-wall0)*1000.0
        else:
            self._accept(state,nxt);self._compact(state)
    async def _decode_turn(self,quantum):
        if not self._decode_work():return False
        if self._lane is not None and (self._lane.rows or self._lane.overlap_busy()):
            for _ in range(quantum):
                if not (self._lane.rows or self._lane.overlap_busy()):break
                await self._gpu(self._lane.decode)
        if not self._active:
            self._stats['decode_priority_turns']+=1;return True
        await self._gpu(self._rebatch)
        for s in list(self._active):
            for _ in range(quantum):
                if s.done:break
                await self._gpu(self._decode,s)
            if s.done and s in self._active:self._active.remove(s)
        self._stats['decode_priority_turns']+=1;return True
    async def _advance_one_prefill(self):
        if not self._prefill_jobs:return False
        budget=self._prefill_budget();job=self._prefill_jobs.pop(0);state=await self._gpu(self._advance_prefill,job,self.cfg.native.prefill_chunks_per_turn,budget)
        if state is None and not job.done:self._prefill_jobs.append(job)
        elif state is not None:self._active.append(state);await self._gpu(self._rebatch)
        return True
    async def _admit_one(self):
        if len(self._active)+len(self._prefill_jobs)>=self.cfg.native.max_active_cohorts:return False
        if self._decode_work() and not self.cfg.native.dynamic_admission:return False
        if self._pending.empty():return False
        cohort=await self._gather(wait=not (self._decode_work() or self._prefill_jobs))
        if not cohort:return False
        if self._decode_work():self._stats['mid_generation_admissions']+=len(cohort)
        if self._lane is not None and bool(getattr(self.cfg.native,'static_decode_packed_prefill',False)):
            n0=len(cohort);cohort=await self._gpu(self._packed_admit,cohort)
            for x in self._lane.take_deferred():await self._pending.put(x)   # P7: rows the overlap prefill could not take
            if len(cohort)<n0:self._stats['cohorts']+=1;self._stats['requests_batched']+=n0-len(cohort);self._stats['max_observed_batch']=max(self._stats['max_observed_batch'],n0-len(cohort))
            if not cohort:return True
        job=await self._gpu(self._prepare_prefill,cohort);self._stats['cohorts']+=1;self._stats['requests_batched']+=len(cohort);self._stats['max_observed_batch']=max(self._stats['max_observed_batch'],len(cohort))
        if job:self._prefill_jobs.append(job)
        return True
    async def _loop(self):
        q=max(1,int(self.cfg.native.decode_quantum));priority=bool(self.cfg.scheduler.decode_priority)
        while not self._stop:
            did=False
            if priority:did=await self._decode_turn(q) or did
            did=await self._advance_one_prefill() or did;ad=await self._admit_one();did=ad or did
            if ad and not self._decode_work():did=await self._advance_one_prefill() or did
            if not priority:did=await self._decode_turn(q) or did
            await asyncio.sleep(0 if did else .0005)
    def _compile_counters(self):
        try:
            from torch._dynamo.utils import counters
            out={}
            for cat,vals in counters.items():
                hits={str(k):int(v) for k,v in vals.items() if 'graph' in str(k).lower() or 'cuda' in str(k).lower()}
                if hits:out[str(cat)]=hits
            return out
        except Exception:return {}
    def diagnostics(self):
        cfg=getattr(self.model,'config',None);text=getattr(cfg,'text_config',cfg);layers=list(getattr(text,'layer_types',()) or [])
        try:
            from inferstack.kernels.gdn_triton import diagnostics as gd
            from inferstack.kernels.qwen_conv_triton import diagnostics as cd
            g=gd();g['conv']=cd()
        except Exception:g=self._gdn_diag
        try:
            from inferstack.kernels.qwen_forward_fusion import diagnostics as ffd
            ff=ffd()
        except Exception:
            ff=self._forward_fusion_diag
        graph={'enabled':bool(self.cfg.native.explicit_cuda_graph),'compile_activated':self._compiled,'compile_error':self._compile_error,'static_cache_enabled':bool(getattr(self.cfg.native,'graph_static_cache',False)),'static_cache_uses':self._graph_static_cache_uses,'mark_step_calls':self._graph_mark_steps,'generation_cloning':bool(getattr(self.cfg.native,'graph_generation_cloning',True)),'configured_batch_sizes':list(self.cfg.native.cuda_graph_batch_sizes),'shape_cache_entries':len(self._graph_shapes),'shape_first_seen':self._graph_first,'shape_reuse_candidates':self._graph_reuse,'bucket_padded_rows_accounted':self._graph_bucket_rows,'torch_compile_counters':self._compile_counters(),'actual_cuda_graph_replay_proven':False,'note':'Actual capture/replay is accepted only by the A100 graph probe, never inferred from configuration.'}
        d={'engine':'qwen_hybrid_continuous','started':self.started,'healthy':self.healthy,'fatal_error':self._fatal_error,'device':self.device,'batching_mode':'live_refill_decode_first_chunked_prefill','paged_kv_active':False,'cache_backend':('transformers_hybrid_static_cache' if getattr(self.cfg.native,'graph_static_cache',False) else 'transformers_hybrid_dynamic_cache'),'hybrid_attention_model':len(set(layers))>1 if layers else None,'layer_types':layers,'quantization':str(getattr(self.model,'_inferstack_quantization','none')),'quantization_config':str(getattr(self.model,'_inferstack_quantization_config','none')),'prefix_state_cache':self._prefix_cache.diagnostics(),'sampler':self._sampler.diagnostics(),'gdn_kernel':g,'forward_fusion':ff,'cuda_graph':graph,'static_decode':({**self._static_diag,**self._lane.diagnostics()} if self._lane is not None else dict(self._static_diag)),'decode_profile':dict(self._profile),'ttft_breakdown':dict(self._ttft),'native':asdict(self.cfg.native),'kernels':asdict(self.cfg.kernels),'context_limit':self._context_limit(),'logits_to_keep_optimized':self._supports_logits_to_keep}
        d.update(self._stats);d['average_cohort_size']=self._stats['requests_batched']/self._stats['cohorts'] if self._stats['cohorts'] else 0.;d['active_batches']=len(self._active);d['prefill_jobs']=len(self._prefill_jobs);return d
