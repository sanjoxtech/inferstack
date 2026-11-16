from __future__ import annotations
import copy, time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


def _tensor_nbytes(x)->int:
    try:return int(x.numel()*x.element_size())
    except Exception:return 0

def cache_nbytes(cache)->int:
    total=0; seen=set()
    for layer in getattr(cache,'layers',[]) or []:
        for name in ('keys','values','indexer_keys'):
            x=getattr(layer,name,None)
            if x is not None and id(x) not in seen: total+=_tensor_nbytes(x);seen.add(id(x))
        for name in ('conv_states','recurrent_states'):
            d=getattr(layer,name,None)
            if isinstance(d,dict):
                for x in d.values():
                    if x is not None and id(x) not in seen: total+=_tensor_nbytes(x);seen.add(id(x))
    return total

def _clone_mutable_state(layer):
    dst=copy.copy(layer)
    for name in ('conv_states','recurrent_states'):
        d=getattr(layer,name,None)
        if isinstance(d,dict): setattr(dst,name,{k:(v.clone() if hasattr(v,'clone') else copy.deepcopy(v)) for k,v in d.items()})
    for name in ('is_conv_states_initialized','is_recurrent_states_initialized','has_previous_state'):
        d=getattr(layer,name,None)
        if isinstance(d,dict): setattr(dst,name,dict(d))
    return dst

def fork_cache_cow(cache):
    if cache is None:return None
    out=copy.copy(cache)
    if hasattr(cache,'layers'): out.layers=[_clone_mutable_state(x) for x in cache.layers]
    return out

def _select_tensor(x,idx):
    return x.index_select(0,idx.to(x.device)) if x is not None else None

def select_cache_rows(cache, idx):
    if cache is None:return None
    # Work on a COW fork so a cached prefix snapshot can never be mutated by reordering.
    out=fork_cache_cow(cache)
    if hasattr(out,'batch_select_indices'):
        try:
            maybe=out.batch_select_indices(idx)
            return out if maybe is None else maybe
        except Exception: pass
    if hasattr(out,'reorder_cache'):
        try:
            maybe=out.reorder_cache(idx)
            return out if maybe is None else maybe
        except Exception: pass
    for layer in getattr(out,'layers',[]) or []:
        for name in ('keys','values','indexer_keys'):
            x=getattr(layer,name,None)
            if x is not None:setattr(layer,name,_select_tensor(x,idx))
        for name in ('conv_states','recurrent_states'):
            d=getattr(layer,name,None)
            if isinstance(d,dict):
                for k,v in list(d.items()):
                    if v is not None:d[k]=_select_tensor(v,idx)
    return out

def repeat_cache_rows(cache, repeats:int, device=None):
    import torch
    if repeats<=1:return cache
    idx=torch.zeros((repeats,),dtype=torch.long,device=device)
    return select_cache_rows(cache,idx)

def _pad_seq(x,target):
    import torch
    if x is None or x.ndim<2:return x
    cur=int(x.shape[-2])
    if cur>=target:return x
    shape=list(x.shape);shape[-2]=target-cur
    return torch.cat([x,torch.zeros(shape,dtype=x.dtype,device=x.device)],dim=-2)

def merge_cache_rows(caches, target_seq_len:int|None=None):
    import torch
    caches=[c for c in caches if c is not None]
    if not caches:return None
    if len(caches)==1:return caches[0]
    if any(type(c) is not type(caches[0]) for c in caches):raise RuntimeError('cache class mismatch')
    lists=[getattr(c,'layers',None) for c in caches]
    if any(x is None for x in lists):raise RuntimeError('cache does not expose layers')
    if len({len(x) for x in lists})!=1:raise RuntimeError('cache layer count mismatch')
    out=copy.copy(caches[0]);out.layers=[]
    for li in range(len(lists[0])):
        src=[xs[li] for xs in lists];dst=copy.copy(src[0])
        for name in ('keys','values','indexer_keys'):
            vals=[getattr(x,name,None) for x in src]
            if all(v is None for v in vals):continue
            if any(v is None for v in vals):raise RuntimeError(f'mixed {name} at layer {li}')
            target=int(target_seq_len or max(int(v.shape[-2]) for v in vals))
            setattr(dst,name,torch.cat([_pad_seq(v,target) for v in vals],dim=0))
        for name in ('conv_states','recurrent_states'):
            ds=[getattr(x,name,None) for x in src]
            if all(d is None for d in ds):continue
            if any(not isinstance(d,dict) for d in ds):raise RuntimeError(f'mixed {name} at layer {li}')
            merged={}
            for key in set().union(*(d.keys() for d in ds)):
                vals=[d.get(key) for d in ds]
                if all(v is None for v in vals):merged[key]=None
                elif any(v is None for v in vals):raise RuntimeError(f'mixed state {name}[{key}]')
                else:merged[key]=torch.cat(vals,dim=0)
            setattr(dst,name,merged)
        out.layers.append(dst)
    return out

@dataclass
class PrefixEntry:
    tokens:tuple[int,...]
    cache:Any
    last_logits:Any
    cache_width:int
    bytes:int
    created_at:float

class HybridPrefixStateCache:
    def __init__(self,max_entries=64,min_tokens=128,block_tokens=128,max_bytes=1024*1024*1024):
        self.max_entries=max(0,int(max_entries));self.min_tokens=max(1,int(min_tokens));self.block_tokens=max(1,int(block_tokens));self.max_bytes=max(0,int(max_bytes))
        self._entries:OrderedDict[tuple[int,...],PrefixEntry]=OrderedDict();self.hits=0;self.misses=0;self.evictions=0;self.cow_forks=0
    @property
    def enabled(self):return self.max_entries>0 and self.max_bytes>0
    def _aligned(self,tokens):return len(tokens)>=self.min_tokens and len(tokens)%self.block_tokens==0
    def put(self,tokens,cache,cache_width=None,last_logits=None):
        tokens=tuple(tokens)
        if not self.enabled or not self._aligned(tokens) or cache is None:return False
        snap=fork_cache_cow(cache);size=cache_nbytes(snap)+_tensor_nbytes(last_logits)
        e=PrefixEntry(tokens,snap,last_logits.detach().clone() if hasattr(last_logits,'detach') else last_logits,int(cache_width or len(tokens)),size,time.time())
        self._entries.pop(tokens,None);self._entries[tokens]=e
        while len(self._entries)>self.max_entries or sum(x.bytes for x in self._entries.values())>self.max_bytes:
            self._entries.popitem(last=False);self.evictions+=1
        return tokens in self._entries
    def longest(self,tokens):
        tokens=tuple(tokens);best=None
        for key,e in self._entries.items():
            if len(key)<=len(tokens) and tokens[:len(key)]==key and (best is None or len(key)>len(best.tokens)):best=e
        if best is None:self.misses+=1;return None
        self.hits+=1;self._entries.move_to_end(best.tokens);return best
    def has_prefix(self,tokens):
        """True when some entry is a prefix of tokens (no hit/miss counting, no LRU touch)."""
        tokens=tuple(tokens);return any(len(k)<=len(tokens) and tokens[:len(k)]==k for k in self._entries)
    def fork(self,entry):self.cow_forks+=1;return fork_cache_cow(entry.cache)
    def diagnostics(self):
        cur=sum(x.bytes for x in self._entries.values())
        return {'enabled':self.enabled,'entries':len(self._entries),'hits':self.hits,'misses':self.misses,'evictions':self.evictions,'cow_forks':self.cow_forks,'current_bytes':cur,'current_mb':cur/1024/1024,'max_bytes':self.max_bytes,'block_tokens':self.block_tokens,'min_tokens':self.min_tokens}
