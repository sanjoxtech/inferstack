from __future__ import annotations
from dataclasses import dataclass, fields, field
from pathlib import Path
try: import tomllib
except ModuleNotFoundError: import tomli as tomllib

@dataclass
class SchedulerConfig:
    max_num_batched_tokens:int=8192
    max_running_requests:int=128
    max_prefill_chunk_tokens:int=128
    decode_priority:bool=True
    async_overlap:bool=False

@dataclass
class KernelConfig:
    attention:str="auto"
    rmsnorm:str="auto"
    rope:str="auto"
    cuda_graphs:bool=False
    torch_compile:bool=False
    compile_mode:str="reduce-overhead"

@dataclass
class NativeConfig:
    enabled:bool=True
    max_pending_requests:int=512
    max_requests_per_batch:int=16
    max_batch_tokens:int=8192
    batch_wait_ms:float=2.0
    dynamic_admission:bool=True
    live_batch_refill:bool=True
    active_batch_compaction:bool=True
    max_active_cohorts:int=4
    decode_quantum:int=1
    prefill_chunk_tokens:int=128
    prefill_chunks_per_turn:int=1
    # N3.1 token-budget prefill: tokens (rows x span) per prefill forward. 0 keeps the
    # fixed prefill_chunk_tokens policy. "idle" applies when no request is decoding,
    # "busy" when decode work shares the GPU. Spans are multiples of prefill_chunk_tokens.
    prefill_token_budget_idle:int=0
    prefill_token_budget_busy:int=0
    # N4.1 backlog-aware busy budget: while decoding, a forward may take up to
    # ceil(queued prefill tokens / drain_steps) (between busy and idle), so a burst of long
    # prompts is drained in about drain_steps prefill forwards instead of being throttled
    # to the busy budget. 0 keeps the fixed busy budget.
    prefill_backlog_drain_steps:int=0
    # N3.3: GDN prefill (T>1) backend: reference (upstream chunked algebra), fla (fused
    # FLA kernel; startup fails if it is not installed), auto (fla when available on CUDA).
    gdn_prefill_backend:str="reference"
    # N4.2: GDN short conv for prefill (T>1): reference (upstream F.conv1d + SiLU) or fla
    # (fused Triton conv + SiLU on [B, T, D]; startup fails if FLA is not installed).
    gdn_conv_backend:str="reference"
    # fla conv only when batch x tokens of the forward reaches this (0 = always); below it the
    # per-call Python cost of the FLA wrapper (~0.5 ms x 18 layers) outweighs the kernel gain.
    gdn_conv_fla_min_tokens:int=0
    # Run a few dummy prefill forwards at startup so Triton/FLA kernels compile and autotune
    # before the first request (CUDA only; failures are recorded in diagnostics, not raised).
    prefill_warmup:bool=False
    prefix_state_cache:bool=True
    prefix_state_cache_max_entries:int=64
    prefix_state_cache_max_mb:int=1024
    prefix_cache_min_tokens:int=128
    prefix_cache_block_tokens:int=128
    custom_gdn_triton:bool=True
    strict_custom_kernels:bool=False
    fused_sampler:bool=True
    compile_model:bool=False
    compile_dynamic:bool=True
    strict_compile:bool=False
    explicit_cuda_graph:bool=False
    cuda_graph_batch_sizes:tuple[int,...]=(1,2,4,8,16)
    kv_padding_interval_size:int=8192
    quantization:str="none"
    quantization_group_size:int=128
    use_hf_kernels:bool=False
    graph_static_cache:bool=False
    graph_cache_max_tokens:int=256
    graph_generation_cloning:bool=True
    profile_decode:bool=False

    # v1.7 static decode lane: decode by CUDA-graph replay over a fixed slot pool
    # (runtime/qwen_static_decode.py, runtime/qwen_static_lane.py). Rows that do not
    # fit the pool fall back to the Transformers decode path.
    static_decode:bool=False
    static_decode_slots:int=0                      # 0 = max_requests_per_batch
    static_decode_max_tokens:int=4096              # prompt + max_new_tokens per pooled row
    static_decode_batch_buckets:tuple[int,...]=(1,2,4,8,16)
    static_decode_len_buckets:tuple[int,...]=(256,512,1024,2048,4096)
    static_decode_gdn_backend:str="auto"
    static_decode_cuda_graph:bool=True
    static_decode_strict:bool=False

    # v1.6 Qwen forward-fusion engine. These are inference-only structural
    # optimizations installed after weights load; every path has a safe fallback.
    forward_fusion:bool=False
    fuse_rmsnorm:bool=True
    fuse_add_rmsnorm:bool=True
    fuse_gated_rmsnorm:bool=True
    fuse_mlp_gate_up:bool=True
    fuse_mlp_activation:bool=True
    fuse_gdn_projections:bool=True
    fuse_gdn_beta_g:bool=True
    grouped_gdn_decode:bool=True
    fuse_attention_qkv:bool=True
    forward_fusion_strict:bool=False

@dataclass
class HardwareConfig:
    device:str="auto"
    dtype:str="auto"
    adaptive:bool=True

@dataclass
class Profile:
    name:str="production-v1.6.0"
    scheduler:SchedulerConfig=field(default_factory=SchedulerConfig)
    kernels:KernelConfig=field(default_factory=KernelConfig)
    native:NativeConfig=field(default_factory=NativeConfig)
    hardware:HardwareConfig=field(default_factory=HardwareConfig)


def _build(cls,data):
    names={f.name for f in fields(cls)}
    return cls(**{k:v for k,v in (data or {}).items() if k in names})

def load_profile(path:str|Path)->Profile:
    with open(path,"rb") as f:d=tomllib.load(f)
    return Profile(
        name=d.get("name","production-v1.6.0"),
        scheduler=_build(SchedulerConfig,d.get("scheduler")),
        kernels=_build(KernelConfig,d.get("kernels")),
        native=_build(NativeConfig,d.get("native")),
        hardware=_build(HardwareConfig,d.get("hardware")),
    )
