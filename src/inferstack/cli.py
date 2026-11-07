from __future__ import annotations
import argparse,json
from dataclasses import asdict
from inferstack import __version__
from inferstack.config import load_profile
from inferstack.foundry import FileSystemFoundryRepository, FoundryModelLoader
from inferstack.engines.catalog import capabilities, launch_plan

def _serve(a):
    import uvicorn
    from inferstack.runtime.qwen_continuous import QwenHybridContinuousEngine
    from inferstack.serving.service import ModelService,ModelServiceRegistry
    from inferstack.api.app import create_app
    cfg=load_profile(a.profile);repo=FileSystemFoundryRepository(a.foundry_root);loader=FoundryModelLoader(repo,a.device);manifest,model,tok,plan=loader.load(a.model,cfg)
    eng=QwenHybridContinuousEngine(model,tok,cfg,device=a.device);reg=ModelServiceRegistry();aliases={manifest.get('model_id','model'),manifest.get('source',{}).get('uri','')};aliases.discard('');reg.register(ModelService(manifest.get('model_id','model'),tok,eng),*aliases)
    uvicorn.run(create_app(reg),host=a.host,port=a.port,log_level=a.log_level)

def main(argv=None):
    p=argparse.ArgumentParser(prog='inferstack');p.add_argument('--version',action='version',version=__version__);sub=p.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('serve');s.add_argument('--profile',default='configs/profiles/production-v1.6.0.toml');s.add_argument('--foundry-root',default='examples/model_foundry');s.add_argument('--model',default='foundry://factorylm-2b@candidate');s.add_argument('--device',default='cuda:0');s.add_argument('--host',default='127.0.0.1');s.add_argument('--port',type=int,default=8000);s.add_argument('--log-level',default='info')
    i=sub.add_parser('inspect-foundry');i.add_argument('--profile',default='configs/profiles/production-v1.6.0.toml');i.add_argument('--foundry-root',default='examples/model_foundry');i.add_argument('--model',default='foundry://factorylm-2b@candidate');i.add_argument('--device',default='cuda:0')
    sub.add_parser('engines')
    pc=sub.add_parser('profile-command');pc.add_argument('--engine',required=True,choices=['vllm','sglang']);pc.add_argument('--model',required=True);pc.add_argument('--host',default='127.0.0.1');pc.add_argument('--port',type=int,default=8100);pc.add_argument('--dtype',default='bfloat16')
    n=sub.add_parser('native');nsub=n.add_subparsers(dest='native_cmd',required=True);np=nsub.add_parser('plan');np.add_argument('--profile',default='configs/profiles/production-v1.6.0.toml')
    a=p.parse_args(argv)
    if a.cmd=='serve':return _serve(a)
    if a.cmd=='inspect-foundry':
        cfg=load_profile(a.profile);loader=FoundryModelLoader(FileSystemFoundryRepository(a.foundry_root),a.device);m,src,plan=loader.inspect(a.model,cfg);print(json.dumps({'manifest':m,'runtime_plan':asdict(plan)},indent=2));return
    if a.cmd=='engines':print(json.dumps(capabilities(),indent=2));return
    if a.cmd=='profile-command':print(json.dumps(launch_plan(a.engine,a.model,a.host,a.port,a.dtype),indent=2));return
    if a.cmd=='native' and a.native_cmd=='plan':
        cfg=load_profile(a.profile)
        cache=('Transformers hybrid StaticCache for compiled decode' if cfg.native.graph_static_cache else 'Transformers hybrid DynamicCache + InferStack prefix-state COW')
        print(json.dumps({'profile':cfg.name,'scheduler':asdict(cfg.scheduler),'native':asdict(cfg.native),'kernels':asdict(cfg.kernels),'architecture_routing':{'qwen3.5/qwen3.6_hybrid':{'engine':'qwen_hybrid_continuous','batching':'live refill + decode-first chunked prefill' if cfg.native.live_batch_refill else 'fixed active cohort + decode-first prefill','cache':cache,'paged_kv':False,'quantization':cfg.native.quantization,'decode_profiler':cfg.native.profile_decode,'forward_fusion':cfg.native.forward_fusion,'fusion_plan':{'rmsnorm':cfg.native.fuse_rmsnorm,'add_rmsnorm':cfg.native.fuse_add_rmsnorm,'gated_rmsnorm':cfg.native.fuse_gated_rmsnorm,'mlp_gate_up':cfg.native.fuse_mlp_gate_up,'gdn_projections':cfg.native.fuse_gdn_projections,'grouped_gdn_decode':cfg.native.grouped_gdn_decode,'attention_qkv':cfg.native.fuse_attention_qkv}}}},indent=2));return

if __name__=='__main__':main()
