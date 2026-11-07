from __future__ import annotations
import shutil

def capabilities():
    return [
        {'engine':'inferstack_native','installed':True,'modes':['in_process'],'cache_features':['chunked_prefill','continuous_batching','hybrid_prefix_state'],'api_features':['completions','chat_completions','streaming']},
        {'engine':'vllm','installed':bool(shutil.which('vllm')),'modes':['openai_server'],'cache_features':['paged_kv','continuous_batching','prefix_cache']},
        {'engine':'sglang','installed':bool(shutil.which('sglang')),'modes':['openai_server'],'cache_features':['paged_kv','continuous_batching','radix_prefix_cache']},
        {'engine':'generic_openai','installed':True,'modes':['remote_openai']},
    ]

def launch_plan(engine,model,host='127.0.0.1',port=8100,dtype='bfloat16'):
    if engine=='vllm':argv=['vllm','serve',model,'--host',host,'--port',str(port),'--dtype',dtype]
    elif engine=='sglang':argv=['python','-m','sglang.launch_server','--model-path',model,'--host',host,'--port',str(port),'--dtype',dtype]
    else:raise ValueError(f'no local launch plan for {engine}')
    return {'engine':engine,'model':model,'host':host,'port':port,'argv':argv,'base_url':f'http://{host}:{port}'}
