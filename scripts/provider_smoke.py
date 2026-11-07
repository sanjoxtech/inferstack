#!/usr/bin/env python
"""Live planning smoke test against a configured optimizer provider.

Creates a throwaway state directory with a baseline and candidate record, asks the
provider for an experiment plan through the real ToolPlanner (tools enabled if the
provider config says so), validates it and prints the plan, token usage and the
audit artifact digest. No serving endpoint is contacted and nothing is executed.

  export DEEPSEEK_API_KEY=...      # or OPENROUTER_API_KEY
  python scripts/provider_smoke.py --kind deepseek --model <model-id> --key-env DEEPSEEK_API_KEY
  python scripts/provider_smoke.py --kind openrouter --model <vendor/model> --key-env OPENROUTER_API_KEY --tools
"""
import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from inferstack.optimization.planner import ToolPlanner
from inferstack.providers import create_provider
from inferstack.storage.repository import Repository


async def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--kind', required=True, choices=['deepseek', 'openrouter', 'openai_compatible'])
    p.add_argument('--model', required=True)
    p.add_argument('--key-env', required=True)
    p.add_argument('--base-url')
    p.add_argument('--tools', action='store_true', help='declare tool support (OpenRouter depends on the model)')
    a = p.parse_args()
    state = Path(tempfile.mkdtemp(prefix='inferstack-smoke-'))
    repo = Repository(state / 'control.db', state / 'artifacts')
    for name, profile in (('baseline', {'max_num_seqs': 64}), ('candidate', {'max_num_seqs': 128})):
        repo.put_record('deployment', name, {'id': name, 'engine': 'vllm', 'model': 'Qwen/Qwen3.5-2B-Base',
                                             'base_url': f'http://{name}.invalid', 'engine_version': '0.11.0',
                                             'profile': profile})
    config = {'kind': a.kind, 'model': a.model, 'api_key_env': a.key_env}
    if a.base_url:
        config['base_url'] = a.base_url
    if a.kind != 'deepseek':
        config['tools'] = a.tools
    planner = ToolPlanner(repo, create_provider(config), candidate_ids=['candidate'], max_tokens=40000)
    try:
        plan = await planner.plan({'deployment_id': 'baseline', 'goal': 'Reduce p95 client TTFT under 32 '
                                   'concurrent requests without quality regressions', 'source': 'smoke'}, [])
        print(json.dumps({'result': 'plan_validated', 'plan': plan, 'usage': planner.last_usage}, indent=2))
    except Exception as exc:
        print(json.dumps({'result': 'failed', 'error': type(exc).__name__, 'message': str(exc)[:500],
                          'usage': planner.last_usage, 'audit': getattr(planner, 'last_audit', None),
                          'state': str(state)}, indent=2))
        raise SystemExit(1)
    finally:
        await planner.close()


if __name__ == '__main__':
    asyncio.run(main())
