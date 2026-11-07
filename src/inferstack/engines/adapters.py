"""Engine adapters: launch plans, runtime identity, observed features, telemetry.

Lifecycle (attach/launch, ready, complete/stream, stop) is provided by
ManagedEngine; adapters add what differs per engine:

- `launch_argv` builds the pinned launch command (immutable model/tokenizer
  revision, host, port, dtype); it never accepts a mutable revision.
- `runtime_identity` asks the running server what it actually is (version,
  model, flags) so a deployment can be attested against its record.
- `observed_features` reports effective feature activation with the evidence it
  rests on; configured flags alone are not activation.
- `telemetry` scrapes and normalizes engine metrics (raw retained by hash).

Supported version ranges are declarations until a qualification run records
them as tested (Package J).
"""
from __future__ import annotations

import re

import httpx

from inferstack.contracts.records import MUTABLE_REVISIONS
from inferstack.telemetry import normalize

SUPPORTED_VERSIONS = {  # declared, not yet qualified on hardware
    'vllm': '>=0.10,<1.0',
    'sglang': '>=0.4,<1.0',
    'inferstack_native': '>=1.6',
}


class AdapterError(RuntimeError):
    pass


def _immutable(revision, what):
    if not revision or revision in MUTABLE_REVISIONS:
        raise AdapterError(f'{what} must be an immutable revision, got {revision!r}')
    return revision


class EngineAdapter:
    engine = 'generic_openai'
    identity_path = None

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    def launch_argv(self, *, model, revision, tokenizer_revision=None, host='127.0.0.1', port, dtype='bfloat16',
                    extra=()):
        raise AdapterError(f'{self.engine} has no local launch plan')

    async def _get_json(self, path):
        try:
            response = await self.client.get(path.lstrip('/'))
            if response.status_code != 200:
                return None, f'HTTP {response.status_code} from {path}'
            return response.json(), None
        except (httpx.HTTPError, ValueError) as exc:
            return None, f'{type(exc).__name__} from {path}'

    async def runtime_identity(self):
        models, error = await self._get_json('/v1/models')
        served = [m.get('id') for m in (models or {}).get('data', [])] if models else []
        return {'engine': self.engine, 'version': None, 'served_models': served, 'flags': {},
                'errors': [error] if error else []}

    async def observed_features(self, identity=None, scrape=None):
        return {}

    async def scrape(self):
        try:
            response = await self.client.get('metrics')
            return response.text if response.status_code == 200 else None
        except httpx.HTTPError:
            return None

    async def telemetry(self, previous_text=None, engine_version=None):
        text = await self.scrape()
        if text is None:
            return {'engine': self.engine, 'available': False, 'reason': 'metrics endpoint unavailable', 'metrics': {}}
        return {**normalize(self.engine, text, previous_text=previous_text, engine_version=engine_version),
                'raw_text': text}


class VLLMAdapter(EngineAdapter):
    engine = 'vllm'

    def launch_argv(self, *, model, revision, tokenizer_revision=None, host='127.0.0.1', port, dtype='bfloat16',
                    extra=()):
        _immutable(revision, 'model revision')
        return ['vllm', 'serve', model, '--revision', revision,
                '--tokenizer-revision', _immutable(tokenizer_revision or revision, 'tokenizer revision'),
                '--host', host, '--port', str(port), '--dtype', dtype, *extra]

    async def runtime_identity(self):
        identity = await super().runtime_identity()
        version, error = await self._get_json('/version')
        if version:
            identity['version'] = version.get('version')
        elif error:
            identity['errors'].append(error)
        return identity

    async def observed_features(self, identity=None, scrape=None):
        text = scrape if scrape is not None else await self.scrape()
        if text is None:
            return {'prefix_cache': {'effective': 'unknown', 'evidence': 'metrics unavailable'}}
        queried = re.search(r'^vllm:prefix_cache_queries_total(?:\{[^}]*\})?\s+(\S+)', text, re.M)
        if queried is None:
            return {'prefix_cache': {'effective': 'unknown', 'evidence': 'prefix cache counters absent'}}
        return {'prefix_cache': {'effective': 'active' if float(queried.group(1)) > 0 else 'inactive_or_unused',
                                 'evidence': 'vllm:prefix_cache_queries_total'}}


class SGLangAdapter(EngineAdapter):
    engine = 'sglang'

    def launch_argv(self, *, model, revision, tokenizer_revision=None, host='127.0.0.1', port, dtype='bfloat16',
                    extra=()):
        _immutable(revision, 'model revision')
        if tokenizer_revision not in (None, revision):
            raise AdapterError('SGLang uses one --revision for model and tokenizer; they must match')
        return ['python', '-m', 'sglang.launch_server', '--model-path', model, '--revision', revision,
                '--host', host, '--port', str(port), '--dtype', dtype, '--enable-metrics', *extra]

    async def runtime_identity(self):
        identity = await super().runtime_identity()
        info, error = await self._get_json('/get_server_info')
        if info:
            identity['version'] = info.get('version')
            identity['flags'] = {k: info[k] for k in ('disable_radix_cache', 'disable_cuda_graph', 'enable_torch_compile',
                                                      'attention_backend', 'dtype', 'kv_cache_dtype',
                                                      'enable_metrics') if k in info}
        elif error:
            identity['errors'].append(error)
        return identity

    async def observed_features(self, identity=None, scrape=None):
        identity = identity or await self.runtime_identity()
        flags = identity.get('flags') or {}
        out = {}
        if 'disable_radix_cache' in flags:
            out['prefix_cache'] = {'effective': 'disabled' if flags['disable_radix_cache'] else 'enabled_by_config',
                                   'evidence': 'get_server_info.disable_radix_cache (configuration, not usage)'}
        if 'disable_cuda_graph' in flags:
            out['cuda_graph_replay'] = {'effective': 'disabled' if flags['disable_cuda_graph'] else 'enabled_by_config',
                                        'evidence': 'get_server_info.disable_cuda_graph (configuration, not replay proof)'}
        return out


class NativeAdapter(EngineAdapter):
    engine = 'inferstack_native'

    def launch_argv(self, *, model, revision=None, tokenizer_revision=None, host='127.0.0.1', port, dtype='bfloat16',
                    extra=()):
        # Native resolves revisions through the Foundry manifest; onboarding rejects mutable ones.
        if not model.startswith('foundry://'):
            raise AdapterError('native engine launches Foundry references (foundry://model@release)')
        return ['inferstack', 'serve', '--model', model, '--host', host, '--port', str(port), *extra]

    async def runtime_identity(self):
        identity = await super().runtime_identity()
        status, error = await self._get_json('/v1/native/status')
        if status:
            identity['version'] = status.get('version')
            identity['flags'] = {'models': status.get('models', {})}
        elif error:
            identity['errors'].append(error)
        return identity

    async def observed_features(self, identity=None, scrape=None):
        identity = identity or await self.runtime_identity()
        out = {}
        for model, diag in (identity.get('flags') or {}).get('models', {}).items():
            if not isinstance(diag, dict):
                continue
            if 'paged_kv_active' in diag:
                out[f'{model}.paged_kv'] = {'effective': 'active' if diag['paged_kv_active'] else 'inactive',
                                            'evidence': 'native status paged_kv_active'}
            if 'actual_cuda_graph_replay_proven' in diag:
                out[f'{model}.cuda_graph_replay'] = {
                    'effective': 'proven' if diag['actual_cuda_graph_replay_proven'] else 'not_proven',
                    'evidence': 'native status actual_cuda_graph_replay_proven'}
        return out

    async def telemetry(self, previous_text=None, engine_version=None):
        identity = await self.runtime_identity()
        return {'engine': self.engine, 'available': bool(identity['flags']), 'metrics': {},
                'reason': 'native exposes JSON diagnostics, not Prometheus; see runtime identity',
                'diagnostics': identity['flags']}


ADAPTERS = {'vllm': VLLMAdapter, 'sglang': SGLangAdapter, 'inferstack_native': NativeAdapter,
            'generic_openai': EngineAdapter}


def adapter_for(engine, client):
    return ADAPTERS[engine](client)


async def attest(deployment: dict, adapter: EngineAdapter) -> dict:
    """Compare what the server reports with the deployment record. Unknown is not a pass."""
    identity = await adapter.runtime_identity()
    checks = {'model_served': deployment['model'] in identity['served_models']}
    expected = deployment.get('engine_version')
    if expected is None:
        checks['engine_version'] = None
    elif identity['version'] is None:
        checks['engine_version'] = None
    else:
        checks['engine_version'] = identity['version'] == expected
    status = 'mismatch' if False in checks.values() else 'unverified' if None in checks.values() else 'attested'
    return {'status': status, 'checks': checks, 'identity': identity}
