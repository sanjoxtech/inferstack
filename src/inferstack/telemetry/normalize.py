"""Engine metric normalization with explicit semantics and raw retention.

Each engine has a mapping table from its Prometheus families to InferStack
observation names. Mapping tables are tagged with the engine versions they were
written against; their status stays `declared` until a qualification run on that
version confirms the names. A family absent from a scrape yields `None` with a
reason, never zero. Raw scrape text is retained by hash for audit.

Semantics differ by source and are carried with every value:
- `engine_ttft` is the engine's arrival-to-first-token, not client first content.
- `engine_tpot` is per-output-token time measured by the engine.
- queue/running gauges are instantaneous at scrape time.
"""
from __future__ import annotations

import hashlib
import time

from .prometheus import Scrape, histogram_delta, quantile

# observation -> (kind, family, unit scale to ms or ratio, semantics)
MAPPINGS = {
    'vllm': {
        'tested_against': [], 'declared_for': '>=0.10',
        'families': {
            'engine_ttft_ms': ('histogram', 'vllm:time_to_first_token_seconds', 1000.0, 'engine arrival to first token'),
            'engine_tpot_ms': ('histogram', ('vllm:inter_token_latency_seconds', 'vllm:time_per_output_token_seconds'),
                               1000.0, 'engine inter-token time'),
            'engine_e2e_ms': ('histogram', 'vllm:e2e_request_latency_seconds', 1000.0, 'engine request latency'),
            'engine_queue_ms': ('histogram', 'vllm:request_queue_time_seconds', 1000.0, 'time waiting before scheduling'),
            'engine_prefill_ms': ('histogram', 'vllm:request_prefill_time_seconds', 1000.0, 'prefill phase time'),
            'engine_decode_ms': ('histogram', 'vllm:request_decode_time_seconds', 1000.0, 'decode phase time'),
            'requests_running': ('gauge', 'vllm:num_requests_running', 1.0, 'instantaneous'),
            'requests_waiting': ('gauge', 'vllm:num_requests_waiting', 1.0, 'instantaneous'),
            'kv_cache_usage_ratio': ('gauge', ('vllm:kv_cache_usage_perc', 'vllm:gpu_cache_usage_perc'), 1.0,
                                     'fraction of KV blocks in use'),
            'prompt_tokens_total': ('counter', 'vllm:prompt_tokens_total', 1.0, 'cumulative prompt tokens'),
            'generation_tokens_total': ('counter', 'vllm:generation_tokens_total', 1.0, 'cumulative output tokens'),
            'prefix_cache_hits_total': ('counter', 'vllm:prefix_cache_hits_total', 1.0, 'cumulative prefix-cache hit tokens'),
            'prefix_cache_queries_total': ('counter', 'vllm:prefix_cache_queries_total', 1.0, 'cumulative prefix-cache queried tokens'),
        }},
    'sglang': {
        'tested_against': [], 'declared_for': '>=0.4',
        'families': {
            'engine_ttft_ms': ('histogram', 'sglang:time_to_first_token_seconds', 1000.0, 'engine arrival to first token'),
            'engine_tpot_ms': ('histogram', ('sglang:inter_token_latency_seconds', 'sglang:time_per_output_token_seconds'),
                               1000.0, 'engine inter-token time'),
            'engine_e2e_ms': ('histogram', 'sglang:e2e_request_latency_seconds', 1000.0, 'engine request latency'),
            'requests_running': ('gauge', 'sglang:num_running_reqs', 1.0, 'instantaneous'),
            'requests_waiting': ('gauge', 'sglang:num_queue_reqs', 1.0, 'instantaneous'),
            'kv_cache_usage_ratio': ('gauge', 'sglang:token_usage', 1.0, 'fraction of KV token slots in use'),
            'prefix_cache_hit_ratio': ('gauge', 'sglang:cache_hit_rate', 1.0, 'engine-reported cache hit rate'),
            'prompt_tokens_total': ('counter', 'sglang:prompt_tokens_total', 1.0, 'cumulative prompt tokens'),
            'generation_tokens_total': ('counter', 'sglang:generation_tokens_total', 1.0, 'cumulative output tokens'),
        }},
}


def _family(scrape, family):
    for name in (family if isinstance(family, tuple) else (family,)):
        if scrape.has_family(name):
            return name
    return None


def normalize(engine, text, *, previous_text=None, engine_version=None, label_filter=None, clock=time.time):
    """Normalize one scrape. With `previous_text`, histogram/counter values cover the window."""
    if engine not in MAPPINGS:
        return {'engine': engine, 'available': False, 'reason': f'no metric mapping for {engine}', 'metrics': {}}
    current = Scrape(text)
    prior = Scrape(previous_text) if previous_text is not None else None
    metrics = {}
    for observation, (kind, family, scale, semantics) in MAPPINGS[engine]['families'].items():
        name = _family(current, family)
        if name is None:
            metrics[observation] = {'value': None, 'reason': f'family {family} absent from scrape'}
            continue
        entry = {'source_metric': name, 'semantics': semantics, 'kind': kind}
        if kind == 'histogram':
            hist = current.histogram(name, label_filter)
            if prior is not None:
                hist = histogram_delta(prior.histogram(name, label_filter), hist)
                entry['window'] = 'delta'
            else:
                entry['window'] = 'cumulative_since_start'
            if hist is None or hist['count'] <= 0:
                metrics[observation] = {**entry, 'value': None, 'reason': 'no observations in window'}
                continue
            q50, q99 = quantile(hist, .5), quantile(hist, .99)
            entry.update({'value': hist['sum'] / hist['count'] * scale, 'aggregation': 'mean', 'count': hist['count'],
                          'p50_estimate': None if q50 is None else q50 * scale,
                          'p99_estimate': None if q99 is None else q99 * scale,
                          'quantile_method': 'bucket_interpolation'})
        elif kind == 'counter':
            value = current.total(name, label_filter)
            if prior is not None:
                before = prior.total(name, label_filter)
                value = None if before is None or value < before else value - before
                entry['window'] = 'delta'
            entry['value'] = value
        else:
            entry['value'] = current.total(name, label_filter)
            entry['aggregation'] = 'sum_over_label_sets'
        metrics[observation] = entry
    return {'engine': engine, 'engine_version': engine_version, 'available': True, 'collected_at': clock(),
            'mapping_status': 'tested' if engine_version in MAPPINGS[engine]['tested_against'] else 'declared',
            'raw_sha256': hashlib.sha256(text.encode()).hexdigest(), 'metrics': metrics}
