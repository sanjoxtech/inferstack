"""Canonical metric definitions.

Every metric that can appear in a trial, objective or verdict must be registered
here with its unit, aggregation semantics and measurement source. A metric name
that is not registered cannot be used as an objective, so two components can
never silently disagree about what "ttft" means.

Timestamp convention for all contracts: UTC epoch seconds as float (`*_at`).
Durations use explicit unit suffixes (`_ms`, `_s`). Byte counts use `_bytes`.
"""
from __future__ import annotations

from dataclasses import dataclass

TIMESTAMP_UNIT = 'utc_epoch_seconds'


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    unit: str
    definition: str
    source: str            # client | engine | gpu | infrastructure | billing | evaluator
    better: str            # min | max
    aggregation: str       # per-trial aggregation applied before cross-trial comparison


METRICS: dict[str, MetricDefinition] = {m.name: m for m in [
    MetricDefinition('ttft_client_ms', 'ms', 'Request send to first non-empty content event at the client', 'client', 'min', 'median'),
    MetricDefinition('ttft_engine_ms', 'ms', 'Engine-reported arrival to first generated token', 'engine', 'min', 'median'),
    MetricDefinition('tpot_engine_ms', 'ms', 'Engine token-emission interval after the first token; undefined for 1-token outputs', 'engine', 'min', 'median'),
    MetricDefinition('itl_engine_ms', 'ms', 'Engine inter-token latency per emitted token', 'engine', 'min', 'p50'),
    MetricDefinition('chunk_gap_client_ms', 'ms', 'Gap between client content events; NOT a token latency', 'client', 'min', 'median'),
    MetricDefinition('e2e_ms', 'ms', 'Request send to terminal event or full response at the client', 'client', 'min', 'mean'),
    MetricDefinition('output_tokens_per_s', 'tokens/s', 'Authoritative engine completion tokens divided by wall time of the trial', 'engine', 'max', 'sum_over_wall'),
    MetricDefinition('goodput_rps', 'requests/s', 'Completed requests meeting every declared SLO per second of trial wall time', 'evaluator', 'max', 'sum_over_wall'),
    MetricDefinition('failure_rate', 'ratio', 'Failed or incomplete requests divided by requests issued', 'client', 'min', 'ratio'),
    MetricDefinition('gpu_memory_peak_bytes', 'bytes', 'Peak allocated device memory reported by the engine or allocator; sampled nvidia-smi is labeled separately', 'gpu', 'min', 'max'),
    MetricDefinition('gpu_memory_sampled_peak_bytes', 'bytes', 'Peak of periodic nvidia-smi samples per GPU UUID; not an exact allocation maximum', 'gpu', 'min', 'max'),
    MetricDefinition('cost_usd_per_million_output_tokens', 'usd', 'Identified spend divided by authoritative output tokens; unavailable when price is unknown', 'billing', 'min', 'ratio'),
    MetricDefinition('exact_match_rate', 'ratio', 'Fraction of fixture cases whose output equals expected text exactly', 'evaluator', 'max', 'ratio'),
]}


def require_metric(name: str) -> MetricDefinition:
    try:
        return METRICS[name]
    except KeyError:
        raise ValueError(f'unregistered metric: {name}') from None
