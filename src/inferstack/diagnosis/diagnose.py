"""Deterministic, evidence-only diagnosis of an inference symptom.

A diagnosis is a hypothesis with its evidence, never an invented cause:

- `owner`: who should act. `inference` (serving/engine/capacity: InferStack),
  `model` (model behaviour: Model Foundry), `application` (prompt, tool protocol,
  JSON contract: FactoryOS), or `unknown`.
- `bottleneck` (inference only): `queue_admission`, `prefill`, `decode`,
  `kv_cache_reuse`, `network_streaming`, `capacity`, or `uncertain`.
- `confidence`: `evidenced` (a measured signal crosses a stated threshold),
  `suspected` (only weak or indirect signals), `unknown`.
- `signals`: every measurement used, with its value and the rule it satisfied.
- `alternatives`: other explanations the evidence does not rule out.
- `measurement_requests`: what to measure next when the evidence is insufficient.

Inputs are reported observations (FactoryOS/operator numbers) and, when available,
the durable live serving-evidence window for the deployment. Free text is never read
as evidence or instruction.
"""
from __future__ import annotations

import math

OWNERS = ('inference', 'model', 'application', 'unknown')
BOTTLENECKS = ('queue_admission', 'prefill', 'decode', 'kv_cache_reuse', 'network_streaming', 'capacity',
               'uncertain')
MIN_LIVE_REQUESTS = 50  # below this a live window is too small to attribute a bottleneck

# Observation keys accepted as numbers (FactoryOS/operator reports) and their meaning.
LATENCY_KEYS = ('p95_client_ttft_ms', 'ttft_p95_ms', 'p95_ms', 'e2e_p95_ms', 'p50_ms', 'ttft_ms', 'e2e_ms',
                'itl_p95_ms', 'stall_p95_ms', 'queue_ms', 'tokens_per_s')
APPLICATION_ERRORS = {'tool_call_invalid', 'json_schema', 'json_parse', 'prompt_too_long', 'client_timeout',
                      'application'}
MODEL_ERRORS = {'refusal', 'hallucination', 'wrong_answer', 'off_topic', 'format_drift'}
INFERENCE_ERRORS = {'http_5xx', 'engine_error', 'timeout', 'stream_stall', 'oom', 'overloaded', 'rejected'}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def numeric_observations(observations):
    return {k: v for k, v in (observations or {}).items() if _number(v)}


def _ratio(a, b):
    return a / b if _number(a) and _number(b) and b > 0 else None


def diagnose(category, observations=None, live=None, *, objectives=None):
    """Return a diagnosis dict (see module doc). `live` is a serving-evidence window summary."""
    observations = observations or {}
    numbers = numeric_observations(observations)
    signals, alternatives, requests = [], [], []
    owner, bottleneck, confidence = 'unknown', 'uncertain', 'unknown'
    live_ok = bool(live) and (live.get('requests') or 0) >= MIN_LIVE_REQUESTS

    if category in ('quality', 'error'):
        kinds = {str(k).lower() for k in observations.get('error_types') or []} | (
            {str(observations['error_type']).lower()} if observations.get('error_type') else set())
        if category == 'error' and live:
            served = live.get('requests') or 0
            failed = (live.get('errors') or 0) + (live.get('timeouts') or 0)
            if served and failed / served >= 0.01:
                signals.append({'signal': 'live_error_rate', 'value': failed / served, 'rule': '>= 1% engine errors'})
                kinds.add('engine_error')
            if (live.get('rejected') or 0) > 0:
                signals.append({'signal': 'admission_rejections', 'value': live['rejected'],
                                'sources': live.get('rejection_sources'), 'rule': '> 0 rejections'})
                kinds.add('rejected')
        inference = kinds & INFERENCE_ERRORS
        application = kinds & APPLICATION_ERRORS
        model = kinds & MODEL_ERRORS
        if observations.get('after_release') and category == 'quality':
            inference.add('regression_after_release')
            signals.append({'signal': 'after_release', 'value': observations['after_release'],
                            'rule': 'quality changed after an inference release'})
        owners = [o for o, hit in (('inference', inference), ('application', application), ('model', model)) if hit]
        for name, hit in (('inference', inference - {'regression_after_release'}), ('application', application),
                          ('model', model)):
            if hit:
                signals.append({'signal': f'{name}_error_types', 'value': sorted(hit),
                                'rule': f'error types classified as {name}'})
        if len(owners) == 1:
            owner, confidence = owners[0], 'evidenced'
        elif len(owners) > 1:
            owner, confidence = owners[0], 'suspected'
            alternatives += [f'{o} also implicated' for o in owners[1:]]
        else:
            requests += ['classify failures: error types per request (tool/JSON/model/engine) with trace IDs',
                         'compare outputs from the current and previous inference release on the same inputs']
        if owner == 'inference' and inference & {'rejected', 'overloaded'}:
            bottleneck = 'capacity'
        if category == 'quality' and owner == 'model':
            alternatives.append('prompt or application change (verify the application did not change)')
        return _result(owner, bottleneck, confidence, signals, alternatives, requests, numbers)

    # Latency / throughput / capacity / cost: an inference symptom when measured.
    if not numbers and not live_ok:
        requests += ['report numeric symptoms (e.g. p95_client_ttft_ms, e2e_p95_ms, tokens_per_s) with trace IDs',
                     f'record at least {MIN_LIVE_REQUESTS} requests through the InferStack gateway (live evidence)']
        return _result('unknown', 'uncertain', 'unknown', signals, alternatives, requests, numbers)
    owner, confidence = 'inference', 'suspected'
    for key, value in numbers.items():
        signals.append({'signal': f'reported_{key}', 'value': value, 'rule': 'reported symptom'})
    if live_ok:
        served = live['requests']
        rejected = live.get('rejected') or 0
        ttft, e2e = live.get('ttft_p95_ms'), live.get('e2e_p95_ms')
        stall, itl = live.get('stall_p95_ms'), live.get('itl_p95_ms')
        first_byte_gap = live.get('first_byte_to_content_ms')
        if rejected and rejected / (served + rejected) >= 0.01:
            bottleneck, confidence = 'queue_admission', 'evidenced'
            signals.append({'signal': 'rejection_rate', 'value': rejected / (served + rejected),
                            'sources': live.get('rejection_sources'), 'rule': '>= 1% of requests rejected'})
            alternatives.append('capacity: more replicas may be needed (see scaling advisory)')
        elif _number(stall) and _number(ttft) and stall > max(250.0, 0.5 * ttft):
            bottleneck, confidence = 'decode', 'evidenced'
            signals.append({'signal': 'stall_p95_ms', 'value': stall, 'rule': '> max(250 ms, 0.5 x TTFT p95)'})
            alternatives.append('network buffering can look like decode stalls; compare engine TPOT')
        elif _number(ttft) and _number(e2e) and ttft > 0.5 * e2e:
            bottleneck, confidence = 'prefill', 'evidenced'
            signals.append({'signal': 'ttft_share_of_e2e', 'value': _ratio(ttft, e2e), 'rule': 'TTFT p95 > 50% of E2E'})
            alternatives += ['queueing before prefill (check engine queue time)',
                             'kv_cache_reuse: low prefix-cache hit rate on repeated prefixes']
            requests.append('engine telemetry: queue time and prefix-cache hit rate for the same window')
        elif _number(first_byte_gap) and _number(ttft) and first_byte_gap > 0.3 * ttft:
            bottleneck, confidence = 'network_streaming', 'suspected'
            signals.append({'signal': 'first_byte_to_content_ms', 'value': first_byte_gap,
                            'rule': '> 30% of TTFT'})
        else:
            requests.append('bounded profile of prefill and decode on the reported workload')
        if _number(itl):
            signals.append({'signal': 'itl_p95_ms', 'value': itl, 'rule': 'context'})
    else:
        requests.append(f'live gateway evidence for the deployment (>= {MIN_LIVE_REQUESTS} requests) to attribute '
                        'the bottleneck; reported numbers alone establish the symptom, not its cause')
    queue = numbers.get('queue_ms')
    e2e_reported = numbers.get('e2e_p95_ms') or numbers.get('p95_ms')
    if bottleneck == 'uncertain' and _number(queue) and _number(e2e_reported) and queue > 0.3 * e2e_reported:
        bottleneck = 'queue_admission'
        signals.append({'signal': 'queue_share', 'value': _ratio(queue, e2e_reported), 'rule': 'queue > 30% of E2E'})
    return _result(owner, bottleneck, confidence, signals, alternatives, requests, numbers)


def _result(owner, bottleneck, confidence, signals, alternatives, requests, numbers):
    return {'version': 1, 'owner': owner, 'bottleneck': bottleneck if owner == 'inference' else None,
            'confidence': confidence, 'signals': signals, 'alternatives': alternatives,
            'measurement_requests': requests, 'symptom_evidenced': bool(numbers) or bool(signals)}
