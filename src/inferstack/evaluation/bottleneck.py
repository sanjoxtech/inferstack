"""G4: why a run missed its SLO, measured, so the planner reasons from a bottleneck instead of from settings.

Two bounded measurements, both deterministic and evidence-only (no model, no text generation):

  knee     From a goodput sweep (evaluation/goodput.py): the first rate step below 99% good, which SLO the
           missing requests broke (TTFT or per-token time), and, for engines that expose them, the engine's
           own counters over that step (native /v1/native/status: TTFT split into queue / wait / prefill,
           decode steps and other prefills between a request's admission and its first token, spec lane
           rows per step and step time). Labels follow diagnosis/diagnose.py: queue_admission, prefill,
           decode, capacity, uncertain; `none` when the whole rate grid met the SLO.

  kernels  From a decode-step profile (scripts/decode_step_profile.py, torch.profiler on the GPU host): the
           share of GPU time per kernel family (gdn, attention, gemm, conv, norm_activation, sampling,
           memory, elementwise, other) at each profiled batch size, plus the top kernels.

Measurements are stored per deployment (record kind `measurement`, id `<deployment>--<kind>`, revisions keep
history, the raw evidence is a content-addressed artifact). The context pack and the planner read the latest
ones; a plan made while measurements exist must name the bottleneck it targets.
"""
from __future__ import annotations

import json
import re
import time

GOOD = 0.99
MIN_SHARE = 0.6          # a phase owning at least this share of the TTFT is the bottleneck (evidenced)

# This repository's own Triton kernels have short generic names; map them exactly (src/inferstack/kernels).
REPO_KERNELS = {'_banked_kernel': 'gdn', '_fixup_kernel': 'gdn', '_beta_g_kernel': 'gdn',     # gdn_banked, fusion
                '_split_kernel': 'attention', '_merge_kernel': 'attention',                 # decode_attention
                '_rmsnorm_kernel': 'norm_activation', '_gated_rmsnorm_kernel': 'norm_activation',
                '_add_rmsnorm_kernel': 'norm_activation', '_silu_mul_kernel': 'norm_activation'}

KERNEL_FAMILIES = (      # first match wins; order matters (e.g. attention softmax before sampling)
    ('gdn', r'gated_delta|delta_rule|gdn|fused_recurrent|chunk_(fwd|bwd|scaled|local|gated)|recurrent_'),
    ('attention', r'flash|attn|attention|sdpa|fmha|paged'),
    ('conv', r'conv'),
    ('gemm', r'gemm|gemv|cutlass|cublas|matmul|xmma|splitk|sm[0-9]{2}_|ampere_|hopper|turing_|_mm_|\bmm\b|addmm|bmm'),
    ('norm_activation', r'rms|norm|silu|gelu|softplus|sigmoid|swiglu|act_'),
    ('sampling', r'sampl|argmax|topk|top_k|multinomial|softmax|exponential'),
    ('memory', r'copy|memcpy|memset|cat_|catarray|index|gather|scatter|fill|clone|contiguous'),
    ('elementwise', r'elementwise|vectorized|unrolled|reduce|triton_'),
)


def _knee(steps):
    for i, s in enumerate(steps):
        if s['good_frac'] < GOOD:
            return i, s
    return None, None


def _lane(c, wall_s):
    steps, rows = c.get('lane_spec_steps', 0), c.get('lane_spec_row_steps', 0)
    out = {}
    if steps:
        out['rows_per_step'] = round(rows / steps, 2)
        out['spec_step_ms'] = round(c.get('lane_spec_step_ms', 0) / steps, 3)
    if rows:
        out['spec_acceptance'] = round(c.get('lane_spec_accepted', 0) / rows, 3)
    if c.get('lane_packed_prefills'):
        out['packed_ms_per_prefill'] = round(c.get('lane_packed_ms', 0) / c['lane_packed_prefills'], 2)
    if wall_s and c.get('lane_gpu_thread_ms') is not None:
        out['gpu_busy_frac'] = round(c['lane_gpu_thread_ms'] / (wall_s * 1000), 3)
    return out


def diagnose_case(case_result):
    """Knee diagnosis of one case's sweep: {'bottleneck', 'confidence', 'knee_rps', 'signal'}."""
    steps = case_result.get('steps') or []
    i, k = _knee(steps)
    if k is None:
        return {'bottleneck': 'none', 'confidence': 'evidenced', 'knee_rps': None,
                'signal': {'max_rate_met': steps[-1]['rate'] if steps else None,
                           'note': 'every swept rate met the SLO; the grid is too low to find the knee'}}
    misses = k.get('misses') or {}
    signal = {'knee_rps': k['rate'], 'last_good_rps': steps[i - 1]['rate'] if i else None,
              'good_frac': round(k['good_frac'], 3), 'misses': misses, 'requests': k.get('requests'),
              'ttft_ms_p50': k.get('ttft_ms_p50'),
              'ttft_limit_ms': (case_result.get('case') or {}).get('ttft_ms'),
              'tpot_limit_ms': (case_result.get('case') or {}).get('tpot_ms')}
    c = k.get('counters') or {}
    split = None
    if c.get('requests'):
        n = c['requests']
        split = {'queue_ms': c.get('queue_ms', 0) / n, 'wait_ms': c.get('wait_ms', 0) / n,
                 'prefill_ms': c.get('prefill_ms', 0) / n}
        signal['ttft_split_ms_mean'] = {key: round(v, 1) for key, v in split.items()}
        signal['decode_steps_before_first_token'] = round(c.get('decode_steps', 0) / n, 1)
        signal['other_prefills_before_first_token'] = round(c.get('other_prefill_forwards', 0) / n, 2)
    lane = _lane(c, k.get('wall_s'))
    if lane:
        signal['lane'] = lane
    result = {'knee_rps': k['rate'], 'signal': signal}
    if not misses:
        return {**result, 'bottleneck': 'uncertain', 'confidence': 'unknown'}
    dominant, count = max(misses.items(), key=lambda kv: kv[1])
    signal['dominant_miss'] = dominant
    signal['dominant_miss_share'] = round(count / sum(misses.values()), 3)
    if dominant == 'error':
        return {**result, 'bottleneck': 'capacity', 'confidence': 'evidenced'}
    if dominant == 'no_tokens':
        return {**result, 'bottleneck': 'uncertain', 'confidence': 'unknown'}
    if dominant == 'tpot':
        return {**result, 'bottleneck': 'decode', 'confidence': 'evidenced'}
    # TTFT misses: which part of the time to first token grew
    if split is None:
        return {**result, 'bottleneck': 'uncertain', 'confidence': 'suspected',
                'note': 'TTFT limit missed; the engine exposes no TTFT split'}
    phases = {'queue_admission': split['queue_ms'] + split['wait_ms'], 'prefill': split['prefill_ms']}
    total = sum(phases.values())
    if total <= 0:
        return {**result, 'bottleneck': 'uncertain', 'confidence': 'unknown'}
    label, value = max(phases.items(), key=lambda kv: kv[1])
    signal['phase_share'] = round(value / total, 3)
    if value / total >= MIN_SHARE:
        return {**result, 'bottleneck': label, 'confidence': 'evidenced'}
    return {**result, 'bottleneck': label, 'confidence': 'suspected'}


def diagnose_sweep(goodput):
    """Per-case knee diagnosis of one sweep plus the binding case (lowest knee)."""
    cases = {name: diagnose_case(c) for name, c in (goodput.get('cases') or {}).items()}
    bound = [(d['knee_rps'], name) for name, d in cases.items() if d['knee_rps'] is not None]
    binding = min(bound)[1] if bound else None
    return {'cases': cases, 'binding_case': binding,
            'bottleneck': cases[binding]['bottleneck'] if binding else 'none'}


def diagnose_trials(goodputs):
    """Majority knee diagnosis over repeated sweeps of one side (each repeat is diagnosed independently)."""
    per = [diagnose_sweep(g) for g in goodputs if g]
    if not per:
        return None
    out = {}
    for name in per[0]['cases']:
        labels = [p['cases'][name]['bottleneck'] for p in per if name in p['cases']]
        top = max(set(labels), key=labels.count)
        agree = labels.count(top) / len(labels)
        example = next(p['cases'][name] for p in per if p['cases'].get(name, {}).get('bottleneck') == top)
        knees = [p['cases'][name]['knee_rps'] for p in per if name in p['cases']]
        out[name] = {'bottleneck': top, 'agreement': round(agree, 2),
                     'confidence': example['confidence'] if agree >= 0.66 else 'suspected',
                     'knee_rps': knees, 'signal': example['signal']}
    bound = [(min([k for k in d['knee_rps'] if k is not None], default=float('inf')), n) for n, d in out.items()]
    bound = [b for b in bound if b[0] != float('inf')]
    binding = min(bound)[1] if bound else None
    return {'cases': out, 'binding_case': binding, 'bottleneck': out[binding]['bottleneck'] if binding else 'none',
            'repeats': len(per)}


def kernel_family(name):
    if name.strip() in REPO_KERNELS:
        return REPO_KERNELS[name.strip()]
    low = name.lower()
    for family, pattern in KERNEL_FAMILIES:
        if re.search(pattern, low):
            return family
    return 'other'


def summarize_kernels(decode_step):
    """Kernel-family shares from scripts/decode_step_profile.py's decode_step.json."""
    tables = {}
    for rows, table in (decode_step.get('kernels') or {}).items():
        fam = {}
        for name, ms in table.get('top_kernels_ms') or []:
            fam[kernel_family(name)] = fam.get(kernel_family(name), 0.0) + ms
        total = sum(fam.values()) or 1.0
        tables[str(rows)] = {'eager_gpu_ms': round(table.get('eager_gpu_ms_sum', total), 3),
                             'shares': {f: round(v / total, 3) for f, v in sorted(fam.items(), key=lambda kv: -kv[1])},
                             'top': [{'family': kernel_family(n), 'kernel': n[:80], 'ms': round(ms, 3)}
                                     for n, ms in (table.get('top_kernels_ms') or [])[:5]]}
    scaling = [{k: r[k] for k in ('rows', 'position', 'graph_ms', 'ms_per_row') if k in r}
               for r in decode_step.get('scaling') or []]
    dominant = None
    if tables:
        largest = tables[max(tables, key=lambda r: int(r))]
        fam, share = next(iter(largest['shares'].items()))
        dominant = {'family': fam, 'share': share, 'rows': int(max(tables, key=lambda r: int(r)))}
    meta = decode_step.get('meta') or {}
    return {'gpu': meta.get('gpu'), 'profile': meta.get('profile'), 'by_rows': tables, 'scaling': scaling,
            'dominant': dominant}


# --- storage -----------------------------------------------------------------------------------------------

def record_measurement(repository, deployment_id, kind, summary, raw=None, *, source=None):
    """Store the latest `kind` measurement for a deployment (revisions keep history). Returns the body."""
    deployment = repository.read('deployment', deployment_id)
    raw_sha = repository.artifact(json.dumps(raw, sort_keys=True, default=str).encode())['sha256'] \
        if raw is not None else None
    body = {'deployment_id': deployment_id, 'deployment_revision': deployment['revision'], 'kind': kind,
            'hardware_ref': deployment['body'].get('hardware_ref'), 'measured_at': time.time(),
            'summary': summary, 'raw_sha256': raw_sha, 'source': source}
    key = f'{deployment_id}--{kind}'
    try:
        current = repository.read('measurement', key)['revision']
    except KeyError:
        current = None
    repository.put('measurement', key, body, expected_revision=current)
    return body


def latest_measurements(repository, deployment_id):
    """{kind: body} of the latest measurements for this deployment; `stale` when made on another revision."""
    try:
        current = repository.read('deployment', deployment_id)['revision']
    except KeyError:
        return {}
    out = {}
    for kind in ('knee', 'kernels'):
        try:
            body = repository.read('measurement', f'{deployment_id}--{kind}')['body']
        except KeyError:
            continue
        out[kind] = {**body, 'stale': body.get('deployment_revision') != current}
    return out


def bottleneck_terms(measurements):
    """Words a hypothesis may use to name a measured bottleneck (knee labels and dominant kernel families)."""
    terms = set()
    knee = (measurements.get('knee') or {}).get('summary') or {}
    for d in (knee.get('cases') or {}).values():
        if d.get('bottleneck') not in (None, 'none', 'uncertain'):
            terms.add(d['bottleneck'])
    kernels = (measurements.get('kernels') or {}).get('summary') or {}
    for table in (kernels.get('by_rows') or {}).values():
        for fam, share in (table.get('shares') or {}).items():
            if share >= 0.15 and fam != 'other':
                terms.add(fam)
    return terms


def names_bottleneck(hypothesis, terms):
    text = (hypothesis or '').lower().replace('-', '_')
    spaced = text.replace('_', ' ')
    return any(t in text or t.replace('_', ' ') in spaced for t in terms)
