# Colab L4 engine scorecard (N0/N1) — 13 November 2026

First complete run of `docs/notebooks/colab_a100_engine_scorecard.ipynb`.

- Evidence archive: `inferstack-colab-l4-evidence-20261113T061946Z.zip` (user download, raw
  files not committed), sha256 `901c0cc9bc7c2cc31f15092864dca780ce43016c25478e114e87fe4b7d7f154c`.
- Committed extracts: `evidence/colab-l4-scorecard-787c567-20261113.json` (scorecard
  summary, every cell with median/min/max per metric) and `…manifest.json` (run manifest).
- Source pinned at `787c567` (clean). Run id `l4-1794553433`, workload sha256 `9621fc72…`.
- Hardware: 1 × NVIDIA L4 (23034 MiB), driver 580.82.07, Python 3.13.18.
- Model: `Qwen/Qwen3.5-2B-Base@b1485b2f`, BF16, 8192 context, every engine.
- Engines: native (torch 2.14.1+cu130, Triton 3.8.0), vLLM 0.33.0, SGLang 0.5.23
  (Python 3.12 venv). Engine defaults otherwise (prefix/radix cache, CUDA graphs as shipped).

## Outcome

- Stages: native_unfused, native_full, vLLM and SGLang all **passed** install, conformance and smoke.
- Waiver: vLLM `metric_families_present` (missing `engine_tpot_ms`) only, recorded in
  `waivers.json`. The fix in `88a7d22` is not in the pinned commit; latency is measured
  client-side, so the scorecard is unaffected.
- Fair run: 3 rounds, rotated order, every profile ready in every round,
  12 × 480 = 5760 scored requests, 0 failed. `unsupported_or_failed` is empty.

## Results (median of 3 rounds; output tokens/s ratio vs `native_unfused`)

| case | native_full | vLLM | SGLang |
|---|---|---|---|
| interactive (c=1…16) | 1.33–1.52× | 2.53–2.80× | 2.56–2.83× |
| moderate (c=1…16) | 1.30–1.52× | 2.62–2.92× | 2.72–2.96× |
| prefill_heavy (c=1…16) | 1.15–1.32× | 2.98–4.19× | 3.15–4.69× |
| shared_prefix (c=1…16) | 1.25–1.48× | 2.70–2.98× | 2.84–2.97× |

Client decode ms/token at c=1: native_unfused ~45, native_full ~29–30, vLLM ~16, SGLang ~16.

Largest measured native gap: **prefill**. prefill_heavy TTFT p50 at c=16: native_unfused
4074 ms, native_full 3626 ms, vLLM 466 ms, SGLang 162 ms. Native decode also degrades
with concurrency on this case (107 ms/token at c=16 vs 22 for SGLang).

Round-to-round ranges are narrow (most within a few percent), so the relations above
meet the scorecard's ≥3-round, non-overlapping-range rule.

## Caveats

- L4 only. Not comparable with A100 numbers; no A100 run of this notebook exists yet.
- `shared_prefix` c=1 native_full has `equal_work = False`: mean output 44.5 vs 41.5
  baseline tokens (>5%), consistent with the known native_full greedy-output divergence.
  Its ratio there is not a like-for-like comparison. All other cells are equal-work.
- Peak memory (native ~7.4–8.0 GiB, vLLM ~9.9 GiB, SGLang ~16.7 GiB) reflects each
  engine's preallocation policy, not need.
- vLLM profiled its unused vision encoder at startup (see 2026-11-11 log); text-only
  launch flags were not applied, so vLLM startup time is pessimistic. Steady-state
  serving numbers are unaffected.
- Measurement only: no quality evaluation, not a release decision.
