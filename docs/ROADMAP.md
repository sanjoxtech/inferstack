# InferStack roadmap (2026-11-17)

Goal: the most efficient engine for the models we serve, judged by SLO v2
(`configs/tune/slo-v2-l4.json`): per cell, TTFT p95 <= min(industry class limit, best of vLLM /
SGLang), TTFT p99 and per-token p95 <= industry limits (MLPerf / DistServe), throughput >= best
rival. Today (v1.18, L4, Qwen3.5-2B): 9 of 16 cells met.

| phase | what | status | exit test |
|---|---|---|---|
| P5 | prefix cache (first-sight, single-pass snapshots) | done, v1.18 | kept on 3-round A/B |
| P6 | host-time cuts (tail-fill compaction), SLO tuner (Optuna), SLO v1/v2 | done; tail-fill held (TTFT trade) | tuner: settings max out at 9/16 |
| **P7** | **prefill/decode overlap**: long prompts read on a low-priority stream into spare slots while decode keeps running on a high-priority stream | v1 (2 slots, 1 job, admission blocked): decode p95 -40% but TTFT x4, not kept; v2 building (6 slots, multi-row jobs, short prompts admitted during a job) | SLO v2 cells met > 9, no cell behind v1.18 on tok/s |
| P8 | close the remaining c1 TTFT gaps (moderate c1 77 vs 47 ms: larger prefill graph buckets, fused GDN gating) | planned | moderate/prefill_heavy c1 TTFT p95 <= rival |
| P9 | goodput + cost: open-loop Poisson arrivals, max req/s at >= 99% SLO attainment (MLPerf/vLLM style), and **$ per 1,000 requests at the SLO** (GPU $/h / goodput) for us, vLLM, SGLang | planned | one cost-at-SLO table per GPU, rivals re-measured |
| P10 | kernel loop first: study RightNow-AI/autokernel (MIT, edit-benchmark-keep/revert for Triton kernels) and try it on one InferStack kernel (GQA decode, GDN gating or sampler) under our L4 A/B + correctness gates; then the gated code agent (phase 3): OpenRouter/DeepSeek model proposes one change at a time; tests + honest A/B + reward-hack breaker; opens PRs | planned | catches a planted cheat; one accepted PR |
| P11 | shared improve engine for Model Foundry and FactoryOS; adapter layer for new models; auto-tune per (model, GPU) | planned | new model/GPU tuned without hand edits |

Rules that stay: rival study before each change, A/B over 3 rounds with non-overlapping ranges,
ask before any GPU campaign over $1, never touch sessions we did not create, push only on "push".
