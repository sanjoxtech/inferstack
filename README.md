# InferStack

**An open-source LLM inference engine you can read, plus a fair benchmark harness that runs it next to vLLM and SGLang.**

InferStack has a native serving engine (PyTorch + custom Triton kernels, CUDA-graph decode,
continuous batching, prefix caching, MTP speculative decoding) and a scorecard harness. The
harness launches the native engine, vLLM and SGLang one after another on the same GPU, model and
weights, sends each the same workload, and reports which engine is ahead only when the measured
ranges don't overlap.

The goal: a native engine small enough to understand, that matches production engines on every
workload, with every speedup backed by a reproducible scorecard rather than a single lucky run.

> **Status:** `2.0.0rc1`, under active development. Qwen3.5 (hybrid Gated DeltaNet + attention)
> is the supported model family today. See [what's not done yet](#whats-not-done-yet).

---

## Results

Single **NVIDIA L4** on Colab, Qwen3.5-2B-Base, BF16, greedy decoding. Each engine runs 3 rounds of the
same workload, with equal work and 100% request success in every cell.
Rivals: **vLLM 0.33.0** and **SGLang 0.5.23** using their own default fast paths
(CUDA graphs, FlashAttention / FlashInfer, Triton GDN kernels).

Output throughput (tokens/s, higher is better; median of 3 rounds):

| Workload | Concurrency | **InferStack** | SGLang | vLLM | InferStack vs best rival |
|---|---:|---:|---:|---:|---:|
| interactive | 1 | **96.7** | 63.4 | 62.4 | **1.53×** |
| interactive | 16 | **987.9** | 805.1 | 794.0 | **1.23×** |
| moderate | 8 | **573.6** | 434.8 | 421.5 | **1.32×** |
| moderate | 16 | **901.7** | 776.9 | 747.7 | **1.16×** |
| shared prefix | 8 | **333.7** | 289.7 | 295.4 | **1.13×** |
| prefill-heavy | 16 | 236.1 | **323.9** | 290.5 | 0.73× |
| shared prefix | 16 | 496.8 | **556.4** | 531.6 | 0.89× |

- **Ahead of both vLLM and SGLang in 12 of 16 cells.** Margins range from ×1.13 to ×1.55.
- **Single-stream decode is about 10 ms/token**, versus about 16 ms/token for both rivals.
- **Behind in 4 cells, all with heavy prefill under load.** Our decode slows to 61 ms/token while
  long prompts are being prefilled, but SGLang's stays at 21.6 ms/token. This is the current focus (see
  [roadmap](#roadmap)).

Source: [v1.16 full scorecard](docs/verification/V116_SCORECARD_L4_2026-11-16.md). The raw
per-request data and both rivals' conformance results are in
[`docs/verification/evidence/p4-full-l4-3d08aa0-20261116/`](docs/verification/evidence/p4-full-l4-3d08aa0-20261116/).
Later native-only rounds ([v1.18](docs/verification/V118_SINGLE_PASS_SNAPSHOT_L4_2026-11-17.md))
improved cold-prompt TTFT by 10–20%. On another GPU, model or workload your numbers will be
different. Run the scorecard yourself instead of trusting this table.

---

## How it works

```text
             ┌──────────────────────────────────────────────────────────────┐
  client ──► │ OpenAI-compatible API  (/v1/chat/completions, /v1/completions)│
             └───────────────┬──────────────────────────────────────────────┘
                             │
             ┌───────────────▼──────────────┐     managed engines
             │ scheduler                    │ ──► vLLM / SGLang
             │ continuous batching,         │     (same API, for comparison)
             │ decode-first, prefix cache   │
             └───────────────┬──────────────┘
                             │
             ┌───────────────▼──────────────────────────────────────────┐
             │ native runtime (Qwen3.5)                                 │
             │  • static decode lane: CUDA-graph replay over a slot pool│
             │  • packed, graphed prefill + prefill/decode overlap      │
             │  • radix-style prefix snapshots (KV + conv + GDN state)  │
             │  • MTP speculative decoding                              │
             └───────────────┬──────────────────────────────────────────┘
                             │
             ┌───────────────▼──────────────────────────────────────────┐
             │ Triton kernels: fused RMSNorm / add+RMSNorm / gated norm,│
             │ fused MLP gate-up + SiLU, fused GDN projections, grouped │
             │ GDN decode, decode attention, greedy sampler             │
             └──────────────────────────────────────────────────────────┘
```

All the code is under [`src/inferstack/`](src/inferstack/):

| Path | What's there |
|---|---|
| `api/`, `serving/` | FastAPI server, OpenAI-compatible routes, gateway |
| `runtime/` | Native Qwen runtime: continuous batching, static decode lane, speculative decoding |
| `kernels/` | Triton kernels (norm/MLP/GDN fusions, decode attention, FLA state capture) |
| `engines/` | Adapters that launch and talk to vLLM and SGLang (llama.cpp [planned](docs/LLAMA_CPP_INTEGRATION_PLAN.md)) |
| `evaluation/`, `scripts/engine_scorecard.py` | Fair multi-engine scorecard |
| `optimization/`, `control/` | Planner-executor-verifier loop for bounded optimization experiments |

---

## Quickstart

### Run the tests (CPU, no GPU needed)

```bash
git clone https://github.com/sanjoxtech/inferstack
cd inferstack
pip install -e ".[dev]"
pytest -q
```

On CPU the tests check the reference math, config and wiring. The tests that need CUDA and Qwen3.5
skip automatically.

### Serve a model (CUDA GPU)

```bash
pip install -e ".[native]"
inferstack serve \
  --profile configs/profiles/production-v1.18.0.toml \
  --foundry-root examples/model_foundry \
  --model foundry://factorylm-2b@candidate \
  --device cuda:0 --port 8000
```

`foundry://factorylm-2b@candidate` points to a manifest that pins Qwen3.5-2B-Base to an exact
Hugging Face revision. Then use the server like any OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "factorylm-2b", "messages": [{"role": "user", "content": "Explain KV caching in one sentence."}]}'
```

Each file in [`configs/profiles/`](configs/profiles/) is one engine version. Every flag is
commented with what it does and the measurement that justified it.

### Reproduce the scorecard (Colab L4 / A100)

```bash
python scripts/colab_fast_check.py \
  --profiles native_single vllm sglang --rounds 3 --workload default
```

Or open [`docs/notebooks/colab_a100_engine_scorecard.ipynb`](docs/notebooks/colab_a100_engine_scorecard.ipynb).
Scorecard rules ([`scripts/engine_scorecard.py`](scripts/engine_scorecard.py)):

- Every engine gets the same deterministic workload, and the profile order rotates each round.
- All timings are measured on the client side and calculated the same way for every engine.
- A result is called `ahead` or `behind` only with ≥ 3 rounds and non-overlapping ranges.
  Otherwise it's `overlap`.
- Rivals must pass a conformance check first, so we don't benchmark a misconfigured engine.

---

## How changes get accepted

Each optimization starts as a profile flag and has to earn its place:

1. **Correctness:** CUDA numerical equivalence against the PyTorch reference.
2. **Health:** zero request errors and equal work across engines.
3. **Speed:** a full scorecard run on a GPU, compared against the previous kept version.
4. **Write-up:** a dated report in [`docs/verification/`](docs/verification/) with the
   commit SHA, the evidence, and a keep or revert decision. Reverted ideas are recorded too.

There are more than a dozen of these reports so far. They are the best place to learn *why* the engine looks
the way it does.

---

## Roadmap

- [x] Forward fusion: norm, MLP, GDN and QKV Triton kernels
- [x] Static CUDA-graph decode lane, graphed packed prefill
- [x] Radix-style prefix snapshots with single-pass capture
- [x] MTP speculative decoding
- [ ] **Decode under prefill load:** chunked long prefill interleaved with decode (closes the 4 losing cells)
- [ ] Triton/FLA GDN prefill as the default path
- [ ] More model families and GPUs (A100/H100 scorecards in the repo)
- [ ] Send proven ideas upstream to vLLM / SGLang

### What's not done yet

- Only Qwen3.5 has the native fast path. Other models go through managed engines.
- All published numbers are from a single L4. A100 and H100 scorecards are next.
- The v2 control plane (automated optimization, remote GPU providers) is still being built.
  See [V2 build status](docs/V2_BUILD.md).

---

## Docs

- [Architecture](docs/ARCHITECTURE.md) · [Roadmap](docs/ROADMAP.md) · [Current-state audit](docs/CURRENT_STATE_AUDIT.md)
- [Why vLLM and SGLang beat native, and what to change](docs/NATIVE_GAP_ANALYSIS_2026-11-13.md)
- [v1.6 forward-fusion design notes](docs/V1_6_FORWARD_FUSION.md)
- [Inference harness plan](docs/INFERENCE_HARNESS_PLAN.md) · [Control runbook](docs/CONTROL_RUNBOOK.md)

## Contributing

Issues and PRs are welcome, especially for kernels, new model families, and scorecard runs on
other GPUs. A PR that claims a speedup needs a scorecard showing it (≥ 3 rounds). One that touches
kernel math needs a numerical-equivalence test.

## License

[Apache License 2.0](LICENSE). `src/inferstack/kernels/fla_gdr_states.py` mirrors part of
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT).
