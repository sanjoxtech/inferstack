# DeepSeek Ultrafast: what applies to InferStack Native on one L4 — 15 November 2026

Follow-up to docs/research/DEEPSEEK_ULTRAFAST_CODE_STUDY_2026-11-15.md. That study maps the whole
project. This note asks a narrower question: which of its ideas move *our* scorecard
(Qwen3.5-2B, one L4, c=1..16)? The source is a shallow clone of `basetenlabs/deepseek-ultrafast`
at `76766a9` (14 Nov 2026), studied outside the repo. It is Elastic License 2.0: ideas only, no
code copied.

## What the engine is, measured against our problem

- **Target hardware is Blackwell only** (`sm_100a`/`sm_103a`, CUDA 13, TP4/TP8). Its kernels
  (crates/kernels/cu, ~40 files), PDL, NVLS communication and RDMA KV transfer do not run on an
  L4 (sm_89).
- **It serves one request at a time.** crates/server/src/engine.rs holds `job: Mutex<Option<Job>>`.
  request_queue.rs prepares up to `MAX_PREPARED_REQUESTS = 6` jobs (tokenize, fetch KV) while one
  executes. It is a single-stream latency engine built around speculative decoding (README:
  5.6 accepted tokens per 2.7–3.3 ms step), not a batched-throughput server.
- So it says nothing about our largest current gap, batched decode at c=8/16
  (V112_SCORECARD_L4_2026-11-15.md). It says a lot about the c=1/c=4 gap, where native is
  ×0.86–0.92 of vLLM.

## Ideas that transfer

### A. Speculative decoding with the model's own MTP head — new, high value

DSUF's decode loop is "draft a block, verify it in one captured graph". Its speedup comes from
accepted tokens per step, not from a faster step.

**Our checkpoint already has a draft head.** The pinned `Qwen/Qwen3.5-2B-Base@b1485b2f` config
sets `mtp_num_hidden_layers: 1`, and the safetensors index holds 15 `mtp.*` tensors:

- `fc`;
- `pre_fc_norm_embedding` and `pre_fc_norm_hidden`;
- one full-attention layer (q/k/v/o with q/k norms), its MLP and norms;
- the final `norm`.

transformers drops them (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`), so InferStack Native
has never loaded them.

Each decode step would:

1. draft k tokens with the MTP layer, which reuses the main model's last hidden state and the
   shared embeddings/lm_head;
2. verify k+1 tokens in one main-model forward;
3. accept the longest greedy-matching prefix.

Output stays token-identical to greedy (lossless), so the comparison with vLLM/SGLang at default
settings stays fair.

Hybrid-model subtlety: the GDN recurrent and conv states advance per token and cannot be "rolled
back" like a KV cache. The verify forward must either:

- keep per-position states (FLA can return intermediate states), or
- re-run the accepted prefix.

This is the central design item. It is also why transformers' cache has a `record_past` mode.

Expected effect: decode at c=1 is weight-bandwidth bound (~15 ms floor; native 17.0 ms, vLLM 15.8).
A verify step of 2–3 tokens costs about one decode step. Every accepted draft token is close to a
free token, so c=1/c=4 decode could move well past both rivals' default configurations. Unknowns
to measure before building:

- the MTP acceptance rate on the scorecard prompts;
- how much a 2–4-token verify forward costs on the static lane.

Metric requirement (DSUF records it, and the code study's DU1 asks for it): report step time,
accepted tokens/step and output tok/s separately.

### B. Prepare/execute overlap — relevant to N5

DSUF overlaps host-side preparation of the next job with GPU execution. Our equivalent is per
decode step:

1. `_accept` copies tokens to the CPU (a synchronisation);
2. Python emits them;
3. only then is the next step launched.

vLLM's async scheduling issues step t+1 before the host has processed step t. N5.0 (static-lane
step vs bandwidth floor) measures how much of the c=16 decode time this accounts for.

### C. Capture the whole step, sampling included

DSUF's spec graph captures draft + verify + sampling. Our static lane replays the model step as a
graph, but sampling runs outside it. Pulling greedy sampling (argmax) into the captured graph
removes per-step launches. This is small but cheap, and it is also needed for A.

### D. Startup capture/warmup as an explicit, logged stage — already adopted

DSUF logs `CudaGraphCaptureStart/End`. InferStack now warms prefill kernels at startup
(`prefill_warmup`, v1.12) after a measured ~15 s cold-start stall. Recording the warmup time in
startup diagnostics follows the same practice.

## Ideas that do not transfer now

| DSUF technique | Why not on this project now |
|---|---|
| Blackwell kernels, FP4 weights/KV, PDL | L4 is sm_89: no FP4 tensor cores, no PDL; Hopper/Blackwell only |
| TP4/TP8, NVLS comm | one GPU |
| RDMA KV sharing + Redis ownership | single server; already listed as a later multi-node milestone (DU3) |
| Rust control plane rewrite | only if post-N5 profiling still shows Python overhead as material (DU4) |
| 1M-context bounded-replay prefill | scorecard context is 8192 |

## Proposed order

1. **N5.0** (running): decode step at batch 1/4/8/16 vs the bandwidth floor, with a kernel table.
   Decides how much of B/C is worth doing inside N5.
2. **N5.1**: mixed prefill+decode forwards (the c=8/16 gap).
3. **N6 (new, from this study)**: MTP speculative decoding.
   - N6.0 measures MTP acceptance offline on the scorecard prompts (load `mtp.*` next to the
     model, greedy draft vs target). Small and cheap; can run alongside N5.
   - Build only if acceptance is high enough that a verify step pays off.

N6.0 can be done in parallel with N5.1, since it touches no serving code.
