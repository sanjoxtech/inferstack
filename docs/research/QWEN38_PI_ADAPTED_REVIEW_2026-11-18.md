# Review: Qwen3.8-27B-pi-adapted (Hugging Face harness-hub), 2026-11-18

Sources read: model card https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted, blog
https://huggingface.co/blog/harness-hub/harness-adapters (6 Nov 2026), org page
https://huggingface.co/harness-hub. The release blog with the full recipe
(huggingface.co/blog/harness-hub/harness-adapters-at-scale) returned HTTP 401 and was not read.
Announcement: post by @ClementDelangue. Numbers below are as stated by the authors, not re-measured.

## Claim

A LoRA adapter trained with reinforcement learning makes Qwen3.8-27B better at driving the pi coding
agent: SWE-bench Pro (hard 51) 29.4% -> 62.7%, Terminal-Bench 2.1 39.5% -> 55.8%, 55-62% fewer tool
calls per task (pass@1).

## Method (as stated)

- Base Qwen/Qwen3.8-27B (BF16); LoRA r=32, alpha=64 on q, k, v, o, gate, up, down; LR 2e-5. Apache-2.0.
- RL: AsyncGRPO (TRL, experimental), 8 rollouts per task, temperature 0.8, thinking off, up to 50 agent
  steps and 1,800 s per rollout; 16 micro-batches of up to 65,536 tokens per step; rollouts up to 16
  policy versions stale. Checkpoint step 160 of a run stopped at 191 of 1,000 planned.
- Reward: task verifier correctness + tool-efficiency bonus (weight 0.1). In the blog's other runs the
  bonus is 0.1 x (1 - calls/15) on correct rollouts only, and any rollout with a safety-guard block
  notice scores 0.
- Data: 750 scale800 + 300 TermGrade tasks (Harbor format, served by OpenEnv, rollouts in HF sandboxes,
  tool calls screened by the safeagents reverse proxy). The harness (pi) runs unmodified; training uses
  exactly the tokens the harness saw.
- Compute: 2x NVIDIA B200 per harness (vLLM on one, trainer on the other). Total GPU-hours not stated.
- Not stated: KL coefficient, loss variant, optimizer/schedule, seeds, contamination controls.

## How much to trust the numbers

| result | trust | why |
|---|---|---|
| SWE-bench Pro hard 51: 29.4 -> 62.7% (+33.3 +/- 14.2) | low | checkpoint selected on this same benchmark; 51 tasks, one attempt each |
| Terminal-Bench 2.1: 39.5 -> 55.8% | medium | an 18-task subset was used for selection |
| Terminal-Bench 2.1, 68 tasks not used for selection: 39.7 -> 55.9% (+16.2 +/- 10.6, sign test p=0.0074) | good | the only held-out view |
| 55-62% fewer tool calls | good | consistent across both benchmarks |

Honest reading: a real gain of about +16 points on held-out tasks with roughly half the tool calls; the
+34 headline is inflated by checkpoint selection. Card inconsistencies: "89 tasks" in prose vs 86 in the
table; serve/merge commands reference repo names that differ from the published repo.

## What it means for InferStack

- Agent RL is a serving workload: thousands of long multi-turn sessions that keep re-reading their own
  history. The card's vLLM setup needs `--enable-prefix-caching --mamba-cache-mode align` for this hybrid
  model; without it "every agent turn re-reads its whole history". Our P5 prefix cache with single-pass
  GDN snapshots solves the same problem for Qwen3.5 hybrids.
- Multi-LoRA serving by name (one base model, adapters swapped per request) is part of their training
  loop; InferStack does not serve LoRA adapters yet. Candidate roadmap item if we target RL rollouts.
- Their open Harbor/OpenEnv tasks could become an "agentic multi-turn" scorecard workload (long shared
  history, many short turns), which our current four cases do not cover.
- Training recipe (RL + LoRA + efficiency reward) belongs to Model Foundry, not this repo.
