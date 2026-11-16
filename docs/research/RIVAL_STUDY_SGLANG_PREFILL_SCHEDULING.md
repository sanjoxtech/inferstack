# Rival study: SGLang prefill scheduling vs ours (prefill_heavy gap), 16 November 2026

Source: SGLang main (Apache-2.0): `arg_groups/fields/schedule.py` and `managers/scheduler.py`,
partly read. Design ideas only, no code copied.

## SGLang defaults

- **Batch choice.** Each scheduler iteration runs either a prefill (extend) batch or a decode
  batch. A waiting prefill is tried first, which costs that iteration's decode step. Mixed
  prefill and decode is off (`enable_mixed_chunk = False`), and `prefill_decode_interval` is
  unset.
- **Admission.** Requests are admitted first-come-first-served under a token budget
  (`max_prefill_tokens = 16384`), with no limit on requests per prefill. A request that does not
  fit is chunked, and its remainder is resumed first next iteration.
  `chunked_prefill_size` is unset in the field defaults. Its L4 value comes from a resolution
  step we did not read, so it is unverified.

## Ours (v1.16)

- **Batch choice.** `decode_priority = true` and `decode_quantum = 1`: each loop runs one
  decode step, then admits every waiting request as one packed prefill (budget 16384 tokens).
  A new request waits at most one decode step.
- **Admission budget.** It is the same as SGLang's (16384).

## Reading

- **Scheduling.** Both engines admit a waiting prompt within about one step. Scheduling order
  alone does not explain SGLang's prefill_heavy c4–c16 lead (TTFT 122–165 vs our 245–257 ms,
  tok/s 160–327 vs 140–243, 1 round).
- **Per-token prefill cost.** At c1 the gap is small (119 vs 134 ms), so the difference grows
  with load. Candidates are long packed prefills that stall decode (prefill_heavy has only 32
  output tokens, so prefill dominates), and our eager long-prompt prefill reaching about 50% of
  ideal (c1 1200 tokens: 100 ms measured, about 48 ms ideal).
- **Next.** Read the per-cell `ttft_breakdown` (queue vs prefill vs first decode) for
  prefill_heavy in the running 3-round scorecard before choosing between
  (a) a faster long-prompt prefill and (b) splitting big packed admissions into chunks
  interleaved with decode.
