"""Static decode lane: the native engine's decode path on the CUDA-graph static decoder.

The engine still prefills with the Transformers modules (chunked, prefix-cached). When a
cohort finishes prefill, its rows are copied into free slots of one fixed decode pool
(runtime/qwen_static_decode.py) and decoded there by graph replay, one step for every
live row together. Rows that do not fit (pool full, or prompt + max_new_tokens longer
than the pool's max_len) stay on the original Transformers decode path, so the lane
never rejects work.

Live rows stay packed in slots 0..n-1: when rows finish, the survivors move down in
order. The lane's batch object is an ActiveBatch without a cache, so the engine's own
token acceptance (_accept) is reused unchanged.

N6.2: with `spec` (a SpecState), every step is the MTP verify+draft graph
(runtime/qwen_spec_decode.py) and emits one or two tokens per row. Only greedy rows are
admitted, since the draft check is an argmax. The MTP history comes from the last prefill
columns' hidden states (engine._spec_hidden_hook).
"""
from __future__ import annotations

from inferstack.core.request import RequestState


class StaticDecodeLane:
    def __init__(self, engine, decoder, state, spec=None):
        import torch
        from inferstack.runtime.qwen_continuous import ActiveBatch
        self.e = engine
        self.dec = decoder
        self.st = state
        self.sp = spec
        self.batch = ActiveBatch([], [], None, None, None, None, [], [], 0, 0)
        self.next_ids = torch.zeros(state.batch, dtype=torch.long, device=decoder.device)
        self.pos = torch.zeros(state.batch, dtype=torch.long, device=decoder.device)
        self.positions: list[int] = []        # host copy of pos for live rows
        self.stats = {'rows_admitted': 0, 'rows_rejected_full': 0, 'rows_rejected_length': 0,
                      'decode_steps': 0, 'max_live_rows': 0, 'rows_rejected_sampling': 0,
                      'spec_steps': 0, 'spec_row_steps': 0, 'spec_accepted': 0}

    @property
    def rows(self) -> int:
        return len(self.batch.requests)

    def try_admit(self, state) -> bool:
        """Move rows of a freshly prefilled ActiveBatch into the pool.

        Rows whose prompt + max_new_tokens fit the pool move while free slots last; the
        rest stay in `state` (Transformers path). Returns True when nothing is left."""
        import torch
        spec = self.sp is not None
        if spec and state.prompt_hidden is None:
            return False
        extra = 2 if spec else 0                   # the spec lane also writes the draft column
        fit = [i for i, r in enumerate(state.requests)
               if len(r.prompt_token_ids) + r.sampling.max_new_tokens + extra <= self.st.max_len]
        self.stats['rows_rejected_length'] += len(state.requests) - len(fit)
        if spec:
            greedy = [i for i in fit if float(state.requests[i].sampling.temperature) <= 0]
            self.stats['rows_rejected_sampling'] += len(fit) - len(greedy)
            fit = greedy
        free = self.st.batch - self.rows
        if len(fit) > free:
            self.stats['rows_rejected_full'] += len(fit) - free
            fit = fit[:free]
        if not fit:
            return False
        base, n = self.rows, len(fit)
        lengths = self.dec.load_rows(self.st, state.past, state.attention_mask, state.physical_width,
                                     fit, range(base, base + n))
        idx = torch.tensor(fit, dtype=torch.long, device=self.pos.device)
        if spec:
            # N6.3: one MTP forward for every admitted row; firsts stay on the GPU.
            hid = state.prompt_hidden
            cols = int(hid.shape[1])
            real = state.attention_mask[:, :state.physical_width][:, -cols:].sum(-1).tolist()
            ws = [max(1, min(int(real[i]), lengths[j], cols)) for j, i in enumerate(fit)]
            self.dec.load_spec_rows(self.st, self.sp, base,
                                    [state.requests[i].prompt_token_ids for i in fit],
                                    [hid[i, -w:] for i, w in zip(fit, ws)],
                                    state.previous_gpu.index_select(0, idx), lengths)
        self.next_ids[base:base + n].copy_(state.previous_gpu.index_select(0, idx))
        self.pos[base:base + n].copy_(torch.tensor(lengths, dtype=torch.long, device=self.pos.device))
        b = self.batch
        b.requests += [state.requests[i] for i in fit]
        b.queues += [state.queues[i] for i in fit]
        b.generated += [state.generated[i] for i in fit]
        b.stop_sets += [state.stop_sets[i] for i in fit]
        self.positions += lengths
        self.stats['rows_admitted'] += n
        self.stats['max_live_rows'] = max(self.stats['max_live_rows'], self.rows)
        rest = [i for i in range(len(state.requests)) if i not in set(fit)]
        if not rest:
            state.requests, state.queues, state.generated, state.stop_sets, state.past = [], [], [], [], None
            return True
        from inferstack.runtime.hybrid_prefix_cache import select_cache_rows
        keep = torch.tensor(rest, dtype=torch.long, device=self.pos.device)
        state.past = select_cache_rows(state.past, keep)
        state.attention_mask = state.attention_mask.index_select(0, keep)
        state.previous_gpu = state.previous_gpu.index_select(0, keep)
        state.decode_positions = state.decode_positions.index_select(0, keep)
        if state.prompt_hidden is not None:
            state.prompt_hidden = state.prompt_hidden.index_select(0, keep)
        state.requests = [state.requests[i] for i in rest]
        state.queues = [state.queues[i] for i in rest]
        state.generated = [state.generated[i] for i in rest]
        state.stop_sets = [state.stop_sets[i] for i in rest]
        return False

    def _sample(self, logits):
        import torch
        b, sampler = self.batch, self.e._sampler
        if all(float(r.sampling.temperature) <= 0 for r in b.requests):
            return sampler.sample(logits, b.requests, 0)
        # Per-row step keeps each request's seeded stream identical to the cohort path.
        return torch.cat([sampler.sample(logits[i:i + 1], [r], g)
                          for i, (r, g) in enumerate(zip(b.requests, b.generated))])

    def decode(self):
        """One decode step for every live row (runs on the engine's GPU thread)."""
        self.compact()
        n = self.rows
        if not n:
            return
        if self.sp is not None:
            self._spec_decode(n)
            return
        logits = self.dec.step(self.st, self.next_ids[:n], self.pos[:n], n, length=max(self.positions) + 1)
        nxt = self._sample(logits)
        self.next_ids[:n].copy_(nxt)
        self.pos[:n] += 1
        self.positions = [p + 1 for p in self.positions]
        self.stats['decode_steps'] += 1
        self.e._stats['decode_forward_calls'] += 1
        self.e._accept(self.batch, nxt)
        self.compact()

    def _spec_decode(self, n):
        y, acc = self.dec.spec_step(self.st, self.sp, n, length=max(self.positions) + 2)
        yh, ah = y.tolist(), acc.tolist()            # the step's only host sync
        e = self.e
        e._accept_host(self.batch, [r[0] for r in yh])
        e._accept_host(self.batch, [r[1] if a else None for r, a in zip(yh, ah)])
        self.positions = [p + 1 + int(a) for p, a in zip(self.positions, ah)]
        k = sum(map(bool, ah))
        self.stats['decode_steps'] += 1
        self.stats['spec_steps'] += 1
        self.stats['spec_row_steps'] += n
        self.stats['spec_accepted'] += k
        e._stats['decode_forward_calls'] += 1
        self.compact()

    def compact(self):
        """Drop finished/cancelled rows and pack survivors into slots 0..k-1 (order kept)."""
        import torch
        b = self.batch
        live = [i for i, r in enumerate(b.requests) if r.state == RequestState.RUNNING]
        if len(live) == len(b.requests):
            return
        for i, r in enumerate(b.requests):
            if i not in live and r.state == RequestState.CANCELLED and r.request_id in self.e._queues:
                self.e._emit(b.queues[i], None)
                self.e._requests.pop(r.request_id, None)
                self.e._queues.pop(r.request_id, None)
        for j, src in enumerate(live):
            if src != j:
                if self.sp is not None:
                    self.dec.move_spec_row(self.st, self.sp, src, j, self.positions[src] + 2)
                else:
                    self.dec.move_row(self.st, src, j, self.positions[src] + 1)
        if live:
            idx = torch.tensor(live, dtype=torch.long, device=self.pos.device)
            k = len(live)
            self.next_ids[:k] = self.next_ids.index_select(0, idx)
            self.pos[:k] = self.pos.index_select(0, idx)
        self.e._stats['compactions'] += 1
        self.e._stats['rows_compacted'] += len(b.requests) - len(live)
        b.requests = [b.requests[i] for i in live]
        b.queues = [b.queues[i] for i in live]
        b.generated = [b.generated[i] for i in live]
        b.stop_sets = [b.stop_sets[i] for i in live]
        self.positions = [self.positions[i] for i in live]

    def clear(self):
        b = self.batch
        b.requests, b.queues, b.generated, b.stop_sets = [], [], [], []
        self.positions = []

    def diagnostics(self):
        st = self.st
        return {**self.stats, 'live_rows': self.rows, 'slots': st.batch, 'max_len': st.max_len,
                'batch_buckets': list(st.batch_buckets), 'len_buckets': list(st.len_buckets),
                'decoder': dict(self.dec.stats), 'graphs': len(self.dec.graphs) + len(getattr(self.dec, 'spec_graphs', {})),
                'spec': self.sp is not None,
                'spec_acceptance': (self.stats['spec_accepted'] / self.stats['spec_row_steps'])
                if self.stats['spec_row_steps'] else None}
