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
                      'spec_steps': 0, 'spec_row_steps': 0, 'spec_accepted': 0, 'spec_step_ms': 0.0,
                      'packed_prefills': 0, 'packed_rows': 0, 'packed_prefix_hits': 0,
                      'packed_prefix_groups': 0, 'packed_prefix_shared_rows': 0, 'packed_prefix_single_pass': 0, 'spec_host_ms': 0.0, 'gpu_thread_ms': 0.0, 'gpu_thread_calls': 0, 'packed_tokens_saved': 0, 'packed_ms': 0.0, 'spec_flush_ms': 0.0, 'spec_pipelined_steps': 0}
        n = engine.cfg.native
        self.packed_graphs = bool(getattr(n, 'static_decode_packed_graphs', False))
        self.store_first_sight = bool(getattr(n, 'prefix_cache_store_first_sight', False))
        self.single_pass_snapshot = bool(getattr(n, 'prefix_cache_single_pass_snapshot', False))
        self.compact_from_tail = bool(getattr(n, 'static_decode_compact_from_tail', False))
        self.admit_timing = bool(getattr(n, 'static_decode_admit_timing', False))   # P8 diag: syncs per stage
        self.budget_new_tokens = bool(getattr(n, 'static_decode_budget_new_tokens', False))   # P9
        # P9b: rows that fit the lane wait in the queue while every slot is taken, instead of
        # running on the Transformers path beside the lane (open-loop goodput: that side cohort
        # slowed every decode step from ~12 to ~190 ms per token at 6 req/s)
        self.wait_for_slot = bool(getattr(n, 'static_decode_wait_for_slot', False))
        # P9f: launch spec step N+1 before the host work (accept, emit, compact) of step N, so the
        # GPU runs while the host works. Tokens reach clients one step later; a row that finished
        # in step N runs one extra (discarded) step, so it needs 2 more columns.
        self.pipeline = bool(getattr(n, 'static_decode_spec_pipeline', False)) and spec is not None
        self.margin = 4 if self.pipeline else 2       # columns past prompt + max_new_tokens (spec)
        self._pending = None                        # (tokens, accepts) of a synced step not yet accepted
        # P7: overlap long packed prefills with decode. New rows prefill into spare top slots
        # (>= self.top, never touched by decode graphs) on a low-priority stream from a second
        # thread while decode keeps replaying on a high-priority stream; finished rows move down.
        extra = int(getattr(n, 'static_decode_overlap_slots', 0) or 0)
        self.overlap = bool(getattr(n, 'static_decode_overlap_prefill', False)) and extra > 0
        self.overlap_min = int(getattr(n, 'static_decode_overlap_min_tokens', 512) or 512)
        self.top = state.batch - extra if self.overlap else state.batch
        self._job = None
        self.deferred = []
        self.long_queue = []                       # P7 v2: long prompts waiting for the next overlap job
        self._hi = self._lo = None
        if self.overlap:
            from concurrent.futures import ThreadPoolExecutor
            self._prefill_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='inferstack-prefill')
            if decoder.device.type == 'cuda':
                self._hi = torch.cuda.Stream(device=decoder.device, priority=-1)
                self._lo = torch.cuda.Stream(device=decoder.device, priority=0)
                torch.cuda.set_stream(self._hi)        # the engine's GPU thread: decode runs high priority
        self.pcache = None
        if bool(getattr(n, 'static_decode_packed_prefill', False)) and bool(getattr(n, 'prefix_state_cache', False)):
            from inferstack.runtime.pool_prefix_cache import PoolPrefixCache
            self.pcache = PoolPrefixCache(n.prefix_state_cache_max_entries, n.prefix_cache_min_tokens,
                                          n.prefix_cache_block_tokens, int(n.prefix_state_cache_max_mb) * 2 ** 20)
        from collections import deque
        self._recent = deque(maxlen=64)            # recent packed prompts: a prefix seen twice gets stored

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
        extra = self.margin if spec else 0         # the spec lane also writes the draft column
        fit = [i for i, r in enumerate(state.requests)
               if len(r.prompt_token_ids) + r.sampling.max_new_tokens + extra <= self.st.max_len]
        self.stats['rows_rejected_length'] += len(state.requests) - len(fit)
        if spec:
            greedy = [i for i in fit if float(state.requests[i].sampling.temperature) <= 0]
            self.stats['rows_rejected_sampling'] += len(fit) - len(greedy)
            fit = greedy
        free = self.top - self.rows - self.reserved
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

    def full(self):
        """P9b: no slot free for a new row (the overlap path manages its own reserved slots)."""
        return not self.overlap and self.top - self.rows - self.reserved <= 0

    def admit_packed(self, items):
        """P1: prefill (request, queue) pairs as one flat token batch straight into free slots
        (QwenStaticDecoder.prefill_packed) and emit their first tokens. Returns the pairs that
        must take the Transformers prefill path instead: no free slot, prompt + max_new_tokens
        longer than the pool, over the packed token budget, or sampled rows under spec.

        P2 (pool prefix cache): a row whose prompt starts with a cached block-aligned prefix
        gets that state copied into its slot and only computes the rest. Rows without a hit
        that are long enough have their prefix up to a block boundary computed first (once per
        group of rows sharing it), snapshotted, then continued."""
        import time
        import torch
        e = self.e
        spec = self.sp is not None
        extra = self.margin if spec else 0
        budget = int(getattr(e.cfg.native, 'static_decode_packed_max_tokens', 8192) or 8192)
        pc = e._prefix_cache
        free = self.top - self.rows - self.reserved
        if self.overlap and (self.rows or self._job is not None):
            # P7 v2: long prompts go to the overlap job (or wait for the next one); short ones
            # keep the normal synchronous path below, into slots not reserved for the job, but only
            # when no job is running: Triton's autotuner (FLA kernels) is not thread-safe, so two
            # threads must never prefill at once (crash seen on L4, run b1af725).
            busy = self._job is not None
            longs = [(r, q) for r, q in items if busy or len(r.prompt_token_ids) >= self.overlap_min]
            items = [(r, q) for r, q in items if not busy and len(r.prompt_token_ids) < self.overlap_min]
            self.long_queue.extend(longs)
            if self._job is None and self.long_queue:
                self._start_overlap()
                free = self.top - self.rows - self.reserved
            if not items:
                return []
        take, rest, T = [], [], 0
        for r, q in items:
            n = len(r.prompt_token_ids)
            # P9: only uncached tokens count toward the packed budget (a cached shared prefix is restored,
            # not recomputed); counting full prompts split a 16-row shared-prefix burst into 6+7+3 prefills
            new = n - (self.pcache.peek(r.prompt_token_ids) if self.budget_new_tokens and self.pcache is not None
                       and self.pcache.enabled else 0)
            fits = n + r.sampling.max_new_tokens + extra <= self.st.max_len and not (spec and float(r.sampling.temperature) > 0)
            if fits and len(take) < free and T + new <= budget:
                take.append((r, q))
                T += new
            elif fits and ((take and self.budget_new_tokens) or (self.wait_for_slot and len(take) >= free)):
                # P9: over this prefill's token budget or slots, but servable here: re-queue for the next
                # prefill instead of sending it down the Transformers path
                self.deferred.append((r, q))
            else:
                rest.append((r, q))
        if not take:
            return rest
        rs, qs = [x[0] for x in take], [x[1] for x in take]
        now = time.perf_counter()
        t_admit = now
        st = e._stats
        for r in rs:
            r.state = RequestState.RUNNING
            r._t_prep = r._t_pf = now
            r._ds0 = st['decode_forward_calls']
            r._pf0 = r._pf1 = st['prefill_forward_calls']
        base, k = self.rows, len(rs)
        lengths = [len(r.prompt_token_ids) for r in rs]
        prompts = [r.prompt_token_ids for r in rs]
        slots = list(range(base, base + k))
        tm = self.admit_timing and self.dec.device.type == 'cuda'

        def lap(name, t=[time.perf_counter()]):   # P8 diag: per-stage wall time (synchronised)
            if tm:
                torch.cuda.synchronize()
                now2 = time.perf_counter()
                self.stats[name] = self.stats.get(name, 0.0) + (now2 - t[0]) * 1000
                t[0] = now2
        lap('admit_setup_ms')
        with torch.inference_mode():
            starts, snap = (self._prefix_plan(slots, prompts) if self.pcache is not None and self.pcache.enabled
                            else (None, None))
            lap('admit_plan_ms')
            out = None
            if snap:   # P5: cold prompts leave their snapshot from this single pass
                so = {}
                if self.packed_graphs:   # P8: long graphs record the snapshot inside the graph
                    out = self.dec.prefill_graphed(self.st, slots, prompts, starts=starts, snap=snap, snap_out=so)
                if out is None:
                    out = self.dec.prefill_packed(self.st, slots, prompts, starts=starts, snap=snap, snap_out=so)
                for r, j in enumerate(so['rows']):
                    self.pcache.put(prompts[j][:snap[j]], self.st, slots[j],
                                    conv={i: c[r] for i, c in so['conv'].items()},
                                    rec={i: x[r] for i, x in so['rec'].items()})
                    self.stats['packed_prefix_single_pass'] += 1
            elif self.packed_graphs:
                out = self.dec.prefill_graphed(self.st, slots, prompts, starts=starts)
            if out is None:
                out = self.dec.prefill_packed(self.st, slots, prompts, starts=starts)
            logits, hid, cu = out
            lap('admit_prefill_ms')
            nxt = e._sampler.sample(logits, rs, 0)
            lap('admit_sample_ms')
            if spec:
                win = int(getattr(e.cfg.native, 'static_decode_spec_window', 256) or 256)
                ws = [min(cu[j + 1] - cu[j], win) for j in range(k)]
                self.dec.load_spec_rows(self.st, self.sp, base, prompts,
                                        [hid[cu[j + 1] - w:cu[j + 1]] for j, w in enumerate(ws)], nxt, lengths)
                lap('admit_spec_ms')
        self.next_ids[base:base + k].copy_(nxt)
        self.pos[base:base + k].copy_(torch.tensor(lengths, dtype=torch.long, device=self.pos.device))
        b = self.batch
        b.requests += rs
        b.queues += qs
        b.generated += [0] * k
        b.stop_sets += [e._stop_ids(r) for r in rs]
        self.positions += lengths
        for r in rs:
            r.prompt_cursor = len(r.prompt_token_ids)
        st['prefill_forward_calls'] += 1
        st['prefill_chunks'] += 1
        st['prefill_tokens'] += cu[-1]
        self.stats['rows_admitted'] += k
        self.stats['packed_rows'] += k
        self.stats['packed_prefills'] += 1
        self.stats['max_live_rows'] = max(self.stats['max_live_rows'], self.rows)
        firsts = nxt.tolist()                         # host sync: the packed prefill is done here
        self.stats['packed_ms'] += (time.perf_counter() - t_admit) * 1000
        e._accept_host(b, [None] * base + firsts)
        self.compact()
        lap('admit_finish_ms')
        return rest

    # ---- P7: prefill/decode overlap -------------------------------------------------------
    def overlap_busy(self) -> bool:
        return self._job is not None or bool(self.long_queue)

    @property
    def reserved(self) -> int:
        """Slots below `top` promised to the in-flight overlap job's rows."""
        return len(self._job[0]) if self._job is not None else 0

    def _start_overlap(self):
        """Launch the packed prefill of up to `spare` queued long prompts into the top slots on
        the prefill thread. Prompts that cannot fit the pool after the job lands, are not greedy,
        or exceed the packed token budget wait (non-greedy ones go back to the engine)."""
        import time
        spare = self.st.batch - self.top
        budget = int(getattr(self.e.cfg.native, 'static_decode_packed_max_tokens', 8192) or 8192)
        room = self.top - self.rows
        extra = self.margin if self.sp is not None else 0
        go, keep, T = [], [], 0
        for r, q in self.long_queue:
            n = len(r.prompt_token_ids)
            if float(r.sampling.temperature) > 0 or n + r.sampling.max_new_tokens + extra > self.st.max_len:
                self.deferred.append((r, q))
            elif len(go) < min(spare, room) and T + n <= budget:
                go.append((r, q))
                T += n
            else:
                keep.append((r, q))
        self.long_queue = keep
        if not go:
            return
        e, st = self.e, self.e._stats
        now = time.perf_counter()
        for r, _ in go:
            r.state = RequestState.RUNNING
            r._t_prep = r._t_pf = now
            r._ds0 = st['decode_forward_calls']
            r._pf0 = r._pf1 = st['prefill_forward_calls']
        slots = list(range(self.top, self.top + len(go)))
        self._job = (go, slots, now, self._prefill_pool.submit(self._overlap_prefill, go, slots))
        self.stats['overlap_jobs'] = self.stats.get('overlap_jobs', 0) + 1
        self.stats['overlap_rows'] = self.stats.get('overlap_rows', 0) + len(go)

    def _overlap_prefill(self, go, slots):
        """Prefill thread: same work as admit_packed's prefill, into the top slots, on the
        low-priority stream. Touches only those slots (pool tensors, prefix cache, spec rows)."""
        import contextlib
        import torch
        rs = [r for r, _ in go]
        prompts = [r.prompt_token_ids for r in rs]
        lengths = [len(p) for p in prompts]
        ctx = torch.cuda.stream(self._lo) if self._lo is not None else contextlib.nullcontext()
        with torch.inference_mode(), ctx:
            starts, snap = (self._prefix_plan(slots, prompts) if self.pcache is not None and self.pcache.enabled
                            else (None, None))
            if snap:
                so = {}
                logits, hid, cu = self.dec.prefill_packed(self.st, slots, prompts, starts=starts, snap=snap, snap_out=so)
                for r, j in enumerate(so['rows']):
                    self.pcache.put(prompts[j][:snap[j]], self.st, slots[j],
                                    conv={i: c[r] for i, c in so['conv'].items()},
                                    rec={i: x[r] for i, x in so['rec'].items()})
                    self.stats['packed_prefix_single_pass'] += 1
            else:
                logits, hid, cu = self.dec.prefill_packed(self.st, slots, prompts, starts=starts)
            nxt = logits.argmax(-1)                      # overlap rows are greedy
            if self.sp is not None:
                win = int(getattr(self.e.cfg.native, 'static_decode_spec_window', 256) or 256)
                ws = [min(cu[j + 1] - cu[j], win) for j in range(len(rs))]
                self.dec.load_spec_rows(self.st, self.sp, slots[0], prompts,
                                        [hid[cu[j + 1] - w:cu[j + 1]] for j, w in enumerate(ws)], nxt, lengths)
            ev = None
            if self._lo is not None:
                ev = torch.cuda.Event()
                ev.record(self._lo)
            firsts = nxt.tolist()                        # syncs this (low-priority) stream only
        return {'nxt': nxt, 'firsts': firsts, 'lengths': lengths, 'tokens': cu[-1], 'event': ev}

    def poll_overlap(self, wait=False):
        """Between decode steps: if the overlap prefill finished (or `wait`), move its rows down
        to slots rows..rows+k-1 and emit their first tokens. True when rows were added."""
        import time
        import torch
        if self._job is None or not (wait or self._job[3].done()):
            return False
        go, slots, t0, fut = self._job
        self._job = None
        res = fut.result()                               # re-raises a prefill failure here
        if res['event'] is not None:
            torch.cuda.current_stream().wait_event(res['event'])
            res['nxt'].record_stream(torch.cuda.current_stream())
        e, st = self.e, self.e._stats
        rs, qs = [r for r, _ in go], [q for _, q in go]
        base, k = self.rows, len(go)
        for j, (src, n) in enumerate(zip(slots, res['lengths'])):
            if self.sp is not None:
                self.dec.move_spec_row(self.st, self.sp, src, base + j, n + 2)
            else:
                self.dec.move_row(self.st, src, base + j, n + 1)
        self.next_ids[base:base + k].copy_(res['nxt'])
        self.pos[base:base + k].copy_(torch.tensor(res['lengths'], dtype=torch.long, device=self.pos.device))
        b = self.batch
        b.requests += rs
        b.queues += qs
        b.generated += [0] * k
        b.stop_sets += [e._stop_ids(r) for r in rs]
        self.positions += res['lengths']
        for r in rs:
            r.prompt_cursor = len(r.prompt_token_ids)
        st['prefill_forward_calls'] += 1
        st['prefill_chunks'] += 1
        st['prefill_tokens'] += res['tokens']
        self.stats['rows_admitted'] += k
        self.stats['packed_rows'] += k
        self.stats['packed_prefills'] += 1
        self.stats['max_live_rows'] = max(self.stats['max_live_rows'], self.rows)
        self.stats['packed_ms'] += (time.perf_counter() - t0) * 1000
        e._accept_host(b, [None] * base + res['firsts'])
        self.compact()
        if self.long_queue:                              # P7 v2: next job starts right away
            self._start_overlap()
        return True

    def take_deferred(self):
        out, self.deferred = self.deferred, []
        return out

    def _prefix_plan(self, slots, prompts):
        """Fill slots from the pool prefix cache; compute and store missing shared prefixes.
        Returns (per-row start columns for the continuation prefill, per-row snapshot boundaries
        for that same prefill or None). P5: a row alone in its cohort is snapshotted inside the
        main prefill (single pass) when the decoder can record its boundary; shared prefixes
        still take the prefix pass, since members copy the leader's state before continuing."""
        pc, st, dec = self.pcache, self.st, self.dec
        k = len(slots)
        starts = [0] * k
        for j in range(k):
            e = pc.longest(prompts[j])
            if e is not None:
                pc.restore(e, st, slots[j])
                starts[j] = len(e.tokens)
                self.stats['packed_prefix_hits'] += 1
                self.stats['packed_tokens_saved'] += starts[j]
        blk = pc.block_tokens

        def lcp(a, b):
            m = min(len(a), len(b))
            i = 0
            while i < m and a[i] == b[i]:
                i += 1
            return i
        todo = [j for j in range(k) if not starts[j] and pc.boundary(len(prompts[j]))]
        recent = list(self._recent)
        self._recent.extend(tuple(p) for p in prompts)
        groups = []                                  # (leader, [members], b)
        while todo:
            j0 = todo.pop(0)
            b0 = pc.boundary(len(prompts[j0]))
            shared = {l: min(lcp(prompts[j0], prompts[l]) // blk * blk, b0, pc.boundary(len(prompts[l])))
                      for l in todo}
            bc = max([b for b in shared.values() if b >= pc.min_tokens], default=0)
            if bc:
                members = [l for l, b in shared.items() if b >= bc]
                todo = [l for l in todo if l not in members]
                groups.append((j0, members, bc))
                continue
            # Alone in this cohort: only pay the extra prefix pass when a recent prompt shared
            # the prefix (it is hot). Cold prompts stay one pass with no snapshot, unless
            # store_first_sight (P5): then every prompt's own boundary is stored, so an exact repeat
            # (retry, re-sent conversation, rerun) restores on its first return.
            if self.store_first_sight:
                bs = b0
            else:
                bs = max((min(lcp(prompts[j0], q) // blk * blk, b0) for q in recent), default=0)
            if bs >= pc.min_tokens:
                groups.append((j0, [], bs))
        snap = None
        al = dec.snap_align() if self.single_pass_snapshot else 0
        if al:
            alone = [(j, b) for j, m, b in groups if not m and b % al == 0]
            if alone:
                snap = [0] * k
                for j, b in alone:
                    snap[j] = b
                groups = [x for x in groups if x[1] or x[2] % al]
        if not groups:
            return starts, snap
        dec.prefill_packed(st, [slots[j] for j, _, _ in groups], [prompts[j][:b] for j, _, b in groups])
        for j0, members, b in groups:
            pc.put(prompts[j0][:b], st, slots[j0])
            for l in members:
                pc.copy_slot(st, slots[j0], slots[l], b)
                starts[l] = b
                self.stats['packed_tokens_saved'] += b
            starts[j0] = b
            self.stats['packed_prefix_groups'] += 1
            self.stats['packed_prefix_shared_rows'] += len(members)
        return starts, snap

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
        if self.overlap and not self.rows and self._job is None and self.long_queue:
            self._start_overlap()                        # P7 v2: queued long prompts, nothing decoding
        self.poll_overlap(wait=not self.rows)          # P7: nothing to decode -> wait for the prefill
        if self._pending is None:
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
        import time
        t0 = time.perf_counter()
        y, acc = self.dec.spec_step(self.st, self.sp, n, length=max(self.positions) + 2)
        if self.pipeline:
            # P9f: the step is queued on the GPU; accept/emit the previous step and compact now.
            # compact's row moves are queued after this step, so moved rows carry its update.
            tf = time.perf_counter()
            live = self.compact()                       # flushes the pending step first
            self.stats['spec_flush_ms'] += (time.perf_counter() - tf) * 1000
            yh, ah = y.tolist(), acc.tolist()           # the step's only host sync
            if live is not None:                        # rows were dropped/moved: same order as batch
                yh, ah = [yh[i] for i in live], [ah[i] for i in live]
            t1 = time.perf_counter()
            self.stats['spec_step_ms'] += (t1 - t0) * 1000
            self.positions = [p + 1 + int(a) for p, a in zip(self.positions, ah)]
            self._pending = (yh, ah)
            self.stats['spec_pipelined_steps'] += 1
            self.stats['decode_steps'] += 1
            self.stats['spec_steps'] += 1
            self.e._stats['decode_forward_calls'] += 1
            self.stats['spec_host_ms'] += (time.perf_counter() - t1) * 1000
            return
        yh, ah = y.tolist(), acc.tolist()            # the step's only host sync
        t1 = time.perf_counter()
        self.stats['spec_step_ms'] += (t1 - t0) * 1000   # launch + GPU step + sync
        self._pending = (yh, ah)
        self._flush(positions=True)
        self.stats['decode_steps'] += 1
        self.stats['spec_steps'] += 1
        self.e._stats['decode_forward_calls'] += 1
        self.compact()
        # P6: host work after the step's sync (accept, emit, compact); the GPU idles through it
        self.stats['spec_host_ms'] += (time.perf_counter() - t1) * 1000

    def _flush(self, positions=False):
        """Accept and emit a synced spec step's tokens (rows 0..len-1 of the batch)."""
        if self._pending is None:
            return
        yh, ah = self._pending
        self._pending = None
        e = self.e
        buf = []   # P6: one event-loop wakeup per step for all rows' tokens
        e._accept_host(self.batch, [r[0] for r in yh], buf)
        e._accept_host(self.batch, [r[1] if a else None for r, a in zip(yh, ah)], buf)
        e._emit_many(buf)
        if positions:
            self.positions = [p + 1 + int(a) for p, a in zip(self.positions, ah)]
        self.stats['spec_row_steps'] += len(yh)
        self.stats['spec_accepted'] += sum(map(bool, ah))

    def compact(self):
        """Drop finished/cancelled rows and pack survivors into slots 0..k-1 (order kept, or with
        compact_from_tail the tail rows fill the holes). Returns the new order (old row index per
        new row), or None when nothing changed. A pending pipelined step is accepted first."""
        import torch
        self._flush()
        b = self.batch
        live = [i for i, r in enumerate(b.requests) if r.state == RequestState.RUNNING]
        if len(live) == len(b.requests):
            return None
        for i, r in enumerate(b.requests):
            if i not in live and r.state == RequestState.CANCELLED and r.request_id in self.e._queues:
                self.e._emit(b.queues[i], None)
                self.e._requests.pop(r.request_id, None)
                self.e._queues.pop(r.request_id, None)
        if self.compact_from_tail:
            # P6: fill each hole below k with a live row from the tail: one row move per finished
            # row instead of shifting every survivor after it. A move copies the row's KV, conv,
            # recurrent (and spec) state per layer, ~1 ms of launches at 1.6k tokens on L4, so the
            # order-keeping shift cost ~9 ms per step in prefill_heavy c16. Row order is not used.
            k = len(live)
            alive = set(live)
            holes = [i for i in range(k) if i not in alive]
            order = list(range(k))
            for h, src in zip(holes, [i for i in live if i >= k]):
                order[h] = src
            live = order
        for j, src in enumerate(live):
            if src != j:
                if self.sp is not None:
                    self.dec.move_spec_row(self.st, self.sp, src, j, self.positions[src] + self.margin)
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
        return live

    def clear(self):
        self._pending = None
        b = self.batch
        b.requests, b.queues, b.generated, b.stop_sets = [], [], [], []
        self.positions = []

    def diagnostics(self):
        st = self.st
        return {**self.stats, 'live_rows': self.rows, 'slots': st.batch, 'max_len': st.max_len,
                'batch_buckets': list(st.batch_buckets), 'len_buckets': list(st.len_buckets),
                'decoder': dict(self.dec.stats), 'graphs': len(self.dec.graphs) + len(getattr(self.dec, 'spec_graphs', {})),
                'spec': self.sp is not None,
                'pool_prefix_cache': self.pcache.diagnostics() if self.pcache is not None else None,
                'spec_acceptance': (self.stats['spec_accepted'] / self.stats['spec_row_steps'])
                if self.stats['spec_row_steps'] else None}
