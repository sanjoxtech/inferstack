"""P2: prefix cache in the static pool's own format.

An entry is the state a pool slot holds after its first b prompt tokens (b a multiple of the
block size): full-attention K/V columns 0..b-1, and each GDN layer's conv inputs and
recurrent state after token b-1. A hit copies the entry into a slot and packed prefill
continues from column b (QwenStaticDecoder.prefill_packed(starts=...)). This follows the
"state at block boundaries" idea used for hybrid models by vLLM's align mode and LMDeploy
(#4465); no code is shared with them.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PoolPrefixEntry:
    tokens: tuple
    kv: dict = field(default_factory=dict)      # layer -> (k [Hkv, b, D], v)
    conv: dict = field(default_factory=dict)    # layer -> [C, K]
    rec: dict = field(default_factory=dict)     # layer -> [Hv, dk, dv] fp32
    bytes: int = 0
    created_at: float = 0.0


def _nbytes(x) -> int:
    return int(x.numel() * x.element_size())


class PoolPrefixCache:
    def __init__(self, max_entries=64, min_tokens=128, block_tokens=128, max_bytes=1 << 30):
        self.max_entries = max(0, int(max_entries))
        self.min_tokens = max(1, int(min_tokens))
        self.block_tokens = max(1, int(block_tokens))
        self.max_bytes = max(0, int(max_bytes))
        self._entries: OrderedDict[tuple, PoolPrefixEntry] = OrderedDict()
        self.stats = {'hits': 0, 'misses': 0, 'stores': 0, 'evictions': 0, 'restores': 0, 'hit_tokens': 0}

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.max_bytes > 0

    def boundary(self, n: int) -> int:
        """Largest block boundary strictly inside a prompt of n tokens (0 if below min_tokens)."""
        b = (int(n) - 1) // self.block_tokens * self.block_tokens
        return b if b >= self.min_tokens else 0

    def longest(self, tokens, limit=None):
        """Longest entry that is a prefix of tokens and shorter than `limit` (default len)."""
        tokens = tuple(tokens)
        limit = len(tokens) if limit is None else int(limit)
        best = None
        for key, e in self._entries.items():
            if len(key) < limit and tokens[:len(key)] == key and (best is None or len(key) > len(best.tokens)):
                best = e
        if best is None:
            self.stats['misses'] += 1
            return None
        self.stats['hits'] += 1
        self.stats['hit_tokens'] += len(best.tokens)
        self._entries.move_to_end(best.tokens)
        return best

    def has(self, tokens) -> bool:
        return tuple(tokens) in self._entries

    def put(self, tokens, st, slot: int) -> bool:
        """Snapshot slot's first len(tokens) columns and its GDN state (call right after the
        slot was prefilled with exactly these tokens)."""
        tokens = tuple(tokens)
        b = len(tokens)
        if not self.enabled or b < self.min_tokens or b % self.block_tokens or tokens in self._entries:
            return False
        e = PoolPrefixEntry(tokens, created_at=time.time())
        for i, (k, v) in st.kv.items():
            e.kv[i] = (k[slot, :, :b].clone(), v[slot, :, :b].clone())
        for i, c in st.conv.items():
            e.conv[i] = c[slot].clone()
        for i, r in st.rec.items():
            e.rec[i] = r[slot].clone()
        e.bytes = sum(_nbytes(x) for kv in e.kv.values() for x in kv) + \
            sum(_nbytes(x) for x in e.conv.values()) + sum(_nbytes(x) for x in e.rec.values())
        self._entries[tokens] = e
        self.stats['stores'] += 1
        while len(self._entries) > self.max_entries or sum(x.bytes for x in self._entries.values()) > self.max_bytes:
            self._entries.popitem(last=False)
            self.stats['evictions'] += 1
        return tokens in self._entries

    def restore(self, e: PoolPrefixEntry, st, slot: int):
        b = len(e.tokens)
        for i, (k, v) in e.kv.items():
            st.kv[i][0][slot, :, :b].copy_(k)
            st.kv[i][1][slot, :, :b].copy_(v)
        for i, c in e.conv.items():
            st.conv[i][slot].copy_(c)
        for i, r in e.rec.items():
            st.rec[i][slot].copy_(r)
        self.stats['restores'] += 1

    def copy_slot(self, st, src: int, dst: int, b: int):
        """Give dst the state src holds after its first b tokens (src was prefilled to exactly b)."""
        for k, v in st.kv.values():
            k[dst, :, :b].copy_(k[src, :, :b])
            v[dst, :, :b].copy_(v[src, :, :b])
        for t in list(st.conv.values()) + list(st.rec.values()):
            t[dst].copy_(t[src])

    def diagnostics(self) -> dict:
        cur = sum(x.bytes for x in self._entries.values())
        return {**self.stats, 'enabled': self.enabled, 'entries': len(self._entries), 'current_mb': cur / 2 ** 20,
                'block_tokens': self.block_tokens, 'min_tokens': self.min_tokens}
