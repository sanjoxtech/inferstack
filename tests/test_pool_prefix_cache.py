"""PoolPrefixCache.peek (P9): longest cached prefix length with no stats or LRU side effects."""
from inferstack.runtime.pool_prefix_cache import PoolPrefixCache, PoolPrefixEntry


def test_peek_returns_longest_cached_prefix_without_side_effects():
    pc = PoolPrefixCache(8, min_tokens=4, block_tokens=4)
    for key in ((1, 2, 3, 4), (1, 2, 3, 4, 5, 6, 7, 8), (9, 9, 9, 9)):
        pc._entries[key] = PoolPrefixEntry(key, created_at=0.0)
    order = list(pc._entries)
    stats = dict(pc.stats)
    assert pc.peek([1, 2, 3, 4, 5, 6, 7, 8, 0]) == 8
    assert pc.peek([1, 2, 3, 4, 0]) == 4
    assert pc.peek([1, 2, 3, 4, 5, 6, 7, 8]) == 4      # never the whole prompt (as longest())
    assert pc.peek([7, 7, 7, 7, 7]) == 0
    assert list(pc._entries) == order and pc.stats == stats
