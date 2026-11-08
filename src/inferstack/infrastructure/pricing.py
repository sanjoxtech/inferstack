"""Operator-supplied price table for spend accounting.

Prices are configuration with provenance (source and date), never constants in
code. An allocation is stamped with its price at intent time; spend is derived
from allocation active time x GPU count x price. If a job has a spend budget and
any of its allocations has no known price, spend is unknown and the budget is
treated as exhausted rather than guessed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    usd_per_gpu_hour: float
    source: str
    as_of: str


class PriceTable:
    def __init__(self, entries: dict):
        """entries: {"<provider>/<gpu name>": {"usd_per_gpu_hour", "source", "as_of"}}"""
        self.prices = {}
        for key, value in entries.items():
            price = Price(float(value['usd_per_gpu_hour']), str(value['source']), str(value['as_of']))
            if price.usd_per_gpu_hour < 0 or not price.source or not price.as_of:
                raise ValueError(f'price for {key} needs a non-negative rate, source and as_of date')
            self.prices[key.lower()] = price

    def lookup(self, provider, gpu_name):
        return self.prices.get(f'{provider}/{gpu_name}'.lower())

    def stamp(self, provider, gpu_name, gpu_count):
        """Allocation spec fields; price fields are absent (unknown) when no entry matches."""
        if not gpu_count:
            return {'price': {'usd_per_gpu_hour': 0.0, 'source': 'no GPUs allocated', 'as_of': None}}
        price = self.lookup(provider, gpu_name)
        if price is None:
            return {'price': None}
        return {'price': {'usd_per_gpu_hour': price.usd_per_gpu_hour, 'source': price.source, 'as_of': price.as_of}}
