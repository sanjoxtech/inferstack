"""Prometheus text exposition parsing and histogram arithmetic.

Only what normalization needs: samples with labels, family types, counter and
histogram deltas between two scrapes, and bucket-interpolated quantiles. Quantiles
from buckets are estimates bounded by bucket resolution and are labelled as such.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

_SAMPLE = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)(?:\s+\d+)?$')
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Sample:
    name: str
    labels: tuple
    value: float


def _value(text):
    lowered = text.lower()
    if lowered in ('+inf', 'inf'):
        return math.inf
    if lowered == '-inf':
        return -math.inf
    return float(text)


def parse(text: str) -> tuple[list[Sample], dict[str, str]]:
    samples, types = [], {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            parts = line.split()
            if len(parts) >= 4 and parts[1] == 'TYPE':
                types[parts[2]] = parts[3]
            continue
        match = _SAMPLE.match(line)
        if not match:
            raise ValueError(f'unparsable metrics line: {line[:120]}')
        labels = tuple(sorted((k, v.encode().decode('unicode_escape')) for k, v in _LABEL.findall(match['labels'] or '')))
        samples.append(Sample(match['name'], labels, _value(match['value'])))
    return samples, types


class Scrape:
    def __init__(self, text: str):
        self.samples, self.types = parse(text)
        self.names = {s.name for s in self.samples}

    def has_family(self, family):
        return any(n == family or n.startswith(family + '_') for n in self.names)

    def total(self, name, label_filter=None):
        values = [s.value for s in self.samples if s.name == name
                  and (label_filter is None or all(dict(s.labels).get(k) == v for k, v in label_filter.items()))]
        return sum(values) if values else None

    def histogram(self, family, label_filter=None):
        """Aggregate a histogram across label sets: {'count','sum','buckets':[(le, cumulative)]}."""
        def keep(sample):
            labels = dict(sample.labels)
            return label_filter is None or all(labels.get(k) == v for k, v in label_filter.items())
        count = self.total(family + '_count', label_filter)
        total = self.total(family + '_sum', label_filter)
        if count is None or total is None:
            return None
        buckets = defaultdict(float)
        for s in self.samples:
            if s.name == family + '_bucket' and keep(s):
                buckets[_value(dict(s.labels)['le'])] += s.value
        return {'count': count, 'sum': total, 'buckets': sorted(buckets.items())}


def histogram_delta(before, after):
    if before is None or after is None:
        return None
    if after['count'] < before['count']:
        return None  # counter reset (engine restart) inside the window: not comparable
    prior = dict(before['buckets'])
    return {'count': after['count'] - before['count'], 'sum': after['sum'] - before['sum'],
            'buckets': [(le, c - prior.get(le, 0.0)) for le, c in after['buckets']]}


def quantile(hist, q):
    """Linear interpolation inside the bucket that crosses rank q*count."""
    if not hist or hist['count'] <= 0 or not hist['buckets']:
        return None
    rank, lower, previous = q * hist['count'], 0.0, 0.0
    for le, cumulative in hist['buckets']:
        if cumulative >= rank:
            if math.isinf(le):
                return None  # falls in the overflow bucket: no upper bound
            span = cumulative - previous
            return lower + (le - lower) * ((rank - previous) / span if span else 1.0)
        lower, previous = le, cumulative
    return None
