from __future__ import annotations

import json
import math
from dataclasses import dataclass, field


@dataclass
class StreamMeasurement:
    """Measure transport content timing; never equate SSE events with tokens."""
    started: float
    text: str = ''
    content_times: list[float] = field(default_factory=list)
    completion_tokens: int | None = None
    engine_tpot_ms: float | None = None
    done: bool = False
    finish_reason: str | None = None

    def feed(self, payload: str, now: float):
        if self.done:
            raise ValueError('data after terminal event')
        if payload == '[DONE]':
            self.done = True
            return
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError('stream event must be an object')
        if 'error' in data:
            raise ValueError('engine stream error: ' + str(data['error']))
        usage = data.get('usage')
        if usage is not None:
            count = usage.get('completion_tokens')
            if count is not None:
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError('invalid completion token count')
                if self.completion_tokens is not None and count != self.completion_tokens:
                    raise ValueError('conflicting usage counts')
                self.completion_tokens = count
        timing=data.get('inferstack_timing')
        if timing is not None:
            n=timing.get('token_count')
            elapsed=timing.get('decode_elapsed_ms')
            if isinstance(n,int) and not isinstance(n,bool) and n>1 and n==self.completion_tokens and finite_number(elapsed):
                self.engine_tpot_ms=elapsed/(n-1)
        choices = data.get('choices') or []
        if len(choices) > 1:
            raise ValueError('benchmark supports one completion per request')
        if choices:
            choice = choices[0]
            if choice.get('index', 0) != 0:
                raise ValueError('unexpected completion index')
            piece = choice.get('text', (choice.get('delta') or {}).get('content', '')) or ''
            if not isinstance(piece, str):
                raise ValueError('invalid stream content')
            if piece:
                self.text += piece
                self.content_times.append(now)
            if choice.get('finish_reason') is not None:
                self.finish_reason = choice['finish_reason']

    def result(self, ended: float, error: str | None = None):
        if not self.done and error is None:
            error = 'incomplete stream: missing terminal event'
        first = self.content_times[0] if self.content_times else None
        gaps = [(b-a)*1000 for a,b in zip(self.content_times, self.content_times[1:])]
        # Grouped content events cannot establish true per-token decode latency.
        return {
            'success': error is None, 'error': error,
            'output_text': self.text, 'output_tokens': self.completion_tokens,
            'token_count_source': 'engine_usage' if self.completion_tokens is not None else 'unavailable',
            'ttft_ms': (first-self.started)*1000 if first is not None else None,
            'ttft_definition': 'client_first_nonempty_content',
            'e2e_ms': (ended-self.started)*1000, 'tpot_ms': self.engine_tpot_ms,
            'tpot_definition': 'engine_token_emission_interval' if self.engine_tpot_ms is not None else 'unavailable_without_engine_token_timing',
            'chunk_gaps_ms': gaps, 'content_chunks': len(self.content_times),
            'finish_reason': self.finish_reason,
        }


def finite_number(value, *, nonnegative=True):
    return (not isinstance(value, bool) and isinstance(value, (float, int))
            and math.isfinite(value) and (not nonnegative or value >= 0))
