"""Secret rejection, redaction and retention settings.

Records must reference credentials indirectly (an environment-variable name in a
field ending with `_env`). A record body that carries a credential-like key with a
literal value is rejected at write time rather than silently stored.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SECRET_KEY = re.compile(r'(^|_)(api_?key|token|secret|password|passwd|authorization|credential|private_key|access_key)s?$', re.I)
_SECRET_VALUE = re.compile(r'^(sk-[A-Za-z0-9_-]{16,}|Bearer\s+\S{8,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})$')
REDACTED = '[redacted]'


class SecretInRecord(ValueError):
    pass


def secret_paths(value, path='$'):
    """Yield JSON paths whose key or value looks like a literal credential."""
    if isinstance(value, dict):
        for key, item in value.items():
            here = f'{path}.{key}'
            if isinstance(key, str) and not key.endswith('_env') and _SECRET_KEY.search(key) and item not in (None, ''):
                yield here
            else:
                yield from secret_paths(item, here)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from secret_paths(item, f'{path}[{index}]')
    elif isinstance(value, str) and _SECRET_VALUE.match(value.strip()):
        yield path


def reject_secrets(body):
    found = list(secret_paths(body))
    if found:
        raise SecretInRecord('record contains literal credentials at ' + ', '.join(found) +
                         '; reference them with an *_env field instead')


def redact(value):
    """Return a deep copy with credential-like keys and values replaced."""
    if isinstance(value, dict):
        return {k: (REDACTED if isinstance(k, str) and not k.endswith('_env') and _SECRET_KEY.search(k)
                    and v not in (None, '') else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str) and _SECRET_VALUE.match(value.strip()):
        return REDACTED
    return value


@dataclass(frozen=True)
class RetentionPolicy:
    """Operator retention settings. Content capture is opt-in.

    `None` TTL means retain indefinitely. Enforcement of TTLs is performed by the
    maintenance job (Package B/J); this contract fixes the semantics now.
    """
    capture_request_content: bool = False
    event_ttl_days: int | None = None
    artifact_ttl_days: int | None = None
    provider_raw_output_ttl_days: int | None = 30

    def __post_init__(self):
        for name in ('event_ttl_days', 'artifact_ttl_days', 'provider_raw_output_ttl_days'):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v <= 0):
                raise ValueError(f'{name} must be a positive integer or None')
