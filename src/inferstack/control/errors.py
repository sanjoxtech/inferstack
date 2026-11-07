"""Typed job errors and retry classification.

Only explicitly transient conditions are retryable. Unknown exceptions are treated
as permanent so a bug cannot cause unbounded retries. Messages are redacted and
truncated before persistence.
"""
from __future__ import annotations

import asyncio

from inferstack.contracts.redaction import scrub_text


class JobError(Exception):
    code = 'job_error'
    retryable = False


class TransientError(JobError):
    code = 'transient'
    retryable = True


class ProviderUnavailable(TransientError):
    code = 'provider_unavailable'


class ResourceUnavailable(TransientError):
    code = 'resource_unavailable'


class ConfigurationError(JobError):
    code = 'configuration'


class BudgetExhausted(JobError):
    code = 'budget_exhausted'


class DeadlineExceeded(JobError):
    code = 'deadline_exceeded'


class ReconciliationRequired(JobError):
    """An interrupted side-effecting step could not be proven clean."""
    code = 'reconciliation_required'


def classify(exc: BaseException) -> dict:
    import httpx
    if isinstance(exc, JobError):
        code, retryable = exc.code, exc.retryable
    elif isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, asyncio.TimeoutError, ConnectionError)):
        code, retryable = 'transient_io', True
    elif isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        code, retryable = f'http_{status}', status in (408, 429) or status >= 500
    else:
        code, retryable = 'internal', False
    message = scrub_text(str(exc))[:500]
    return {'type': type(exc).__name__, 'code': code, 'retryable': retryable, 'message': message}
