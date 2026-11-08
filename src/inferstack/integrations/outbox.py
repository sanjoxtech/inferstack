"""Durable outbox for authorized outcome callbacks (FactoryOS InferenceOutcome v1).

The outcome row is written in the same transaction that makes the job terminal,
so a crash can never lose or duplicate it. `deliver_due` sends due rows with the
row's idempotency key in the `Idempotency-Key` header; the receiver deduplicates
on it, so retries never repeat an effect. Transient failures (network, 429, 5xx)
back off exponentially; non-retryable rejections and exhausted attempts move the
row to `dead`, visible to operators and re-armable with `retry_dead`.

Destination and credential are runtime configuration (callback URL, bearer token
from an environment variable), never stored in the payload. HTTPS is required
except for loopback receivers.
"""
from __future__ import annotations

import json
import os
from urllib.parse import urlparse

OUTCOME_CONTRACT = 'inferstack.outcome/v1'


def outcome_body(job_id, payload, state, result):
    callback = payload.get('outcome_callback') or {}
    result = result if isinstance(result, dict) else {}
    experiments = result.get('experiments') or result.get('partial', {}).get('experiments') or []
    last = experiments[-1] if experiments else {}
    evidence = (last.get('verdict') or {}).get('evidence')
    return {'contract': OUTCOME_CONTRACT, 'issue_id': callback.get('issue_id'),
            'idempotency_key': callback.get('idempotency_key'), 'trace_ids': callback.get('trace_ids', []),
            'job_id': job_id, 'status': state, 'decision': result.get('decision'),
            'stop_reason': result.get('stop_reason'), 'experiments': len(experiments),
            'evidence_sha256': (evidence or {}).get('sha256'),
            'error': (result.get('error') or {}).get('type') if isinstance(result.get('error'), dict) else None,
            'note': 'keep means verified, not live: traffic moves only through a progressive rollout'}


class Outbox:
    def __init__(self, store, *, max_attempts=8, base_backoff=5.0, max_backoff=900.0):
        self.store, self.max_attempts = store, max_attempts
        self.base_backoff, self.max_backoff = base_backoff, max_backoff

    def list(self, state=None, limit=100):
        sql, params = 'SELECT * FROM outbox', []
        if state:
            sql, params = sql + ' WHERE state=?', [state]
        with self.store.connection() as db:
            rows = db.execute(sql + ' ORDER BY created_at LIMIT ?', (*params, limit)).fetchall()
        return [{**{k: r[k] for k in r.keys()}, 'payload': json.loads(r['payload'])} for r in rows]

    def _claim(self, limit):
        """Take due rows under the writer lock, marking them in flight (crash -> due again later)."""
        now = self.store.clock()
        with self.store.transaction() as db:
            rows = db.execute("SELECT * FROM outbox WHERE state IN ('pending','sending') AND next_attempt_at<=? "
                              "ORDER BY next_attempt_at LIMIT ?", (now, limit)).fetchall()
            for r in rows:  # lease: if this process dies mid-send the row is retried after 120 s
                db.execute("UPDATE outbox SET state='sending',attempts=attempts+1,next_attempt_at=? WHERE id=?",
                           (now + 120, r['id']))
        return [dict({k: r[k] for k in r.keys()}, attempts=r['attempts'] + 1) for r in rows]

    def _settle(self, row, *, delivered, retryable=True, error=None, response=None):
        now = self.store.clock()
        with self.store.transaction() as db:
            if delivered:
                db.execute("UPDATE outbox SET state='delivered',delivered_at=?,response=?,last_error=NULL WHERE id=?",
                           (now, (response or '')[:2000], row['id']))
                return 'delivered'
            dead = not retryable or row['attempts'] >= self.max_attempts
            delay = min(self.max_backoff, self.base_backoff * 2 ** (row['attempts'] - 1))
            db.execute('UPDATE outbox SET state=?,next_attempt_at=?,last_error=? WHERE id=?',
                       ('dead' if dead else 'pending', now + delay, (error or '')[:2000], row['id']))
            return 'dead' if dead else 'retry'

    async def deliver_due(self, send, limit=50):
        """send(row) -> (status_code, text). Returns per-row outcomes."""
        outcomes = []
        for row in self._claim(limit):
            try:
                status, text = await send(row)
            except Exception as exc:  # noqa: BLE001 - transport failures are retryable by definition
                outcomes.append(self._settle(row, delivered=False, error=f'{type(exc).__name__}: {exc}'[:500]))
                continue
            if 200 <= status < 300:
                outcomes.append(self._settle(row, delivered=True, response=text))
            else:
                outcomes.append(self._settle(row, delivered=False, retryable=status == 429 or status >= 500,
                                             error=f'HTTP {status}: {text[:300]}'))
        return outcomes

    def retry_dead(self, outbox_id):
        with self.store.transaction() as db:
            cursor = db.execute("UPDATE outbox SET state='pending',attempts=0,next_attempt_at=? WHERE id=? "
                                "AND state='dead'", (self.store.clock(), outbox_id))
            if not cursor.rowcount:
                raise ValueError(f'{outbox_id} is not dead')


def http_sender(url, *, token_env, transport=None, timeout=15.0):
    """Authorized HTTP delivery. The token is read from the environment at send time."""
    import httpx
    parsed = urlparse(url)
    if parsed.scheme != 'https' and parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('outcome callbacks require https (loopback excepted)')

    async def send(row):
        token = os.environ.get(token_env)
        if not token:
            raise RuntimeError(f'callback credential {token_env} is not set')
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            response = await client.post(url, json=row['payload'], headers={
                'Authorization': 'Bearer ' + token, 'Idempotency-Key': row['idempotency_key'],
                'X-InferStack-Contract': OUTCOME_CONTRACT})
            return response.status_code, response.text
    return send
