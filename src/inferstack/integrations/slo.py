"""SLO observation for triggers from the gateway's per-deployment admission window."""
from __future__ import annotations

import time


def gateway_slo_observer(gateway_url, token, transport=None, clock=time.time):
    """observe(deployment_id, metric, window_s) -> float | None. Supported: error_rate, ttfb_p50_ms."""
    import httpx

    async def observe(deployment_id, metric, window_s):
        async with httpx.AsyncClient(base_url=gateway_url, transport=transport, timeout=10,
                                     headers={'Authorization': 'Bearer ' + token}) as client:
            response = await client.get('/gateway/observe', params={'deployment_id': deployment_id,
                                                                    'since': clock() - window_s})
            response.raise_for_status()
            window = response.json()
        if metric == 'error_rate':
            return window['errors'] / window['requests'] if window.get('requests') else None
        if metric == 'ttfb_p50_ms':
            return window.get('ttfb_p50_ms')
        return None  # unsupported metric never fires
    return observe
