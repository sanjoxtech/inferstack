"""Optimizer provider contract: capability-declared, normalized chat completions.

The optimizer model is interchangeable and independent of the served model.
Adapters normalize: response content, tool calls, finish reason, token usage,
and errors (retryable vs not). Raw responses are kept for audit with credential-
like content redacted. No provider model ID or credential lives in code.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import httpx

from inferstack.contracts.redaction import redact, scrub_text
from inferstack.control.errors import JobError, ProviderUnavailable


class ProviderRejected(JobError):
    code = 'provider_rejected'


class MalformedProviderResponse(JobError):
    code = 'provider_malformed'


@dataclass(frozen=True)
class ProviderCapabilities:
    tools: bool
    json_mode: bool


@dataclass
class ProviderResponse:
    content: str | None
    tool_calls: list[dict]
    finish_reason: str | None
    usage: dict
    raw: dict = field(repr=False)


class ChatProvider:
    name = 'openai_compatible'
    default_base_url = None
    extra_headers: dict = {}

    def __init__(self, *, model, api_key_env, base_url=None, tools=None, json_mode=True, timeout=120,
                 max_output_tokens=4096, transport=None):
        if not model:
            raise ValueError('provider model id must be configured')
        key = os.environ.get(api_key_env or '')
        if not key:
            raise ValueError(f'missing provider credential environment variable {api_key_env!r}')
        base = (base_url or self.default_base_url or '').rstrip('/')
        if not base.startswith('https://') and not base.startswith('http://127.0.0.1') and transport is None:
            raise ValueError('provider base_url must use https')
        self.model, self.max_output_tokens = model, max_output_tokens
        self.capabilities = ProviderCapabilities(tools=self.default_tools() if tools is None else bool(tools),
                                                 json_mode=bool(json_mode))
        self.client = httpx.AsyncClient(base_url=base + '/', timeout=timeout, transport=transport,
                                        headers={'Authorization': 'Bearer ' + key, **self.extra_headers})

    def default_tools(self):
        return False  # unknown until configured: never assume tool support

    def payload(self, messages, tools, json_mode):
        body = {'model': self.model, 'messages': messages, 'max_tokens': self.max_output_tokens, 'temperature': 0}
        if tools and self.capabilities.tools:
            body['tools'] = tools
        if json_mode and self.capabilities.json_mode and not body.get('tools'):
            body['response_format'] = {'type': 'json_object'}
        return body

    def error_from_body(self, data):
        error = data.get('error') if isinstance(data, dict) else None
        if not error:
            return None
        code = error.get('code') if isinstance(error, dict) else None
        message = scrub_text(str(error.get('message') if isinstance(error, dict) else error))[:300]
        if code in (429, 502, 503, 'rate_limit_exceeded', 'server_error'):
            return ProviderUnavailable(f'{self.name}: {message}')
        return ProviderRejected(f'{self.name}: {message}')

    async def chat(self, messages, *, tools=None, json_mode=True) -> ProviderResponse:
        try:
            response = await self.client.post('chat/completions', json=self.payload(messages, tools, json_mode))
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise ProviderUnavailable(f'{self.name}: {type(exc).__name__}') from exc
        if response.status_code in (408, 429) or response.status_code >= 500:
            raise ProviderUnavailable(f'{self.name}: HTTP {response.status_code}')
        try:
            data = response.json()
        except ValueError as exc:
            raise MalformedProviderResponse(f'{self.name}: non-JSON response (HTTP {response.status_code})') from exc
        error = self.error_from_body(data)
        if error is not None:
            raise error
        if response.status_code >= 400:
            raise ProviderRejected(f'{self.name}: HTTP {response.status_code}')
        try:
            choice = data['choices'][0]
            message = choice['message']
        except (KeyError, IndexError, TypeError) as exc:
            raise MalformedProviderResponse(f'{self.name}: response has no choices[0].message') from exc
        calls = []
        for call in message.get('tool_calls') or []:
            fn = call.get('function') or {}
            if not isinstance(fn.get('name'), str):
                raise MalformedProviderResponse(f'{self.name}: tool call without a name')
            calls.append({'id': call.get('id') or f'call_{len(calls)}', 'name': fn['name'],
                          'arguments': fn.get('arguments') or '{}'})
        usage = data.get('usage') or {}
        normalized = {k: usage[k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')
                      if isinstance(usage.get(k), int)}
        return ProviderResponse(content=message.get('content'), tool_calls=calls,
                                finish_reason=choice.get('finish_reason'), usage=normalized, raw=redact(data))

    async def close(self):
        await self.client.aclose()


class DeepSeekProvider(ChatProvider):
    """https://api-docs.deepseek.com — OpenAI-compatible chat completions with JSON mode and tools."""
    name = 'deepseek'
    default_base_url = 'https://api.deepseek.com'

    def default_tools(self):
        return True


class OpenRouterProvider(ChatProvider):
    """https://openrouter.ai/api/v1 — tool and JSON support depend on the routed model,
    so both must be declared in configuration; errors may arrive in a 200 body."""
    name = 'openrouter'
    default_base_url = 'https://openrouter.ai/api/v1'
    extra_headers = {'X-Title': 'InferStack optimizer'}


PROVIDERS = {'deepseek': DeepSeekProvider, 'openrouter': OpenRouterProvider, 'openai_compatible': ChatProvider}


def create_provider(config: dict, transport=None) -> ChatProvider:
    config = dict(config)
    kind = config.pop('kind', 'openai_compatible')
    if kind not in PROVIDERS:
        raise ValueError(f'unknown provider kind {kind!r}; choose from {sorted(PROVIDERS)}')
    return PROVIDERS[kind](transport=transport, **config)


def audit_record(responses, messages, provider=None) -> bytes:
    """Serialized, redacted provider transcript for the artifact store."""
    meta = {'provider': provider.name, 'model': provider.model} if provider is not None else {}
    return json.dumps({**meta, 'messages': redact(messages), 'responses': [r.raw for r in responses]},
                      sort_keys=True, default=str).encode()
