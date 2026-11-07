"""Bounded, SSRF-resistant URL retrieval for research sources.

Guarantees:
- Only http/https, no embedded credentials, optional port allowlist.
- Every TCP connection (first request and every redirect hop) resolves the host
  inside the transport, rejects it if ANY resolved address fails the address
  policy (default: globally routable only), and dials the vetted IP itself. A DNS
  answer cannot change between validation and connect (no rebinding window).
- Redirects are followed manually with a hop limit; https->http downgrade is refused.
- Total time and response bytes are capped; content types are allowlisted.
- Environment proxies are ignored: a proxy would connect on our behalf and bypass
  the address policy.
- No login-wall bypass; non-2xx and blocked responses are outcomes, not errors.

Fetched content is untrusted evidence. Nothing here interprets it as instructions.
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpcore
import httpx

TEXT_TYPES = {'text/plain', 'text/markdown', 'text/html', 'application/xhtml+xml', 'application/json',
              'text/x-rst'}


class BlockedAddress(httpcore.ConnectError):
    pass


def is_public(ip: ipaddress._BaseAddress) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


async def system_resolve(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({info[4][0] for info in infos})


class GuardedBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, policy, resolver):
        self.inner = httpcore.AnyIOBackend()
        self.policy, self.resolver = policy, resolver
        self.dialed: list[tuple[str, str]] = []
        self.blocked: str | None = None  # httpx re-wraps transport errors, so keep the reason here

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        try:
            literal = ipaddress.ip_address(host)
            addresses = [str(literal)]
        except ValueError:
            addresses = await self.resolver(host, port)
        if not addresses:
            self.blocked = f'{host} did not resolve'
            raise BlockedAddress(self.blocked)
        bad = [a for a in addresses if not self.policy(ipaddress.ip_address(a))]
        if bad:
            self.blocked = f'{host} resolves to disallowed address(es): {",".join(bad)}'
            raise BlockedAddress(self.blocked)
        self.dialed.append((host, addresses[0]))
        return await self.inner.connect_tcp(addresses[0], port, timeout=timeout, local_address=local_address,
                                            socket_options=socket_options)

    async def connect_unix_socket(self, *a, **k):
        self.blocked = 'unix sockets are not allowed'
        raise BlockedAddress(self.blocked)

    async def sleep(self, seconds):
        await self.inner.sleep(seconds)


class _Text(HTMLParser):
    SKIP = {'script', 'style', 'noscript', 'template', 'svg', 'head'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.title, self._skip, self._in_title = [], None, 0, False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag == 'title':
            self._in_title = True
        if tag in ('p', 'br', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'pre', 'tr', 'section', 'article'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag == 'title':
            self._in_title = False

    def handle_data(self, data):
        if self._in_title and self.title is None:
            self.title = data.strip()[:300] or None
        elif not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str | None]:
    parser = _Text()
    parser.feed(html)
    lines = [' '.join(line.split()) for line in ''.join(parser.parts).splitlines()]
    return '\n'.join(line for line in lines if line), parser.title


@dataclass
class FetchResult:
    url: str
    status: str               # retrieved | blocked | unavailable | unsupported | too_large | invalid
    reason: str = ''
    final_url: str | None = None
    http_status: int | None = None
    content_type: str | None = None
    text: str | None = None
    title: str | None = None
    raw_bytes: int = 0
    redirects: list[str] = field(default_factory=list)


class SafeFetcher:
    def __init__(self, *, timeout_seconds=15.0, max_bytes=2_000_000, max_redirects=3, allowed_ports=(80, 443),
                 policy=is_public, resolver=system_resolve, user_agent='InferStack-research/2.0'):
        if not 0 < timeout_seconds <= 120 or not 0 < max_bytes <= 50_000_000 or not 0 <= max_redirects <= 10:
            raise ValueError('fetch bounds out of range')
        self.timeout, self.max_bytes, self.max_redirects = timeout_seconds, max_bytes, max_redirects
        self.allowed_ports = None if allowed_ports is None else set(allowed_ports)
        self.policy, self.resolver, self.user_agent = policy, resolver, user_agent

    def check_url(self, url: str) -> str | None:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return 'scheme must be http or https'
        if not parsed.hostname:
            return 'missing host'
        if parsed.username or parsed.password:
            return 'embedded credentials are not allowed'
        try:
            port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        except ValueError:
            return 'invalid port'
        if self.allowed_ports is not None and port not in self.allowed_ports:
            return f'port {port} is not allowed'
        return None

    def _client(self, backend):
        transport = httpx.AsyncHTTPTransport(trust_env=False, retries=0)
        # httpx 0.28 builds its pool internally; replace it with one using the guarded backend.
        transport._pool = httpcore.AsyncConnectionPool(ssl_context=httpx.create_ssl_context(), network_backend=backend,
                                                       max_connections=4, retries=0)
        return httpx.AsyncClient(transport=transport, trust_env=False, follow_redirects=False,
                                 timeout=httpx.Timeout(self.timeout, connect=min(10.0, self.timeout)),
                                 headers={'User-Agent': self.user_agent, 'Accept': 'text/html,text/plain,*/*;q=0.1'})

    async def fetch(self, url: str) -> FetchResult:
        try:
            return await asyncio.wait_for(self._fetch(url), self.timeout)
        except asyncio.TimeoutError:
            return FetchResult(url, 'unavailable', 'total time limit exceeded')

    async def _fetch(self, url):
        result = FetchResult(url, 'invalid')
        problem = self.check_url(url)
        if problem:
            result.reason = problem
            return result
        backend = GuardedBackend(self.policy, self.resolver)
        current = url
        async with self._client(backend) as client:
            for hop in range(self.max_redirects + 1):
                try:
                    async with client.stream('GET', current) as response:
                        if response.is_redirect:
                            target = urljoin(current, response.headers.get('location', ''))
                            problem = self.check_url(target)
                            if problem:
                                return FetchResult(url, 'blocked', f'redirect target rejected: {problem}',
                                                   redirects=result.redirects + [target])
                            if urlparse(current).scheme == 'https' and urlparse(target).scheme == 'http':
                                return FetchResult(url, 'blocked', 'https to http redirect refused',
                                                   redirects=result.redirects + [target])
                            result.redirects.append(target)
                            current = target
                            continue
                        result.final_url, result.http_status = current, response.status_code
                        ctype = response.headers.get('content-type', '').split(';')[0].strip().lower()
                        result.content_type = ctype or None
                        if response.status_code >= 400:
                            result.status, result.reason = 'unavailable', f'HTTP {response.status_code}'
                            return result
                        if ctype not in TEXT_TYPES:
                            result.status, result.reason = 'unsupported', f'content type {ctype or "missing"}'
                            return result
                        declared = response.headers.get('content-length')
                        if declared and declared.isdigit() and int(declared) > self.max_bytes:
                            result.status, result.reason = 'too_large', f'declared {declared} bytes'
                            return result
                        body = bytearray()
                        async for chunk in response.aiter_raw():
                            body.extend(chunk)
                            if len(body) > self.max_bytes:
                                result.status, result.reason = 'too_large', f'exceeded {self.max_bytes} bytes'
                                return result
                        result.raw_bytes = len(body)
                        text = bytes(body).decode(response.encoding or 'utf-8', errors='replace')
                        if ctype in ('text/html', 'application/xhtml+xml'):
                            text, result.title = html_to_text(text)
                        result.status, result.text = 'retrieved', text
                        return result
                except (httpx.HTTPError, httpcore.NetworkError, OSError) as exc:
                    if backend.blocked:
                        return FetchResult(url, 'blocked', backend.blocked, redirects=result.redirects)
                    return FetchResult(url, 'unavailable', f'{type(exc).__name__}: {exc}'[:300],
                                       redirects=result.redirects)
        return FetchResult(url, 'blocked', f'more than {self.max_redirects} redirects', redirects=result.redirects)
