"""Minimal real OpenAI-compatible HTTP server used as an engine process in tests.

--flavor adds engine-specific endpoints (vllm: /version + /metrics, sglang:
/get_server_info + /metrics, native: /v1/native/status). Streaming responses send
chunks slowly and record a client disconnect to --events.
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VLLM_METRICS = """# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{model_name="m",le="0.05"} 2
vllm:time_to_first_token_seconds_bucket{model_name="m",le="0.1"} 3
vllm:time_to_first_token_seconds_bucket{model_name="m",le="+Inf"} 4
vllm:time_to_first_token_seconds_count{model_name="m"} 4
vllm:time_to_first_token_seconds_sum{model_name="m"} 0.4
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m"} 3
vllm:prefix_cache_queries_total{model_name="m"} 120
"""
SGLANG_METRICS = """# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="m"} 2
sglang:token_usage{model_name="m"} 0.25
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--model', default='m')
    parser.add_argument('--answer', default='four')
    parser.add_argument('--delay', type=float, default=0.0)
    parser.add_argument('--flavor', default='generic')
    parser.add_argument('--version', default='0.11.0')
    parser.add_argument('--fail-status', type=int, default=0)
    parser.add_argument('--events', default=None)
    parser.add_argument('--dump-env', default=None)
    parser.add_argument('--stream-tokens', type=int, default=200)
    parser.add_argument('--chunk-delay', type=float, default=0.02)
    parser.add_argument('--first-delay', type=float, default=0.0)
    parser.add_argument('--slots', type=int, default=0, help='concurrent streams served; more queue (0 = no limit)')
    parser.add_argument('--usage', action='store_true', help='report usage in the final stream chunk')
    args, unknown = parser.parse_known_args()  # engine flags (e.g. --max-num-seqs) are accepted and recorded
    if args.dump_env:
        import os
        import sys
        # '~/' is the candidate's HOME (its sandboxed workspace): writes elsewhere are refused.
        path = os.path.join(os.environ['HOME'], args.dump_env[2:]) if args.dump_env.startswith('~/') else args.dump_env
        with open(path, 'w') as f:
            json.dump({'env': dict(os.environ), 'argv': sys.argv, 'executable': sys.executable, 'unknown': unknown}, f)

    lane = threading.BoundedSemaphore(args.slots) if args.slots else None

    def note(text):
        if args.events:
            with open(args.events, 'a') as f:
                f.write(text + '\n')

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *a):
            pass

        def send(self, body, status=200, ctype='application/json'):
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            routes = {'/v1/models': {'data': [{'id': args.model}]}}
            if args.flavor == 'vllm':
                routes['/version'] = {'version': args.version}
                routes['/metrics'] = VLLM_METRICS
            if args.flavor == 'sglang':
                routes['/get_server_info'] = {'version': args.version, 'disable_radix_cache': False,
                                              'disable_cuda_graph': True, 'dtype': 'bfloat16'}
                routes['/metrics'] = SGLANG_METRICS
            if args.flavor == 'native':
                routes['/v1/native/status'] = {'version': args.version, 'ready': True, 'models': {args.model: {
                    'paged_kv_active': False, 'actual_cuda_graph_replay_proven': False}}}
            if self.path in routes:
                body = routes[self.path]
                return self.send(body, ctype='text/plain' if isinstance(body, str) else 'application/json')
            self.send({'error': 'not found'}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            note('request ' + self.headers.get('X-Request-Id', '-') + ' ' + self.headers.get('traceparent', '-'))
            if args.fail_status:
                return self.send({'error': 'injected'}, args.fail_status)
            if args.delay:
                time.sleep(args.delay)
            if self.path != '/v1/completions':
                return self.send({'error': 'not found'}, 404)
            if not body.get('stream'):
                return self.send({'choices': [{'text': args.answer, 'finish_reason': 'stop'}],
                                  'usage': {'completion_tokens': 1},
                                  'echo_request_id': self.headers.get('X-Request-Id')})
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Transfer-Encoding', 'chunked')
            self.end_headers()
            try:
                if lane:
                    lane.acquire()
                if args.first_delay:
                    time.sleep(args.first_delay)
                count = args.stream_tokens
                if args.usage:  # scorecard mode: honour max_tokens like a real engine
                    count = min(count, int(body.get('max_tokens') or count))
                for i in range(count):
                    chunk = f'data: {json.dumps({"choices": [{"text": str(i)}]})}\n\n'.encode()
                    self.wfile.write(b'%x\r\n%s\r\n' % (len(chunk), chunk))
                    self.wfile.flush()
                    time.sleep(args.chunk_delay)
                if args.usage:
                    final = {'choices': [], 'usage': {'completion_tokens': count, 'prompt_tokens': 10}}
                    chunk = f'data: {json.dumps(final)}\n\n'.encode()
                    self.wfile.write(b'%x\r\n%s\r\n' % (len(chunk), chunk))
                done = b'data: [DONE]\n\n'
                self.wfile.write(b'%x\r\n%s\r\n' % (len(done), done))
                self.wfile.write(b'0\r\n\r\n')
                note('stream completed')
            except (BrokenPipeError, ConnectionResetError):
                note('stream disconnected')
            finally:
                if lane:
                    lane.release()

    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
