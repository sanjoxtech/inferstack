"""Minimal real OpenAI-compatible HTTP server used as an engine process in tests."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--model', default='m')
    parser.add_argument('--answer', default='four')
    parser.add_argument('--delay', type=float, default=0.0)
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/v1/models':
                return self.send({'data': [{'id': args.model}]})
            self.send_error(404)

        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length', 0)))
            if self.path == '/v1/completions':
                if args.delay:
                    import time
                    time.sleep(args.delay)
                return self.send({'choices': [{'text': args.answer, 'finish_reason': 'stop'}],
                                  'usage': {'completion_tokens': 1}})
            self.send_error(404)

    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
