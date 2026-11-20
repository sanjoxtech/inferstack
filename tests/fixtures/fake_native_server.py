"""Fake native engine for tuning tests: reads the TOML profile (as `inferstack serve --profile` would) and
serves through fake_openai_server with behaviour set by the knobs:
  static_decode_slots >= 64       -> exits before ready (a crash, like an OOM in graph capture)
  static_decode_slots < 32        -> slow first token (queueing)
  gemm_blas_library == 'cublaslt' -> different greedy text (a quality break)
"""
import os
import sys
import tomllib
from pathlib import Path

argv = sys.argv[1:]
profile = tomllib.loads(Path(argv[argv.index('--profile') + 1]).read_text())
native = profile.get('native') or {}
slots = native.get('static_decode_slots', 16)
if slots >= 64:
    sys.exit('CUDA out of memory (fake)')
port = argv[argv.index('--port') + 1]
fake = str(Path(__file__).with_name('fake_openai_server.py'))
args = [sys.executable, fake, '--port', port, '--usage', '--stream-tokens', '4', '--chunk-delay', '0.002',
        '--first-delay', '0.7' if slots < 32 else '0.0',
        '--answer', 'five' if native.get('gemm_blas_library') == 'cublaslt' else 'four']
os.execv(sys.executable, args)
