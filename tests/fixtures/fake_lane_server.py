"""Fake engine whose behaviour lives in patchable source (G7 tests): `inferstack.runtime.fake_lane` on
PYTHONPATH sets SLOTS (concurrent streams; more queue), FIRST_DELAY (seconds per request) and ANSWER (the
greedy text). A code change to that module is what the A/B measures."""
import os
import sys
from pathlib import Path

from inferstack.runtime import fake_lane as lane

argv = sys.argv[1:]
port = argv[argv.index('--port') + 1]
fake = str(Path(__file__).with_name('fake_openai_server.py'))
os.execv(sys.executable, [sys.executable, fake, '--port', port, '--usage', '--stream-tokens', '4',
                          '--chunk-delay', '0.005', '--first-delay', str(lane.FIRST_DELAY),
                          '--slots', str(lane.SLOTS), '--answer', lane.ANSWER])
