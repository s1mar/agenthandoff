"""A stand-in for llama-server, for tests. Standard library only.

    python fake_llama_server.py --port N --pid-file PATH [--load-seconds S] [--fail]

Answers /health (503 while "loading", then 200), /v1/models and
/v1/chat/completions (always {"ok": true}). Writes its pid to --pid-file so a test
can check that it was stopped. --fail exits at once with code 3, like a server that
cannot load its model.
"""
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--pid-file", required=True)
parser.add_argument("--load-seconds", type=float, default=0.3)
parser.add_argument("--fail", action="store_true")
args, _ = parser.parse_known_args()

with open(args.pid_file, "w") as f:
    f.write(str(os.getpid()))
if args.fail:
    print("error: failed to load model", flush=True)
    sys.exit(3)
ready_at = time.monotonic() + args.load_seconds


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            if time.monotonic() < ready_at:
                self.reply(503, {"error": {"message": "Loading model"}})
            else:
                self.reply(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self.reply(200, {"data": [{"id": "fake"}]})
        else:
            self.reply(404, {})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.reply(200, {"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 5, "completion_tokens": 3}})


ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
