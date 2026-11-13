"""A stand-in for a model CLI (like `ollama run`, `agy -p` or `claude -p`), for tests.

    python fake_cli.py MODE [--model M] [--count-file PATH]

It reads the prompt from stdin. MODE:
  echo     print {"answer": "<model>: <first 40 chars of the prompt>", "chars": <prompt length>}
  text     print the prompt's last line back as plain text
  fail     print "boom" on stderr and exit 1
  sleep    sleep 30 s (for timeouts)
  empty    print nothing
  garbage-once  print garbage the first time (per count file), JSON after
  garbage  always print garbage
  huge     print 300,000 characters
Each run adds one line to --count-file, so a test can count the calls.
"""
import argparse
import json
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("mode")
parser.add_argument("--model", default="none")
parser.add_argument("--count-file")
parser.add_argument("--prompt", help="take the prompt from this argument, not stdin (like agy -p)")
args = parser.parse_args()

prompt = args.prompt if args.prompt is not None else sys.stdin.buffer.read().decode("utf-8")
calls = 0
if args.count_file:
    with open(args.count_file, "a", encoding="utf-8") as f:
        f.write("call\n")
    with open(args.count_file, encoding="utf-8") as f:
        calls = len(f.readlines())

if args.mode == "echo":
    print(json.dumps({"answer": f"{args.model}: {prompt[:40]}", "chars": len(prompt)}))
elif args.mode == "text":
    print(prompt.strip().splitlines()[-1])
elif args.mode == "fail":
    print("boom", file=sys.stderr)
    sys.exit(1)
elif args.mode == "sleep":
    time.sleep(30)
elif args.mode == "empty":
    pass
elif args.mode == "garbage-once":
    print("sure! here you go" if calls == 1 else '{"ok": true}')
elif args.mode == "garbage":
    print("no json here")
elif args.mode == "huge":
    sys.stdout.write("x" * 300_000)
