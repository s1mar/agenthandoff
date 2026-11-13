"""The MCP server and the CLI as an agent sees them: real subprocesses, a temporary home."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COUNTER = """import sys
for p in sys.argv[1:]:
    print(p.replace("\\\\", "/").rsplit("/", 1)[-1], sum(1 for _ in open(p, encoding="utf-8")), "lines")
"""


class Home:
    """A temporary AGENTHANDOFF_HOME with a config.json, plus a work folder to run in."""

    def __init__(self, config: dict | None = None):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name)
        self.work = self.path / "work"
        self.work.mkdir()
        self.counter = self.path / "counter.py"
        self.counter.write_text(COUNTER, encoding="utf-8")
        if config is not None:
            self.write(config)

    def write(self, config: dict):
        (self.path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    def env(self) -> dict:
        env = {k: v for k, v in os.environ.items() if not k.startswith("AGENTHANDOFF_")}
        env.update(AGENTHANDOFF_HOME=str(self.path), PYTHONPATH=str(ROOT / "src"))
        return env

    def file(self, name: str, text: str) -> str:
        p = self.work / name
        p.write_text(text, encoding="utf-8")
        return str(p)

    def counter_script(self, **extra) -> dict:
        return {"name": "count", "argv": [sys.executable, str(self.counter), "{paths}"],
                "description": "count lines", **extra}

    def cleanup(self):
        self._tmp.cleanup()


class McpServerTests(unittest.TestCase):
    def setUp(self):
        self.home = Home()
        self.addCleanup(self.home.cleanup)

    def talk(self, *messages) -> dict:
        lines = [json.dumps({"jsonrpc": "2.0", **m}) for m in messages]
        out = subprocess.run([sys.executable, "-m", "agenthandoff.mcp_server"], input="\n".join(lines) + "\n",
                             capture_output=True, text=True, env=self.home.env(), cwd=self.home.work,
                             timeout=60).stdout
        return {r["id"]: r for r in map(json.loads, out.splitlines())}

    def call(self, name, arguments) -> tuple[str, bool]:
        r = self.talk({"id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}})[1]
        return r["result"]["content"][0]["text"], r["result"]["isError"]

    def listed(self) -> list[str]:
        return [t["name"] for t in self.talk({"id": 1, "method": "tools/list"})[1]["result"]["tools"]]

    def test_not_set_up_lists_code_tools_only(self):
        self.assertEqual(self.listed(), ["handoff_read", "handoff_profile"])

    def test_set_up_lists_the_digest_too(self):
        self.home.write({"backend": "ollama", "model": "m"})
        self.assertEqual(self.listed(), ["handoff_read", "handoff_profile", "handoff_digest"])

    def test_digest_still_answers_when_not_listed(self):
        text, is_error = self.call("handoff_digest", {"paths": ["x"], "question": "q"})
        self.assertTrue(is_error)
        self.assertIn("agenthandoff setup", text)

    def test_read_outlines_code_with_no_model(self):
        path = self.home.file("app.py", 'def handler(event):\n    """Handle one event."""\n    return 1\n')
        text, is_error = self.call("handoff_read", {"paths": [path], "root": str(self.home.work)})
        self.assertFalse(is_error, text)
        self.assertTrue(text.startswith("ROUTE: outline"), text)
        self.assertIn("def handler(event): Handle one event.", text)

    def test_read_finds_a_named_identifier(self):
        path = self.home.file("app.py", "x = 1\nretry_limit = 5\ny = 2\n")
        text, _ = self.call("handoff_read", {"paths": [path], "question": "where is `retry_limit` set?",
                                           "root": str(self.home.work)})
        self.assertTrue(text.startswith("ROUTE: find"), text)
        self.assertIn("app.py:2: retry_limit = 5", text)

    def test_read_of_prose_without_a_model_says_read_it_yourself(self):
        path = self.home.file("notes.txt", "Some thoughts about the plan.\nMore of them.\n")
        text, is_error = self.call("handoff_read", {"paths": [path]})
        self.assertTrue(is_error)
        self.assertIn("no model is set up", text)
        self.assertIn("needs reading", text)                 # one file: "needs", "Read it"
        self.assertIn("Read it yourself", text)
        self.assertIn("Read the files yourself", text)
        self.assertIn("agenthandoff setup", text)            # a beginner learns how to get one

    def test_bad_tool_name_is_refused_before_running(self):
        text, is_error = self.call("handoff_read", {"paths": ["x"], "tool": "grep"})
        self.assertTrue(is_error)
        self.assertIn("'tool' must be one of", text)

    def test_script_is_listed_and_runs(self):
        self.home.write({"scripts": [self.home.counter_script()]})
        self.assertIn("script_count", self.listed())
        path = self.home.file("a.log", "one\ntwo\nthree\n")
        text, is_error = self.call("script_count", {"paths": [path]})
        self.assertFalse(is_error, text)
        self.assertIn("a.log 3 lines", text)

    def test_unknown_script_is_an_unknown_tool(self):
        self.home.write({"scripts": [self.home.counter_script()]})
        r = self.talk({"id": 1, "method": "tools/call", "params": {"name": "script_nope", "arguments": {}}})[1]
        self.assertEqual(r["error"]["code"], -32602)

    def test_read_picks_a_script_whose_question_fits(self):
        self.home.write({"scripts": [self.home.counter_script(question="how many lines", match=["*.log"])]})
        path = self.home.file("a.log", "one\ntwo\n")
        text, _ = self.call("handoff_read", {"paths": [path], "question": "How many lines are there?"})
        self.assertTrue(text.startswith("ROUTE: script count"), text)
        self.assertIn("a.log 2 lines", text)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.home = Home()
        self.addCleanup(self.home.cleanup)

    def run_cli(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-m", "agenthandoff.cli", *args], capture_output=True, text=True,
                              encoding="utf-8", env=self.home.env(), cwd=self.home.work, timeout=60)

    def test_read_routes_logs_to_the_profile(self):
        path = self.home.file("svc.log", "".join(f"2026-11-11 10:00:{i % 60:02d} INFO tick n={i}\n" for i in range(200))
                              + "2026-11-11 10:00:09 ERROR boom\n")
        r = self.run_cli("read", path, "--root", str(self.home.work))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.startswith("ROUTE: profile (1 file of logs)"), r.stdout)
        self.assertIn("ERROR boom", r.stdout)

    def test_read_forced_tool(self):
        path = self.home.file("data.json", '{"a": 1, "b": [1, 2]}')
        r = self.run_cli("read", path, "--tool", "shape", "--root", str(self.home.work))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ROUTE: shape (asked for)", r.stdout)
        self.assertIn("$.b[]: 2 items", r.stdout)

    def test_set_changes_one_run_and_bad_values_exit_2(self):
        path = self.home.file("app.py", "".join(f"def f{i}():\n    pass\n" for i in range(10)))
        r = self.run_cli("read", path, "--set", "outline.max_items=3")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("(7 more definitions not shown)", r.stdout)
        self.assertIn("def f2()", r.stdout)
        self.assertNotIn("def f3()", r.stdout)
        bad = self.run_cli("read", path, "--set", "outline.max_items=nope")
        self.assertEqual(bad.returncode, 2)
        self.assertIn("outline.max_items", bad.stderr)
        typo = self.run_cli("read", path, "--set", "outline.max_itmes=3")
        self.assertEqual(typo.returncode, 2)
        self.assertIn("outline.max_items", typo.stderr)   # suggests the right key
        shape = self.run_cli("read", path, "--set", "no-equals-sign")
        self.assertEqual(shape.returncode, 2)
        self.assertIn("KEY=VALUE", shape.stderr)

    def test_config_set_get_unset(self):
        self.assertEqual(self.run_cli("config", "set", "profile.rare_count", "7").returncode, 0)
        self.assertEqual(json.loads((self.home.path / "config.json").read_text())["profile.rare_count"], 7)
        got = self.run_cli("config", "get", "profile.rare_count")
        self.assertIn("7", got.stdout)
        bad = self.run_cli("config", "set", "profile.rare_count", "0")
        self.assertEqual(bad.returncode, 2)
        self.assertEqual(json.loads((self.home.path / "config.json").read_text())["profile.rare_count"], 7)
        self.assertEqual(self.run_cli("config", "unset", "profile.rare_count").returncode, 0)
        self.assertNotIn("profile.rare_count", json.loads((self.home.path / "config.json").read_text()))

    def test_script_command(self):
        self.home.write({"scripts": [self.home.counter_script()]})
        path = self.home.file("a.txt", "x\ny\n")
        r = self.run_cli("script", "count", path)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("a.txt 2 lines", r.stdout)
        missing = self.run_cli("script", "nope", path)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("scripts: count", missing.stderr)

    def test_failing_script_exits_2(self):
        self.home.write({"scripts": [{"name": "boom", "argv": [sys.executable, "-c", "import sys; sys.exit(3)"]}]})
        r = self.run_cli("script", "boom")
        self.assertEqual(r.returncode, 2)
        self.assertIn("exited with code 3", r.stderr)
        self.assertIn("Do this task yourself", r.stderr)

    def test_stats_counts_reads_and_repeated_model_questions(self):
        path = self.home.file("app.py", "def f():\n    pass\n")
        for _ in range(2):
            self.run_cli("read", path)
        logs = self.home.path / "logs"
        with open(logs / "digest.jsonl", "w", encoding="utf-8") as f:
            for q in ("what does section 3 say?", "what does section 12 say?", "summarize"):
                f.write(json.dumps({"question": q, "provider": "local"}) + "\n")
            f.write(json.dumps({"question": "summarize", "cache": "hit"}) + "\n")
            f.write("not json\n")
        r = self.run_cli("stats")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("read: 2 runs: outline x2", r.stdout)
        self.assertIn("digest (model): 4 runs, 0 not done, 1 answered from the cache", r.stdout)
        self.assertIn("x2  what does section <n> say?", r.stdout)
        self.assertNotIn("summarize", r.stdout)   # once from the model, once from the cache

    def test_stats_with_no_logs(self):
        r = self.run_cli("stats")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("read: 0 runs", r.stdout)


if __name__ == "__main__":
    unittest.main()
