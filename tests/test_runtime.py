"""Tests for everything around the digest: settings, the lock, the GPU check, the
api order of steps, the MCP server, the managed llama-server, model downloads,
setup and check. No real model server, GPU or internet is used: fakes stand in.

Run all tests: python -m unittest discover -s tests
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from support import ROOT, TESTS, FakeBackend, FakeManagedServer, make_settings

from agenthandoff import api, check, config, hardware, models, setup_wizard
from agenthandoff.backend import ModelStatus
from agenthandoff.errors import BackendError, BadSetting, GpuBusy, LockBusy, NoInput, NotConfigured
from agenthandoff.lock import GpuLock, pid_alive


class TempHome(unittest.TestCase):
    """Each test gets an empty AGENTHANDOFF_HOME and no other AGENTHANDOFF_* variables."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        clean = {k: v for k, v in os.environ.items() if not k.startswith("AGENTHANDOFF_")}
        self.env = mock.patch.dict(os.environ, {**clean, "AGENTHANDOFF_HOME": str(self.home)}, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()


# ---------------------------------------------------------------- settings

class ConfigTests(TempHome):
    def test_not_set_up_says_run_setup(self):
        with self.assertRaises(NotConfigured) as e:
            config.load()
        self.assertIn("agenthandoff setup", str(e.exception))

    def test_env_beats_config_file_beats_default(self):
        config.write_config_file({"backend": "ollama", "chunk_tokens": 4000})
        s = config.load()
        self.assertEqual((s.chunk_tokens, s.sources["chunk_tokens"]), (4000, "config.json"))
        self.assertEqual(s.model, config.DEFAULT_OLLAMA_MODEL)
        os.environ["AGENTHANDOFF_CHUNK_TOKENS"] = "3000"
        self.assertEqual(config.load().chunk_tokens, 3000)

    def test_bad_values_name_the_setting(self):
        os.environ["AGENTHANDOFF_BACKEND"] = "ollama"
        for value, words in (("abc", "not a number"), ("0", "between 500 and 8000"), ("9000", "between 500 and 8000")):
            os.environ["AGENTHANDOFF_CHUNK_TOKENS"] = value
            with self.assertRaises(BadSetting) as e:
                config.load()
            self.assertIn(words, str(e.exception))

    def test_llamacpp_needs_a_model_unless_it_is_your_own_server(self):
        os.environ["AGENTHANDOFF_BACKEND"] = "llamacpp"
        with self.assertRaises(NotConfigured):
            config.load()
        os.environ["AGENTHANDOFF_URL"] = "http://127.0.0.1:8080"
        self.assertEqual(config.load().url, "http://127.0.0.1:8080")

    def test_home_folder_paths_are_expanded(self):
        os.environ.update(AGENTHANDOFF_BACKEND="llamacpp", AGENTHANDOFF_MODEL="~/m.gguf",
                          AGENTHANDOFF_LLAMA_SERVER="~/bin/llama-server")
        s = config.load()
        self.assertEqual(Path(s.model), Path.home() / "m.gguf")
        self.assertEqual(Path(s.llama_server), Path.home() / "bin" / "llama-server")

    def test_llama_args_as_list_or_string_keep_windows_paths(self):
        config.write_config_file({"backend": "ollama", "llama_args": ["--lora", r"C:\my models\a.gguf"]})
        self.assertEqual(config.load().llama_args, ["--lora", r"C:\my models\a.gguf"])
        if os.name == "nt":
            os.environ["AGENTHANDOFF_LLAMA_ARGS"] = r'-ngl 20 --lora C:\models\a.gguf --mmproj "C:\my models\b.gguf"'
            expected = ["-ngl", "20", "--lora", r"C:\models\a.gguf", "--mmproj", r"C:\my models\b.gguf"]
        else:
            os.environ["AGENTHANDOFF_LLAMA_ARGS"] = '-ngl 20 --lora /models/a.gguf --mmproj "/my models/b.gguf"'
            expected = ["-ngl", "20", "--lora", "/models/a.gguf", "--mmproj", "/my models/b.gguf"]
        self.assertEqual(config.load().llama_args, expected)


# ---------------------------------------------------------------- the lock

def sleeper():
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])


class LockTests(unittest.TestCase):
    def test_live_foreign_lock_refuses_and_stale_is_taken(self):
        with tempfile.TemporaryDirectory() as t:
            other = sleeper()
            try:
                Path(t, "gpu.lock.json").write_text(json.dumps(
                    {"pid": other.pid, "model": "m", "who": "other", "ts": time.time(), "lease_s": 3600}))
                with self.assertRaises(LockBusy) as e:
                    GpuLock(Path(t), "m", "me").acquire()
                self.assertIn("gpu.lock.json", str(e.exception))
            finally:
                other.kill()
                other.wait()
            lock = GpuLock(Path(t), "m", "me").acquire()   # that pid is gone now: stale, taken
            self.assertEqual(json.loads(Path(t, "gpu.lock.json").read_text())["pid"], os.getpid())
            lock.release()
            self.assertFalse(Path(t, "gpu.lock.json").exists())

    def test_two_takers_at_once_only_one_wins(self):
        with tempfile.TemporaryDirectory() as t:
            results, start = [], threading.Barrier(8)

            def take():
                start.wait()
                try:
                    GpuLock(Path(t), "m", "me").acquire()
                    results.append("got")
                except LockBusy:
                    results.append("busy")
            threads = [threading.Thread(target=take) for _ in range(8)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            self.assertEqual(results.count("got"), 1)

    def test_garbage_lock_is_held_at_first_then_stale(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t, "gpu.lock.json")
            path.write_text("[1, 2]")
            with self.assertRaises(LockBusy):
                GpuLock(Path(t), "m", "me").acquire()
            old = time.time() - 120
            os.utime(path, (old, old))       # as if a crash left it two minutes ago
            GpuLock(Path(t), "m", "me").acquire().release()

    def test_external_lock_honoured(self):
        with tempfile.TemporaryDirectory() as t:
            other = sleeper()
            try:
                ext = Path(t, "OTHER_LOCK.json")
                ext.write_text(json.dumps({"pid": other.pid, "model": "big", "who": "panel"}))
                with self.assertRaises(LockBusy):
                    GpuLock(Path(t), "m", "me", external=str(ext)).acquire()
            finally:
                other.kill()
                other.wait()

    def test_pid_alive(self):
        self.assertTrue(pid_alive(os.getpid()))
        dead = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(259)"])  # 259 = Windows STILL_ACTIVE
        dead.wait()
        self.assertFalse(pid_alive(dead.pid))
        self.assertFalse(pid_alive("not a pid"))


# ---------------------------------------------------------------- the GPU check

class GpuBusyTests(unittest.TestCase):
    def reason(self, gpus, loaded=False, size=6000):
        with mock.patch.object(hardware, "nvidia_gpus", return_value=gpus), \
             mock.patch.object(hardware.time, "sleep"):
            return hardware.busy_reason(make_settings(".", ignore_gpu_busy=False), ModelStatus(loaded, size))

    def gpu(self, util, free, total=12282):
        return [hardware.GpuStatus(0, "Test GPU", util, free, total)]

    def test_busy_gpu_refused_even_if_our_model_is_loaded(self):
        self.assertIn("busy", self.reason(self.gpu(100, 600)))
        self.assertIn("busy", self.reason(self.gpu(100, 600), loaded=True))

    def test_low_memory_refused_unless_size_unknown(self):
        self.assertIn("needs about", self.reason(self.gpu(0, 3000)))
        self.assertIsNone(self.reason(self.gpu(0, 3000), size=None))   # e.g. -ngl: part of it on the CPU

    def test_free_gpu_passes(self):
        self.assertIsNone(self.reason(self.gpu(0, 11000)))
        self.assertIsNone(self.reason(self.gpu(0, 500), loaded=True))

    def test_no_nvidia_is_unknown_not_busy(self):
        self.assertIsNone(self.reason([]))

    def test_gpu_name_with_a_comma(self):
        out = "0, 3, 11000, 12282, NVIDIA RTX 4000 Ada, Laptop\n"
        with mock.patch.object(hardware.shutil, "which", return_value="nvidia-smi"), \
             mock.patch.object(hardware.subprocess, "run", return_value=mock.Mock(stdout=out)):
            gpus = hardware.nvidia_gpus()
        self.assertEqual((gpus[0].name, gpus[0].free_mb), ("NVIDIA RTX 4000 Ada, Laptop", 11000))


# ---------------------------------------------------------------- api: order of steps and logging

class ApiTests(TempHome):
    def setUp(self):
        super().setUp()
        (self.home / "run.log").write_text("2026 ERROR disk quota exceeded on /var/spool\nok\n", encoding="utf-8")
        self.settings = make_settings(self.home)

    def log(self):
        return [json.loads(x) for x in (self.home / "logs" / "digest.jsonl").read_text().splitlines()]

    def test_success_is_logged(self):
        backend = FakeBackend(lambda p: [{"claim": "quota", "quote": "disk quota exceeded"}])
        res = api.run_digest([str(self.home / "run.log")], "q", settings=self.settings, backend=backend, seed=1)
        self.assertEqual(len(res.claims), 1)
        self.assertEqual(self.log()[-1]["claims"], 1)

    def test_bad_path_fails_before_the_backend_starts(self):
        backend = FakeBackend(lambda p: [])
        with self.assertRaises(NoInput):
            api.run_digest([str(self.home / "nope.log")], "q", settings=self.settings, backend=backend)
        self.assertFalse(backend.entered)
        self.assertIn("NoInput", self.log()[-1]["failed"])

    def test_gpu_busy_is_refused_and_logged(self):
        with mock.patch.object(api.hardware, "busy_reason", return_value="GPU busy"):
            with self.assertRaises(GpuBusy):
                api.run_digest([str(self.home / "run.log")], "q", settings=self.settings, backend=FakeBackend(None))
        self.assertIn("GPU busy", self.log()[-1]["failed"])

    def test_server_failure_mid_run_is_logged_with_progress(self):
        backend = FakeBackend(None, fail=BackendError("server died"))
        with self.assertRaises(BackendError):
            api.run_digest([str(self.home / "run.log")], "q", settings=self.settings, backend=backend)
        record = self.log()[-1]
        self.assertIn("server died", record["failed"])
        self.assertEqual(record["chunks"], 1)


# ---------------------------------------------------------------- request bodies

class RequestBodyTests(unittest.TestCase):
    """What each backend sends. Found on a real llama-server: Gemma 4 thinks by
    default and returned empty content, so the llama.cpp body must turn that off."""

    REPLY = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 2}}

    def sent_body(self, backend):
        with mock.patch("agenthandoff.backend.http_json", return_value=self.REPLY) as call:
            reply = backend.chat("system", "user", {"type": "object"}, 10)
        self.assertEqual((reply.text, reply.input_tokens, reply.output_tokens), ("{}", 5, 2))
        return call.call_args.args[1]

    def test_llamacpp_turns_thinking_off(self):
        from agenthandoff.backend import LlamaCppBackend
        backend = LlamaCppBackend(make_settings(Path("."), backend="llamacpp", url="http://127.0.0.1:1"))
        body = self.sent_body(backend)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(body["response_format"]["type"], "json_schema")

    def test_plain_openai_server_gets_no_extra_fields(self):
        from agenthandoff.backend import OpenAIBackend
        body = self.sent_body(OpenAIBackend("http://127.0.0.1:1", "m", 5))
        self.assertNotIn("chat_template_kwargs", body)


# ---------------------------------------------------------------- the managed llama-server

class ManagedServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def server_pid(self):
        return int((self.dir / "pid").read_text())

    def wait_dead(self, pid, seconds=20):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not pid_alive(pid):
                return True
            time.sleep(0.2)
        return False

    def make(self, extra_args=()):
        from agenthandoff.llama_server import ManagedServer
        return ManagedServer("llama-server", "m.gguf", 4096, list(extra_args), self.dir / "log")

    def test_the_server_is_told_to_end_thinking_at_once_when_it_can(self):
        # 2026-11-12: with enable_thinking false, Gemma 4 still thought through every chunk of
        # a "find the smallest value" question until the reply limit; --reasoning-budget 0 fixed it.
        from agenthandoff import llama_server as ls
        with mock.patch.object(ls, "supports_reasoning_budget", return_value=True):
            cmd = self.make(["--threads", "4"]).command("llama-server", 9000)
        self.assertIn("--reasoning-budget", cmd)
        self.assertEqual(cmd[cmd.index("--reasoning-budget") + 1], "0")
        self.assertEqual(cmd[-2:], ["--threads", "4"])

    def test_an_older_server_or_the_users_own_budget_gets_no_budget_flag(self):
        from agenthandoff import llama_server as ls
        with mock.patch.object(ls, "supports_reasoning_budget", return_value=False):
            self.assertNotIn("--reasoning-budget", self.make().command("llama-server", 9000))
        with mock.patch.object(ls, "supports_reasoning_budget", return_value=True):
            cmd = self.make(["--reasoning-budget", "200"]).command("llama-server", 9000)
            self.assertEqual(cmd.count("--reasoning-budget"), 1)
            self.assertEqual(cmd[cmd.index("--reasoning-budget") + 1], "200")
            cmd = self.make(["--reasoning-budget=50"]).command("llama-server", 9000)
            self.assertNotIn("--reasoning-budget", cmd)

    def test_the_budget_probe_reads_the_programs_help(self):
        from agenthandoff import llama_server as ls
        script = self.dir / "fake_help.py"
        script.write_text("print('--reasoning-budget N   token budget for thinking')", encoding="utf-8")
        self.assertTrue(ls._help_mentions([sys.executable, str(script)], "--reasoning-budget"))
        old = self.dir / "old_help.py"
        old.write_text("print('usage: llama-server [options]')", encoding="utf-8")
        self.assertFalse(ls._help_mentions([sys.executable, str(old)], "--reasoning-budget"))
        self.assertFalse(ls._help_mentions([str(self.dir / "missing.exe")], "--reasoning-budget"))

    def test_start_answers_then_stop_ends_the_server(self):
        server = FakeManagedServer(self.dir / "pid", self.dir / "server.log")
        url = server.start()
        try:
            from agenthandoff.net import local_json
            self.assertEqual(local_json(f"{url}/v1/models")["data"][0]["id"], "fake")
        finally:
            server.stop()
        self.assertTrue(self.wait_dead(self.server_pid()))

    def test_server_that_never_gets_ready_is_stopped(self):
        server = FakeManagedServer(self.dir / "pid", self.dir / "server.log", load_seconds=3600)
        with mock.patch("agenthandoff.llama_server.STARTUP_TIMEOUT_S", 3):
            with self.assertRaises(BackendError) as e:
                server.start()
        self.assertIn("did not finish loading", str(e.exception))
        self.assertTrue(self.wait_dead(self.server_pid()), "a server that timed out was left running")

    def test_server_that_fails_to_load_reports_its_log(self):
        server = FakeManagedServer(self.dir / "pid", self.dir / "server.log", fail=True)
        with self.assertRaises(BackendError) as e:
            server.start()
        self.assertIn("failed to load model", str(e.exception))

    def test_server_dies_when_agenthandoff_is_killed(self):
        helper = subprocess.Popen([sys.executable, str(TESTS / "support.py"), "start-and-wait", str(self.dir / "pid")],
                                  stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(helper.stdout.readline().strip(), "up")
            pid = self.server_pid()
            self.assertTrue(pid_alive(pid))
            helper.kill()            # like Ctrl+Break, a command timeout or a closed MCP client
            helper.wait()
            self.assertTrue(self.wait_dead(pid), "llama-server outlived agenthandoff")
            # The watcher exits just after the server (it polls every 0.5 s) and holds the log
            # until then; on Windows the log cannot be deleted while it is held.
            log, deadline = self.dir / "pid.log", time.monotonic() + 10
            while True:
                try:
                    log.unlink()
                    break
                except PermissionError:
                    self.assertLess(time.monotonic(), deadline, "the watcher outlived llama-server")
                    time.sleep(0.2)
        finally:
            if helper.poll() is None:
                helper.kill()
                helper.wait()
            helper.stdout.close()


# ---------------------------------------------------------------- model downloads

class FileServer:
    """Serves one byte string over HTTP with Range support. `cut_after` stops a reply early."""

    def __init__(self, data: bytes):
        self.data, self.cut_after, self.requests = data, None, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                start = 0
                rng = self.headers.get("Range")
                outer.requests.append(rng)
                if rng:
                    start = int(rng.split("=")[1].split("-")[0])
                    if start >= len(outer.data):
                        self.send_response(416)
                        self.end_headers()
                        return
                body = outer.data[start:]
                self.send_response(206 if rng else 200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body[:outer.cut_after] if outer.cut_after else body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/model.gguf"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.data = os.urandom(300_000)
        self.server = FileServer(self.data)
        url = self.server.url

        class TestModel(models.CatalogModel):
            @property
            def url(self):
                return url
        self.model = TestModel("t", "test", "repo", "model.gguf", len(self.data),
                               hashlib.sha256(self.data).hexdigest(), 1)

    def tearDown(self):
        self.server.close()
        self.tmp.cleanup()

    def test_full_download_is_checked_and_marked(self):
        real_hash = models.sha256_of
        with mock.patch.object(models, "sha256_of", side_effect=real_hash) as hashed:
            path = models.download(self.model, self.home, progress=False)
            self.assertEqual(path.read_bytes(), self.data)
            self.assertTrue(models.is_ready(self.model, self.home))
            models.download(self.model, self.home, progress=False)
        self.assertEqual(len(self.server.requests), 1)        # second time: no download...
        self.assertEqual(hashed.call_count, 1)                # ...and no re-hash of 4.6 GB

    def test_dropped_connection_keeps_the_part_and_resumes(self):
        self.server.cut_after = 100_000
        with self.assertRaises(BackendError) as e:
            models.download(self.model, self.home, progress=False)
        self.assertIn("Run setup again", str(e.exception))
        part = models.models_dir(self.home) / "model.gguf.part"
        self.assertEqual(part.stat().st_size, 100_000)
        self.server.cut_after = None
        self.assertEqual(models.download(self.model, self.home, progress=False).read_bytes(), self.data)
        self.assertEqual(self.server.requests[-1], "bytes=100000-")

    def test_complete_part_file_is_finished_without_a_request(self):
        part = models.models_dir(self.home) / "model.gguf.part"
        part.parent.mkdir(parents=True)
        part.write_bytes(self.data)                            # e.g. Ctrl+C while hashing
        models.download(self.model, self.home, progress=False)
        self.assertEqual(self.server.requests, [])

    def test_corrupt_download_is_deleted(self):
        target = models.models_dir(self.home) / "model.gguf"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"x" * len(self.data))              # right size, wrong content
        with self.assertRaises(BackendError) as e:
            models.download(self.model, self.home, progress=False)
        self.assertIn("checksum", str(e.exception))
        self.assertFalse(target.exists())


# ---------------------------------------------------------------- setup and check

class SetupTests(TempHome):
    def test_without_a_terminal_nothing_is_installed_or_downloaded(self):
        ask = setup_wizard.Prompter(assume_yes=False, interactive=False)
        with redirect_stdout(io.StringIO()):
            self.assertFalse(ask.confirm("Download 4.6 GB?"))
            self.assertTrue(setup_wizard.Prompter(assume_yes=True, interactive=False).confirm("Download?"))

    def test_own_gguf_and_llama_server_flags_ask_nothing(self):
        gguf = self.home / "my.gguf"
        gguf.write_bytes(b"GGUF")
        ask = setup_wizard.Prompter(assume_yes=False, interactive=False)
        with mock.patch("agenthandoff.check.run_check", return_value=0), redirect_stdout(io.StringIO()):
            code = setup_wizard.run_setup("llamacpp", str(gguf), None, sys.executable, prompter=ask)
        self.assertEqual(code, 0)
        saved = json.loads(config.config_path().read_text())
        self.assertEqual(saved, {"backend": "llamacpp", "llama_server": sys.executable, "model": str(gguf.resolve())})

    def test_broken_config_is_backed_up_not_a_crash(self):
        config.config_path().write_text("{not json")
        with mock.patch.object(setup_wizard, "_setup_ollama", return_value={"backend": "ollama", "model": "m"}), \
             mock.patch("agenthandoff.check.run_check", return_value=0), redirect_stdout(io.StringIO()):
            self.assertEqual(setup_wizard.run_setup("ollama", prompter=setup_wizard.Prompter(True, False)), 0)
        self.assertTrue(config.config_path().with_suffix(".broken.json").exists())

    def test_setup_says_how_to_let_read_use_the_model(self):
        for saved, told in (({}, True), ({"route.allow_model": True}, False)):
            config.write_config_file(saved)
            out = io.StringIO()
            with mock.patch.object(setup_wizard, "_setup_ollama", return_value={"backend": "ollama", "model": "m"}), \
                 mock.patch("agenthandoff.check.run_check", return_value=0), redirect_stdout(out):
                setup_wizard.run_setup("ollama", prompter=setup_wizard.Prompter(True, False))
            self.assertEqual("config set route.allow_model true" in out.getvalue(), told, out.getvalue())

    def test_switching_route_keeps_tuning_but_drops_old_route(self):
        config.write_config_file({"backend": "llamacpp", "model": "/x.gguf", "llama_server": "/ls", "timeout": 900})
        with mock.patch.object(setup_wizard, "_setup_ollama", return_value={"backend": "ollama", "model": "m"}), \
             mock.patch("agenthandoff.check.run_check", return_value=0), redirect_stdout(io.StringIO()):
            setup_wizard.run_setup("ollama", prompter=setup_wizard.Prompter(True, False))
        self.assertEqual(json.loads(config.config_path().read_text()),
                         {"timeout": 900, "backend": "ollama", "model": "m"})

    def test_new_model_goes_first_and_other_providers_stay_as_fallbacks(self):
        agy = {"type": "command", "argv": ["agy", "-p"]}
        config.write_config_file({"providers": {"agy": agy}, "provider": ["agy", "default"], "allow_remote": True})
        with mock.patch.object(setup_wizard, "_setup_ollama", return_value={"backend": "ollama", "model": "m"}), \
             mock.patch("agenthandoff.check.run_check", return_value=0), redirect_stdout(io.StringIO()):
            setup_wizard.run_setup("ollama", prompter=setup_wizard.Prompter(True, False))
        saved = json.loads(config.config_path().read_text())
        self.assertEqual(saved["provider"], ["default", "agy"])
        self.assertEqual(saved["providers"], {"agy": agy})
        self.assertEqual(config.load().provider_chain, ["default", "agy"])


class CheckTests(TempHome):
    def run_check(self, backend):
        config.write_config_file({"backend": "ollama"})
        out = io.StringIO()
        with mock.patch.object(check, "make_backend", return_value=backend), \
             mock.patch.object(check.hardware, "nvidia_gpus", return_value=[]), \
             mock.patch.object(check.hardware, "memory_for_models", return_value=None), redirect_stdout(out):
            return check.run_check(), out.getvalue()

    def test_all_steps_pass(self):
        backend = FakeBackend(None)
        backend.chat = lambda *a, **k: mock.Mock(text='{"ok": true}')
        code, out = self.run_check(backend)
        self.assertEqual(code, 0, out)
        self.assertIn("[  ok] code tools", out)
        self.assertIn("[  ok] default: test prompt", out)
        self.assertIn("[  --] default: GPU", out)

    def test_missing_model_fails_with_the_fix(self):
        backend = FakeBackend(None)
        backend.describe_model = mock.Mock(side_effect=BackendError("m is not downloaded. Run `ollama pull m`."))
        code, out = self.run_check(backend)
        self.assertEqual(code, 2)
        self.assertIn("[FAIL] default: model: m is not downloaded", out)

    def test_no_model_still_reports_the_code_tools(self):
        config.write_config_file({})
        out = io.StringIO()
        with redirect_stdout(out):
            code = check.run_check()
        self.assertEqual(code, 0)       # no model is a normal state: the code tools need none
        self.assertIn("[  ok] code tools", out.getvalue())
        self.assertIn("none set up (optional)", out.getvalue())
        self.assertIn("handoff_read and handoff_profile still work", out.getvalue())

    def test_a_failing_provider_does_not_hide_the_next_one(self):
        from agenthandoff.providers import Capabilities
        config.write_config_file({"providers": {
            "a": {"type": "command", "argv": ["x"]}, "b": {"type": "command", "argv": ["y"]}},
            "provider": ["a", "b"], "allow_remote": True})
        good = FakeBackend(None)
        good.chat = lambda *a, **k: mock.Mock(text='{"ok": true}')
        good.capabilities = Capabilities(False, False, True)
        bad = FakeBackend(None)
        bad.describe_model = mock.Mock(side_effect=BackendError("a is broken"))
        bad.capabilities = good.capabilities
        out = io.StringIO()
        with mock.patch.object(check, "make_backend", side_effect=[bad, good]), redirect_stdout(out):
            code = check.run_check()
        text = out.getvalue()
        self.assertEqual(code, 2)
        self.assertIn("[FAIL] a: model: a is broken", text)
        self.assertIn("[  ok] b: test prompt", text)
        self.assertNotIn("b: GPU", text)       # a remote command provider never touches the GPU
        self.assertIn("falls back in order", text)


# ---------------------------------------------------------------- the MCP server and the CLI

class McpTests(unittest.TestCase):
    def talk(self, messages, extra_env=None):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), **(extra_env or {}))
        lines = [m if isinstance(m, str) else json.dumps(m) for m in messages]
        out = subprocess.run([sys.executable, "-m", "agenthandoff.mcp_server"], input="\n".join(lines) + "\n",
                             capture_output=True, text=True, env=env, timeout=60).stdout
        return [json.loads(x) for x in out.splitlines()]

    def test_initialize_list_and_unknown_method(self):
        replies = self.talk([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "nope"}])
        self.assertEqual([r["id"] for r in replies], [1, 2, 3])
        self.assertEqual(replies[0]["result"]["protocolVersion"], "2025-06-18")
        self.assertEqual(replies[1]["result"]["tools"][0]["name"], "handoff_read")   # code first
        self.assertEqual(replies[2]["error"]["code"], -32601)

    def test_non_object_message_does_not_kill_the_server(self):
        replies = self.talk(["[1]", "5", {"jsonrpc": "2.0", "id": 7, "method": "ping"}])
        self.assertEqual(replies[-1], {"jsonrpc": "2.0", "id": 7, "result": {}})

    def test_bad_arguments_are_named(self):
        replies = self.talk([{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": "handoff_digest", "arguments": {"paths": ["x"]}}}])
        text = replies[0]["result"]["content"][0]["text"]
        self.assertTrue(replies[0]["result"]["isError"])
        self.assertIn("bad arguments", text)
        self.assertIn("'question'", text)

    def test_not_set_up_says_do_it_yourself(self):
        with tempfile.TemporaryDirectory() as home:
            env = {k: "" for k in os.environ if k.startswith("AGENTHANDOFF_")}
            env["AGENTHANDOFF_HOME"] = home
            replies = self.talk([{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
                "name": "handoff_digest", "arguments": {"paths": ["x"], "question": "q"}}}], env)
        text = replies[0]["result"]["content"][0]["text"]
        self.assertTrue(replies[0]["result"]["isError"])
        self.assertIn("agenthandoff setup", text)
        self.assertIn("Do this task yourself", text)

    def test_profile_is_listed_and_runs_without_setup_or_question(self):
        with tempfile.TemporaryDirectory() as home:
            Path(home, "a.log").write_text("tick\n" * 5 + "FATAL once\n")
            env = {k: "" for k in os.environ if k.startswith("AGENTHANDOFF_")}
            env["AGENTHANDOFF_HOME"] = home              # not set up: profile needs no model
            replies = self.talk([
                {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                    "name": "handoff_profile", "arguments": {"paths": [str(Path(home, "a.log"))], "root": home}}},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                    "name": "handoff_profile", "arguments": {"paths": [str(Path(home, "nope.log"))]}}},
                {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
                    "name": "handoff_profile", "arguments": {}}}], env)
        by_id = {r["id"]: r for r in replies}
        self.assertEqual([t["name"] for t in by_id[1]["result"]["tools"]], ["handoff_read", "handoff_profile"])   # no model set up: no digest
        self.assertFalse(by_id[2]["result"]["isError"])
        self.assertIn("LOG PROFILE: a.log - 6 lines", by_id[2]["result"]["content"][0]["text"])
        self.assertIn("L6: FATAL once", by_id[2]["result"]["content"][0]["text"])
        self.assertTrue(by_id[3]["result"]["isError"])
        self.assertIn("Read the log yourself", by_id[3]["result"]["content"][0]["text"])
        self.assertTrue(by_id[4]["result"]["isError"])
        self.assertIn("'paths'", by_id[4]["result"]["content"][0]["text"])

    def test_unknown_tool_is_an_error(self):
        replies = self.talk([{"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                              "params": {"name": "nope", "arguments": {}}}])
        self.assertEqual(replies[0]["error"]["code"], -32602)


class CliTests(unittest.TestCase):
    def test_missing_path_exits_2_with_the_reason(self):
        with tempfile.TemporaryDirectory() as home:
            Path(home, "config.json").write_text(json.dumps({"backend": "ollama"}))
            env = {k: v for k, v in os.environ.items() if not k.startswith("AGENTHANDOFF_")}
            env.update(AGENTHANDOFF_HOME=home, PYTHONPATH=str(ROOT / "src"))
            r = subprocess.run([sys.executable, "-m", "agenthandoff.cli", "digest", str(Path(home, "nope.log")),
                                "-q", "q"], capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(r.returncode, 2)
        self.assertIn("not found", r.stderr)
        self.assertIn("Do this task yourself", r.stderr)

    def run_profile(self, *paths):
        env = {k: v for k, v in os.environ.items() if not k.startswith("AGENTHANDOFF_")}
        env["PYTHONPATH"] = str(ROOT / "src")
        env["AGENTHANDOFF_HOME"] = tempfile.mkdtemp()     # never the user's real folder
        return subprocess.run([sys.executable, "-m", "agenthandoff.cli", "profile", *paths],
                              capture_output=True, text=True, env=env, timeout=60)

    def test_profile_prints_and_exits_0_without_setup(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "a.log").write_text("tick\n" * 5)
            r = self.run_profile(str(Path(d, "a.log")), "--root", d)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("LOG PROFILE: a.log - 5 lines", r.stdout)

    def test_profile_of_a_missing_file_exits_2_and_says_read_it_yourself(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.run_profile(str(Path(d, "nope.log")))
        self.assertEqual(r.returncode, 2)
        self.assertIn("Read the log yourself", r.stderr)


if __name__ == "__main__":
    unittest.main()
