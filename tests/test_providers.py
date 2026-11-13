"""Tests for providers: every adapter answers the same chat() contract, capabilities decide
the GPU lock and the remote rule, and the fallback chain. A fake CLI, a fake HTTP server and
a plugin module stand in for real models."""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from support import FakeBackend, TESTS, make_settings

from agenthandoff import api, config, providers as pv
from agenthandoff.backend import Reply
from agenthandoff.errors import BackendError, BadSetting, GpuBusy, RemoteRefused, ThinkingOnly, Unavailable

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
FAKE_CLI = str(TESTS / "fake_cli.py")


def settings_with(providers, chain=None, allow_remote=False, **kw):
    return make_settings(tempfile.gettempdir(), providers=providers, provider_chain=chain or list(providers),
                         allow_remote=allow_remote, **kw)


def cli(mode, *extra, **spec):
    return {"type": "command", "argv": [sys.executable, FAKE_CLI, mode, *extra], "remote": False, **spec}


class Recorder(BaseHTTPRequestHandler):
    """An OpenAI-compatible server that records what it was sent."""
    seen: list = []
    reply: dict | None = None      # a test may set the next chat reply

    def log_message(self, *a):
        pass

    def do_GET(self):
        self._reply({"data": [{"id": "m"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Recorder.seen.append({"headers": dict(self.headers), "body": body, "path": self.path})
        self._reply(Recorder.reply or {"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
                                       "usage": {"prompt_tokens": 7, "completion_tokens": 2}})

    def _reply(self, data):
        raw = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class HttpServerCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Recorder)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        Recorder.seen = []


class ContractTests(HttpServerCase):
    """Every adapter: chat() returns a Reply whose text holds the JSON the schema asks for."""

    def specs(self):
        return {"openai": {"type": "openai", "url": self.url, "model": "m"},
                "command": cli("garbage-once", "--count-file", str(Path(self.tmp.name, "n"))),
                "python": {"type": "python", "factory": "plugin_provider:make", "options": {"prefix": "p"},
                           "remote": False}}

    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_every_adapter_answers_chat(self):
        for name, spec in self.specs().items():
            with self.subTest(adapter=name):
                with pv.make_backend(settings_with({name: spec}), name) as backend:
                    reply = backend.chat("system", "user text", SCHEMA if name != "python" else None, max_output=50)
                self.assertIsInstance(reply, Reply)
                self.assertTrue(reply.text.strip())
                self.assertIsNotNone(pv.parse_json(reply.text), reply.text)
                self.assertTrue(backend.describe_model())


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.count = str(Path(self.tmp.name, "calls"))

    def backend(self, spec, **kw):
        return pv.make_backend(settings_with({"c": spec}, **kw), "c")

    def calls(self):
        return len(Path(self.count).read_text().splitlines()) if Path(self.count).exists() else 0

    def test_prompt_goes_on_stdin_and_model_is_substituted(self):
        b = self.backend(cli("echo", "--model", "{model}", model="tiny"))
        text = b.chat("SYS", "héllo " + "y" * 50_000).text     # far above the Windows command-line limit
        data = json.loads(text)
        self.assertTrue(data["answer"].startswith("tiny: SYS"))
        self.assertGreater(data["chars"], 50_000)

    @unittest.skipUnless(os.name == "nt", "cmd.exe runs .cmd and .bat files only on Windows")
    def test_a_batch_file_never_gets_the_prompt_on_its_command_line(self):
        shim = Path(self.tmp.name, "echoargs.cmd")
        shim.write_bytes(b"@echo off\r\necho got: %*\r\n")
        marker = Path(self.tmp.name, "pwned.txt")
        b = self.backend({"type": "command", "argv": [str(shim), "-p", "{prompt}"], "remote": False})
        with self.assertRaises(pv.BackendError) as e:
            b.chat("SYS", f'notes" & echo PWNED > "{marker}" & echo "')
        self.assertIn("cmd.exe", str(e.exception))
        self.assertIn("stdin", str(e.exception))
        self.assertFalse(marker.exists())
        on_stdin = self.backend({"type": "command", "argv": [str(shim), "-p"], "remote": False})
        self.assertIn("got: -p", on_stdin.chat("SYS", 'text with " & | < >').text)

    def test_a_digest_names_a_cli_provider_that_has_no_model(self):
        with tempfile.TemporaryDirectory() as home:
            path = Path(home, "notes.txt")
            path.write_text("The retry limit is five.\n", encoding="utf-8")
            settings = make_settings(home, providers={"c": cli("echo")}, provider_chain=["c"])
            result = api.run_digest([str(path)], "what is the retry limit?", settings=settings, recall_check=False)
        self.assertEqual(result.stats["model"], "c")

    def test_prompt_placeholder_puts_the_prompt_in_argv(self):
        b = self.backend(cli("echo", "--prompt", "{prompt}"))
        data = json.loads(b.chat("SYS", "héllo").text)
        self.assertTrue(data["answer"].startswith("none: SYS"), data)
        self.assertEqual(data["chars"], len("SYS\n\nhéllo"))
        with self.assertRaises(BackendError) as e:
            b.chat("SYS", "y" * 40_000)
        self.assertIn("too long for a command line", str(e.exception))
        self.assertIn("chunk_tokens", str(e.exception))

    def test_schema_is_asked_for_in_the_prompt(self):
        b = self.backend(cli("text"))
        text = b.chat("SYS", "user", SCHEMA).text
        self.assertIn("JSON", text)

    def test_failure_modes_are_backend_errors_with_the_reason(self):
        for mode, words in (("fail", "boom"), ("empty", "nothing"), ("sleep", "no answer after")):
            with self.subTest(mode=mode):
                b = self.backend(cli(mode, timeout=2))
                with self.assertRaises(BackendError) as e:
                    b.chat("s", "u")
                self.assertIn(words, str(e.exception))

    def test_unparseable_json_is_retried_once(self):
        b = self.backend(cli("garbage-once", "--count-file", self.count))
        self.assertEqual(json.loads(b.chat("s", "u", SCHEMA).text), {"ok": True})
        self.assertEqual(self.calls(), 2)
        b = self.backend(cli("garbage", "--count-file", self.count))
        self.assertEqual(b.chat("s", "u", SCHEMA).text.strip(), "no json here")   # the digest then counts it unreadable
        self.assertEqual(self.calls(), 4)

    def test_huge_output_is_cut_and_marked(self):
        reply = self.backend(cli("huge")).chat("s", "u")
        self.assertTrue(reply.truncated)
        self.assertLessEqual(len(reply.text), pv.MAX_REPLY_CHARS)

    def test_missing_program_is_reported_by_describe_and_chat(self):
        b = self.backend({"type": "command", "argv": ["no-such-program-xyz"], "remote": False})
        with self.assertRaises(BackendError) as e:
            b.describe_model()
        self.assertIn("no-such-program-xyz", str(e.exception))
        with self.assertRaises(BackendError):
            b.chat("s", "u")


class OpenAITests(HttpServerCase):
    def test_a_reply_spent_on_thinking_says_how_to_turn_it_off(self):
        # what Ollama's /v1 returned for Gemma 4 with a JSON schema, 2026-11-11
        Recorder.reply = {"choices": [{"message": {"content": "", "reasoning": "The user wants JSON..."},
                                       "finish_reason": "length"}], "usage": {"completion_tokens": 20}}
        self.addCleanup(setattr, Recorder, "reply", None)
        backend = pv.make_backend(settings_with({"o": {"type": "openai", "url": self.url, "model": "m"}}), "o")
        with self.assertRaises(ThinkingOnly) as e:
            backend.chat("s", "u", SCHEMA)
        # its own type, so a digest can skip one such reply instead of failing
        self.assertIn("spent the reply thinking", str(e.exception))
        self.assertIn('"reasoning_effort": "none"', str(e.exception))

    def test_key_from_env_headers_and_extra_body_are_sent(self):
        spec = {"type": "openai", "url": self.url + "/", "model": "m", "api_key_env": "MY_TEST_KEY",
                "headers": {"X-Team": "t1"}, "extra_body": {"top_k": 5}}
        with mock.patch.dict(os.environ, {"MY_TEST_KEY": "sk-abc"}):
            reply = pv.make_backend(settings_with({"o": spec}), "o").chat("s", "u", SCHEMA)
        seen = Recorder.seen[-1]
        self.assertEqual(seen["path"], "/v1/chat/completions")
        self.assertEqual(seen["headers"]["Authorization"], "Bearer sk-abc")
        self.assertEqual(seen["headers"]["X-Team"], "t1")
        self.assertEqual(seen["body"]["top_k"], 5)
        self.assertEqual((reply.input_tokens, reply.output_tokens), (7, 2))

    def test_missing_key_variable_says_which(self):
        spec = {"type": "openai", "url": self.url, "model": "m", "api_key_env": "UNSET_KEY_VAR_123"}
        with self.assertRaises(BackendError) as e:
            pv.make_backend(settings_with({"o": spec}), "o").chat("s", "u")
        self.assertIn("UNSET_KEY_VAR_123", str(e.exception))

    def test_schema_can_be_turned_off_for_servers_that_reject_it(self):
        spec = {"type": "openai", "url": self.url, "model": "m", "supports_schema": False}
        pv.make_backend(settings_with({"o": spec}), "o").chat("s", "u", SCHEMA)
        body = Recorder.seen[-1]["body"]
        self.assertNotIn("response_format", body)
        self.assertIn("JSON", body["messages"][0]["content"])


class PythonProviderTests(unittest.TestCase):
    def test_bad_factories_say_what_is_wrong(self):
        for factory, words in (("nope_module_xyz:make", "cannot import"), ("plugin_provider:missing", "missing"),
                               ("plugin_provider:NOT_CALLABLE", "callable"), ("plugin_provider", "module:name")):
            with self.subTest(factory=factory):
                with self.assertRaises(BackendError) as e:
                    pv.make_backend(settings_with({"p": {"type": "python", "factory": factory, "remote": False}}), "p")
                self.assertIn(words, str(e.exception))

    def test_a_plugin_error_in_describe_model_is_a_backend_error(self):
        b = pv.make_backend(settings_with({"p": {"type": "python", "factory": "plugin_provider:BrokenDescribe",
                                                 "remote": False}}), "p")
        with self.assertRaises(BackendError) as e:
            b.describe_model()
        self.assertIn("model gone. Run `pull it`.", str(e.exception))

    def test_the_example_plugin_loads(self):
        example = Path(__file__).resolve().parents[1] / "examples" / "providers" / "ollama_http.py"
        b = pv.make_backend(settings_with({"p": {"type": "python", "factory": f"{example}:make",
                                                 "options": {"model": "m"}}}), "p")
        self.assertEqual(b.capabilities, pv.Capabilities(True, True, False))

    def test_plugin_capabilities_are_used(self):
        b = pv.make_backend(settings_with({"p": {"type": "python", "factory": "plugin_provider:WithCapabilities"}}), "p")
        self.assertTrue(b.capabilities.uses_local_gpu)
        self.assertFalse(b.capabilities.remote)


class FlatSettingsTests(unittest.TestCase):
    """Settings built by hand (the Python API) with no providers table."""

    def test_each_flat_backend_makes_a_backend(self):
        with tempfile.TemporaryDirectory() as home:
            for backend, kind in (("ollama", "OllamaBackend"), ("openai", "OpenAIBackend")):
                with self.subTest(backend=backend):
                    made = pv.make_backend(make_settings(home, backend=backend, url="http://localhost:9",
                                                         llama_args=["-ngl", "9"],
                                                         llama_server="/x/llama-server"))
                    self.assertEqual(type(made).__name__, kind)
            made = pv.make_backend(make_settings(home, backend="llamacpp", url=None, model=__file__,
                                                 llama_args=["-ngl", "9"], llama_server=sys.executable))
            self.assertEqual(type(made).__name__, "LlamaCppBackend")


class CapabilityTests(unittest.TestCase):
    def caps(self, spec):
        return pv.capabilities(spec)

    def test_local_and_remote_by_type_and_url(self):
        self.assertEqual(self.caps({"type": "llamacpp", "model": "m.gguf"}), pv.Capabilities(True, True, False))
        self.assertEqual(self.caps({"type": "ollama", "url": "http://127.0.0.1:11434"}), pv.Capabilities(True, True, False))
        self.assertEqual(self.caps({"type": "openai", "url": "http://localhost:1234", "model": "m"}),
                         pv.Capabilities(True, True, False))
        self.assertEqual(self.caps({"type": "openai", "url": "https://api.example.com", "model": "m"}),
                         pv.Capabilities(False, True, True))
        self.assertEqual(self.caps({"type": "command", "argv": ["x"]}), pv.Capabilities(False, False, True))
        self.assertEqual(self.caps({"type": "command", "argv": ["x"], "remote": False, "uses_local_gpu": True}),
                         pv.Capabilities(True, False, False))

    def test_lookalike_hosts_are_not_loopback(self):
        for url in ("http://localhost.evil.com", "http://127.0.0.1.nip.io", "http://10.0.0.5:11434"):
            self.assertTrue(self.caps({"type": "openai", "url": url, "model": "m"}).remote, url)
        for url in ("http://127.0.0.2:8080", "http://[::1]:8080", "http://LOCALHOST:1"):
            self.assertFalse(self.caps({"type": "openai", "url": url, "model": "m"}).remote, url)

    def test_remote_provider_needs_allow_remote(self):
        spec = {"type": "command", "argv": [sys.executable, FAKE_CLI, "echo"]}
        with self.assertRaises(RemoteRefused) as e:
            pv.make_backend(settings_with({"cloud": spec}), "cloud")
        self.assertIn("allow_remote", str(e.exception))
        self.assertTrue(pv.make_backend(settings_with({"cloud": spec}, allow_remote=True), "cloud"))

    def test_unknown_type_is_never_treated_as_openai(self):
        with self.assertRaises(BadSetting):
            pv.make_backend(settings_with({"x": {"type": "mystery"}}), "x")


class ApiProviderTests(unittest.TestCase):
    """run_digest takes the GPU lock and GPU check only for providers that use the local GPU,
    and walks the fallback chain."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name, "a.txt")
        self.file.write_text("alpha line\nbeta line\n")

    def settings(self, providers, chain):
        return make_settings(self.tmp.name, providers=providers, provider_chain=chain, allow_remote=True,
                             ignore_gpu_busy=False)

    def run_with(self, settings, backends):
        made = []

        def fake_make(s, name):
            made.append(name)
            b = backends[name]
            if isinstance(b, Exception):
                raise b
            return b
        with mock.patch.object(api.pv, "make_backend", side_effect=fake_make):
            result = api.run_digest([str(self.file)], "q", settings=settings, recall_check=False)
        return result, made

    def fake(self, gpu):
        b = FakeBackend(lambda prompt: [])
        b.capabilities = pv.Capabilities(gpu, True, not gpu)
        b.name = "x"
        return b

    def test_no_gpu_lock_or_check_for_a_provider_off_the_gpu(self):
        s = self.settings({"cli": {"type": "command", "argv": ["x"]}}, ["cli"])
        with mock.patch.object(api, "GpuLock", side_effect=AssertionError("lock taken")), \
             mock.patch.object(api.hardware, "busy_reason", side_effect=AssertionError("GPU checked")):
            result, _ = self.run_with(s, {"cli": self.fake(gpu=False)})
        self.assertEqual(result.stats["provider"], "cli")

    def test_gpu_provider_takes_the_lock_and_checks_the_gpu(self):
        s = self.settings({"g": {"type": "ollama", "model": "m"}}, ["g"])
        with mock.patch.object(api.hardware, "busy_reason", return_value="busy") as check:
            with self.assertRaises(GpuBusy):
                self.run_with(s, {"g": self.fake(gpu=True)})
        check.assert_called_once()

    def test_fallback_chain_uses_the_next_provider_and_says_why(self):
        s = self.settings({"g": {"type": "ollama", "model": "m"}, "cli": {"type": "command", "argv": ["x"]}}, ["g", "cli"])
        with mock.patch.object(api.hardware, "busy_reason", return_value="the GPU is busy"):
            result, made = self.run_with(s, {"g": self.fake(gpu=True), "cli": self.fake(gpu=False)})
        self.assertEqual(made, ["g", "cli"])
        self.assertEqual(result.stats["provider"], "cli")
        self.assertIn("g: the GPU is busy", result.stats["fell_back"][0])

    def test_all_providers_failing_lists_each_reason(self):
        s = self.settings({"a": {"type": "command", "argv": ["x"]}, "b": {"type": "command", "argv": ["y"]}}, ["a", "b"])
        with self.assertRaises(Unavailable) as e:
            self.run_with(s, {"a": BackendError("a broke"), "b": RemoteRefused("b refused")})
        self.assertIn("a broke", str(e.exception))
        self.assertIn("b refused", str(e.exception))

    def test_input_errors_do_not_try_other_providers(self):
        s = self.settings({"a": {"type": "command", "argv": ["x"]}}, ["a"])
        with mock.patch.object(api.pv, "make_backend") as make, self.assertRaises(Unavailable):
            api.run_digest([str(Path(self.tmp.name, "missing.txt"))], "q", settings=s)
        make.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class BatchGuardTests(unittest.TestCase):
    def test_every_character_cmd_treats_as_special_is_refused(self):
        from agenthandoff import batch
        with mock.patch.object(batch.os, "name", "nt"):
            for ch in '"&|<>^%!\r\n':
                self.assertIsNotNone(batch.unsafe_arg("C:/tools/x.cmd", [f"a{ch}b"]), repr(ch))
                self.assertIsNotNone(batch.unsafe_arg("C:/tools/X.BAT", [f"a{ch}b"]), repr(ch))
                self.assertIsNone(batch.unsafe_arg("C:/tools/x.exe", [f"a{ch}b"]), repr(ch))
            self.assertIsNone(batch.unsafe_arg("C:/tools/x.cmd", ["C:/My Files/build (1).log", "why it failed?"]))
        with mock.patch.object(batch.os, "name", "posix"):
            self.assertIsNone(batch.unsafe_arg("/usr/bin/x.cmd", ['a"&b']))


class ContextSizeTests(unittest.TestCase):
    """A provider's context_tokens sets the server's window where agenthandoff starts or
    asks for it, and caps the file text sent per call everywhere, so a small window never
    cuts a chunk silently."""

    def test_context_tokens_is_checked(self):
        for bad in (100, "big", True, 10**9):
            with self.subTest(bad=bad):
                with self.assertRaises(BadSetting) as e:
                    pv.check_spec("x", {"type": "ollama", "context_tokens": bad}, "settings")
                self.assertIn("context_tokens", str(e.exception))
        pv.check_spec("x", {"type": "openai", "url": "http://h", "model": "m", "context_tokens": 4096}, "settings")

    def test_a_window_in_llama_args_is_refused_for_context_tokens(self):
        # A -c in llama_args would fight context_tokens, and chunks would be sized for the wrong window.
        for args in (["-c", "4096"], ["--ctx-size", "4096"], ["--ctx-size=4096"], ["-ngl", "20", "-c4096"]):
            with self.subTest(args=args):
                with self.assertRaises(BadSetting) as e:
                    pv.check_spec("x", {"type": "llamacpp", "model": "m.gguf", "llama_args": args}, "settings")
                self.assertIn("context_tokens", str(e.exception))
        pv.check_spec("x", {"type": "llamacpp", "model": "m.gguf", "llama_args": ["-ngl", "20", "--cont-batching"]},
                      "settings")

    def test_the_chunk_size_used_is_recorded(self):
        with tempfile.TemporaryDirectory() as home:
            path = Path(home, "notes.txt")
            path.write_text("the retry limit is five\n" * 50, encoding="utf-8")
            fake = FakeBackend(lambda prompt: [])
            fake.context_tokens = 4096
            result = api.run_digest([str(path)], "q", recall_check=False, settings=make_settings(home), backend=fake)
        self.assertEqual(result.stats["chunk_tokens"], 2048)

    def test_the_window_reaches_ollama_and_the_managed_llama_server(self):
        s = settings_with({"o": {"type": "ollama", "context_tokens": 32768},
                           "l": {"type": "llamacpp", "model": "m.gguf", "context_tokens": 16384}})
        o = pv.make_backend(s, "o")
        with mock.patch("agenthandoff.backend.local_json", return_value={"message": {"content": "{}"}}) as call:
            o.chat("s", "u")
        self.assertEqual(call.call_args.args[1]["options"]["num_ctx"], 32768)
        self.assertEqual(pv.make_backend(s, "l").server.context, 16384)
        self.assertEqual(pv.make_backend(settings_with({"d": {"type": "ollama"}}), "d").context_tokens,
                         config.CONTEXT_TOKENS)

    def test_the_smallest_window_in_the_chain_caps_each_chunk(self):
        big = {"type": "ollama"}
        small = {"type": "openai", "url": "http://h", "model": "m", "context_tokens": 4096}
        self.assertEqual(pv.chunk_limit(settings_with({"a": big}, chunk_tokens=6000)), 6000)
        self.assertEqual(pv.chunk_limit(settings_with({"a": big, "b": small}, chunk_tokens=6000)), 2048)
        self.assertEqual(pv.chunk_limit(settings_with({"b": small}, chunk_tokens=1000)), 1000)

    def test_a_digest_sends_chunks_that_fit_a_small_window(self):
        with tempfile.TemporaryDirectory() as home:
            path = Path(home, "notes.txt")
            path.write_text("".join(f"Line {i}: the retry limit for job {i} is five.\n" for i in range(1500)),
                            encoding="utf-8")
            chunks = {}
            for window in (4096, None):
                fake = FakeBackend(lambda prompt: [])
                if window:
                    fake.context_tokens = window
                result = api.run_digest([str(path)], "what is the retry limit?", recall_check=False,
                                        settings=make_settings(home), backend=fake)
                chunks[window] = result.stats["chunks"]
        self.assertGreater(chunks[4096], chunks[None])
        self.assertGreaterEqual(chunks[4096], 2 * chunks[None] - 1)
