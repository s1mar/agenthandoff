"""Tests for settings: the registry, the layers, project files, providers and scripts in the
config, and the `agenthandoff config` command. No model, no network."""
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import support  # noqa: F401  (puts src on the path)

from agenthandoff import config
from agenthandoff.config_cli import run_config
from agenthandoff.errors import BadSetting, NotConfigured


class Home(unittest.TestCase):
    """An empty AGENTHANDOFF_HOME, no other AGENTHANDOFF_* variables, and an empty project folder."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name, "home")
        self.project = Path(self.tmp.name, "project")
        self.project.mkdir()
        clean = {k: v for k, v in os.environ.items() if not k.startswith("AGENTHANDOFF_")}
        self.env = mock.patch.dict(os.environ, {**clean, "AGENTHANDOFF_HOME": str(self.home)}, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def save(self, values):
        self.home.mkdir(exist_ok=True)
        (self.home / "config.json").write_text(json.dumps(values), encoding="utf-8")

    def project_file(self, values, where=None):
        (where or self.project).joinpath(".agenthandoff.json").write_text(json.dumps(values), encoding="utf-8")

    def load(self, **kw):
        kw.setdefault("project_dir", self.project)
        return config.load(**kw)


class RegistryTests(Home):
    def test_every_field_has_a_unique_env_name_and_help(self):
        envs = [f.env for f in config.FIELDS]
        self.assertEqual(len(envs), len(set(envs)))
        self.assertEqual(config.FIELD["route.policy"].env, "AGENTHANDOFF_ROUTE_POLICY")
        self.assertTrue(all(f.help for f in config.FIELDS))

    def test_defaults_without_any_setup_when_no_model_is_needed(self):
        s = self.load(need_model=False)
        self.assertEqual(s.get("route.policy"), "code_first")
        self.assertEqual(s.get("profile.rare_count"), 3)
        self.assertEqual(s.provider_chain, [])
        self.assertEqual(s.sources["route.policy"], "default")

    def test_a_model_job_without_setup_still_says_run_setup(self):
        with self.assertRaises(NotConfigured) as e:
            self.load()
        self.assertIn("agenthandoff setup", str(e.exception))

    def test_unknown_key_names_the_closest_key(self):
        self.save({"backend": "ollama", "route.polcy": "code_only"})
        with self.assertRaises(BadSetting) as e:
            self.load()
        self.assertIn("route.polcy", str(e.exception))
        self.assertIn("route.policy", str(e.exception))

    def test_ranges_and_choices_are_enforced_with_the_allowed_values(self):
        cases = [("route.policy", "fast", "code_first"), ("profile.rare_count", "0", "between 1 and"),
                 ("tools.max_output_chars", "10", "between"), ("route.allow_model", "maybe", "true or false"),
                 ("chunk_tokens", "9000", "between")]
        for key, value, words in cases:
            with self.subTest(key=key), mock.patch.dict(os.environ, {config.FIELD[key].env: value}):
                with self.assertRaises(BadSetting) as e:
                    self.load(need_model=False)
                self.assertIn(words, str(e.exception))
                self.assertIn(config.FIELD[key].env, str(e.exception))

    def test_non_default_lists_only_changed_values(self):
        os.environ["AGENTHANDOFF_PROFILE_RARE_COUNT"] = "5"
        s = self.load(need_model=False)
        self.assertEqual(s.non_default(), {"profile.rare_count": 5})


class LayerTests(Home):
    def test_a_utf16_file_is_read(self):
        # Windows PowerShell 5 `>` and Out-File write UTF-16 with a BOM.
        self.home.mkdir(exist_ok=True)
        (self.home / "config.json").write_bytes(json.dumps({"backend": "ollama", "timeout": 77}).encode("utf-16"))
        self.assertEqual(self.load().get("timeout"), 77.0)

    def test_context_tokens_reaches_the_provider_setup_made(self):
        # Setup writes flat keys; the window must be settable without writing a providers block.
        from agenthandoff import providers as pv
        self.save({"backend": "llamacpp", "model": "m.gguf", "context_tokens": 8192})
        s = self.load()
        self.assertEqual(pv.make_backend(s).server.context, 8192)
        self.assertEqual(pv.chunk_limit(s), 4096)
        self.save({"backend": "ollama", "model": "m"})
        self.assertEqual(pv.make_backend(self.load()).context_tokens, config.CONTEXT_TOKENS)

    def test_a_file_saved_with_a_bom_is_read(self):
        # Windows PowerShell 5 (Set-Content -Encoding utf8) and older Notepad write a BOM first.
        self.home.mkdir(exist_ok=True)
        (self.home / "config.json").write_bytes(b"\xef\xbb\xbf" + json.dumps({"backend": "ollama", "timeout": 77}).encode())
        (self.project / ".agenthandoff.json").write_bytes(b"\xef\xbb\xbf" + b'{"find.max_matches": 5}')
        s = self.load()
        self.assertEqual(s.get("timeout"), 77.0)
        self.assertEqual(s.get("find.max_matches"), 5)

    def test_later_layers_win_and_each_source_is_named(self):
        self.save({"backend": "ollama", "profile.rare_count": 4, "profile.max_patterns": 30, "find.max_matches": 9,
                   "timeout": 100})
        self.project_file({"profile.rare_count": 5, "profile.max_patterns": 20})
        os.environ["AGENTHANDOFF_PROFILE_MAX_PATTERNS"] = "10"
        s = self.load(overrides={"find.max_matches": "7"})
        self.assertEqual((s.get("timeout"), s.sources["timeout"]), (100.0, "config.json"))
        self.assertEqual((s.get("profile.rare_count"), s.sources["profile.rare_count"]), (5, "project file"))
        self.assertEqual((s.get("profile.max_patterns"), s.sources["profile.max_patterns"]), (10, "env"))
        self.assertEqual((s.get("find.max_matches"), s.sources["find.max_matches"]), (7, "--set"))

    def test_project_file_is_found_in_a_parent_folder(self):
        deep = self.project / "a" / "b"
        deep.mkdir(parents=True)
        self.project_file({"route.policy": "code_only"})
        self.assertEqual(config.load(need_model=False, project_dir=deep).get("route.policy"), "code_only")

    def test_project_file_may_not_set_anything_that_runs_or_sends(self):
        for key, value in (("providers", {"x": {"type": "command", "argv": ["evil"]}}), ("scripts", []),
                           ("url", "http://evil.example"), ("allow_remote", True), ("backend", "openai"),
                           ("llama_server", "/tmp/evil"), ("llama_args", ["--x"]), ("provider", "x"),
                           ("model", "/tmp/m.gguf"), ("project_config", False), ("external_lock", "/x")):
            with self.subTest(key=key):
                self.project_file({key: value})
                with self.assertRaises(BadSetting) as e:
                    self.load(need_model=False)
                self.assertIn(key, str(e.exception))
                self.assertIn("project", str(e.exception))
                self.assertIn("config set project_config false", str(e.exception))   # the way out

    def test_project_file_may_only_make_things_safer(self):
        # A cloned repo may lower a limit, turn the model off or pick code_only...
        self.project_file({"tools.max_output_chars": 4000, "route.allow_model": False, "route.policy": "code_only"})
        s = self.load(need_model=False)
        self.assertEqual((s.get("tools.max_output_chars"), s.get("route.allow_model"), s.get("route.policy")),
                         (4000, False, "code_only"))
        # ...but not raise a limit above yours, force the model, or silence the recall check.
        self.save({"tools.max_output_chars": 8000, "route.allow_model": False})
        for key, value in (("tools.max_output_chars", 1_000_000), ("tools.max_file_bytes", 4_000_000_000),
                           ("route.policy", "model_only"), ("route.allow_model", True),
                           ("digest.recall_check", False)):
            with self.subTest(key=key):
                self.project_file({key: value})
                with self.assertRaises(BadSetting) as e:
                    self.load(need_model=False)
                self.assertIn(key, str(e.exception))
                self.assertIn("config set project_config false", str(e.exception))
        self.project_file({"tools.max_output_chars": 8000})                       # the same as yours is fine
        self.assertEqual(self.load(need_model=False).get("tools.max_output_chars"), 8000)

    def test_project_files_can_be_turned_off_by_the_user(self):
        self.save({"project_config": False})
        self.project_file({"route.policy": "code_only"})
        self.assertEqual(self.load(need_model=False).get("route.policy"), "code_first")

    def test_broken_project_file_is_an_error_naming_it(self):
        (self.project / ".agenthandoff.json").write_text("{nope", encoding="utf-8")
        with self.assertRaises(BadSetting) as e:
            self.load(need_model=False)
        self.assertIn(".agenthandoff.json", str(e.exception))

    def test_unknown_override_key_names_the_closest(self):
        with self.assertRaises(BadSetting) as e:
            self.load(need_model=False, overrides={"profile.rarecount": "2"})
        self.assertIn("profile.rare_count", str(e.exception))


class ProviderConfigTests(Home):
    def test_flat_keys_become_one_provider_named_default(self):
        self.save({"backend": "ollama", "model": "m"})
        s = self.load()
        self.assertEqual(s.provider_chain, ["default"])
        self.assertEqual(s.providers["default"]["type"], "ollama")
        self.assertEqual(s.providers["default"]["model"], "m")

    def test_named_providers_and_a_fallback_chain(self):
        self.save({"providers": {"fast": {"type": "command", "argv": ["mycli", "--json"]},
                                 "lms": {"type": "openai", "url": "http://127.0.0.1:1234", "model": "x"}},
                   "provider": ["fast", "lms"]})
        s = self.load()
        self.assertEqual(s.provider_chain, ["fast", "lms"])
        self.assertEqual(s.backend, "command")
        os.environ["AGENTHANDOFF_PROVIDER"] = "lms"
        self.assertEqual(self.load().provider_chain, ["lms"])

    def test_unknown_provider_name_lists_the_known_ones(self):
        self.save({"providers": {"a": {"type": "command", "argv": ["x"]}}, "provider": "b"})
        with self.assertRaises(BadSetting) as e:
            self.load()
        self.assertIn("'b'", str(e.exception))
        self.assertIn("a", str(e.exception))

    def test_bad_provider_specs_say_what_is_wrong(self):
        cases = [({"type": "nope"}, "type"), ({"type": "command"}, "argv"),
                 ({"type": "command", "argv": "mycli -x"}, "argv"), ({"type": "openai", "model": "m"}, "url"),
                 ({"type": "python"}, "factory"), ({"type": "openai", "url": "http://h", "model": "m",
                                                     "api_key": "sk-123"}, "do not put keys in config"),
                 ({"type": "command", "argv": ["x"], "colour": 1}, "colour")]
        for spec, words in cases:
            with self.subTest(spec=spec):
                self.save({"providers": {"p": spec}, "provider": "p"})
                with self.assertRaises(BadSetting) as e:
                    self.load()
                self.assertIn(words, str(e.exception))

    def test_providers_only_from_the_users_own_config(self):
        os.environ["AGENTHANDOFF_PROVIDERS"] = "{}"
        with self.assertRaises(BadSetting) as e:
            self.load(need_model=False)
        self.assertIn("config.json", str(e.exception))


class ScriptConfigTests(Home):
    def test_scripts_are_checked(self):
        good = {"name": "git_log", "argv": ["git", "log", "--stat", "-n", "20"], "match": ["*"],
                "description": "recent commits"}
        self.save({"scripts": [good]})
        self.assertEqual(self.load(need_model=False).scripts[0]["name"], "git_log")
        bad = [({**good, "name": "has space"}, "name"), ({**good, "argv": "git log"}, "argv"),
               ({**good, "match": "*.log"}, "match"), ({**good, "timeout": 0}, "timeout"),
               ({**good, "question": "([unclosed"}, "question"), ([good, good], "twice"),
               ({**good, "shell": True}, "shell")]
        for scripts, words in bad:
            with self.subTest(words=words):
                self.save({"scripts": scripts if isinstance(scripts, list) else [scripts]})
                with self.assertRaises(BadSetting) as e:
                    self.load(need_model=False)
                self.assertIn(words, str(e.exception))


class WriteTests(Home):
    def test_writing_keeps_every_known_key_and_refuses_unknown_ones(self):
        values = {"backend": "ollama", "route.policy": "code_only", "providers": {"a": {"type": "command",
                                                                                        "argv": ["x"]}}}
        config.write_config_file(values)
        self.assertEqual(json.loads(config.config_path().read_text()), values)
        with self.assertRaises(BadSetting):
            config.write_config_file({"nope": 1})


class ConfigCommandTests(Home):
    def run_cmd(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = run_config(list(args), project_dir=self.project)
        return code, out.getvalue(), err.getvalue()

    def test_set_get_unset_round_trip(self):
        self.assertEqual(self.run_cmd("set", "profile.rare_count", "6")[0], 0)
        self.assertEqual(self.run_cmd("get", "profile.rare_count")[1].strip(), "6")
        self.assertEqual(json.loads(config.config_path().read_text()), {"profile.rare_count": 6})
        code, out, _ = self.run_cmd("unset", "profile.rare_count")
        self.assertEqual(code, 0)
        self.assertIn("the default applies: 3", out)
        self.assertEqual(self.run_cmd("get", "profile.rare_count")[1].strip(), "3")

    def test_set_takes_json_for_lists_and_objects(self):
        code, _, err = self.run_cmd("set", "providers", '{"cli": {"type": "command", "argv": ["mycli"]}}')
        self.assertEqual(code, 0, err)
        self.assertEqual(self.run_cmd("set", "provider", "cli")[0], 0)
        self.assertEqual(config.load(project_dir=self.project).provider_chain, ["cli"])

    def test_a_bad_value_is_refused_and_the_file_is_unchanged(self):
        self.run_cmd("set", "route.policy", "code_only")
        before = config.config_path().read_text()
        code, _, err = self.run_cmd("set", "route.policy", "fast")
        self.assertEqual(code, 2)
        self.assertIn("code_first", err)
        self.assertEqual(config.config_path().read_text(), before)
        code, _, err = self.run_cmd("set", "route.polcy", "x")
        self.assertEqual(code, 2)
        self.assertIn("route.policy", err)

    def test_show_lists_every_setting_with_its_source(self):
        self.run_cmd("set", "find.max_matches", "9")
        code, out, _ = self.run_cmd("show")
        self.assertEqual(code, 0)
        self.assertIn("find.max_matches = 9  (config.json)", out)
        self.assertIn("route.policy = code_first  (default)", out)
        self.assertEqual(sum(1 for line in out.splitlines() if " = " in line), len(config.FIELDS))

    def test_path_prints_the_user_config_file(self):
        self.assertEqual(self.run_cmd("path")[1].strip(), str(config.config_path()))


class OldHomeTests(unittest.TestCase):
    """The folder from before the rename (~/.local-offload) moves to the new one, once."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.old, self.new = self.root / "old home", self.root / "new home"

    def make_old(self):
        (self.old / "models").mkdir(parents=True)
        model = self.old / "models" / "m.gguf"
        model.write_text("x")
        (self.old / "config.json").write_text(json.dumps({"backend": "llamacpp", "model": str(model)}),
                                              encoding="utf-8")

    def test_the_old_folder_moves_and_its_model_path_follows(self):
        self.make_old()
        with redirect_stderr(io.StringIO()) as err:
            self.assertTrue(config.move_old_home(self.old, self.new))
        self.assertFalse(self.old.exists())
        saved = json.loads((self.new / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["model"], str(self.new / "models" / "m.gguf"))
        self.assertTrue(Path(saved["model"]).is_file())
        self.assertIn("moved your settings", err.getvalue())

    def test_an_existing_new_folder_is_never_overwritten(self):
        self.make_old()
        self.new.mkdir()
        self.assertFalse(config.move_old_home(self.old, self.new))
        self.assertTrue((self.old / "config.json").exists())
        self.assertFalse((self.new / "config.json").exists())

    def test_no_old_folder_does_nothing(self):
        self.assertFalse(config.move_old_home(self.old, self.new))
        self.assertFalse(self.new.exists())

    def test_a_set_home_variable_skips_the_move(self):
        self.make_old()
        with mock.patch.dict(os.environ, {"AGENTHANDOFF_HOME": str(self.new)}), \
             mock.patch.object(config, "OLD_HOME", str(self.old)):
            self.assertEqual(config.home_dir(), self.new)
        self.assertTrue(self.old.exists())

class ModelDefaultTests(unittest.TestCase):
    def test_the_model_digest_is_off_by_default(self):
        # A small model can misread prose, so read sends plain text to it only when asked.
        field = next(f for f in config.FIELDS if f.key == "route.allow_model")
        self.assertIs(field.default, False)


class ReadmeTests(unittest.TestCase):
    def test_the_reference_lists_every_setting(self):
        readme = (Path(__file__).resolve().parent.parent / "docs" / "reference.md").read_text(encoding="utf-8")
        missing = [f.key for f in config.FIELDS if f"`{f.key}`" not in readme]
        self.assertEqual(missing, [], "settings missing from docs/reference.md")

if __name__ == "__main__":
    unittest.main()
