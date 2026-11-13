"""Tests for the code tools (find, outline, shape), the file sniffer, the router and user
scripts. None of them may call a model: the router tests run with a backend that fails if
it is used."""
import json
import os
import re
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from support import TESTS, make_settings

from agenthandoff import chunk as ck
from agenthandoff import find, outline, router, shape, sniff
from agenthandoff import profile as pf


MODULE = '"""Module doc."""\n' + "".join(f"def f{i}(x):\n    y = x * {i}\n    return y + 1\n\n" for i in range(8))


def lines_of(text, path="f.log"):
    return [ck.Line(path, i, t) for i, t in enumerate(text.splitlines(), 1)]
from agenthandoff.errors import NoInput, ToolFailed, Unavailable
from agenthandoff.scripts import run_script

FAKE_CLI = str(TESTS / "fake_cli.py")


class Files(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def write(self, name, text):
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
        return path

    def lines_of(self, path):
        return path.read_text(encoding="utf-8").splitlines()

    def assert_real_line_numbers(self, out, path):
        """Every 'L<n>' or 'name:<n>' the tool printed points at a real line of the file."""
        n_lines = len(self.lines_of(path))
        nums = [int(n) for n in re.findall(r"\bL(\d+)\b", out)]
        nums += [int(n) for n in re.findall(re.escape(path.name) + r"[:-](\d+)\b", out)]
        self.assertTrue(nums, out)
        for n in nums:
            self.assertTrue(1 <= n <= n_lines, f"L{n} is not a line of {path.name} ({n_lines} lines)")


# ---------------------------------------------------------------- sniffing file kinds

class SniffTests(Files):
    def test_kinds_by_name_and_by_content(self):
        cases = {
            "app.py": ("def f():\n    return 1\n", "code"),
            "web.ts": ("export function f() {}\n", "code"),
            "data.json": ('{"a": 1}\n', "json"),
            "rows.csv": ("a,b\n1,2\n", "csv"),
            "events.log": ("".join(f'{{"ts": "2026-01-01T00:00:0{i}", "level": "info"}}\n' for i in range(9)),
                           "jsonl"),
            "iso.txt": ("".join(f"2026-01-01 00:00:0{i} INFO tick\n" for i in range(9)), "log"),
            "syslog": ("".join(f"Nov  1 03:04:0{i} host sshd[12]: Accepted key\n" for i in range(9)), "log"),
            "access": ("".join(f'1.2.3.4 - - [11/Nov/2026:10:00:0{i} +0000] "GET / HTTP/1.1" 200 5\n'
                               for i in range(9)), "log"),
            "levels.out": ("".join(f"[WARN] retry {i}\n" for i in range(9)), "log"),
            "notes.md": ("# Title\n\nSome prose about the system.\nMore prose.\n", "text"),
        }
        for name, (text, kind) in cases.items():
            with self.subTest(name=name):
                path = self.write(name, text)
                self.assertEqual(sniff.kind_of(path, text.splitlines()), kind)


# ---------------------------------------------------------------- find

class FindTests(Files):
    def test_terms_are_quoted_strings_and_identifiers_only(self):
        self.assertEqual(find.terms('Where is `DEFAULT_REDIRECT_LIMIT` set and what does "max retries" mean?'),
                         ["DEFAULT_REDIRECT_LIMIT", "max retries"])
        self.assertEqual(find.terms("What does resolve_redirects do with session.max_redirects?"),
                         ["resolve_redirects", "session.max_redirects"])
        self.assertEqual(find.terms("Why does getUserName fail with ERR_TIMEOUT?"), ["getUserName", "ERR_TIMEOUT"])
        self.assertEqual(find.terms("Why did this CI run fail?"), [])
        self.assertEqual(find.terms("What is the root cause of this build failure?"), [])

    def test_names_that_are_usually_just_words_are_weak_terms(self):
        weak = ["GitHub", "PowerShell", "JavaScript", "macOS", "iOS", "iPhone", "README.md", "node.js",
                "build.log", "config.json"]
        strong = ["DEFAULT_REDIRECT_LIMIT", "session.max_redirects", "getUserName", "--max-depth", "os.path",
                  "ERR_TIMEOUT", "resolve_redirects"]
        for t in weak:
            self.assertTrue(find.weak(t), t)
        for t in strong:
            self.assertFalse(find.weak(t), t)

    def test_matches_with_context_and_real_line_numbers(self):
        path = self.write("a.py", "x = 1\ny = 2\nMAX_RETRIES = 5\nz = 3\nprint(MAX_RETRIES)\n")
        out = find.run_find([str(path)], ["MAX_RETRIES"], context=1, max_matches=10, root=str(self.dir))
        self.assertIn("'MAX_RETRIES': 2 matches in 1 file", out)
        self.assertIn("a.py:3: MAX_RETRIES = 5", out)
        self.assertIn("a.py-2- y = 2", out)
        self.assert_real_line_numbers(out, path)

    def test_too_many_matches_are_capped_with_the_count(self):
        path = self.write("b.log", "".join(f"tick {i} ERR_X\n" for i in range(30)))
        out = find.run_find([str(path)], ["ERR_X"], context=0, max_matches=5, root=str(self.dir))
        self.assertIn("30 matches", out)
        self.assertIn("25 more not shown", out)

    def test_no_exact_match_tries_ignoring_case(self):
        path = self.write("c.py", "Max_Retries = 5\n")    # neither the term's case nor all lower
        out = find.run_find([str(path)], ["MAX_RETRIES"], context=0, max_matches=5, root=str(self.dir))
        self.assertIn("no exact match; ignoring case: 1 match", out)
        self.assertIn("c.py:1: Max_Retries = 5", out)


# ---------------------------------------------------------------- outline

class OutlineTests(Files):
    def test_detail_levels(self):
        lines = MODULE.splitlines()
        names = outline.outline_file(Path("m.py"), lines, "m.py", detail="names")
        self.assertEqual(names.splitlines()[0], f"m.py ({len(lines)} lines): Module doc.")
        self.assertIn("f0 L2, f1 L6", names)
        self.assertNotIn("def f0(x)", names)
        head = outline.outline_file(Path("m.py"), lines, "m.py", detail="head")
        self.assertEqual(head, f"m.py ({len(lines)} lines): Module doc. (8 definitions)")
        js = outline.outline_file(Path("a.js"), ["function go(a) {", "}", "const x = 1;"], "a.js", detail="names")
        self.assertIn("go L1", js)

    def test_python_outline_has_docstrings_defs_constants_and_imports(self):
        path = self.write("mod.py", '''
            """Send HTTP requests. Second sentence is dropped."""
            import os
            from json import dumps

            DEFAULT_LIMIT = 30
            _private = 1


            class Session(Base):
                """Keeps cookies between requests."""

                def get(self, url, **kwargs):
                    """GET a URL."""
                    return url

                def _hidden(self):
                    pass


            async def fetch(url, timeout=10):
                return url
            ''')
        out = outline.outline_file(path, self.lines_of(path), "mod.py", max_items=100)
        self.assertIn("mod.py (21 lines): Send HTTP requests.", out)
        self.assertIn("imports: os, json", out)
        self.assertIn("L5 DEFAULT_LIMIT = 30", out)
        self.assertIn("L9 class Session(Base): Keeps cookies between requests.", out)
        self.assertIn("L12   def get(self, url, **kwargs): GET a URL.", out)
        self.assertIn("L16   def _hidden(self)", out)
        self.assertIn("L20 async def fetch(url, timeout=10)", out)
        self.assertNotIn("_private", out)
        self.assert_real_line_numbers(out, path)

    def test_broken_python_falls_back_to_patterns(self):
        path = self.write("bad.py", "def ok():\n    pass\n\ndef broken(:\n")
        out = outline.outline_file(path, self.lines_of(path), "bad.py", max_items=100)
        self.assertIn("L1 def ok", out)
        self.assertIn("syntax error", out)

    def test_other_languages_by_pattern(self):
        cases = {
            "a.ts": ("// Routes for the shop.\nexport async function getCart(id: string) {\n}\n"
                     "export class CartStore {\n}\nexport const total = (items) => 0;\n",
                     ["Routes for the shop.", "L2 export async function getCart", "L4 export class CartStore",
                      "L6 export const total"]),
            "b.go": ("// Package cache keeps values.\npackage cache\n\nfunc (c *Cache) Get(k string) string {\n}\n"
                     "type Cache struct {\n}\n",
                     ["Package cache keeps values.", "L4 func (c *Cache) Get", "L6 type Cache struct"]),
            "c.rs": ("pub struct Pool {}\nimpl Pool {\n    pub fn get(&self) {}\n}\n",
                     ["L1 pub struct Pool", "L2 impl Pool", "L3 pub fn get"]),
            "D.java": ("public class D {\n    public int size() { return 0; }\n}\n",
                       ["L1 public class D", "L2 public int size()"]),
        }
        for name, (text, expected) in cases.items():
            with self.subTest(name=name):
                path = self.write(name, text)
                out = outline.outline_file(path, self.lines_of(path), name, max_items=100)
                for words in expected:
                    self.assertIn(words, out)
                self.assert_real_line_numbers(out, path)

    def test_item_cap_says_how_many_are_hidden(self):
        path = self.write("many.py", "".join(f"def f{i}():\n    pass\n" for i in range(12)))
        out = outline.outline_file(path, self.lines_of(path), "many.py", max_items=5)
        self.assertIn("(7 more definitions not shown)", out)


# ---------------------------------------------------------------- shape

class ShapeTests(Files):
    def test_json_lines_keys_counts_and_odd_one_out(self):
        rows = [{"ts": f"2026-01-01T00:00:{i:02d}", "level": "info", "ms": 10 + i % 3, "region": "eu"}
                for i in range(20)]
        rows[13]["region"] = "us"
        rows[7].pop("ms")
        text = "\n".join(json.dumps(r) for r in rows) + "\nnot json at all\n"
        path = self.write("e.jsonl", text)
        out = shape.shape_file(path, self.lines_of(path), "e.jsonl", "jsonl", max_keys=50)
        self.assertIn("20 JSON objects", out)
        self.assertIn("1 lines are not JSON (first: L21)", out)
        self.assertIn("ms: in 19 of 20 (missing at L8)", out)
        self.assertIn("odd one out: us at L14", out)
        self.assert_real_line_numbers(out, path)

    def test_a_long_text_value_is_cut_to_one_short_line(self):
        # A Claude Code transcript listed a 4.9 KB multi-line prompt as a "most common value".
        prompt = "You are scoring outputs.\n\nINPUT FILE:\n" + "rubric text " * 400
        rows = [{"type": "user", "content": prompt}] + [{"type": "assistant", "content": f"reply {i}"} for i in range(9)]
        path = self.write("t.jsonl", "\n".join(json.dumps(r) for r in rows))
        out = shape.shape_file(path, self.lines_of(path), "t.jsonl", "jsonl", max_keys=50)
        self.assertIn("type: assistant x9, user x1", out)
        line = next(ln for ln in out.splitlines() if ln.strip().startswith("content:"))
        self.assertLess(len(line), 400, line)
        self.assertIn("You are scoring outputs. INPUT FILE:", line)    # line breaks become spaces
        self.assertNotIn("rubric text " * 20, out)

    def test_json_document_paths_and_types(self):
        path = self.write("c.json", json.dumps({"name": "svc", "replicas": 3, "ports": [{"port": 80}, {"port": 443}],
                                                "limits": {"cpu": "2", "memory": "512Mi"}}, indent=1))
        out = shape.shape_file(path, self.lines_of(path), "c.json", "json", max_keys=50)
        self.assertIn("$.replicas: number 3", out)
        self.assertIn("$.ports[]: 2 items", out)
        self.assertIn("$.ports[].port: number, 2 values, min 80, max 443", out)
        self.assertIn("$.limits.memory: string '512Mi'", out)

    def test_csv_columns_and_bad_rows(self):
        text = "host,cpu,zone\n" + "".join(f"web-{i},{20 + i},a\n" for i in range(10)) + "web-x,99\n"
        path = self.write("h.csv", text)
        out = shape.shape_file(path, self.lines_of(path), "h.csv", "csv", max_keys=50)
        self.assertIn("11 rows, 3 columns: host, cpu, zone", out)
        self.assertIn("cpu: min 20 (L2), median", out)
        self.assertIn("rows with the wrong number of columns: L12", out)
        self.assert_real_line_numbers(out, path)


# ---------------------------------------------------------------- the router

class NoModel:
    """A backend that fails the test if anything calls it."""
    model = "none"

    def __getattr__(self, name):
        raise AssertionError(f"the router used the model ({name})")


class RouterTests(Files):
    def settings(self, **values):
        return make_settings(self.dir, values=values)

    def read(self, paths, question="", **values):
        with mock.patch("agenthandoff.providers.make_backend", side_effect=AssertionError("model used")), \
             mock.patch("agenthandoff.api.run_digest", side_effect=AssertionError("digest used")):
            return router.read([str(p) for p in paths], question, self.settings(**values), root=str(self.dir))

    def test_each_kind_goes_to_its_code_tool(self):
        log = self.write("app.log", "".join(f"2026-01-01 00:00:{i:02d} INFO tick n={i}\n" for i in range(20)))
        code = self.write("m.py", MODULE)
        data = self.write("d.json", json.dumps({"items": [{"id": i, "name": f"n{i}"} for i in range(30)]}))
        for path, tool in ((log, "profile"), (code, "outline"), (data, "shape")):
            with self.subTest(tool=tool):
                out = self.read([path], "what happened?")
                self.assertTrue(out.startswith(f"ROUTE: {tool}"), out[:80])

    def test_named_identifier_goes_to_find(self):
        code = self.write("m.py", "LIMIT = 30\n")
        out = self.read([code], "Where is `LIMIT` defined?")
        self.assertTrue(out.startswith("ROUTE: find"), out[:80])
        self.assertIn("m.py:1: LIMIT = 30", out)

    def test_find_with_no_match_falls_back_to_the_kind_tool(self):
        code = self.write("m.py", MODULE)
        out = self.read([code], "Where is `NOWHERE_X` defined?")
        self.assertIn("ROUTE: outline", out)
        self.assertIn("no match for 'NOWHERE_X'", out)

    def test_mixed_folder_gets_one_tool_per_kind(self):
        self.write("src/m.py", "def f():\n    pass\n")
        self.write("logs/a.log", "".join(f"2026-01-01 00:00:{i:02d} INFO tick\n" for i in range(40)))
        out = self.read([self.dir])
        self.assertIn("ROUTE: outline", out)
        self.assertIn("ROUTE: profile", out)

    def test_prose_without_a_model_says_read_it_yourself(self):
        prose = self.write("notes.md", "Some prose.\nMore prose here.\n")
        with self.assertRaises(Unavailable) as e:
            self.read([prose], "summarize this", **{"route.allow_model": False})
        self.assertIn("read", str(e.exception).lower())

    def test_a_model_that_is_set_up_but_off_names_the_setting_that_turns_it_on(self):
        # The default (route.allow_model false) with a model set up: the way out is one command.
        prose = self.write("notes.md", "Some prose.\nMore prose here.\n")
        with self.assertRaises(Unavailable) as e:
            self.read([prose], "summarize this")
        self.assertIn("route.allow_model is off", str(e.exception))
        self.assertIn("agenthandoff config set route.allow_model true", str(e.exception))

    def test_prose_with_a_model_goes_to_the_digest(self):
        prose = self.write("notes.md", "Some prose.\nMore prose here.\n")
        fake = mock.Mock()
        fake.render.return_value = "ANSWER: prose"
        with mock.patch("agenthandoff.api.run_digest", return_value=fake) as run:
            out = router.read([str(prose)], "summarize", make_settings(self.dir, provider_chain=["x"],
                                                                        providers={"x": {"type": "ollama"}},
                                                                        values={"route.allow_model": True}))
        run.assert_called_once()
        self.assertTrue(out.startswith("ROUTE: digest"))

    def test_code_only_never_uses_the_model_and_model_only_always_does(self):
        prose = self.write("notes.md", "Some prose.\n")
        with self.assertRaises(Unavailable):
            self.read([prose], "q", **{"route.policy": "code_only"})
        log = self.write("a.log", "".join(f"2026-01-01 00:00:{i:02d} INFO tick\n" for i in range(9)))
        fake = mock.Mock()
        fake.render.return_value = "ANSWER: x"
        with mock.patch("agenthandoff.api.run_digest", return_value=fake) as run:
            out = router.read([str(log)], "q", make_settings(self.dir, values={"route.policy": "model_only"},
                                                              provider_chain=["x"], providers={"x": {"type": "ollama"}}))
        run.assert_called_once()
        self.assertTrue(out.startswith("ROUTE: digest"))

    def test_output_is_capped_by_the_setting(self):
        log = self.write("big.log", "".join(f"2026-01-01 00:00:00 INFO unique message number {i} {'x' * 50}\n"
                                            for i in range(300)))
        # too small even at the least profile detail, so the hard cap applies
        out = self.read([log], "", **{"tools.max_output_chars": 400, "profile.rare_count": 1000})
        self.assertLessEqual(len(out), 600)
        self.assertIn("output cut at 400 characters", out)

    def test_profile_settings_reach_the_profile(self):
        log = self.write("a.log", "".join(f"2026-01-01 00:00:{i % 60:02d} INFO tick n={i}\n" for i in range(200))
                         + "2026-01-01 00:01:00 WARN slow disk\n" * 5)
        self.assertIn("RARE LINES", self.read([log], "", **{"profile.rare_count": 10}))
        self.assertNotIn("RARE LINES", self.read([log], ""))

    def test_a_script_with_a_question_pattern_is_picked_first(self):
        xml = self.write("r.xml", "<testsuite/>")
        scripts = [{"name": "junit", "argv": [sys.executable, FAKE_CLI, "text"], "match": ["*.xml"],
                    "question": "fail|test", "description": "", "timeout": 10, "max_output_chars": None}]
        with mock.patch.object(router, "run_script", return_value="2 failed") as run:
            out = router.read([str(xml)], "which tests fail?", make_settings(self.dir, scripts=scripts))
        run.assert_called_once()
        self.assertTrue(out.startswith("ROUTE: script junit"), out)

    def test_a_script_without_a_question_pattern_is_never_picked_by_itself(self):
        log = self.write("a.log", "".join(f"2026-01-01 00:00:{i:02d} INFO tick\n" for i in range(40)))
        scripts = [{"name": "git_log", "argv": ["git", "log"], "match": ["*"], "question": None, "description": "",
                    "timeout": 10, "max_output_chars": None}]
        with mock.patch.object(router, "run_script", side_effect=AssertionError("script run")):
            out = router.read([str(log)], "anything", make_settings(self.dir, scripts=scripts))
        self.assertTrue(out.startswith("ROUTE: profile"))

    def test_a_script_is_picked_only_when_every_file_matches(self):
        data = self.write("report.json", json.dumps({"tests": [{"name": f"t{i}", "ok": i != 3} for i in range(20)]}))
        log = self.write("a.log", "".join(f"2026-01-01 00:00:{i:02d} INFO tick\n" for i in range(40)))
        scripts = [{"name": "junit", "argv": ["x"], "match": ["*.log"], "question": "fail", "description": "",
                    "timeout": 10, "max_output_chars": None}]
        with mock.patch.object(router, "run_script", side_effect=AssertionError("script run")):
            out = router.read([str(data), str(log)], "what failed?", make_settings(self.dir, scripts=scripts))
        self.assertNotIn("ROUTE: script", out)
        self.assertIn("ROUTE: profile", out)
        self.assertIn("ROUTE: shape", out)

    def test_missing_paths_are_no_input(self):
        with self.assertRaises(NoInput):
            self.read([self.dir / "nope.log"], "")

    def test_a_failed_digest_keeps_what_the_code_tools_found(self):
        self.write("notes.txt", "Some prose about the plan.\n" * 30)
        self.write("app.log", "".join(f"2026-01-01 00:00:{i % 60:02d} INFO tick n={i}\n" for i in range(200)))
        with mock.patch.object(router, "_digest", side_effect=Unavailable("the model server is down.")):
            out = router.read([str(self.dir)], "", self.settings(**{"route.allow_model": True}),
                              root=str(self.dir))
        self.assertIn("ROUTE: profile", out)
        self.assertIn("NOTE: the model could not read 1 file of plain text: the model server is down.", out)

    def test_a_file_shorter_than_its_summary_is_shown_whole(self):
        log = self.write("small.log", "2026-01-01 00:00:01 started\n2026-01-01 00:00:02 port=8080 ready\n")
        out = self.read([log], "")
        self.assertTrue(out.startswith("ROUTE: whole file (1 file shorter than its profile)"), out)
        self.assertIn("FILE: small.log (2 lines)\nL1: 2026-01-01 00:00:01 started\nL2: 2026-01-01 00:00:02 port=8080 ready",
                      out)
        big = self.write("big.log", "".join(f"2026-01-01 00:00:{i % 60:02d} INFO tick n={i}\n" for i in range(300)))
        self.assertTrue(self.read([big], "").startswith("ROUTE: profile"))

    def test_the_read_it_yourself_note_counts_the_files(self):
        notes = [self.write(f"n{i}.txt", "Some thoughts about the plan. " * 60 + "\nMore of them.\n") for i in range(2)]
        with self.assertRaises(Unavailable) as e:
            self.read(notes, "", **{"route.allow_model": False})
        self.assertIn("2 files of plain text (n0.txt, n1.txt) need reading", str(e.exception))
        self.assertIn("Read them yourself.", str(e.exception))

    def test_a_long_file_name_does_not_make_the_whole_file_look_bigger(self):
        # A 50-line npm debug log with a 40-character name got a profile bigger than the
        # file, because the name was repeated on every line of the whole-file form.
        name = "2026-11-10T07_43_58_309Z-debug-0.log"
        text = "".join(f"{i} silly step{i} fetch manifest pkg{i}@1.{i}.0 from registry\n" for i in range(50))
        out = self.read([self.write(name, text)], "")
        self.assertTrue(out.startswith("ROUTE: whole file"), out[:120])
        self.assertLess(len(out), len(text) + 400)
        self.assertEqual(out.count(name), 1)          # once, in the file header

    def test_ordinary_questions_still_get_the_kind_tool(self):
        self.write("pkg/a.py", '"""Talks to GitHub from PowerShell on macOS."""\n' + MODULE.split("\n", 1)[1])
        self.write("pkg/b.py", MODULE)
        for q in ["What does it say about PowerShell and GitHub?", "Does this work on macOS?",
                  "What is in README.md?", "Summarize the JavaScript parts"]:
            out = self.read([self.dir / "pkg"], q)
            self.assertIn("ROUTE: outline (2 files of source code)", out, q)
            self.assertNotIn("NOTE: no match", out, q)          # a word that is not there is not news
        out = self.read([self.dir / "pkg"], "What does it say about PowerShell?")
        self.assertIn("ROUTE: find", out)
        self.assertLess(out.index("ROUTE: outline"), out.index("ROUTE: find"))
        self.assertIn("a.py:1:", out)

    def test_a_code_name_still_goes_to_find_alone(self):
        self.write("pkg/a.py", "MAX_RETRIES = 5\n" + MODULE)
        self.write("pkg/b.py", MODULE)
        out = self.read([self.dir / "pkg"], "Where is MAX_RETRIES set?")
        self.assertTrue(out.startswith("ROUTE: find"), out[:80])
        self.assertNotIn("ROUTE: outline", out)
        mixed = self.read([self.dir / "pkg"], "Where is MAX_RETRIES set for GitHub?")
        self.assertTrue(mixed.startswith("ROUTE: find"), mixed[:80])
        self.assertNotIn("ROUTE: outline", mixed)

    def test_fitting_a_profile_reads_each_line_once(self):
        long = "x" * 150
        text = "".join(f"2026-01-01 00:00:00 kind{k} WARN step {k} failed {long} n={i}\n"
                       for k in range(40) for i in range(5))
        log = self.write("wide.log", text)
        with mock.patch.object(pf, "line_pattern", wraps=pf.line_pattern) as split:
            out = self.read([log], "")
        self.assertIn("PROFILE DETAIL:", out)                   # several detail levels were tried
        self.assertEqual(split.call_count, 200)                 # but each line was split into a pattern once

    def test_a_long_profile_drops_detail_instead_of_being_cut(self):
        long = "x" * 150
        text = "".join(f"2026-01-01 00:00:00 kind{k} WARN step {k} failed {long} n={i}\n"
                       for k in range(40) for i in range(5))
        names = ["disk", "net", "auth", "cache", "queue", "store", "mail", "print"]
        text += "".join([f"2026-01-01 00:00:01 ERROR {x} lost its {y} link {long}\n"     # 60 distinct one-offs
                         for x in names for y in names[:8]][:60])
        log = self.write("wide.log", text)
        full = pf.profile_lines("wide.log", lines_of(text, "wide.log"))
        self.assertGreater(len(full), 12000)              # the fixture really is too long at full detail
        out = self.read([log], "")
        self.assertNotIn("[output cut", out)
        self.assertLessEqual(len(out), 12000)
        self.assertIn("PROFILE DETAIL:", out)
        self.assertIn("fail on 200 lines (first L1)", out)
        self.assertIn("error on 60 lines (first L201)", out)
        self.assertIn("RARE LINES WITH PROBLEM WORDS", out)
        # never more detail than the user asked for
        small = self.read([log], "", **{"profile.max_line_chars": 70})     # still too long, so fitted
        self.assertIn("PROFILE DETAIL: lines cut to 70 characters", small)
        shown = [ln.split("] ", 1)[-1].split(": ", 1)[-1] for ln in small.splitlines()
                 if ln.startswith(("[", "   e.g. L", "   L", "      max line", "      min line"))]
        self.assertGreater(len(shown), 20)
        self.assertFalse([s for s in shown if len(s) > 76], "a line longer than the 70 characters asked for")

    def test_many_files_drop_detail_instead_of_dropping_files(self):
        for i in range(60):
            self.write(f"pkg/mod{i:02d}.py", f'"""Module number {i} does task {i}."""\n' + MODULE.split("\n", 1)[1])
        out = self.read([self.dir / "pkg"], "")
        self.assertLessEqual(len(out), 12000)
        self.assertNotIn("[output cut", out)
        for i in range(60):
            self.assertIn(f"mod{i:02d}.py (", out)
        self.assertIn("OUTLINE DETAIL: names only", out)
        self.assertIn("f0 L2", out)
        self.assertIn("ROUTE: outline (60 files of source code)", out)

    def test_large_logs_are_read_by_code_tools(self):
        log = self.write("huge.log", "".join(f"2026-01-01 00:00:00 INFO tick n={i}\n" for i in range(80_000)))
        self.assertGreater(Path(log).stat().st_size, 2_000_000)
        out = self.read([log], "")
        self.assertTrue(out.startswith("ROUTE: profile"), out[:300])
        self.assertIn("80000 lines", out)

    def test_a_question_about_problems_gets_the_short_profile(self):
        body = [f"2026-01-01 00:00:{i % 60:02d} INFO tick n={i}" for i in range(200)]
        body[90] = "2026-01-01 00:01:30 ERROR disk full on /var"
        log = self.write("svc.log", "\n".join(body) + "\n")
        for question in ["What errors or warnings does this log record?", "any failures?", "why did it crash",
                         "What problems are in the log?"]:
            with self.subTest(question=question):
                out = self.read([log], question)
                self.assertIn("ERROR disk full", out)
                self.assertNotIn("INFO tick", out.split("LAST LINES:")[0])   # the last lines are always shown
        for question in ["", "Summarize this log.", "what happened between 00:00 and 00:01?"]:
            with self.subTest(question=question):
                self.assertIn("INFO tick", self.read([log], question))

    def test_each_read_is_logged_with_the_tools_used(self):
        log = self.write("a.log", "".join(f"2026-01-01 00:00:{i:02d} INFO tick\n" for i in range(40)))
        self.read([log], "why?")
        record = json.loads((self.dir / "logs" / "read.jsonl").read_text().splitlines()[-1])
        self.assertEqual(record["tools"], ["profile"])
        self.assertEqual(record["question"], "why?")


# ---------------------------------------------------------------- user scripts

class ScriptRunTests(Files):
    def script(self, mode, **kw):
        return {"name": "s", "argv": [sys.executable, FAKE_CLI, mode], "match": ["*"], "question": None,
                "description": "", "timeout": kw.get("timeout", 10), "max_output_chars": kw.get("cap")}

    def test_output_is_returned(self):
        self.assertIn("answer", run_script(self.script("echo"), []))

    @unittest.skipUnless(os.name == "nt", "cmd.exe runs .cmd and .bat files only on Windows")
    def test_a_batch_file_never_gets_text_cmd_would_run(self):
        shim = self.dir / "echoargs.cmd"
        shim.write_bytes(b"@echo off\r\necho got: %*\r\n")
        marker = self.dir / "pwned.txt"

        def shim_script(*argv):
            return dict(self.script("echo"), argv=[str(shim), *argv])

        attack = f'x" & echo PWNED > "{marker}" & echo "'
        with self.assertRaises(ToolFailed) as e:
            run_script(shim_script("{question}"), [], attack, root=str(self.dir))
        self.assertIn("cmd.exe", str(e.exception))
        with self.assertRaises(ToolFailed):
            run_script(shim_script("{paths}"), [str(self.dir / "a&echo PWNED>pwned.txt")], root=str(self.dir))
        self.assertFalse(marker.exists())
        self.assertIn("hello there", run_script(shim_script("{question}"), [], "hello there", root=str(self.dir)))

    def test_paths_and_question_are_passed_as_separate_arguments(self):
        s = {**self.script("echo"), "argv": [sys.executable, "-c",
                                             "import sys, json; print(json.dumps(sys.argv[1:]))", "{paths}",
                                             "--q={question}"]}
        out = run_script(s, ["a b.log", "c.log"], question="why; rm -rf /")
        self.assertEqual(json.loads(out), ["a b.log", "c.log", "--q=why; rm -rf /"])

    def test_failure_timeout_empty_and_huge_output(self):
        with self.assertRaises(ToolFailed) as e:
            run_script(self.script("fail"), [])
        self.assertIn("boom", str(e.exception))
        with self.assertRaises(ToolFailed) as e:
            run_script(self.script("sleep", timeout=1), [])
        self.assertIn("no result after", str(e.exception))
        with self.assertRaises(ToolFailed):
            run_script(self.script("empty"), [])
        out = run_script(self.script("huge", cap=500), [])
        self.assertIn("output cut at 500", out)
        self.assertLess(len(out), 600)


# ---------------------------------------------------------------- profile: more log formats

class ProfileFormatTests(Files):
    def test_syslog_and_access_log_times_are_masked(self):
        self.assertEqual(pf.line_pattern("Nov  1 03:04:05 host sshd[12]: Accepted key")[0],
                         "host sshd[<n>]: Accepted key")
        access = pf.line_pattern('1.2.3.4 - - [11/Nov/2026:10:00:01 +0000] "GET /a HTTP/1.1" 200 5')[0]
        self.assertNotIn("Nov", access)
        self.assertEqual(access, pf.line_pattern('5.6.7.8 - - [12/Dec/2026:23:59:59 -0500] "GET /b HTTP/1.1" 404 9')[0])
        self.assertEqual(pf.line_pattern("2026-11-11 03:04:05, Info CBS Loaded servicing stack")[0],
                         "Info CBS Loaded servicing stack")
        self.assertEqual(pf.line_pattern("03:04:05.123 WARN slow")[0], "WARN slow")

    def test_windows_paths_guids_codes_and_inner_dates_are_masked(self):
        pattern, fields = pf.line_pattern(
            r"Info CBS Added source: \\?\C:\WINDOWS\Servicing\Packages\, total sources: 2")
        self.assertEqual(pattern, "Info CBS Added source: <path>, total sources: <n>")
        pattern, fields = pf.line_pattern(r"Failed to open C:\Program Files\App\a.dll [HRESULT = 0x80070002]")
        self.assertEqual(pattern, "Failed to open <path> <path> [HRESULT = <hex>]")   # the space splits it
        self.assertEqual(fields["hex1"], "0x80070002")
        pattern, _ = pf.line_pattern("device {4d36e972-e325-11ce-bfc1-08002be10318} added")
        self.assertEqual(pattern, "device <guid> added")
        self.assertEqual(pf.line_pattern(">>>  Section start 2026/10/13 21:38:35.333")[0], ">>> Section start <time>")
        self.assertEqual(pf.line_pattern("Package~amd64~~10.0.26100.9974 staged")[0], "Package~amd64~~<n> staged")

    def test_problem_lines_come_first_with_counts_per_word(self):
        info = [f"Info CBS tick {i}" for i in range(40)]
        bad = ["Error: disk 0x80070002 failed"] * 5 + ["Warning: low space"] * 4
        lines = info[:20] + bad + info[20:] + ["Error: once only, at the end"]
        out = pf.profile_lines("cbs.log", lines_of("\n".join(lines)))
        self.assertIn("PROBLEM WORDS (whole words, any case; failed counts as fail, unable and could not count as cannot): error on 6 lines (first L21), fail on 5 lines (first L21), "
                      "warning on 4 lines (first L26)", out)
        body = out.split("LAST LINES:")[1].split("\n\n", 1)[1]     # after the file's last lines
        self.assertLess(body.index("Error: disk <hex> failed"), body.index("Info CBS tick"))
        self.assertLess(body.index("Warning: low space"), body.index("Info CBS tick"))
        self.assertLess(body.index("L50: Error: once only"), body.index("Info CBS tick"))

    def test_the_label_says_which_words_count_together(self):
        # A reader who saw "cannot on 3 lines" searched for "cannot" and missed "Unable to ...".
        out = pf.profile_lines("s.log", lines_of("\n".join(["dvs: Unable to find devices"] + ["ok"] * 30)))
        self.assertIn("unable and could not count as cannot", out)
        self.assertIn("cannot on 1 line (first L1)", out)

    def test_a_problems_question_shows_only_the_problem_part(self):
        info = [f"2026-01-01 INFO tick n={i}" for i in range(40)]
        rare = ["2026-01-01 DEBUG cache warmed", "2026-01-01 DEBUG config loaded"]
        bad = ["2026-01-01 ERROR disk 7 failed"] * 5
        lines = lines_of("\n".join(info[:20] + bad + info[20:] + rare + ["2026-01-01 WARN once only"]))
        full = pf.profile_lines("a.log", lines)
        short = pf.profile_lines("a.log", lines, problems_only=True)
        self.assertIn("INFO tick", full)
        self.assertIn("ERROR disk <n> failed", short)
        self.assertIn("WARN once only", short)
        above = short.split("LAST LINES:")[0]    # the last lines are always shown
        self.assertNotIn("INFO tick", above)
        self.assertNotIn("cache warmed", above)
        self.assertIn("1 other line pattern and 2 other rare lines not shown", short)
        self.assertLess(len(short), len(full))

    def test_a_problems_question_shows_the_first_line_of_each_kind(self):
        # a reader saw "timeout on 5 lines (first L26562)" but not that
        # line's text, because its pattern was past the pattern limit.
        rows = ([f"ERROR store {i} closed" for i in range(5)] + [f"ERROR writer {i} null" for i in range(5)]
                + [f"INFO request {i} timeout after 9 s" for i in range(5)] + ["ok"] * 20)
        lines = lines_of("\n".join(rows))
        short = pf.profile_lines("a.log", lines, max_patterns=2, problems_only=True)
        self.assertNotIn("[5x, lines 11-15]", short)
        self.assertIn("FIRST LINE OF EACH KIND:", short)
        self.assertIn("   timeout L11: INFO request 0 timeout after 9 s", short)
        self.assertIn("   error L1: ERROR store 0 closed", short)
        self.assertLess(short.index("FIRST LINE OF EACH KIND"), short.index("[5x, lines 1-5]"))
        self.assertIn("(1 more problem pattern not shown)", short)
        self.assertNotIn("FIRST LINE OF EACH KIND", pf.profile_lines("a.log", lines, max_patterns=2))

    def test_the_first_lines_leave_out_a_word_that_is_part_of_the_format(self):
        lines = [f"ERROR: logging before init: step {i} ok" for i in range(99)]
        lines.insert(40, "ERROR: logging before init: warning disk slow")
        out = pf.profile_lines("g.log", lines_of("\n".join(lines)), problems_only=True)
        first = out.split("FIRST LINE OF EACH KIND:\n", 1)[1].split("\n\n", 1)[0]
        self.assertEqual(first, "   warning L41: ERROR: logging before init: warning disk slow")

    def test_a_problems_question_on_a_clean_log_shows_the_whole_profile(self):
        lines = lines_of("\n".join(f"2026-01-01 INFO tick n={i}" for i in range(40)))
        self.assertEqual(pf.profile_lines("a.log", lines, problems_only=True), pf.profile_lines("a.log", lines))

    def test_a_line_that_reports_no_problem_is_not_a_problem(self):
        for text in ["Scan done (error = 0)", "errors: 0", "Build finished with 0 errors and 0 warnings",
                     "no errors found", "completed without error", "failures=0", "Warnings: none",
                     "0 error(s), 0 warning(s)"]:
            self.assertEqual(pf.problem_words(text), set(), text)
        for text in ["errors: 3", "error = 5", "1 error", "Error: no space left", "no route; error 12",
                     "failed (error = 0x80004005)", "Failures: 2", "kind0 WARN step 0 failed", "job 0 error"]:
            self.assertTrue(pf.problem_words(text), text)

    def test_the_problem_count_says_how_many_zero_lines_it_left_out(self):
        lines = [f"run {i} ok (error = 0)" for i in range(30)] + ["run 30 failed (error = 7)"]
        out = pf.profile_lines("r.log", lines_of("\n".join(lines)))
        self.assertIn("error on 1 line (first L31)", out)
        self.assertIn("30 lines that report none, like 'error = 0', are not counted", out)

    def test_a_word_on_nearly_every_line_is_called_part_of_the_format(self):
        lines = [f"ERROR: logging before init: step {i} ok" for i in range(99)]
        lines.insert(40, "ERROR: logging before init: warning disk slow")
        out = pf.profile_lines("g.log", lines_of("\n".join(lines)))
        words = next(ln for ln in out.splitlines() if ln.startswith("PROBLEM WORDS"))
        self.assertIn("warning on 1 line (first L41)", words)
        self.assertIn("error on 100 lines (first L1; on nearly every line, so likely part of the line format)",
                      words)
        self.assertLess(words.index("warning"), words.index("error on 100"))

    def test_day_first_dates_and_urls_are_masked_as_such(self):
        self.assertEqual(pf.line_pattern("Start 13/10/2026 21:38:35 ok")[0], "Start <time> ok")
        self.assertEqual(pf.line_pattern("Start 07/16/2026 ok")[0], "Start <time> ok")
        pattern, fields = pf.line_pattern("GET https://example.com/a/b?x=1 done")
        self.assertEqual(pattern, "GET <url> done")
        self.assertEqual(fields["url1"], "https://example.com/a/b?x=1")

    def test_a_value_that_does_not_end_the_clause_is_a_real_problem(self):
        # "FATAL: 0 bytes free" was once read as "reports none" and hidden.
        real = ["FATAL: 0 bytes free on /dev/sda1, shutting down", "ERROR: 0 rows returned, expected 5",
                "ERROR: 0/5 replicas ready", "Error: 0 is not a valid index", "ERROR: 00:01:02 disk gone",
                "error: None of the hosts answered", "Warning: None of the certificates are valid",
                "error: false positive rate too high", "shard 0 errors: disk full", "worker 0 fails to start"]
        for text in real:
            with self.subTest(text=text):
                self.assertTrue(pf.problem_words(text), text)
        none = ["Build finished (error = 0)", "errors: 0, warnings: 0", "errors: 0 warnings: 0", "error=0",
                "Compiled with 0 errors", "Found 0 errors.", "0 error(s), 0 warning(s)", "errors: none",
                "status: ok; error = 0; retries = 2", "Build: 0 errors, 2 changes"]
        for text in none:
            with self.subTest(text=text):
                self.assertEqual(pf.problem_words(text), set(), text)

    def test_a_log_whose_only_problem_words_report_none_says_so(self):
        out = pf.profile_lines("n.log", lines_of("\n".join(f"run {i} ok (error = 0)" for i in range(30))))
        self.assertIn("PROBLEM WORDS", out)
        self.assertIn("30 lines that report none, like 'error = 0', are not counted", out)

    def test_a_format_word_does_not_put_every_line_first(self):
        # Every line starts "ERROR:" (the log's own prefix); the one FATAL line must still show.
        lines = [f"ERROR: logging before init: item {chr(97 + i % 26)}{chr(97 + i // 26)} loaded" for i in range(100)]
        lines.append("ERROR: logging before init: FATAL out of memory")
        out = pf.profile_lines("f.log", lines_of("\n".join(lines)))
        self.assertIn("L101: ERROR: logging before init: FATAL out of memory", out)
        self.assertIn("RARE LINES WITH PROBLEM WORDS (patterns seen 3 times or fewer), 1:", out)

    def test_more_date_forms_and_url_edges(self):
        self.assertEqual(pf.line_pattern("at 17.06.2026 04:00:35 started")[0], "at <time> started")
        self.assertNotIn("<time>", pf.line_pattern("ratio 3/4/1000 kept")[0])      # not a year
        pattern, fields = pf.line_pattern("see (https://x.com/a), then retry")
        self.assertEqual(fields["url1"], "https://x.com/a")
        self.assertEqual(pattern, "see (<url>), then retry")

    def test_space_padded_times_after_a_date_are_one_time(self):
        # A Windows log pads time fields with spaces ("06/07/2026  2:18: 8.000"), which split
        # one message into several patterns and showed pieces of the time as number stats.
        times = ["06/02/2026 18:37:38.00000976", "06/07/2026  2:18: 8.00000696", "06/13/2026  7:52:36.00000516",
                 "06/17/2026  4: 0:35.00000405", "2026-06-17  4: 0:35"]
        for t in times:
            with self.subTest(t=t):
                self.assertEqual(pf.line_pattern(f"{t}:RegOpenKey failed with (259)")[0],
                                 "<time>:RegOpenKey failed with (<n>)")
        lines = [f"{times[i % 4]}:RegOpenKey failed with (259)" for i in range(40)]
        out = pf.profile_lines("s.log", lines_of("\n".join(lines)))
        self.assertIn("1 line patterns", out)
        self.assertEqual(pf.line_pattern("step 1: 23 items, 2: 5 items")[0], "step <n>: <n> items, <n>: <n> items")

    def test_a_rare_problem_line_is_shown_even_past_the_rare_cap(self):
        rare = [f"Info step{chr(97 + i % 26)}{chr(97 + i // 26)} done" for i in range(100)]
        out = pf.profile_lines("r.log", lines_of("\n".join(rare + ["FATAL: out of memory"])))
        self.assertIn("L101: FATAL: out of memory", out)
        self.assertIn("more)", out)


if __name__ == "__main__":
    unittest.main()
