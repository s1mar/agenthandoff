"""Fixes from real files where the tools missed or misled.

Each test is a small, made-up file with the same shape as the real one. None copies a real file.
"""
import json
import unittest

import test_tools as tt

from agenthandoff import profile as pf
from agenthandoff import shape

ESC = "\x1b"


class RealProfileTests(tt.Files):
    def test_terminal_escape_codes_do_not_split_patterns_or_show(self):
        # A model-pull log: every line wrapped in cursor codes, with a bar and a timer.
        rows = []
        for i in range(60):
            bar = "█" * (i // 10)
            rows.append(f"{ESC}[?25l{ESC}[1Gpulling 0123456789ab: {i:3d}% ▕{bar:<10}▏ "
                        f"{i * 50} MB/5.5 GB  70 MB/s  {i % 3}m{(i * 7) % 60}s{ESC}[K{ESC}[?25h")
        out = pf.profile_lines("pull.log", tt.lines_of("\n".join(rows + [f"success {ESC}[K"])))
        head = out.splitlines()[0]
        self.assertRegex(head, r" [12] line patterns$", head)
        self.assertNotIn(ESC, out)
        self.assertNotIn("[?25l", out)

    def test_timers_are_numbers(self):
        self.assertEqual(pf.line_pattern("eta 1m11s left")[0], pf.line_pattern("eta 0s left")[0])
        self.assertEqual(pf.line_pattern("eta 1m11s left")[1]["#1"], "71")      # in seconds, like "71s"
        self.assertEqual(pf.line_pattern("took 2h3m")[0], "took <n>m")
        self.assertEqual(pf.line_pattern("took 2h3m")[1]["#1"], "123")
        self.assertEqual(pf.line_pattern("took 250ms")[0], "took <n>ms")        # one part: as before
        self.assertEqual(pf.line_pattern("model 7B loaded")[0], "model <n>B loaded")   # not a timer

    def test_a_progress_bar_is_one_pattern_and_its_last_line_is_shown(self):
        # A dashboard redrawn every 10 s: the bar fill made each frame a new pattern, and
        # only the first frame of a pattern was shown, so the final 90/90 was never seen.
        frames = []
        for i in range(10, 91, 10):
            fill = i * 30 // 90
            frames += [" GRID 3 models", f"   [{'#' * fill}{'.' * (30 - fill)}]  {i}/90  {i / 0.9:.1f}%",
                       "   retries 2   dead cells 1", "-" * 40]
        out = pf.profile_lines("progress.log", tt.lines_of("\n".join(frames)))
        bar_blocks = [ln for ln in out.splitlines() if ln.startswith("[") and "<n>%" in ln]
        self.assertEqual(len(bar_blocks), 1, out)
        self.assertRegex(out, r"last L\d+: +\[#+\]  90/90  100\.0%")

    def test_the_last_lines_of_the_file_are_shown_even_for_a_problems_question(self):
        # A docker build: tests inside it failed, but the build succeeded on its last lines.
        rows = [f"step {i} ok" for i in range(30)] + ["--- FAIL: TestWrite (0.00s)", "FAIL pkg/config"]
        rows += [f"step {i} ok" for i in range(30, 60)] + ["Successfully built abcdef012345",
                                                           "Successfully tagged img:pr-1"]
        out = pf.profile_lines("build.log", tt.lines_of("\n".join(rows)), problems_only=True)
        self.assertIn("FAIL: TestWrite", out)
        self.assertIn("LAST LINES:", out)
        self.assertIn("L64: Successfully tagged img:pr-1", out)
        self.assertIn("L63: Successfully built abcdef012345", out)
        # Before the long part, so a small output cap that cuts from the end keeps it.
        self.assertLess(out.index("LAST LINES:"), out.index("FAIL: TestWrite"))

    def test_last_lines_drop_the_time_and_tags_before_cutting(self):
        rows = [f"[2026-07-24 11:36:{i:02d},100] [build.py] [INFO]: step {i}" for i in range(40)]
        rows.append("[2026-07-24 11:37:00,100] [build.py] [INFO]: Successfully built 1a2b3c4d5e6f")
        out = pf.profile_lines("b.log", tt.lines_of("\n".join(rows)), max_line_chars=40)
        self.assertIn("L41: Successfully built 1a2b3c4d5e6f", out)

    def test_a_cut_problem_line_keeps_the_words_around_the_problem(self):
        # A servicing log: a 50-character date and level prefix ate an 80-character cut.
        prefix = "2026-11-11 15:56:34, Info                  CBS    "
        rows = [prefix + f"Loaded package {i}" for i in range(40)]
        rows.insert(20, prefix + "InternalOpenPackage of Package_for_KB5 failed [HRESULT = 0x800f0805 - CBS_E_INVALID]")
        out = pf.profile_lines("cbs.log", tt.lines_of("\n".join(rows)), max_line_chars=80, problems_only=True)
        rare = next(ln for ln in out.splitlines() if ln.strip().startswith("L21:"))
        self.assertIn("failed [HRESULT = 0x800f0805", rare)
        self.assertLess(len(rare), 130, rare)

    def test_a_short_problem_line_is_not_changed(self):
        out = pf.profile_lines("a.log", tt.lines_of("\n".join(["ok"] * 30 + ["ERROR disk full"])), problems_only=True)
        self.assertIn("L31: ERROR disk full", out)


class RealShapeTests(tt.Files):
    def rows_out(self, rows, question="", max_keys=50):
        path = self.write("r.jsonl", "\n".join(json.dumps(r) for r in rows))
        return path, shape.shape_file(path, self.lines_of(path), "r.jsonl", "jsonl", max_keys=max_keys,
                                      question=question)

    def test_lists_of_objects_are_looked_inside(self):
        # A session transcript: the tool error flag lives in message.content[].is_error.
        rows = [{"type": "assistant", "message": {"content": [{"type": "text", "text": f"step {i}"}]}}
                for i in range(6)]
        rows += [{"type": "user", "message": {"content": [{"type": "tool_result", "is_error": False}]}}
                 for _ in range(6)]
        rows.insert(9, {"type": "user", "message": {"content": [{"type": "tool_result", "is_error": True,
                                                                 "content": "InputValidationError"}]}})
        path, out = self.rows_out(rows)
        self.assertIn("message.content[].is_error", out)
        self.assertIn("true x1", out)
        self.assertIn("odd one out: true at L10", out)
        self.assert_real_line_numbers(out, path)

    def test_a_key_in_a_few_rows_shows_its_value(self):
        # A run trace: the one partial step said why only in a key on one row.
        rows = [{"step": "ast:query", "ms": i} for i in range(30)]
        rows[17] = {"step": "review:pass", "ms": 2120, "output": {"skipped": "partial: semgrep skipped: not on PATH"}}
        path, out = self.rows_out(rows)
        self.assertIn("output.skipped: partial: semgrep skipped: not on PATH at L18", out)

    def test_every_top_level_key_is_listed_even_when_its_value_never_changes(self):
        # "which fields?" got 7 of 12 keys, since a key with one value was left out.
        rows = [{"id": f"t-{i}", "source": "contractnli", "split": "test", "gold": "yes" if i % 2 else "no"}
                for i in range(20)]
        path, out = self.rows_out(rows)
        self.assertIn("source: always contractnli", out)
        self.assertIn("split: always test", out)

    def test_keys_named_in_the_question_come_first(self):
        rows = [{**{f"k{j}": j * i for j in range(30)}, "is_error": i == 7} for i in range(20)]
        path, out = self.rows_out(rows, question="Which calls returned errors?", max_keys=5)
        self.assertIn("is_error:", out)

    # Found on files picked for scoring. The score was taken
    # before these two fixes and is not re-claimed after them.
    def test_a_question_word_matches_whole_parts_of_a_key_not_any_substring(self):
        # "missing" stems to "miss", which sits inside "submission": that key went first.
        rows = [{"submission_len": i, "name": f"n{i}"} for i in range(10)]
        keys = shape._order(["name", "submission_len"], "Are any rows missing a field?")
        self.assertEqual(keys, ["name", "submission_len"])
        self.assertEqual(shape._order(["toolName", "isApiError"], "which errors?"), ["isApiError", "toolName"])

    def test_a_key_named_by_a_rare_question_word_comes_before_a_common_one(self):
        # "tool" named twenty keys and pushed the one "error" key down past the output limit.
        keys = [f"tool.part{i}" for i in range(20)] + ["result.is_error"]
        order = shape._order(keys, "Which tool calls returned errors?")
        self.assertEqual(order[0], "result.is_error")
        # Each word takes a turn, so a rarer word that names the wrong key cannot bury it either.
        keys += ["returnCode", "returnText", "error"]
        order = shape._order(keys, "Which tool calls returned errors?")
        self.assertLess(order.index("result.is_error"), 4, order[:6])
        # Data used as keys (a dict keyed by question text) is not matched.
        self.assertEqual(shape._order(["k", "answers.Which errors did you see?"], "errors?")[0], "k")

    def test_the_last_rows_of_a_json_lines_file_are_shown_near_the_top(self):
        # "what did the session do last?" got a key summary that never shows the end.
        rows = [{"uuid": f"id-{i:04d}", "type": "assistant", "text": f"working on step {i}"} for i in range(40)]
        rows.append({"uuid": "id-9999", "type": "assistant", "text": "All done: the report is written."})
        rows.append({"uuid": "id-9998", "type": "system", "subtype": "stop"})
        path, out = self.rows_out(rows, max_keys=2)
        head = out.split("\n   uuid:")[0]
        self.assertIn("LAST ROWS:", head)
        self.assertIn("L41: text=All done: the report is written., type=assistant", head)   # words first
        self.assertIn("L42: type=system, subtype=stop", head)
        self.assertNotIn("id-9999", head)        # a key with a new value on every row is an id: left out
        self.assert_real_line_numbers(out, path)

    def test_a_number_that_changes_on_every_row_is_kept_in_the_last_rows(self):
        rows = [{"model": "m1", "cost": 0.001 * i + 0.0001} for i in range(20)]
        path, out = self.rows_out(rows)
        self.assertRegex(out, r"L20: model=m1, cost=0\.019")


class RealCsvTests(tt.Files):
    # Found on a file picked for scoring, after it was scored.
    def test_doubled_quotes_after_the_sniffed_part_still_parse(self):
        rows = ["id,label,text"] + [f"{i},{'yes' if i % 2 else 'no'},plain text {i}" for i in range(400)]
        rows += [f'{400 + i},yes,"she said ""hi"", then left"' for i in range(5)]
        path = self.write("d.csv", "\n".join(rows))
        out = shape.shape_file(path, self.lines_of(path), "d.csv", "csv")
        self.assertIn("405 rows", out)
        self.assertNotIn("wrong number of columns", out)
        self.assertIn("label: yes x205, no x200", out)

    def test_counts_say_when_bad_rows_are_left_out(self):
        rows = ["a,b"] + [f"{i},x" for i in range(30)] + ["1,2,3"]
        path = self.write("e.csv", "\n".join(rows))
        out = shape.shape_file(path, self.lines_of(path), "e.csv", "csv")
        self.assertIn("counts below are of the 30 rows with 2 columns", out)


class RealRouterTests(tt.Files):
    settings = tt.RouterTests.settings
    read = tt.RouterTests.read

    def test_a_column_named_in_a_question_about_a_data_file_gets_the_shape_first(self):
        # "which config_id has the highest sharpe?" got find on the header line only.
        text = "config_id,model,sharpe\n" + "".join(f"{i},m{i % 3},{(i * 37) % 100 / 10 - 5:.2f}\n" for i in range(50))
        data = self.write("grid.csv", text)
        out = self.read([data], "How many configs, and which config_id has the highest sharpe?")
        self.assertTrue(out.startswith("ROUTE: shape"), out[:80])
        self.assertIn("50 rows, 3 columns", out)
        self.assertIn("sharpe: min", out)


if __name__ == "__main__":
    unittest.main()
