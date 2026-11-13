"""Tests for reading files, finding quotes and the digest steps. A scripted fake model
stands in for the real one, so no server, GPU or network is needed.

Run all tests: python -m unittest discover -s tests
"""
import json
import re
import tempfile
import unittest
from pathlib import Path

from support import FakeBackend, make_settings, sent_ids

from agenthandoff import chunk as ck
from agenthandoff import digest
from agenthandoff.errors import BackendError, InputTooLarge, NoInput, ThinkingOnly


def lines_of(text, path="f.txt"):
    return [ck.Line(path, i, t) for i, t in enumerate(text.splitlines(), 1)]


def probe_quote(prompt):
    """The planted line's text as the model would quote it, or None if not in this chunk."""
    found = [ln for ln in prompt.splitlines() if digest.PROBE_WORDS in ln and ln.startswith("L")]
    return found[0].split(": ", 1)[1] if found else None


# ---------------------------------------------------------------- finding quotes

class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.chunk = ck.Chunk(lines_of("alpha beta gamma\nint x = compute(42);\n  spaced    out   text\nlast line here"))

    def test_exact_quote_gives_code_line(self):
        r = ck.resolve("x = compute(42)", self.chunk)
        self.assertTrue(r.ok)
        self.assertEqual(r.hits, [ck.Hit("f.txt", 2, 2)])

    def test_label_copied_by_model_is_ignored(self):
        self.assertEqual(ck.resolve("L00002: int x = compute(42);", self.chunk).hits, [("f.txt", 2, 2)])

    def test_whitespace_is_normalised(self):
        self.assertTrue(ck.resolve("spaced out text", self.chunk).ok)

    def test_invented_quote_dropped(self):
        r = ck.resolve("x = compute(43)", self.chunk)
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, "quote not found")

    def test_short_quote_dropped(self):
        self.assertEqual(ck.resolve("x", self.chunk).reason, "quote too short")

    def test_only_ellipsis_is_empty_not_a_match_everywhere(self):
        self.assertEqual(ck.resolve("...", self.chunk).reason, "empty quote")
        self.assertEqual(ck.resolve("alpha beta gamma\n...", self.chunk).hits, [("f.txt", 1, 1)])

    def test_multiline_needs_consecutive_lines(self):
        self.assertEqual(ck.resolve("alpha beta gamma\nint x = compute(42);", self.chunk).hits, [("f.txt", 1, 2)])
        self.assertFalse(ck.resolve("alpha beta gamma\nlast line here", self.chunk).ok)

    def test_multiline_skips_blank_lines(self):
        c = ck.Chunk(lines_of("def f():\n\n    return 42"))
        self.assertEqual(ck.resolve("def f():\n\n    return 42", c).hits, [("f.txt", 1, 3)])
        self.assertEqual(ck.resolve("def f():\n    return 42", c).hits, [("f.txt", 1, 3)])

    def test_multiline_skips_the_probe_line(self):
        c = ck.Chunk(lines_of("first real line\nsecond real line"))
        c.lines.insert(1, ck.Line(None, 2, "ERROR recall check: unique marker RC-1 was planted here"))
        self.assertEqual(ck.resolve("first real line\nsecond real line", c).hits, [("f.txt", 1, 2)])

    def test_ellipsis_pieces_in_order(self):
        self.assertTrue(ck.resolve("alpha ... gamma", self.chunk).ok)
        self.assertFalse(ck.resolve("gamma ... alpha", self.chunk).ok)

    def test_quote_in_many_places_is_kept_with_every_place(self):
        # A log line repeated 300 times is often the key fact; it is real, so keep it.
        c = ck.Chunk(lines_of("\n".join(["return None  # ok"] * 5)))
        r = ck.resolve("return None", c)
        self.assertTrue(r.ok)
        self.assertEqual([h.start for h in r.hits], [1, 2, 3, 4, 5])

    def test_multiline_does_not_cross_files(self):
        c = ck.Chunk(lines_of("end of a\n", "a.py") + lines_of("start of b\n", "b.py"))
        self.assertFalse(ck.resolve("end of a\nstart of b", c).ok)

    # The cases below are quotes the real model (Gemma 4 E4B) wrote on the eval suite
    # that the first, stricter matcher dropped, losing required facts.

    def test_two_code_lines_joined_into_one(self):
        c = ck.Chunk(lines_of("    if old.hostname != new.hostname:\n        return True\n    return False"))
        r = ck.resolve("if old.hostname != new.hostname: return True", c)
        self.assertEqual(r.hits, [("f.txt", 1, 2)])

    def test_wrapped_comment_without_its_comment_marks(self):
        c = ck.Chunk(lines_of("x = 1\n# If we get redirected to a new host, we should strip out any\n"
                              "# authentication headers.\ny = 2"))
        r = ck.resolve("If we get redirected to a new host, we should strip out any authentication headers.", c)
        self.assertEqual(r.hits, [("f.txt", 2, 3)])

    def test_respaced_code_inside_brackets(self):
        c = ck.Chunk(lines_of('if (\n    a.scheme == "http"\n    and b.port in (443, None)\n):\n    return False'))
        r = ck.resolve('if (a.scheme == "http" and b.port in (443, None)): return False', c)
        self.assertEqual(r.hits, [("f.txt", 1, 5)])

    def test_full_stop_and_quote_marks_around_a_quote(self):
        self.assertEqual(ck.resolve('"alpha beta gamma."', self.chunk).hits, [("f.txt", 1, 1)])

    def test_probe_quoted_with_a_full_stop_still_counts(self):
        c = ck.Chunk(lines_of("first real line"))
        c.lines.append(ck.Line(None, 1, "ERROR recall check: unique marker RC-7311 was planted here"))
        r = ck.resolve("ERROR recall check: unique marker RC-7311 was planted here.", c)
        self.assertEqual(r.hits, [(None, 1, 1)])

    def test_match_points_at_the_line_not_the_lines_before_it(self):
        self.assertEqual(ck.resolve("spaced out text", self.chunk).hits, [("f.txt", 3, 3)])

    def test_span_is_limited(self):
        c = ck.Chunk(lines_of("\n".join(f"word{i}" for i in range(20))))
        near = " ".join(f"word{i}" for i in range(ck.MAX_SPAN_LINES))
        far = " ".join(f"word{i}" for i in range(ck.MAX_SPAN_LINES + 1))
        self.assertTrue(ck.resolve(near, c).ok)
        self.assertFalse(ck.resolve(far, c).ok)


class TokenEstimateTests(unittest.TestCase):
    def test_digits_count_one_each(self):
        # Real Gemma 4 count for this line is about 30 tokens; length/3 said 21.
        line = "2026-11-05T14:04:06 svc=checkout status=200 dur_ms=1259"
        self.assertGreaterEqual(ck.est_tokens(line), 28)

    def test_words_are_cheaper_than_their_length(self):
        self.assertLess(ck.est_tokens("return the configuration"), len("return the configuration") // 3)


# ---------------------------------------------------------------- reading and packing

class FilesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_folder_and_file_inside_it_read_once(self):
        (self.dir / "a.txt").write_text("hello", encoding="utf-8")
        files, _ = ck.collect_files([str(self.dir), str(self.dir / "a.txt")])
        self.assertEqual(len(files), 1)

    def test_include_filters_folders_but_not_named_files(self):
        (self.dir / "a.py").write_text("x", encoding="utf-8")
        (self.dir / "run.log").write_text("y", encoding="utf-8")
        self.assertEqual([f.name for f in ck.collect_files([str(self.dir)], ["*.py"])[0]], ["a.py"])
        self.assertEqual([f.name for f in ck.collect_files([str(self.dir / "run.log")], ["*.py"])[0]], ["run.log"])

    def test_unreadable_file_is_skipped_not_a_crash(self):
        lines, skipped = ck.read_lines([self.dir / "gone.txt"])   # listed, then deleted before reading
        self.assertEqual((lines, len(skipped)), ([], 1))

    def test_line_numbers_count_newlines_only(self):
        (self.dir / "f.txt").write_text("one\ntwo\fstill two\nthree\n", encoding="utf-8", newline="")
        lines, _ = ck.read_lines([self.dir / "f.txt"])
        self.assertEqual([(ln.num, ln.text) for ln in lines], [(1, "one"), (2, "two\fstill two"), (3, "three")])

    def test_windows_line_endings_are_one_line_each(self):
        (self.dir / "f.txt").write_bytes(b"one\r\ntwo\r\n")
        self.assertEqual([ln.text for ln in ck.read_lines([self.dir / "f.txt"])[0]], ["one", "two"])

    def test_utf16_file_is_text_not_binary(self):
        (self.dir / "ps.log").write_text("ERROR build failed\nline two\n", encoding="utf-16")
        files, skipped = ck.collect_files([str(self.dir)])
        self.assertEqual((len(files), skipped), (1, []))
        self.assertEqual(ck.read_lines(files)[0][0].text, "ERROR build failed")

    def test_real_binary_is_skipped(self):
        (self.dir / "x.bin").write_bytes(b"\x00\x01\x02binary")
        self.assertIn("binary", ck.collect_files([str(self.dir)])[1][0])
        (self.dir / "x.bin").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 20)
        self.assertIn("binary", ck.collect_files([str(self.dir)])[1][0])

    def test_text_log_with_a_block_of_zero_bytes_is_text(self):
        # a real NVIDIA log: a crash left 1,612 zero bytes between two lines
        (self.dir / "nv.log").write_bytes(b"a ok\r\nworker FAIL\r\n" + b"\x00" * 1612 + b"[x] next\r\n")
        files, skipped = ck.collect_files([str(self.dir)])
        self.assertEqual((len(files), skipped), (1, []))
        self.assertEqual([(ln.num, ln.text) for ln in ck.read_lines(files)[0]],
                         [(1, "a ok"), (2, "worker FAIL"), (3, "[x] next")])

    def test_utf16_without_a_byte_order_mark_is_text(self):
        for encoding in ("utf-16-le", "utf-16-be"):
            with self.subTest(encoding=encoding):
                # not plain ASCII: dropping zero bytes would read ASCII UTF-16 right by luck
                (self.dir / "w.log").write_bytes("ÉCHEC: build échoué à 10:00\nline two\n".encode(encoding))
                files, skipped = ck.collect_files([str(self.dir)])
                self.assertEqual((len(files), skipped), (1, []))
                self.assertEqual(ck.read_lines(files)[0][0].text, "ÉCHEC: build échoué à 10:00")

    def test_size_cap_is_per_call(self):
        (self.dir / "big.log").write_text("x" * 5000, encoding="utf-8")
        self.assertEqual(len(ck.collect_files([str(self.dir)])[0]), 1)
        files, skipped = ck.collect_files([str(self.dir)], max_bytes=1000)
        self.assertEqual(files, [])
        self.assertIn("larger than 1,000 bytes", skipped[0])
        self.assertIn("tools.max_file_bytes", skipped[0])


class PackTests(unittest.TestCase):
    def test_never_cuts_a_line_and_respects_budget(self):
        lines = lines_of("\n".join(f"line {i} " + "x" * (i % 50) for i in range(2000)))
        chunks = ck.pack(lines, 1000)
        self.assertGreater(len(chunks), 1)
        self.assertEqual([ln.num for c in chunks for ln in c.lines], list(range(1, 2001)))
        for c in chunks:
            # +1: est_tokens rounds each line up, and render() joins without the final newline
            self.assertLessEqual(ck.est_tokens(c.render()), 1000 + 1)

    def test_render_has_absolute_labels_and_headers(self):
        c = ck.Chunk(lines_of("a\nb", "x/y.py"))
        self.assertEqual(c.render(), "=== FILE: x/y.py ===\nL00001: a\nL00002: b")


# ---------------------------------------------------------------- the digest steps

class DigestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        body = [f"2026-01-01 INFO step {i} ok" for i in range(300)]
        body[150] = "2026-01-01 ERROR disk quota exceeded on /var/spool"
        (self.dir / "run.log").write_text("\n".join(body), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def picker_good(self, prompt):
        out = []
        if "disk quota exceeded" in prompt:
            out.append({"claim": "Disk quota error.", "quote": "ERROR disk quota exceeded on /var/spool"})
        out.append({"claim": "invented", "quote": "this text is not in the file at all"})
        return out

    def digest(self, picker, answer="See [C1].", chunk_tokens=800, **kw):
        return digest.run([str(self.dir / "run.log")], "why did it fail", FakeBackend(picker, answer, **kw),
                          make_settings(self.dir / "home", chunk_tokens), display_root=self.dir, seed=1)

    def test_pointer_comes_from_code_and_probe_is_hidden(self):
        res = self.digest(self.picker_good)
        self.assertTrue(res.recall_ok)
        self.assertEqual(len(res.claims), 1)
        self.assertEqual(res.claims[0].hits, [("run.log", 151, 151)])
        self.assertGreaterEqual(res.stats["dropped_reasons"]["quote not found"], 1)
        self.assertNotIn("recall check", res.render())
        self.assertIn("run.log:151", res.render())

    def test_missed_probe_flags_recall_failed(self):
        # A model that cannot find the code answers with the prompt's example, never a real marker.
        res = self.digest(self.picker_good, recall=lambda p: "RC-0000")
        self.assertFalse(res.recall_ok)
        self.assertIn("RECALL CHECK: FAILED", res.render())

    def test_the_answer_call_never_sees_the_planted_line(self):
        # A small model drops an off-topic line when it has good claims to give (25 real
        # digests, 2026-11-12: the check failed 17 times while the answer was right 24 times),
        # so the planted line goes to its own call and the chunks the answer uses stay clean.
        backend = FakeBackend(self.picker_good)
        digest.run([str(self.dir / "run.log")], "why did it fail", backend,
                   make_settings(self.dir / "home", 800), display_root=self.dir, seed=1)
        self.assertGreater(len(backend.prompts["map"]), 1)
        self.assertFalse([p for p in backend.prompts["map"] if "RC-" in p or digest.PROBE_WORDS in p])

    def test_the_recall_call_reads_one_whole_chunk_with_the_planted_line(self):
        backend = FakeBackend(self.picker_good)
        res = digest.run([str(self.dir / "run.log")], "why did it fail", backend,
                         make_settings(self.dir / "home", 800), display_root=self.dir, seed=1)
        self.assertEqual(len(backend.prompts["recall"]), 1)
        sent = backend.prompts["recall"][0]
        self.assertIn(res.stats["probe_marker"], sent)
        self.assertNotIn("why did it fail", sent)        # the question would pull it off task
        def labelled(prompt):
            return sum(ln.startswith("L") for ln in prompt.splitlines())
        # the largest chunk the answer calls read, plus the planted line (not a small tail chunk)
        self.assertEqual(labelled(sent), max(map(labelled, backend.prompts["map"])) + 1)
        self.assertTrue(res.recall_ok)

    def test_only_the_planted_code_passes(self):
        for reply, ok in [("RC-0000", False), ("I could not find it.", False), ("", False), ("RC-1", False)]:
            with self.subTest(reply=reply):
                self.assertEqual(self.digest(self.picker_good, recall=lambda p, r=reply: r).recall_ok, ok)
        echo = lambda p: "The code is " + re.findall(r"RC-\d{4}", p)[0] + "."  # noqa: E731
        self.assertTrue(self.digest(self.picker_good, recall=echo).recall_ok)

    def test_no_recall_call_when_the_check_is_off(self):
        backend = FakeBackend(self.picker_good)
        res = digest.run([str(self.dir / "run.log")], "why did it fail", backend,
                         make_settings(self.dir / "home", 800), display_root=self.dir, seed=1, recall_check=False)
        self.assertEqual(backend.prompts["recall"], [])
        self.assertIsNone(res.recall_ok)
        self.assertNotIn("RECALL CHECK", res.render())

    def test_the_recall_call_is_counted(self):
        on = self.digest(self.picker_good)
        backend = FakeBackend(self.picker_good)
        off = digest.run([str(self.dir / "run.log")], "why did it fail", backend,
                         make_settings(self.dir / "home", 800), display_root=self.dir, seed=1, recall_check=False)
        self.assertEqual(on.stats["calls"], off.stats["calls"] + 1)

    def test_one_reply_spent_thinking_does_not_end_the_digest(self):
        # Seen 2026-11-12 on the eval suites: one map reply of many came back as thinking
        # only, and the whole digest failed. That part is unreadable; the rest still counts.
        first_chunk = lambda system, user: system == digest.MAP_SYSTEM and "L00001:" in user  # noqa: E731
        res = self.digest(self.picker_good, thinks=first_chunk)
        self.assertEqual(res.stats["thinking_parts"], 1)
        self.assertEqual(res.stats["unreadable_chunks"], 1)
        self.assertIn("WARNING", res.render())
        self.assertEqual(res.claims[0].hits, [("run.log", 151, 151)])

    def test_every_reply_spent_thinking_still_says_how_to_fix_it(self):
        with self.assertRaises(ThinkingOnly) as e:
            self.digest(self.picker_good, thinks=lambda system, user: system == digest.MAP_SYSTEM)
        self.assertIn("thinking", str(e.exception))
        self.assertIsInstance(e.exception, BackendError)   # so a provider chain still falls back

    def test_a_recall_reply_spent_thinking_fails_the_check_only(self):
        res = self.digest(self.picker_good, thinks=lambda system, user: system == digest.RECALL_SYSTEM)
        self.assertFalse(res.recall_ok)
        self.assertEqual(res.claims[0].hits, [("run.log", 151, 151)])

    def test_many_claims_still_fill_the_per_chunk_limit(self):
        def picker(prompt):  # more real claims than the cap
            lines = [ln.split(": ", 1)[1] for ln in prompt.splitlines()
                     if ln.startswith("L") and "INFO" in ln][:digest.MAX_CLAIMS_PER_CHUNK + 2]
            return [{"claim": "step", "quote": q} for q in lines]
        res = self.digest(picker)
        self.assertTrue(res.recall_ok)
        self.assertGreater(res.stats["dropped_reasons"].get("over the per-chunk limit", 0), 0)

    def test_null_fields_are_empty_not_the_word_none(self):
        (self.dir / "run.log").write_text("value is None here\nsecond line", encoding="utf-8")
        res = self.digest(lambda p: [{"claim": None, "quote": None}, {"claim": None, "quote": "value is None here"}])
        self.assertEqual(len(res.claims), 1)
        self.assertEqual(res.claims[0].claim, "")

    def test_unknown_citations_are_replaced(self):
        res = self.digest(self.picker_good, "Bad [C9], range [C1-C7], good [c1].")
        self.assertEqual(res.answer, "Bad [?], range [C1-?], good [C1].")
        self.assertEqual(res.stats["unknown_cites"], 2)

    def test_cut_off_reply_is_reported(self):
        res = self.digest(self.picker_good, truncate_map=True)
        self.assertEqual(res.stats["unreadable_chunks"], res.stats["chunks"])
        self.assertIn("WARNING", res.render())

    def test_cut_off_chunk_is_split_and_retried(self):
        # The fake model "runs out of room" on any prompt with more than 100 lines (a
        # chunk here is about 250), as a real one does when a chunk is more tokens than
        # estimated. Two halvings bring it under.
        too_long = lambda p: sum(ln.startswith("L") for ln in p.splitlines()) > 100  # noqa: E731
        res = self.digest(self.picker_good, chunk_tokens=3000, truncate_map=too_long)
        self.assertGreater(res.stats["split_chunks"], 0)
        self.assertEqual(res.stats["unreadable_chunks"], 0)
        self.assertEqual(res.claims[0].hits, [("run.log", 151, 151)])
        self.assertTrue(res.recall_ok)

    def test_merge_sees_the_checked_quote_not_only_the_sentence(self):
        # On the eval suite the model claimed a 1,259 ms request was "slower than 5
        # seconds"; only the quote shows that is wrong, so the merge must get it.
        backend = FakeBackend(self.picker_good)
        digest.run([str(self.dir / "run.log")], "why did it fail", backend,
                   make_settings(self.dir / "home", 800), display_root=self.dir, seed=1)
        self.assertIn("quote: ERROR disk quota exceeded on /var/spool", backend.merge_prompts[0])

    def test_merge_input_is_capped_by_size(self):
        lines = [f"2026-01-01T00:00:{i % 60:02d} ERROR distinct failure number {i:04d} here" for i in range(900)]
        (self.dir / "run.log").write_text("\n".join(lines), encoding="utf-8")
        picker = lambda p: [{"claim": "e", "quote": ln.split(": ", 1)[1]}  # noqa: E731
                            for ln in p.splitlines() if "distinct failure" in ln and ln.startswith("L")][:12]
        backend = FakeBackend(picker)
        res = digest.run([str(self.dir / "run.log")], "q", backend, make_settings(self.dir / "home", 2000),
                         display_root=self.dir, seed=1)
        self.assertGreater(res.stats["merge_claims_cut"], 0)
        self.assertLessEqual(ck.est_tokens(backend.merge_prompts[0]), digest.MERGE_INPUT_TOKENS + 100)

    def test_a_small_window_gets_a_smaller_merge(self):
        # A 4096-token provider must not be sent the 8000-token merge a 12288 one gets.
        lines = [f"2026-01-01T00:00:{i % 60:02d} ERROR distinct failure number {i:04d} here" for i in range(900)]
        (self.dir / "run.log").write_text("\n".join(lines), encoding="utf-8")
        picker = lambda p: [{"claim": "e", "quote": ln.split(": ", 1)[1]}  # noqa: E731
                            for ln in p.splitlines() if "distinct failure" in ln and ln.startswith("L")][:12]
        backend = FakeBackend(picker)
        backend.context_tokens = 4096
        digest.run([str(self.dir / "run.log")], "q", backend, make_settings(self.dir / "home", 6000),
                   display_root=self.dir, seed=1)
        self.assertEqual(digest.merge_budget(backend), 2048)
        self.assertLessEqual(ck.est_tokens(backend.merge_prompts[0]), 2048 + 100)
        self.assertGreater(len(sent_ids(backend.merge_prompts[0])), 10)
        for prompt in backend.prompts["select"]:
            self.assertLessEqual(ck.est_tokens(prompt), 2048 + 100)

    def test_notes_too_big_to_combine_are_listed(self):
        # 4 notes of about 900 tokens each are more than the 2048 a 4096 window leaves.
        backend = FakeBackend(self.many_files(), answer="mod0 does this [C1].", note=lambda p: "word " * 700)
        backend.context_tokens = 4096
        res = digest.run([str(self.dir)], "summarize each module", backend,
                         make_settings(self.dir / "home", 2000), display_root=self.dir, seed=1)
        self.assertEqual(backend.prompts["combine"], [])
        self.assertTrue(res.answer.startswith("By file:"))
        self.assertEqual(res.stats["combine_skipped"], 1)

    def test_merge_cap_takes_claims_from_every_file(self):
        # One big file used to fill the merge budget, so a package summary saw only it.
        claims = [digest.Claim(f"C{i + 1}", "a long sentence about the big file " * 3, "q" * 150,
                               [ck.Hit("big.py", i + 1, i + 1)]) for i in range(400)]
        claims += [digest.Claim(f"C{401 + n}", f"only in small_{n}", "x", [ck.Hit(f"small_{n}.py", 1, 1)])
                   for n in range(3)]
        listed = digest._pick_for_merge(claims)
        self.assertLess(len(listed), len(claims))
        for n in range(3):
            self.assertIn(f"C{401 + n}", [c.id for c in listed])
        cost = sum(ck.est_tokens(digest._merge_entry(c)) + 1 for c in listed)
        self.assertLessEqual(cost, digest.MERGE_INPUT_TOKENS)
        self.assertEqual(listed, [c for c in claims if c in listed])     # still in file order

    def many_files(self, files=4, per_file=30):
        """`files` modules with `per_file` distinct functions each; the picker claims each def."""
        for f in range(files):
            body = [f"def mod{f}_function_{i:03d}(): return 'a distinct body number {i:03d}'" for i in range(per_file)]
            (self.dir / f"mod{f}.py").write_text("\n".join(body), encoding="utf-8")
        first_three = re.compile(r"_function_00[0-2]\(")       # 3 claims per file, so every file has some
        return lambda p: [{"claim": "defines a function", "quote": ln.split(": ", 1)[1]}  # noqa: E731
                          for ln in p.splitlines() if ln.startswith("L") and (per_file > 30 or first_three.search(ln))
                          ][:12]

    def test_few_claims_are_not_picked_first(self):
        # On a handful of claims picking can only lose one (it dropped the key line on log-npm).
        backend = FakeBackend(self.picker_good)
        digest.run([str(self.dir / "run.log")], "why did it fail", backend,
                   make_settings(self.dir / "home", 800), display_root=self.dir, seed=1)
        self.assertEqual(backend.prompts["select"], [])
        self.assertEqual(len(backend.merge_prompts), 1)

    def test_many_claims_are_picked_and_the_picked_shown_first(self):
        keep = {"C5", "C77"}
        backend = FakeBackend(self.many_files(files=1, per_file=300),
                              select=lambda p: [i for i in sent_ids(p) if i in keep] + ["C99999", 7],
                              answer="See [C77] and [C1].")
        res = digest.run([str(self.dir / "mod0.py")], "q", backend, make_settings(self.dir / "home", 400),
                         display_root=self.dir, seed=1)
        self.assertGreater(len(backend.prompts["select"]), 1)
        self.assertEqual(sent_ids(backend.merge_prompts[0]), ["C5", "C77"])   # unknown IDs ignored
        self.assertEqual(res.picked, keep)
        shown, _ = res.shown_claims()
        self.assertIn("[?]", res.answer)               # C1 was not sent to the merge, so it is not a citation
        self.assertEqual([c.id for c in shown[:3]], ["C77", "C5", "C1"])       # cited, then picked, then the rest

    def test_nothing_picked_keeps_every_claim(self):
        backend = FakeBackend(self.many_files(files=1, per_file=300), select=lambda p: [])
        res = digest.run([str(self.dir / "mod0.py")], "q", backend, make_settings(self.dir / "home", 400),
                         display_root=self.dir, seed=1)
        self.assertEqual(res.picked, set())
        self.assertGreater(len(sent_ids(backend.merge_prompts[0])), 2)

    def test_claims_from_several_files_get_one_note_per_file(self):
        backend = FakeBackend(self.many_files(), answer="mod0 does this [C1].")
        res = digest.run([str(self.dir)], "summarize each module", backend,
                         make_settings(self.dir / "home", 2000), display_root=self.dir, seed=1)
        self.assertEqual(len(backend.prompts["note"]), 4)
        for f in range(4):
            self.assertIn(f"File: mod{f}.py", "\n".join(backend.prompts["note"]))
        self.assertEqual(len(backend.prompts["combine"]), 1)
        self.assertEqual(backend.merge_prompts, [])
        self.assertEqual(res.answer, "mod0 does this [C1].")

    def test_a_combine_that_cites_nothing_is_replaced_by_the_notes(self):
        backend = FakeBackend(self.many_files(), answer="The notes do not give a summary of each module.")
        res = digest.run([str(self.dir)], "summarize each module", backend,
                         make_settings(self.dir / "home", 2000), display_root=self.dir, seed=1)
        self.assertTrue(res.answer.startswith("By file:"))
        for f in range(4):
            self.assertIn(f"- mod{f}.py: note [C", res.answer)
        self.assertEqual(res.stats["combine_refused"], 1)

    def test_pointer_lists_a_few_places_and_counts_the_rest(self):
        hits = [ck.Hit("a.log", n, n) for n in range(1, 8)]
        claim = digest.Claim("C1", "repeated", "q", hits)
        self.assertEqual(claim.pointer(), "a.log:1, a.log:2, a.log:3 (+4 more)")

    def test_probe_line_number_never_reported(self):
        res = self.digest(lambda p: [])
        self.assertTrue(res.recall_ok)
        self.assertEqual(res.claims, [])
        self.assertNotIn(res.stats["probe_marker"], res.render())

    def test_too_big_and_empty_input_are_refused(self):
        with self.assertRaises(InputTooLarge):
            digest.prepare([str(self.dir)], None, None, 800, max_input_tokens=100)
        with self.assertRaises(NoInput) as e:
            digest.prepare([str(self.dir / "missing.log")], None, None, 800)
        self.assertIn("not found", str(e.exception))

    def test_output_lists_cited_claims_first_and_caps_the_rest(self):
        lines = [f"2026 ERROR distinct failure number {i:03d} here" for i in range(40)]
        (self.dir / "run.log").write_text("\n".join(lines), encoding="utf-8")
        picker = lambda p: [{"claim": "e", "quote": ln.split(": ", 1)[1]}  # noqa: E731
                            for ln in p.splitlines() if "distinct failure" in ln and ln.startswith("L")]
        res = self.digest(picker, "Worst is [C30].", chunk_tokens=150)
        shown, omitted = res.shown_claims()
        self.assertEqual(shown[0].id, "C30")
        self.assertEqual(len(shown), digest.MAX_SHOWN_CLAIMS)
        self.assertEqual(omitted, len(res.claims) - digest.MAX_SHOWN_CLAIMS)

    def test_json_output_has_every_claim(self):
        data = self.digest(self.picker_good).to_dict()
        self.assertEqual(json.loads(json.dumps(data))["claims"][0]["pointer"], "run.log:151")


# ---------------------------------------------------------------- the log profile (code only)

from agenthandoff import profile as pf   # noqa: E402


class ProfileTests(unittest.TestCase):
    def test_pattern_masks_time_numbers_ids_paths_and_values(self):
        pattern, fields = pf.line_pattern("2026-01-02 03:04:05 web-4 job 9f3a2c1d took 12ms in src/a.py max_body_mb=1")
        self.assertEqual(pattern, "web-<n> job <id> took <n>ms in <path> max_body_mb=<max_body_mb>")
        self.assertEqual(fields["max_body_mb"], "1")
        self.assertEqual(fields["path1"], "src/a.py")
        self.assertEqual((fields["#1"], fields["#2"]), ("4", "12"))

    def test_the_one_host_set_differently_is_named_with_its_line(self):
        text = "\n".join(f"web-{i} config max_body_mb={'1' if i == 6 else '50'}" for i in range(1, 10))
        out = pf.profile_lines("h.log", lines_of(text))
        self.assertIn("max_body_mb: 50 x8, 1 x1  | odd one out: 1 at L6", out)

    def test_no_odd_one_out_when_values_are_spread(self):
        text = "\n".join(f"req tenant={t}" for t in "aaaaabbbbc")     # c is rare, but a covers only 50%
        out = pf.profile_lines("t.log", lines_of(text))
        self.assertIn("tenant: a x5, b x4, c x1", out)
        self.assertNotIn("odd one out", out)

    def test_a_latency_jump_is_named_at_its_first_line(self):
        values = [10 + i % 7 for i in range(30)] + [900, 950, 15]
        text = "\n".join(f"svc=api latency_ms={v}" for v in values)
        out = pf.profile_lines("l.log", lines_of(text))
        self.assertIn("max 950 (L32)", out)
        self.assertIn("first value over 10x the early median", out)
        self.assertIn("900 at L31", out)
        self.assertIn("max line L32: svc=api latency_ms=950", out)

    def test_no_jump_on_steady_numbers(self):
        text = "\n".join(f"latency_ms={10 + i % 9}" for i in range(40))
        self.assertNotIn("first value over", pf.profile_lines("l.log", lines_of(text)))

    def test_rare_lines_are_listed_with_their_numbers(self):
        text = "\n".join(["tick ok"] * 5 + ["FATAL disk full on /var"] + ["tick ok"] * 5)
        out = pf.profile_lines("r.log", lines_of(text))
        self.assertIn("[10x, lines 1-11] tick ok", out)
        self.assertIn("RARE LINES", out)
        self.assertIn("L6: FATAL disk full on /var", out)

    def test_long_lines_are_cut(self):
        out = pf.profile_lines("x.log", lines_of("x" * 500))
        self.assertIn("[cut]", out)
        self.assertNotIn("x" * 201, out)

    def test_pattern_and_rare_caps_say_how_many_are_hidden(self):
        common = [f"kind{chr(97 + i % 26)}{chr(97 + i // 26)} ok" for i in range(pf.MAX_PATTERNS + 2) for _ in range(4)]
        rare = [f"once{chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(pf.MAX_RARE_LINES + 1)]
        out = pf.profile_lines("c.log", lines_of("\n".join(common + rare)))
        self.assertIn("(2 more patterns not shown)", out)
        self.assertIn("(1 more)", out)

    def test_run_profile_reads_files_and_refuses_empty_input(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "a.log").write_text("tick\n" * 5)
            out = pf.run_profile([str(Path(d, "a.log"))], root=d)
            self.assertIn("LOG PROFILE: a.log - 5 lines", out)
            Path(d, "empty.log").write_text("")
            with self.assertRaises(NoInput):
                pf.run_profile([str(Path(d, "empty.log"))])
            with self.assertRaises(NoInput):
                pf.run_profile([str(Path(d, "missing.log"))])


if __name__ == "__main__":
    unittest.main()
