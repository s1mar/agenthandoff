"""Tests for the exact cache of model answers: same question on unchanged files is answered
with no model call; any change to the files, the question, the provider or the settings that
shape the answer is a miss."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from support import FakeBackend, make_settings

from agenthandoff import api, cache, digest
from agenthandoff import providers as pv


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name, "app.log")
        self.file.write_text("start\nERROR disk full\nend\n")
        self.calls = 0

    def settings(self, **values):
        return make_settings(self.tmp.name, providers={"p": {"type": "command", "argv": ["x"], "remote": False}},
                             provider_chain=["p"], values=values)

    def digest(self, settings, question="why?"):
        def picker(prompt):
            self.calls += 1
            return [{"claim": "the disk is full", "quote": "ERROR disk full"}]
        backend = FakeBackend(picker, answer="Disk full [C1].")
        backend.capabilities, backend.name = pv.Capabilities(False, True, False), "p"
        with mock.patch.object(api.pv, "make_backend", return_value=backend):
            return api.run_digest([str(self.file)], question, settings=settings, recall_check=False, seed=0)

    def test_same_question_same_files_is_a_hit_with_the_same_answer(self):
        s = self.settings()
        first = self.digest(s)
        second = self.digest(s)
        self.assertEqual(self.calls, 1)
        self.assertEqual(second.stats["cache"], "hit")
        self.assertEqual(second.render().split("STATS")[0], first.render().split("STATS")[0])
        self.assertEqual(second.claims[0].pointer(), first.claims[0].pointer())

    def test_changed_file_question_or_setting_is_a_miss(self):
        s = self.settings()
        self.digest(s)
        self.digest(s, question="what else?")
        self.assertEqual(self.calls, 2)
        self.file.write_text("start\nERROR disk full\nend\nmore\n")
        self.digest(s)
        self.assertEqual(self.calls, 3)
        self.digest(make_settings(self.tmp.name, providers=s.providers, provider_chain=["p"], chunk_tokens=3000))
        self.assertEqual(self.calls, 4)

    def test_a_changed_prompt_is_a_miss(self):
        s = self.settings()
        self.digest(s)
        with mock.patch.object(digest, "MERGE_SYSTEM", digest.MERGE_SYSTEM + " Be brief."):
            self.digest(s)
        self.assertEqual(self.calls, 2)

    def test_every_instruction_the_model_reads_is_in_the_key(self):
        before = cache.prompts_hash()
        for module, name in [(digest, n) for n in ("MAP_SYSTEM", "RECALL_SYSTEM", "RECALL_REMINDER", "MERGE_SYSTEM", "SELECT_SYSTEM",
                                                   "FILE_NOTE_SYSTEM", "COMBINE_SYSTEM")] + \
                            [(pv, "JSON_NOTE"), (pv, "RETRY_NOTE")]:
            with self.subTest(name=name), mock.patch.object(module, name, getattr(module, name) + " x"):
                self.assertNotEqual(cache.prompts_hash(), before)
        for name in ("MAP_SCHEMA", "MERGE_SCHEMA", "SELECT_SCHEMA"):
            changed = dict(getattr(digest, name), extra=1)
            with self.subTest(name=name), mock.patch.object(digest, name, changed):
                self.assertNotEqual(cache.prompts_hash(), before)
        self.assertEqual(cache.prompts_hash(), before)

    def test_cache_can_be_turned_off(self):
        s = self.settings(**{"cache.enabled": False})
        self.digest(s)
        self.digest(s)
        self.assertEqual(self.calls, 2)

    def test_a_broken_cache_file_is_a_miss_not_a_crash(self):
        s = self.settings()
        self.digest(s)
        for f in (Path(self.tmp.name) / "cache").glob("*.json"):
            f.write_text("{broken")
        self.digest(s)
        self.assertEqual(self.calls, 2)

    def test_oldest_entries_are_removed_past_the_limit(self):
        s = self.settings(**{"cache.max_entries": 2})
        for q in ("a?", "b?", "c?"):
            self.digest(s, question=q)
        self.assertEqual(len(list((Path(self.tmp.name) / "cache").glob("*.json"))), 2)


if __name__ == "__main__":
    unittest.main()
