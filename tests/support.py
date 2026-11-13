"""Shared test helpers: settings, a scripted fake model, and a fake managed llama-server.

Run as a script (`python support.py start-and-wait PID_FILE`) it starts a fake managed
server and then waits forever, so a test can kill it hard and check the server dies too.
"""
import json
import re
import sys
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path.insert(0, str(ROOT / "src"))

from agenthandoff import config, digest  # noqa: E402
from agenthandoff.backend import ModelStatus, Reply  # noqa: E402
from agenthandoff.errors import ThinkingOnly  # noqa: E402
from agenthandoff.llama_server import ManagedServer  # noqa: E402

# A test run must never touch the user's real settings folder, nor move the old one.
import os  # noqa: E402
import tempfile  # noqa: E402
os.environ.setdefault("AGENTHANDOFF_HOME", tempfile.mkdtemp(prefix="agenthandoff-test-home-"))
config.OLD_HOME = str(Path(tempfile.mkdtemp(prefix="agenthandoff-test-old-")) / "absent")


def make_settings(home, chunk_tokens=6000, **overrides):
    values = dict(home=Path(home), backend="ollama", model="fake", url="http://x", llama_server=None,
                  llama_args=[], chunk_tokens=chunk_tokens, timeout=1.0, external_lock=None,
                  ignore_gpu_busy=True, sources={})
    values.update(overrides)
    return config.Settings(**values)


def sent_ids(prompt):
    """Claim IDs in a merge-step prompt, in order ("C3 (path:12): ...")."""
    return re.findall(r"^(C\d+) \(", prompt, re.M)


class FakeBackend:
    """Answers map calls with picker(prompt) and the merge steps with fixed replies:
    the answer call and the combine call with `answer`, a per-file note with
    note(prompt) (default: the first claim ID it was sent), a pick call with
    select(prompt) (default: every ID it was sent). Records whether it was started
    (entered) so tests can check the order of steps, and every prompt by step."""
    model = "fake"

    def __init__(self, picker, answer="See [C1].", truncate_map=False, fail=None, select=None, note=None,
                 recall=None, thinks=None):
        self.picker, self.answer, self.truncate_map, self.fail = picker, answer, truncate_map, fail
        # thinks(system, user) -> True: that call spends its reply thinking, as a real model sometimes does
        self.thinks = thinks or (lambda system, user: False)
        self.select = select or sent_ids
        self.note = note or (lambda p: "note [" + sent_ids(p)[0] + "]")
        # recall(prompt) answers the recall call; the default reads the code as a good model would
        self.recall = recall or (lambda p: (re.findall(r"RC-\d{4}", p) or ["RC-0000"])[0])
        self.entered = False
        self.merge_prompts = []      # what the answer call was sent, for tests to check
        self.prompts = {"map": [], "recall": [], "select": [], "note": [], "combine": []}

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        return None

    def model_status(self):
        return ModelStatus(False, None)

    def describe_model(self):
        return "fake model"

    def chat(self, system, user, schema=None, max_output=1500):
        if self.fail:
            raise self.fail
        if self.thinks(system, user):
            raise ThinkingOnly("fake spent the reply thinking and wrote no answer.")
        if system == digest.MAP_SYSTEM:
            # truncate_map: True cuts off every map reply; a function decides per prompt.
            cut = self.truncate_map(user) if callable(self.truncate_map) else self.truncate_map
            self.prompts["map"].append(user)
            return Reply(json.dumps({"claims": self.picker(user)}), 100, 10, 0.01, truncated=cut)
        if system == getattr(digest, "RECALL_SYSTEM", None):
            self.prompts["recall"].append(user)
            return Reply(self.recall(user), 100, 5, 0.01)
        if system == digest.SELECT_SYSTEM:
            self.prompts["select"].append(user)
            return Reply(json.dumps({"ids": self.select(user)}), 100, 10, 0.01)
        if system == digest.FILE_NOTE_SYSTEM:
            self.prompts["note"].append(user)
            return Reply(json.dumps({"answer": self.note(user)}), 100, 10, 0.01)
        if system == digest.COMBINE_SYSTEM:
            self.prompts["combine"].append(user)
        else:
            self.merge_prompts.append(user)
        return Reply(json.dumps({"answer": self.answer}), 100, 10, 0.01)


class FakeManagedServer(ManagedServer):
    """A ManagedServer whose command runs fake_llama_server.py instead of llama-server."""

    def __init__(self, pid_file, log_path, fail=False, load_seconds=0.3):
        super().__init__(exe=sys.executable, model_path=sys.executable, context=1024, extra_args=[],
                         log_path=log_path)
        self.pid_file, self.fail, self.load_seconds = Path(pid_file), fail, load_seconds

    def command(self, exe, port):
        cmd = [exe, str(TESTS / "fake_llama_server.py"), "--port", str(port),
               "--pid-file", str(self.pid_file), "--load-seconds", str(self.load_seconds)]
        return cmd + (["--fail"] if self.fail else [])


if __name__ == "__main__" and sys.argv[1:2] == ["start-and-wait"]:
    pid_file = Path(sys.argv[2])
    server = FakeManagedServer(pid_file, pid_file.with_suffix(".log"))
    server.start()
    print("up", flush=True)
    time.sleep(3600)
