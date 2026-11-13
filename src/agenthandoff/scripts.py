"""User scripts: your own programs that answer some questions with no model.

Declared in your own config.json (never in a project file), for example:

    "scripts": [
      {"name": "git_log", "argv": ["git", "log", "--stat", "-n", "20"], "match": ["*"],
       "description": "the last 20 commits with the files they changed"},
      {"name": "junit", "argv": ["python", "C:/tools/junit_summary.py", "{paths}"],
       "match": ["*.xml"], "question": "fail|test"}
    ]

- `argv` is a list run with no shell. "{paths}" becomes one argument per path, "{question}"
  the question, "{root}" the --root folder (or the current folder).
- `match`: file name patterns; the router picks the script only when every file matches.
  `question` (optional): a regular expression the question must match.
- `timeout` (seconds, default 60) and `max_output_chars` (default: tools.max_output_chars).
- A script that fails, times out or prints nothing makes the tool exit 2: do it yourself.
Each script is also an MCP tool named script_<name>.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from . import batch
from .errors import BadSetting, ToolFailed

NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,47}$")
KEYS = {"name", "argv", "match", "question", "description", "timeout", "max_output_chars"}
DEFAULT_TIMEOUT = 60.0


def check_scripts(scripts: list, where: str) -> list[dict]:
    """The scripts with defaults filled in. Raises BadSetting naming the problem."""
    seen, out = set(), []
    for i, s in enumerate(scripts):
        label = f"{where} scripts[{i}]"
        if not isinstance(s, dict):
            raise BadSetting(f"{label}: each script must be a JSON object")
        extra = set(s) - KEYS
        if extra:
            raise BadSetting(f"{label}: unknown key {sorted(extra)[0]!r} (allowed: {', '.join(sorted(KEYS))}). "
                             "Scripts never run through a shell.")
        name = s.get("name")
        if not isinstance(name, str) or not NAME.match(name):
            raise BadSetting(f"{label}: name must be letters, digits, _ or -, starting with a letter")
        if name in seen:
            raise BadSetting(f"{label}: the name {name!r} is used twice")
        seen.add(name)
        argv = s.get("argv")
        if not (isinstance(argv, list) and argv and all(isinstance(a, str) and a for a in argv)):
            raise BadSetting(f"{label}: argv must be a non-empty list of strings, e.g. [\"git\", \"log\"]")
        match = s.get("match", ["*"])
        if not (isinstance(match, list) and match and all(isinstance(m, str) and m for m in match)):
            raise BadSetting(f"{label}: match must be a list of file name patterns, e.g. [\"*.log\"]")
        question = s.get("question")
        if question is not None:
            try:
                re.compile(question)
            except (re.error, TypeError) as e:
                raise BadSetting(f"{label}: question is not a valid regular expression: {e}") from None
        timeout = s.get("timeout", DEFAULT_TIMEOUT)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise BadSetting(f"{label}: timeout must be a number of seconds above 0")
        cap = s.get("max_output_chars")
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 100):
            raise BadSetting(f"{label}: max_output_chars must be a whole number, 100 or more")
        out.append({"name": name, "argv": argv, "match": match, "question": question,
                    "description": s.get("description") or f"runs {argv[0]}", "timeout": float(timeout),
                    "max_output_chars": cap})
    return out


def script_argv(script: dict, paths: list[str], question: str, root: str | None) -> list[str]:
    """The command line for a script. Raises ToolFailed when the program is a .cmd or .bat
    file and a question, path or root holds a character cmd.exe would run (see batch.py)."""
    argv, filled = [], []
    for i, arg in enumerate(script["argv"]):
        if arg == "{paths}":
            argv += paths
            filled += paths
        else:
            value = arg.replace("{question}", question).replace("{root}", root or str(Path.cwd()))
            argv.append(value)
            if i and value != arg:
                filled.append(value)
    exe = shutil.which(argv[0])       # finds .cmd and .exe shims on Windows; no shell needed
    if exe:
        argv[0] = exe
    if batch.unsafe_arg(argv[0], filled) is not None:
        raise ToolFailed(f"script {script['name']}: " + batch.refusal(argv[0], "the question or file names")
                         + " Point argv at the real program behind the .cmd file, or reword the question.")
    return argv


def run_script(script: dict, paths: list[str], question: str = "", root: str | None = None,
               max_chars: int = 12000) -> str:
    """Run one user script and return what it printed. Raises ToolFailed (exit 2) if it
    cannot start, fails, times out or prints nothing."""
    argv = script_argv(script, paths, question, root)
    cap = script.get("max_output_chars") or max_chars
    try:
        done = subprocess.run(argv, capture_output=True, timeout=script["timeout"], cwd=root or None,
                              stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        raise ToolFailed(f"script {script['name']}: program not found: {script['argv'][0]}") from None
    except subprocess.TimeoutExpired:
        raise ToolFailed(f"script {script['name']}: no result after {script['timeout']:g} s") from None
    except OSError as e:
        raise ToolFailed(f"script {script['name']}: cannot start: {e}") from None
    out = done.stdout.decode("utf-8", "replace")
    if done.returncode != 0:
        tail = done.stderr.decode("utf-8", "replace").strip()[-300:]
        raise ToolFailed(f"script {script['name']} exited with code {done.returncode}" + (f": {tail}" if tail else ""))
    if not out.strip():
        raise ToolFailed(f"script {script['name']} printed nothing")
    if len(out) > cap:
        out = out[:cap] + f"\n[output cut at {cap} characters; set max_output_chars to see more]"
    return out.rstrip("\n")
