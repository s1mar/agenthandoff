"""Start and stop llama.cpp's llama-server for one job.

Why we manage it ourselves: a model left loaded holds GPU memory that games, video
calls and other programs need. Starting the server when a job begins and stopping
it when the job ends means the model is never pinned in memory. Loading a
model takes seconds; a digest takes tens of seconds to minutes, so the cost is small.

The server must also stop if agenthandoff itself is killed (Ctrl+C, an agent's
command timeout, an MCP client closing). So llama-server is not started directly.
A small watcher process (`python -m agenthandoff.llama_server --watch -- CMD`)
starts it and reads its own stdin, which is a pipe from us. When we stop, or die
for any reason, the operating system closes that pipe, the watcher reads end of
file, and it stops llama-server. This works the same on Windows, macOS and Linux.

The server listens on 127.0.0.1 only, on a free port picked at start, so it never
clashes with a server the user runs and is not reachable from other machines.
Its output goes to <home>/logs/llama-server.log, the first place to look when it
fails to start.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from .errors import BackendError
from .net import local_get_status

STARTUP_TIMEOUT_S = 300   # first start can be slow on a cold disk or a CPU-only machine
STOP_TIMEOUT_S = 15


def find_llama_server(configured: str | None = None) -> str | None:
    """The llama-server program: the configured path, else PATH, else the places the
    usual installers put it (an already-open terminal does not see PATH changes)."""
    if configured:
        path = Path(configured).expanduser()
        return str(path) if path.is_file() else shutil.which(configured)
    found = shutil.which("llama-server")
    if found:
        return found
    candidates = []
    if os.name == "nt":
        winget = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet"
        candidates.append(winget / "Links" / "llama-server.exe")
        candidates += sorted((winget / "Packages").glob("ggml.llamacpp*/**/llama-server.exe"))
    else:
        candidates += [Path("/opt/homebrew/bin/llama-server"), Path("/usr/local/bin/llama-server")]
    return next((str(p) for p in candidates if p.is_file()), None)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"


_HELP_CACHE: dict[tuple[str, ...], bool] = {}


def _help_mentions(argv: list[str], flag: str) -> bool:
    """True if `argv --help` prints `flag`. A program that is missing, hangs or fails says no."""
    key = (*argv, flag)
    if key not in _HELP_CACHE:
        try:
            out = subprocess.run([*argv, "--help"], capture_output=True, text=True, timeout=30,
                                 encoding="utf-8", errors="replace")
            _HELP_CACHE[key] = flag in (out.stdout or "") + (out.stderr or "")
        except (OSError, subprocess.SubprocessError):
            _HELP_CACHE[key] = False
    return _HELP_CACHE[key]


def supports_reasoning_budget(exe: str) -> bool:
    return _help_mentions([exe], "--reasoning-budget")


class ManagedServer:
    def __init__(self, exe: str | None, model_path: str, context: int, extra_args: list[str], log_path: Path):
        self.exe_setting = exe
        self.model_path = model_path
        self.context = context
        self.extra_args = list(extra_args)
        self.log_path = Path(log_path)
        self.watcher: subprocess.Popen | None = None
        self.url: str | None = None

    def check_ready_to_start(self) -> str:
        """Raise BackendError with a fix if the program or the model file is missing."""
        exe = find_llama_server(self.exe_setting)
        if exe is None:
            where = f" at {self.exe_setting}" if self.exe_setting else " on this computer"
            raise BackendError(f"llama-server was not found{where}. Run `agenthandoff setup` to install it.")
        if not Path(self.model_path).is_file():
            raise BackendError(f"model file not found: {self.model_path}. Run `agenthandoff setup` to get one.")
        return exe

    def command(self, exe: str, port: int) -> list[str]:
        # With enable_thinking false, Gemma 4 still thought through every chunk of a
        # "find the smallest value" question until the reply limit, and wrote no answer.
        # --reasoning-budget 0 ends thinking at once; --reasoning off did not. Older builds
        # without the flag start as before, and a budget in llama_args is the user's choice.
        own = any(a == "--reasoning-budget" or a.startswith("--reasoning-budget=") for a in self.extra_args)
        budget = [] if own or not supports_reasoning_budget(exe) else ["--reasoning-budget", "0"]
        return [exe, "-m", self.model_path, "-c", str(self.context), "--host", "127.0.0.1",
                "--port", str(port), "--parallel", "1", *budget, *self.extra_args]

    def start(self) -> str:
        """Start the server and wait until it has loaded the model. Returns its URL.
        If anything goes wrong on the way, including Ctrl+C, the server is stopped."""
        exe = self.check_ready_to_start()
        port = _free_port()
        cmd = self.command(exe, port)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as log:
            log.write("$ " + subprocess.list2cmdline(cmd) + "\n")
            log.flush()
            self.watcher = subprocess.Popen(
                [sys.executable, "-m", "agenthandoff.llama_server", "--watch", "--", *cmd],
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [
                    str(Path(__file__).resolve().parents[1]), os.environ.get("PYTHONPATH")]))))
        self.url = f"http://127.0.0.1:{port}"
        try:
            self._wait_until_ready()
        except BaseException:
            self.stop()
            raise
        return self.url

    def _wait_until_ready(self):
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.watcher.poll() is not None:
                code = self.watcher.returncode
                raise BackendError(f"llama-server stopped while starting (exit {code}). "
                                   f"Last lines of {self.log_path}:\n{_tail(self.log_path)}")
            # /health answers 503 while the model loads and 200 once it is ready.
            if local_get_status(f"{self.url}/health", timeout=2) == 200:
                return
            time.sleep(0.5)
        raise BackendError(f"llama-server did not finish loading the model within {STARTUP_TIMEOUT_S} s. "
                           f"See {self.log_path}.")

    def stop(self):
        """Stop the server. Safe to call more than once."""
        if self.watcher is None:
            return
        watcher, self.watcher = self.watcher, None
        try:
            watcher.stdin.close()        # the watcher sees end of file and stops llama-server
        except OSError:
            pass
        try:
            watcher.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            watcher.kill()
            watcher.wait()


# ---------------------------------------------------------------- the watcher process

def _watch(cmd: list[str]) -> int:
    """Run cmd; stop it when our stdin closes. Returns cmd's exit code."""
    child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
    parent_gone = threading.Event()

    def wait_for_eof():
        try:
            while sys.stdin.buffer.read(1024):
                pass
        except OSError:
            pass
        parent_gone.set()

    threading.Thread(target=wait_for_eof, daemon=True).start()
    while child.poll() is None:
        if parent_gone.wait(0.5):
            child.terminate()
            try:
                child.wait(timeout=STOP_TIMEOUT_S - 5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            break
    return child.returncode


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "--watch" and sys.argv[2] == "--":
        sys.exit(_watch(sys.argv[3:]))
    print("usage: python -m agenthandoff.llama_server --watch -- COMMAND...", file=sys.stderr)
    sys.exit(1)
