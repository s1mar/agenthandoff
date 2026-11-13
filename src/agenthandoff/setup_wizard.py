"""`agenthandoff setup`: choose a model server, get a model, save the choice.

Written for someone who has never run a local model. It asks one question at a
time, suggests an answer, finds what is already installed, and asks before
installing or downloading anything. Experts can skip every question with flags:

    agenthandoff setup --backend llamacpp --model ~/models/my.gguf --llama-server /opt/llama/llama-server
    agenthandoff setup --backend llamacpp --url http://127.0.0.1:8080     (a llama-server you run)
    agenthandoff setup --backend ollama --model qwen3:8b --yes

Without a terminal (an agent or a script running setup) nothing is installed or
downloaded unless --yes is given.

The result is <home>/config.json. Environment variables still override it.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from . import config, hardware, models
from .backend import OllamaBackend, OpenAIBackend
from .errors import BadSetting, Unavailable
from .llama_server import find_llama_server
from .net import local_get_status, local_stream


class SetupStopped(Exception):
    """Setup cannot go on; the message says what the user should do."""


def has_terminal() -> bool:
    """True if a person can answer questions on stdin. On Windows, isatty() is also
    true for the NUL device that agents and scripts often attach, so ask Windows
    whether stdin is a real console."""
    if not sys.stdin or not sys.stdin.isatty():
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes
        import msvcrt
        mode = ctypes.c_ulong()
        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
        return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
    except (OSError, ValueError, AttributeError):
        return False


class Prompter:
    """Asks questions on the terminal.

    --yes: take every suggested answer, including "yes, install / download".
    No terminal and no --yes: take suggested choices, but answer "no" to anything
    that installs or downloads, so a script never does that by surprise.
    """

    def __init__(self, assume_yes: bool, interactive: bool | None = None):
        self.assume_yes = assume_yes
        self.interactive = has_terminal() if interactive is None else interactive

    def choose(self, question: str, options: list[tuple[str, str]], default: int = 0) -> str:
        """options are (value, description). Returns the chosen value."""
        print(f"\n{question}\n")
        for i, (_, text) in enumerate(options, start=1):
            print(f"  {i}. {text}")
        if self.assume_yes or not self.interactive:
            print(f"\n-> {default + 1} (suggested)")
            return options[default][0]
        while True:
            raw = input(f"\nType a number [{default + 1}]: ").strip()
            if not raw:
                return options[default][0]
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return options[int(raw) - 1][0]
            print(f"Please type a number from 1 to {len(options)}.")

    def confirm(self, question: str) -> bool:
        """A yes/no question about installing or downloading something."""
        if self.assume_yes:
            print(f"{question} -> yes (--yes)")
            return True
        if not self.interactive:
            print(f"{question} -> no (no terminal to ask; add --yes to allow it)")
            return False
        return input(f"{question} [Y/n] ").strip().lower() in ("", "y", "yes")

    def text(self, question: str, flag: str) -> str:
        if self.assume_yes or not self.interactive:
            raise SetupStopped(f"{question}: no terminal to ask. Pass it with {flag}.")
        # A path dragged into the terminal arrives in quotes; remove them.
        return input(f"{question}: ").strip().strip("'\"")


# ---------------------------------------------------------------- llama.cpp

def _llama_install_command() -> list[str] | None:
    """The package-manager command that installs llama.cpp here, if there is one."""
    system = platform.system()
    if system == "Windows" and shutil.which("winget"):
        # The user has just said yes, so the package's terms are accepted for them;
        # without these flags winget can stop at an agreement prompt.
        return ["winget", "install", "--id", "ggml.llamacpp", "--exact",
                "--accept-source-agreements", "--accept-package-agreements"]
    if system in ("Darwin", "Linux") and shutil.which("brew"):
        return ["brew", "install", "llama.cpp"]
    return None


def _get_llama_server(ask: Prompter, given: str | None) -> str:
    if given:
        exe = find_llama_server(given)
        if not exe:
            raise SetupStopped(f"{given} is not a file.")
        return exe
    exe = find_llama_server()
    if exe:
        print(f"\nFound llama-server: {exe}")
        return exe
    command = _llama_install_command()
    if command and ask.confirm(f"\nllama.cpp is not installed. Install it now with `{' '.join(command)}`?"):
        if subprocess.run(command).returncode != 0:
            raise SetupStopped("the install command failed; see its output above.")
        exe = find_llama_server()
        if exe:
            print(f"Installed: {exe}")
            return exe
        print("The install finished, but llama-server is not on PATH yet. Open a new terminal and "
              "run `agenthandoff setup` again.")
    print("\nTo install llama.cpp yourself, download the build for your system from "
          "https://github.com/ggml-org/llama.cpp/releases and unzip it.")
    path = ask.text("Path to llama-server (llama-server.exe on Windows)", "--llama-server")
    exe = find_llama_server(path)
    if not exe:
        raise SetupStopped(f"{path} is not a file.")
    return exe


def _choose_gguf(ask: Prompter, model_arg: str | None, home: Path) -> str:
    if model_arg and Path(model_arg).expanduser().is_file():
        return str(Path(model_arg).expanduser().resolve())
    memory = hardware.memory_for_models()
    if memory:
        print(f"\nThis machine: {memory.description}.")
    if model_arg:
        chosen = models.by_key(model_arg)
        if chosen is None:
            raise SetupStopped(f"{model_arg!r} is neither a .gguf file nor one of: "
                               + ", ".join(m.key for m in models.CATALOG))
    else:
        suggested = models.suggest(memory.usable_mb if memory else None)
        options = []
        for m in models.CATALOG:
            fits = "" if memory is None or memory.usable_mb >= m.min_memory_mb else "  (may not fit; will be slow)"
            options.append((m.key, f"{m.title}, {m.size_gb} download{fits}"))
        options.append(("own", "Use a .gguf file I already have"))
        choice = ask.choose("Which model?", options, default=models.CATALOG.index(suggested))
        if choice == "own":
            path = Path(ask.text("Path to the .gguf file", "--model")).expanduser()
            if not path.is_file():
                raise SetupStopped(f"{path} is not a file.")
            return str(path.resolve())
        chosen = models.by_key(choice)
    if not models.is_ready(chosen, home):
        question = f"Download {chosen.file} ({chosen.size_gb}) from huggingface.co/{chosen.repo}?"
        if not ask.confirm(question):
            raise SetupStopped("no model downloaded. Run setup again when you are ready, or pass --model.")
    return str(models.download(chosen, home))


def _setup_llamacpp(ask: Prompter, model_arg: str | None, url: str | None, server_arg: str | None,
                    home: Path) -> dict:
    if url:   # a llama-server the user runs: nothing to install or download
        if local_get_status(f"{url}/health") != 200:
            raise SetupStopped(f"no llama-server answered at {url}/health. Start it, then run setup again.")
        return {"backend": "llamacpp", "url": url}
    exe = _get_llama_server(ask, server_arg)
    return {"backend": "llamacpp", "llama_server": exe, "model": _choose_gguf(ask, model_arg, home)}


# ---------------------------------------------------------------- Ollama and LM Studio

def _setup_ollama(ask: Prompter, model_arg: str | None, url: str) -> dict:
    if local_get_status(f"{url}/api/tags") != 200:
        raise SetupStopped(f"Ollama is not running at {url}. Install it from https://ollama.com/download, "
                           "start it, then run `agenthandoff setup` again.")
    wanted = model_arg or config.DEFAULT_OLLAMA_MODEL
    backend = OllamaBackend(url, wanted, 30)
    try:
        backend.describe_model()
    except Unavailable:
        if not ask.confirm(f"\nThe model {wanted} is not downloaded yet. Download it now with Ollama?"):
            raise SetupStopped(f"run `ollama pull {wanted}`, then run setup again.") from None
        _ollama_pull(url, wanted)
    values = {"backend": "ollama", "model": wanted}
    if url != config.DEFAULT_URLS["ollama"]:
        values["url"] = url
    return values


def _ollama_pull(url: str, name: str):
    """Pull a model through Ollama's API, printing progress."""
    for raw in local_stream(f"{url}/api/pull", {"model": name}, timeout=3600):
        try:
            event = json.loads(raw)
        except ValueError:
            continue
        if "error" in event:
            raise SetupStopped(f"Ollama could not download {name}: {event['error']}")
        total, done = event.get("total"), event.get("completed")
        line = f"{event.get('status', '')} {done * 100 // total}%" if total and done else event.get("status", "")
        print(f"\r  {line[:70]:70s}", end="", flush=True)
    print()


def _setup_openai(ask: Prompter, model_arg: str | None, url: str) -> dict:
    if local_get_status(f"{url}/v1/models") != 200:
        raise SetupStopped(f"no server answered at {url}. In LM Studio: load a model, open the Developer "
                           "tab and press Start Server. Then run `agenthandoff setup` again.")
    available = [m for m in OpenAIBackend(url, "", 30).available_models() if m]
    if model_arg:
        model = model_arg
    elif available:
        model = ask.choose("Which loaded model should agenthandoff use?", [(m, m) for m in available])
    else:
        raise SetupStopped("the server has no model loaded. Load one in LM Studio, then run setup again.")
    print(f"\nImportant: in LM Studio, set this model's context length to at least {config.CONTEXT_TOKENS} "
          "tokens, or long chunks will be cut off.")
    values = {"backend": "openai", "model": model}
    if url != config.DEFAULT_URLS["openai"]:
        values["url"] = url
    return values


# ---------------------------------------------------------------- entry point

ROUTES = [
    ("llamacpp", "llama.cpp  (recommended) agenthandoff starts the model only while a job runs, so it\n"
                 "                never holds your GPU memory. Most control over speed and memory."),
    ("ollama",   "Ollama     the easiest install; it downloads and manages models for you."),
    ("openai",   "LM Studio  a desktop app with a window, or any other OpenAI-compatible server."),
]
ROUTE_KEYS = {"backend", "model", "url", "llama_server"}   # replaced when the route changes


def _saved_settings() -> dict:
    """The current config.json, or {} if missing. A broken one is kept as a backup."""
    try:
        return config.read_config_file()
    except BadSetting as e:
        backup = config.config_path().with_suffix(".broken.json")
        config.config_path().replace(backup)
        print(f"Note: {e}\nThe old file was moved to {backup}; starting fresh.")
        return {}


def run_setup(backend: str | None = None, model: str | None = None, url: str | None = None,
              llama_server: str | None = None, assume_yes: bool = False,
              prompter: Prompter | None = None) -> int:
    """Interactive setup. Returns an exit code (0 ready, 2 stopped or not working)."""
    ask = prompter or Prompter(assume_yes)
    home = config.home_dir()
    url = url.rstrip("/") if url else None
    try:
        saved = _saved_settings()
        print("agenthandoff setup. Nothing is installed or downloaded without asking.")
        if backend is None:
            backend = ask.choose("How should agenthandoff run local models?", ROUTES, default=0)
        if backend == "llamacpp":
            values = _setup_llamacpp(ask, model, url, llama_server, home)
        elif backend == "ollama":
            values = _setup_ollama(ask, model, url or config.DEFAULT_URLS["ollama"])
        elif backend == "openai":
            values = _setup_openai(ask, model, url or config.DEFAULT_URLS["openai"])
        else:
            raise SetupStopped(f"unknown backend {backend!r}; use one of {', '.join(config.BACKENDS)}")
    except (SetupStopped, Unavailable) as e:
        print(f"\nSetup stopped: {e}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        print("\nSetup cancelled. Nothing was saved.", file=sys.stderr)
        return 2

    kept = {k: v for k, v in saved.items() if k not in ROUTE_KEYS}   # keep the user's tuning
    chain = kept.get("provider")
    if chain:   # the model just set up goes first; the user's other providers stay as fallbacks
        chain = [chain] if isinstance(chain, str) else list(chain)
        kept["provider"] = ["default"] + [name for name in chain if name != "default"]
    path = config.write_config_file({**kept, **values})
    print(f"\nSaved to {path}")
    if not {**kept, **values}.get("route.allow_model"):
        print("`digest` uses this model now. `read` sends plain text to it only after:\n"
              "  agenthandoff config set route.allow_model true")
    print("Checking that it works...\n")
    from .check import run_check
    return run_check()


if __name__ == "__main__":
    sys.exit(run_setup())
