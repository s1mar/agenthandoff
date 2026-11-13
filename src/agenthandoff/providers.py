"""Providers: where a model answer comes from. Plug in any model by endpoint, CLI or Python.

Declared in your own config.json under "providers", and picked with "provider" (one name,
or a list to try in order):

    "providers": {
      "local":  {"type": "llamacpp", "model": "C:/models/gemma-4-E4B-it-Q4_0.gguf"},
      "ollama": {"type": "ollama", "model": "gemma4:e4b-it-q4_K_M"},
      "lms":    {"type": "openai", "url": "http://127.0.0.1:1234", "model": "qwen3-8b"},
      "hosted": {"type": "openai", "url": "https://api.example.com", "model": "m",
                 "api_key_env": "EXAMPLE_API_KEY"},
      "cli":    {"type": "command", "argv": ["ollama", "run", "{model}"], "model": "gemma3:4b",
                 "remote": false, "uses_local_gpu": true},
      "mine":   {"type": "python", "factory": "my_module:make_model", "options": {}}
    },
    "provider": ["local", "cli"]

Types:
- llamacpp, ollama: as set up by `agenthandoff setup`.
- openai: any OpenAI-compatible /v1/chat/completions endpoint (LM Studio, vLLM, llama-server,
  Ollama's /v1, hosted APIs). `api_key_env` names an environment variable holding the key;
  keys are never stored in config. `headers` and `extra_body` are sent as given.
- command: any program that reads a prompt on stdin and prints the answer. No shell.
  "{model}" in argv is replaced by `model`.
- python: "module:name" or "path/to/file.py:name", called with `options`; it returns an
  object with chat(system, user, schema=None, max_output=1500) giving a Reply or a string.

Capabilities decide two rules:
- uses_local_gpu: only these take the GPU lock and the GPU-busy check.
- remote: a provider that may send file text off this machine (a non-loopback URL; any
  command or python provider unless it says "remote": false). Refused unless allow_remote.
`supports_schema` false (default for command) means the JSON shape is asked for in the
prompt instead, and a reply that is not JSON is retried once.
"""
from __future__ import annotations

import importlib
import importlib.util
import ipaddress
import json
import os
import re
import shutil
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from . import batch, config
from .backend import Backend, LlamaCppBackend, OllamaBackend, OpenAIBackend, Reply
from .errors import BackendError, BadSetting, NotConfigured, RemoteRefused

TYPES = ("llamacpp", "ollama", "openai", "command", "python")
COMMON_KEYS = {"type", "model", "timeout", "remote", "uses_local_gpu", "supports_schema", "description",
               "context_tokens"}
TYPE_KEYS = {
    "llamacpp": {"url", "llama_server", "llama_args"},
    "ollama": {"url"},
    "openai": {"url", "api_key_env", "headers", "extra_body"},
    "command": {"argv"},
    "python": {"factory", "options"},
}
MIN_CONTEXT, MAX_CONTEXT = 4096, 1_048_576   # context_tokens range; below 4096 a chunk and its reply do not fit
MAX_REPLY_CHARS = 200_000          # a longer reply is cut and marked truncated
JSON_NOTE = ("\n\nReply with only one JSON object, no other text, matching this JSON schema:\n{schema}")
RETRY_NOTE = "\n\nYour last reply was not a JSON object. Reply again with only the JSON object."


@dataclass(frozen=True)
class Capabilities:
    uses_local_gpu: bool
    supports_schema: bool
    remote: bool


def parse_json(text: str) -> dict | None:
    from .digest import parse_json_reply
    return parse_json_reply(text)


# ---------------------------------------------------------------- checking specs

def check_spec(name: str, spec, where: str) -> None:
    """Raise BadSetting if a provider spec cannot work, naming what to fix."""
    label = f"{where}" if where != "settings" else f"provider {name!r}"
    if not isinstance(spec, dict):
        raise BadSetting(f"{label}: must be a JSON object with a \"type\"")
    kind = spec.get("type")
    if kind not in TYPES:
        raise BadSetting(f"{label}: type must be one of {', '.join(TYPES)}; got {kind!r}")
    if "api_key" in spec:
        raise BadSetting(f"{label}: do not put keys in config. Set \"api_key_env\" to the name of an "
                         "environment variable that holds the key.")
    allowed = COMMON_KEYS | TYPE_KEYS[kind]
    extra = sorted(set(spec) - allowed)
    if extra:
        raise BadSetting(f"{label}: unknown key {extra[0]!r} for type {kind} (allowed: {', '.join(sorted(allowed))})")
    for key in ("remote", "uses_local_gpu", "supports_schema"):
        if key in spec and not isinstance(spec[key], bool):
            raise BadSetting(f"{label}: {key} must be true or false")
    if "timeout" in spec and (isinstance(spec["timeout"], bool) or not isinstance(spec["timeout"], (int, float))
                              or spec["timeout"] <= 0):
        raise BadSetting(f"{label}: timeout must be a number of seconds above 0")
    if "context_tokens" in spec:
        ctx = spec["context_tokens"]
        if isinstance(ctx, bool) or not isinstance(ctx, int) or not MIN_CONTEXT <= ctx <= MAX_CONTEXT:
            raise BadSetting(f"{label}: context_tokens must be a whole number from {MIN_CONTEXT} to {MAX_CONTEXT}, "
                             "the model's context window in tokens")
    for key in ("model", "url", "llama_server", "api_key_env", "description"):
        if key in spec and not isinstance(spec[key], str):
            raise BadSetting(f"{label}: {key} must be a string")
    if kind == "command":
        argv = spec.get("argv")
        if not (isinstance(argv, list) and argv and all(isinstance(a, str) and a for a in argv)):
            raise BadSetting(f"{label}: argv must be a non-empty list of strings, e.g. [\"ollama\", \"run\", \"gemma3\"]. "
                             "Commands never run through a shell.")
    if kind == "python" and not isinstance(spec.get("factory"), str):
        raise BadSetting(f"{label}: factory must be \"module:name\" or \"path/to/file.py:name\"")
    if kind == "openai":
        if not spec.get("url"):
            raise BadSetting(f"{label}: url is required, e.g. \"http://127.0.0.1:1234\"")
        if not spec.get("model"):
            raise BadSetting(f"{label}: model is required (the model id the server knows)")
    for key in ("headers", "extra_body", "options"):
        if key in spec and not isinstance(spec[key], dict):
            raise BadSetting(f"{label}: {key} must be a JSON object")
    if "headers" in spec and not all(isinstance(v, str) for v in spec["headers"].values()):
        raise BadSetting(f"{label}: header values must be strings")
    if "llama_args" in spec and not (isinstance(spec["llama_args"], list)
                                     and all(isinstance(a, str) for a in spec["llama_args"])):
        raise BadSetting(f"{label}: llama_args must be a list of strings")
    window = next((a for a in spec.get("llama_args") or [] if re.fullmatch(r"-c\d*|--ctx-size(=.*)?", a)), None)
    if window:
        raise BadSetting(f"{label}: set the context window with \"context_tokens\", not {window!r} in llama_args, "
                         "so agenthandoff sizes its chunks for the same window")


def resolved_spec(spec: dict) -> dict:
    """The spec with defaults filled in and paths and URLs tidied."""
    out = dict(spec)
    kind = out["type"]
    if kind == "ollama":
        out.setdefault("url", config.DEFAULT_URLS["ollama"])
        out.setdefault("model", config.DEFAULT_OLLAMA_MODEL)
    if kind == "llamacpp" and out.get("model"):
        out["model"] = str(Path(out["model"]).expanduser())
    if out.get("url"):
        out["url"] = out["url"].rstrip("/")
    out.setdefault("llama_args", [])
    return out


def require_model(name: str, spec: dict) -> None:
    kind = spec["type"]
    if kind == "llamacpp" and not spec.get("model") and not spec.get("url"):
        raise NotConfigured("no model is set for the llamacpp backend. Run `agenthandoff setup`.")
    if kind == "openai" and not spec.get("model"):
        raise NotConfigured(f"no model is set for provider {name!r}. Run `agenthandoff setup`.")


def is_loopback(url: str | None) -> bool:
    host = (urllib.parse.urlsplit(url or "").hostname or "").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def capabilities(spec: dict, plugin=None) -> Capabilities:
    kind = spec["type"]
    if kind == "llamacpp":
        base = Capabilities(True, True, False) if not spec.get("url") else _url_caps(spec["url"])
    elif kind in ("ollama", "openai"):
        base = _url_caps(spec.get("url") or config.DEFAULT_URLS.get(kind))
    else:
        declared = getattr(plugin, "capabilities", None) or {}
        base = Capabilities(bool(declared.get("uses_local_gpu", False)),
                            bool(declared.get("supports_schema", False)), bool(declared.get("remote", True)))
    return Capabilities(spec.get("uses_local_gpu", base.uses_local_gpu),
                        spec.get("supports_schema", base.supports_schema), spec.get("remote", base.remote))


def _url_caps(url: str) -> Capabilities:
    local = is_loopback(url)
    return Capabilities(uses_local_gpu=local, supports_schema=True, remote=not local)


# ---------------------------------------------------------------- adapters

MAX_ARG_PROMPT = 30_000    # Windows caps a whole command line at 32,767 characters


class CommandBackend(Backend):
    """Any CLI: the prompt goes on stdin (no command-line length limit, no shell), the
    answer comes back on stdout. For a CLI that only takes the prompt as an argument
    (like `agy -p`), put "{prompt}" in argv: the prompt goes there and stdin is empty."""

    def __init__(self, name: str, spec: dict, timeout: float):
        self.name, self.spec = name, spec
        self.model = spec.get("model", "")
        self.timeout = float(spec.get("timeout", timeout))
        self.argv = [a.replace("{model}", self.model) for a in spec["argv"]]

    def _exe(self) -> str:
        exe = shutil.which(self.argv[0])
        if not exe:
            raise BackendError(f"provider {self.name}: program not found: {self.argv[0]}. Install it, or fix "
                               "argv in config.json.")
        return exe

    def describe_model(self):
        return f"{self._exe()} {' '.join(self.argv[1:])}".strip()

    def _run(self, prompt: str) -> tuple[str, bool]:
        argv = [self._exe(), *self.argv[1:]]
        stdin = prompt.encode("utf-8")
        if any("{prompt}" in a for a in argv[1:]):
            if len(prompt) > MAX_ARG_PROMPT:
                raise BackendError(f"provider {self.name}: the prompt ({len(prompt):,} characters) is too long for a "
                                   f"command line (at most {MAX_ARG_PROMPT:,}). Lower chunk_tokens, e.g. "
                                   "--set chunk_tokens=3000, or use a CLI that reads the prompt on stdin.")
            filled = [a.replace("{prompt}", prompt) for a in argv[1:] if "{prompt}" in a]
            if batch.unsafe_arg(argv[0], filled) is not None:
                raise BackendError(f"provider {self.name}: " + batch.refusal(argv[0], "file text")
                                   + ' Remove "{prompt}" from argv so the prompt goes on stdin, or point argv '
                                   "at the real program behind the .cmd file (for example node and its .js file).")
            argv = [argv[0]] + [a.replace("{prompt}", prompt) for a in argv[1:]]
            stdin = b""
        try:
            done = subprocess.run(argv, input=stdin, capture_output=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise BackendError(f"provider {self.name}: no answer after {self.timeout:g} s") from None
        except OSError as e:
            raise BackendError(f"provider {self.name}: cannot start {argv[0]}: {e}") from None
        if done.returncode != 0:
            tail = done.stderr.decode("utf-8", "replace").strip()[-300:]
            raise BackendError(f"provider {self.name}: {self.argv[0]} exited with code {done.returncode}"
                               + (f": {tail}" if tail else ""))
        text = done.stdout.decode("utf-8", "replace")
        if not text.strip():
            raise BackendError(f"provider {self.name}: {self.argv[0]} printed nothing")
        if len(text) > MAX_REPLY_CHARS:
            return text[:MAX_REPLY_CHARS], True
        return text, False

    def chat(self, system, user, schema=None, max_output=1500):
        started = time.monotonic()
        note = JSON_NOTE.format(schema=json.dumps(schema)) if schema is not None else ""
        prompt = f"{system}{note}\n\n{user}"
        text, cut = self._run(prompt)
        if schema is not None and not cut and parse_json(text) is None:
            text, cut = self._run(prompt + RETRY_NOTE)
        return Reply(text=text.strip(), input_tokens=None, output_tokens=None,
                     seconds=time.monotonic() - started, truncated=cut)


class PythonBackend(Backend):
    """A model given by your own Python code."""

    def __init__(self, name: str, spec: dict):
        self.name, self.spec = name, spec
        self.model = spec.get("model", spec["factory"])
        self.impl = _load_factory(name, spec["factory"])(spec.get("options", {}))
        if not callable(getattr(self.impl, "chat", None)):
            raise BackendError(f"provider {name}: {spec['factory']} returned an object without chat()")

    def _plugin_call(self, method: str, default):
        """Call an optional plugin method; any error it raises becomes a BackendError, so
        `check` and a digest report it instead of crashing."""
        fn = getattr(self.impl, method, None)
        if not callable(fn):
            return default()
        try:
            return fn()
        except Exception as e:
            raise BackendError(f"provider {self.name}: {e}") from e

    def describe_model(self):
        return self._plugin_call("describe_model", lambda: f"python provider {self.spec['factory']}")

    def model_status(self):
        return self._plugin_call("model_status", super().model_status)

    def chat(self, system, user, schema=None, max_output=1500):
        started = time.monotonic()
        try:
            reply = self.impl.chat(system, user, schema=schema, max_output=max_output)
        except Exception as e:
            raise BackendError(f"provider {self.name}: {type(e).__name__}: {e}") from e
        if isinstance(reply, str):
            reply = Reply(reply, None, None, time.monotonic() - started)
        if not isinstance(reply, Reply) or not reply.text.strip():
            raise BackendError(f"provider {self.name}: chat() returned nothing usable")
        return reply


def _load_factory(name: str, factory: str):
    module_name, sep, attr = factory.rpartition(":")
    if not sep or not module_name or not attr:
        raise BackendError(f"provider {name}: factory must look like module:name, got {factory!r}")
    try:
        if module_name.endswith(".py"):
            path = Path(module_name).expanduser()
            spec = importlib.util.spec_from_file_location(f"agenthandoff_plugin_{name}", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"no such file {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)
    except Exception as e:
        raise BackendError(f"provider {name}: cannot import {module_name}: {type(e).__name__}: {e}") from e
    target = getattr(module, attr, None)
    if target is None:
        raise BackendError(f"provider {name}: {module_name} has no {attr!r} (missing)")
    if not callable(target):
        raise BackendError(f"provider {name}: {factory} is not callable")
    return target


def _flat_spec(settings) -> dict:
    """The provider the old flat settings describe (tests and older callers)."""
    spec = {"type": settings.backend, "model": settings.model}
    if settings.url:
        spec["url"] = settings.url
    if settings.backend == "llamacpp":     # the llama keys are refused for every other type
        if settings.llama_args:
            spec["llama_args"] = settings.llama_args
        if settings.llama_server:
            spec["llama_server"] = settings.llama_server
    return spec


def make_backend(settings, name: str | None = None) -> Backend:
    """The backend for provider `name` (default: the first in the chain). It carries
    .name and .capabilities. Raises RemoteRefused for a remote provider without allow_remote."""
    name = name or (settings.provider_chain[0] if settings.provider_chain else "default")
    raw = settings.providers.get(name)
    if raw is None:
        if name != "default" or not settings.backend:
            raise NotConfigured("no local model is set up yet. Run `agenthandoff setup`.")
        raw = _flat_spec(settings)
    check_spec(name, raw, "settings")
    spec = resolved_spec(raw)
    kind = spec["type"]
    timeout = float(spec.get("timeout", settings.timeout))
    context = spec.get("context_tokens", config.CONTEXT_TOKENS)
    plugin = None
    if kind == "llamacpp":
        backend = LlamaCppBackend(_llama_settings(settings, spec, timeout), context=context)
    elif kind == "ollama":
        backend = OllamaBackend(spec["url"], spec["model"], timeout, context=context)
    elif kind == "openai":
        backend = OpenAIBackend(spec["url"], spec["model"], timeout, headers=spec.get("headers"),
                                api_key_env=spec.get("api_key_env"), extra_body=spec.get("extra_body"),
                                supports_schema=spec.get("supports_schema", True))
    elif kind == "command":
        backend = CommandBackend(name, spec, timeout)
    else:
        backend = PythonBackend(name, spec)
        plugin = backend.impl
    caps = capabilities(spec, plugin)
    if caps.remote and not settings.allow_remote:
        raise _refused(name)
    backend.name, backend.capabilities, backend.context_tokens = name, caps, context
    return backend


def chunk_limit(settings) -> int:
    """File text per map call: chunk_tokens, or half the smallest context window in
    the chain if that is less, so the chunk, the prompt and the reply fit every
    provider that may answer. A provider without context_tokens counts as CONTEXT_TOKENS."""
    windows = []
    for name in settings.provider_chain or ["default"]:
        spec = settings.providers.get(name)
        ctx = spec.get("context_tokens") if isinstance(spec, dict) else None
        windows.append(ctx if isinstance(ctx, int) and not isinstance(ctx, bool) else config.CONTEXT_TOKENS)
    return min([settings.chunk_tokens] + [w // 2 for w in windows])


def _refused(name: str) -> RemoteRefused:
    return RemoteRefused(f"provider {name!r} may send file text off this machine. Set \"allow_remote\": true in "
                         f"your own {config.config_path()} to allow it, or mark the provider \"remote\": false "
                         "if it runs locally.")


def _llama_settings(settings, spec: dict, timeout: float):
    from dataclasses import replace
    return replace(settings, backend="llamacpp", model=spec.get("model", ""), url=spec.get("url"),
                   llama_server=spec.get("llama_server") or settings.llama_server,
                   llama_args=spec.get("llama_args") or [], timeout=timeout)
