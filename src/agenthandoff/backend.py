"""Talk to a local model server. Standard library only.

Three backends, all with the same chat() call:

- LlamaCppBackend: llama.cpp's llama-server. By default agenthandoff starts the
  server itself for one job and stops it afterwards (llama_server.py), so the model
  is only in memory while it is working. If `url` is set, it talks to a server you
  run yourself instead. Model "thinking" is turned off, as for Ollama.
- OllamaBackend: Ollama's native /api/chat, so we can set the context size, turn
  "thinking" off and read Ollama's own token counts.
- OpenAIBackend: /v1/chat/completions, for LM Studio and other OpenAI-compatible
  servers. The context size is whatever the server was loaded with.

Use a backend as a context manager (`with make_backend(settings) as backend:`) so a
server it started is always stopped. Every call returns a Reply; numbers the server
did not send stay None and are never guessed.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from .config import CONTEXT_TOKENS, Settings
from .errors import BackendError, ThinkingOnly
from .llama_server import ManagedServer
from .net import http_json, is_local_url, local_json

GPU_LAYER_FLAGS = ("-ngl", "--gpu-layers", "--n-gpu-layers")


@dataclass
class Reply:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    seconds: float
    truncated: bool = False          # the model hit the output limit, so the text is cut off


@dataclass
class ModelStatus:
    """What the GPU check needs to know about our model."""
    loaded: bool                     # already in GPU memory (so that memory is ours)
    size_mb: int | None              # memory it needs on the GPU; None if unknown or partly on the CPU


def _messages(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class Backend:
    """Base class. Subclasses implement chat() and describe_model(), and may start and
    stop a server in __enter__/__exit__."""

    model: str
    context_tokens: int = CONTEXT_TOKENS   # the window the digest sizes its prompts for

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def chat(self, system: str, user: str, schema: dict | None = None, max_output: int = 1500) -> Reply:
        """Send one system + user message. If `schema` is given, the server is asked to
        return JSON matching it. `max_output` caps the reply length in tokens."""
        raise NotImplementedError

    def describe_model(self) -> str:
        """A short description of the model, after checking the server can reach it.
        Raises BackendError (with a fix) if the server or the model is missing."""
        raise NotImplementedError

    def model_status(self) -> ModelStatus:
        return ModelStatus(loaded=False, size_mb=None)


class OllamaBackend(Backend):
    def __init__(self, url: str, model: str, timeout: float, context: int = CONTEXT_TOKENS):
        self.url, self.model, self.timeout = url, model, timeout
        self.context_tokens = context

    def chat(self, system, user, schema=None, max_output=1500):
        body = {
            "model": self.model,
            "messages": _messages(system, user),
            "stream": False,
            "think": False,
            "keep_alive": "5m",  # Ollama frees the model after 5 idle minutes; never pinned
            "options": {"temperature": 0, "num_ctx": self.context_tokens, "num_predict": max_output},
        }
        if schema is not None:
            body["format"] = schema
        started = time.monotonic()
        data = local_json(f"{self.url}/api/chat", body, self.timeout)
        return Reply(
            text=(data.get("message") or {}).get("content", ""),
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            seconds=time.monotonic() - started,
            truncated=data.get("done_reason") == "length",
        )

    def _same(self, name: str) -> bool:
        """'gemma4' and 'gemma4:latest' are the same model to Ollama."""
        return name in (self.model, f"{self.model}:latest")

    def installed_models(self) -> list[dict]:
        return local_json(f"{self.url}/api/tags", timeout=10).get("models", [])

    def describe_model(self):
        if not any(self._same(m.get("name", "")) for m in self.installed_models()):
            raise BackendError(f"{self.model} is not downloaded. Run `ollama pull {self.model}`.")
        return self.model

    def model_status(self):
        try:
            running = local_json(f"{self.url}/api/ps", timeout=5).get("models", [])
            loaded = [m for m in running if self._same(m.get("name", ""))]
            if loaded:
                return ModelStatus(True, int(loaded[0].get("size_vram") or 0) // 2**20 or None)
            for m in self.installed_models():
                if self._same(m.get("name", "")):
                    # Download size is a rough floor for the memory needed; the cache adds more.
                    return ModelStatus(False, int(m.get("size") or 0) // 2**20 or None)
        except BackendError:
            pass  # unknown; the GPU check then only looks at how busy the GPU is
        return ModelStatus(False, None)


class OpenAIBackend(Backend):
    # Fields added to every request. Empty here, because servers differ in what extra
    # fields they accept; LlamaCppBackend fills it in.
    EXTRA_BODY: dict = {}

    def __init__(self, url: str | None, model: str, timeout: float, headers: dict | None = None,
                 api_key_env: str | None = None, extra_body: dict | None = None, supports_schema: bool = True):
        self.url, self.model, self.timeout = url, model, timeout
        self.headers, self.api_key_env = dict(headers or {}), api_key_env
        self.extra_body, self.supports_schema = dict(extra_body or {}), supports_schema

    def _headers(self) -> dict:
        headers = dict(self.headers)
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "")
            if not key:
                raise BackendError(f"the environment variable {self.api_key_env} (api_key_env) is not set, so "
                                   "there is no API key to send.")
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _call(self, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        url = f"{self.url}{path}"
        return http_json(url, body, timeout or self.timeout, self._headers(), local=is_local_url(url))

    def chat(self, system, user, schema=None, max_output=1500):
        if schema is not None and not self.supports_schema:
            system += "\n\nReply with only one JSON object, no other text, matching this JSON schema:\n" + \
                json.dumps(schema)
        body = {"model": self.model, "messages": _messages(system, user),
                "temperature": 0, "max_tokens": max_output, **self.EXTRA_BODY, **self.extra_body}
        if schema is not None and self.supports_schema:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "result", "schema": schema}}
        started = time.monotonic()
        data = self._call("/v1/chat/completions", body)
        usage = data.get("usage") or {}
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        if not (message.get("content") or "").strip() and (message.get("reasoning")
                                                           or message.get("reasoning_content")):
            raise ThinkingOnly(
                f"{self.model} spent the reply thinking and wrote no answer. Turn thinking off for this "
                "provider: for Ollama's OpenAI endpoint add \"extra_body\": {\"reasoning_effort\": \"none\"}; "
                "in LM Studio turn thinking off for the model; for llama.cpp use a llama-server new enough to list "
                "--reasoning-budget in its --help (agenthandoff then passes 0).")
        return Reply(
            text=message.get("content") or "",
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            seconds=time.monotonic() - started,
            truncated=choice.get("finish_reason") == "length",
        )

    def available_models(self) -> list[str]:
        return [m.get("id", "") for m in self._call("/v1/models", timeout=10).get("data", [])]

    def describe_model(self):
        ids = self.available_models()
        if self.model not in ids:
            raise BackendError(f"{self.model} is not loaded on the server (it has: {', '.join(ids) or 'none'}). "
                               "Load it in LM Studio, or run `agenthandoff setup`.")
        return self.model


class LlamaCppBackend(OpenAIBackend):
    """llama-server speaks the OpenAI API, so chat() is inherited. What this adds is
    starting the server for the job (managed mode, the default) and stopping it after."""

    # Models such as Gemma 4 "think" first by default. llama-server puts that text in a
    # separate reasoning_content field, so the output budget is spent and content comes
    # back empty. Turn thinking off, as OllamaBackend does with "think": False.
    # Templates without a thinking switch ignore this.
    EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

    def __init__(self, settings: Settings, context: int = CONTEXT_TOKENS):
        super().__init__(settings.url, settings.model, settings.timeout)
        self.context_tokens = context
        self.managed = settings.url is None
        self.extra_args = settings.llama_args
        self.server = None
        if self.managed:
            self.server = ManagedServer(
                exe=settings.llama_server, model_path=settings.model, context=context,
                extra_args=settings.llama_args, log_path=settings.home / "logs" / "llama-server.log")

    def __enter__(self):
        if self.managed:
            self.url = self.server.start()
        return self

    def __exit__(self, *exc):
        if self.managed:
            self.server.stop()
            self.url = None

    def describe_model(self):
        if not self.managed:
            local_json(f"{self.url}/health", timeout=10)
            return f"the model loaded in your llama-server at {self.url}"
        self.server.check_ready_to_start()
        return f"{self.model} ({os.path.getsize(self.model) // 2**20} MB)"

    def model_status(self):
        if not self.managed:
            return ModelStatus(True, None)  # your server already holds the model
        if any(flag in self.extra_args for flag in GPU_LAYER_FLAGS):
            return ModelStatus(False, None)  # you chose to keep part of it on the CPU
        try:
            return ModelStatus(False, os.path.getsize(self.model) // 2**20)
        except OSError:
            return ModelStatus(False, None)


def make_backend(settings: Settings) -> Backend:
    """The first provider in the settings' chain (see providers.py)."""
    from .providers import make_backend as make
    return make(settings)
