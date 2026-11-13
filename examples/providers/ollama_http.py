"""An example `python` provider: any model you can reach from Python, here Ollama's HTTP API.

Copy this file, change chat(), and point config.json at it:

    "providers": {
      "mine": {"type": "python", "factory": "C:/path/to/ollama_http.py:make",
               "options": {"model": "gemma4:e4b-it-q4_K_M"}}
    },
    "provider": "mine"

The factory gets `options` and returns an object with chat(). chat() returns the reply text
(or an agenthandoff Reply). When `schema` is given, return JSON matching it. The optional
`capabilities` dict tells agenthandoff what this provider does: here it runs on this
machine and uses its GPU, so agenthandoff takes the GPU lock and checks the GPU is free.
Standard library only, like agenthandoff itself.
"""
from __future__ import annotations

import json
import urllib.request


class OllamaHttp:
    capabilities = {"remote": False, "uses_local_gpu": True, "supports_schema": True}

    def __init__(self, options: dict):
        self.model = options.get("model", "gemma4:e4b-it-q4_K_M")
        self.url = options.get("url", "http://localhost:11434").rstrip("/")
        self.timeout = float(options.get("timeout", 300))

    def _call(self, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None    # None: a GET
        request = urllib.request.Request(self.url + path, data, {"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # local: skip any proxy
        with opener.open(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def describe_model(self) -> str:
        """Called by `agenthandoff check`. Raise with a fix if the model is missing."""
        names = [m["name"] for m in self._call("/api/tags").get("models", [])]
        if self.model not in names:
            raise RuntimeError(f"{self.model} is not in Ollama. Run `ollama pull {self.model}`.")
        return f"{self.model} at {self.url} (example python provider)"

    def chat(self, system: str, user: str, schema=None, max_output: int = 1500) -> str:
        body = {"model": self.model, "stream": False, "think": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"num_predict": max_output, "temperature": 0}}
        if schema is not None:
            body["format"] = schema
        return self._call("/api/chat", body)["message"]["content"]


def make(options: dict) -> OllamaHttp:
    return OllamaHttp(options)
