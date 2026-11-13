"""HTTP helpers for talking to the model server on this machine.

All local calls go through one opener that ignores proxy settings. Without that,
urllib uses HTTP_PROXY or the Windows system proxy for 127.0.0.1 too, and on a
machine behind a company proxy every call to the local server goes to the proxy and
fails with a confusing error. Downloads from the internet (models.py) do use the
proxy, as they should.
"""
from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request

from .errors import BackendError

_local = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_remote = urllib.request.build_opener()       # uses the system proxy, as internet calls should


def is_local_url(url: str) -> bool:
    from .providers import is_loopback
    return is_loopback(url)


def local_json(url: str, body: dict | None = None, timeout: float = 30) -> dict:
    """GET (body None) or POST JSON to a local server and return the decoded reply.
    Every failure becomes a BackendError that names the URL."""
    return http_json(url, body, timeout, local=True)


def http_json(url: str, body: dict | None = None, timeout: float = 30, headers: dict | None = None,
              local: bool = True) -> dict:
    """Like local_json, with extra headers. Remote URLs (local=False) go through the system
    proxy; local ones never do. Error messages never include the headers (they may hold keys)."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with (_local if local else _remote).open(req, timeout=timeout) as resp:
            reply = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise BackendError(f"HTTP {e.code} from {url}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise BackendError(f"cannot reach {url}: {e}") from e
    except (ValueError, http.client.HTTPException) as e:
        raise BackendError(f"bad reply from {url}: {e}") from e
    if not isinstance(reply, dict):
        raise BackendError(f"bad reply from {url}: expected a JSON object")
    if "error" in reply and not reply.get("choices") and not reply.get("message"):
        err = reply["error"]
        raise BackendError(f"{url} returned an error: {err.get('message', err) if isinstance(err, dict) else err}")
    return reply


def local_stream(url: str, body: dict, timeout: float):
    """POST JSON and yield each line of a streamed reply (Ollama's pull progress)."""
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with _local.open(req, timeout=timeout) as resp:
            yield from resp
    except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
        raise BackendError(f"{url} failed: {e}") from e


def local_get_status(url: str, timeout: float = 3) -> int | None:
    """The HTTP status of a GET, or None if nothing answered."""
    try:
        with _local.open(url, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, http.client.HTTPException):
        return None
