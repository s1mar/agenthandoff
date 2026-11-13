"""The one function the CLI, the MCP server and the evals all call to run a digest.

Order matters:
1. Read and chunk the files first. A wrong path or an input that is too big is
   reported at once, before any model is started.
2. If the cache holds an answer to this question about these exact files (same content,
   same provider and settings), return it: no model call.
3. For each provider in the chain, in order:
   - make it (a remote provider is refused unless allow_remote is on);
   - only if it uses the local GPU: take the GPU lock, so no other agenthandoff job runs
     at the same time, and check the GPU for programs that do not use our lock;
   - start it (for llama.cpp this starts llama-server), run the model steps, stop it.
   If the provider is unavailable (busy GPU, server down, refused), try the next one and
   note why in the result's stats.
4. Log the run to <home>/logs/digest.jsonl, whether it worked or not.
Any errors.Unavailable raised here means "do this job yourself".
"""
from __future__ import annotations

import contextlib
import time
from pathlib import Path

from . import cache, config, digest, hardware
from . import providers as pv
from .backend import Backend
from .config import Settings
from .errors import BackendError, GpuBusy, LockBusy, RemoteRefused, Unavailable
from .lock import GpuLock

# Errors that mean "this provider cannot do it now"; the next provider may. Input errors
# (no input, too big, bad settings) would fail the same way everywhere, so they stop the run.
PROVIDER_ERRORS = (BackendError, GpuBusy, LockBusy, RemoteRefused)


def _run_one(prepared, question, backend: Backend, settings: Settings, recall_check, seed, name):
    caps = getattr(backend, "capabilities", None)
    uses_gpu = caps.uses_local_gpu if caps is not None else True
    guard = GpuLock(settings.home, backend.model or name, "agenthandoff digest", settings.external_lock) \
        if uses_gpu else contextlib.nullcontext()
    with guard:
        if uses_gpu:
            reason = hardware.busy_reason(settings, backend.model_status())
            if reason:
                raise GpuBusy(reason)
        with backend:
            return digest.run_prepared(prepared, question, backend, recall_check, seed)


def run_digest(paths: list[str], question: str, include: list[str] | None = None,
               root: str | None = None, recall_check: bool | None = None, settings: Settings | None = None,
               seed: int | None = None, backend: Backend | None = None) -> digest.DigestResult:
    """Answer `question` about `paths` with a model. `backend` is for tests; normally the
    providers come from the settings."""
    settings = settings or config.load()
    if recall_check is None:
        recall_check = settings.get("digest.recall_check")
    chain = list(settings.provider_chain) or ["default"]
    record = {"question": question, "paths": paths, "backend": settings.backend, "chain": chain,
              "settings": settings.non_default()}
    started = time.monotonic()
    prepared = None
    try:
        limit = pv.chunk_limit(settings) if backend is None else digest.chunk_limit(backend, settings.chunk_tokens)
        prepared = digest.prepare(paths, include, Path(root) if root else None, limit)
        if backend is not None:
            result = _run_one(prepared, question, backend, settings, recall_check, seed, chain[0])
            result.stats["provider"] = getattr(backend, "name", chain[0])
        else:
            result = _with_chain(prepared, question, settings, chain, recall_check, seed)
    except Exception as e:
        partial = dict(prepared.stats) if prepared else {}
        digest.log_record(settings, {**record, **partial, "failed": f"{type(e).__name__}: {e}",
                                     "seconds": time.monotonic() - started})
        raise
    digest.log_record(settings, {**record, "recall_ok": result.recall_ok, "claims": len(result.claims),
                                 **result.stats})
    return result


def _with_chain(prepared, question, settings, chain, recall_check, seed):
    key = cache.key_for(prepared, question, settings, chain, recall_check, seed)
    if settings.get("cache.enabled"):
        hit = cache.get(settings, key)
        if hit is not None:
            hit.stats["cache"] = "hit"
            return hit
    tried, errors = [], []
    for name in chain:
        try:
            backend = pv.make_backend(settings, name)
            result = _run_one(prepared, question, backend, settings, recall_check, seed, name)
        except PROVIDER_ERRORS as e:
            tried.append(f"{name}: {e}")
            errors.append(e)
            continue
        result.stats["provider"] = name
        if tried:
            result.stats["fell_back"] = tried
        if settings.get("cache.enabled"):
            cache.put(settings, key, result)
        return result
    if len(errors) == 1:
        raise errors[0]                    # one provider: keep its own error and message
    raise Unavailable("no provider could do it: " + "; ".join(tried))
