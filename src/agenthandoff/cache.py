"""An exact cache of model answers, so the same question about unchanged files never costs a
model call twice.

The key is a SHA-256 of everything that shapes the answer: the text of every line read
(with its path and number), the question, the provider chain's specs, the chunk size, the
recall check and the seed. Change any of them and it is a miss. It is not a "similar question"
cache (rejected by design): only an identical job is reused.

Entries live in <home>/cache/<key>.json. Past cache.max_entries, the oldest are removed.
A cache file that cannot be read is a miss, never an error.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from . import chunk as ck
from . import digest
from .digest import Claim, DigestResult

VERSION = 1      # bump when the digest's output changes meaning, so old entries are not reused
def prompts_hash() -> str:
    """The model's fixed instructions (system prompts, reply shapes, the JSON notes a provider
    adds), so changing one never reuses old answers. The question and file text are in the
    key separately; a change to how the user message is laid out needs a VERSION bump."""
    from . import providers as pv
    parts = [digest.MAP_SYSTEM, digest.RECALL_SYSTEM, digest.RECALL_REMINDER, digest.MERGE_SYSTEM, digest.SELECT_SYSTEM,
             digest.FILE_NOTE_SYSTEM, digest.COMBINE_SYSTEM, pv.JSON_NOTE, pv.RETRY_NOTE]
    parts += [json.dumps(s, sort_keys=True) for s in (digest.MAP_SCHEMA, digest.MERGE_SCHEMA, digest.SELECT_SCHEMA)]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:16]


def key_for(prepared, question: str, settings, chain: list[str], recall_check: bool, seed) -> str:
    h = hashlib.sha256()
    head = {"v": VERSION, "prompts": prompts_hash(), "q": question, "chain": [settings.providers.get(n, {"flat": settings.backend,
                                                                              "model": settings.model})
                                                   for n in chain],
            "chunk_tokens": settings.chunk_tokens, "recall": bool(recall_check), "seed": seed}
    h.update(json.dumps(head, sort_keys=True, default=str).encode("utf-8"))
    for chunk in prepared.chunks:
        h.update(b"\x00chunk")
        for line in chunk.lines:
            h.update(f"\x00{line.path}\x01{line.num}\x01".encode("utf-8"))
            h.update(line.text.encode("utf-8"))
    return h.hexdigest()


def _folder(settings) -> Path:
    return Path(settings.home) / "cache"


def get(settings, key: str) -> DigestResult | None:
    path = _folder(settings) / f"{key}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        claims = [Claim(c["id"], c["claim"], c["quote"], [ck.Hit(*h) for h in c["hits"]]) for c in data["claims"]]
        result = DigestResult(data["question"], data["answer"], claims, data["recall_ok"], dict(data["stats"]),
                              list(data["skipped"]), set(data["picked"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    try:
        os.utime(path)               # recently used entries are kept longest
    except OSError:
        pass
    return result


def put(settings, key: str, result: DigestResult) -> None:
    folder = _folder(settings)
    data = {"question": result.question, "answer": result.answer, "recall_ok": result.recall_ok,
            "stats": dict(result.stats), "skipped": result.skipped, "picked": sorted(result.picked),
            "claims": [{"id": c.id, "claim": c.claim, "quote": c.quote, "hits": [list(h) for h in c.hits]}
                       for c in result.claims]}
    try:
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / f"{key}.tmp"
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, folder / f"{key}.json")
        _trim(folder, settings.get("cache.max_entries"))
    except OSError:
        pass                         # a cache that cannot be written only costs a model call later


def _trim(folder: Path, keep: int) -> None:
    entries = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime_ns)
    for old in entries[:max(0, len(entries) - keep)]:
        try:
            old.unlink()
        except OSError:
            pass
