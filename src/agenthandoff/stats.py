"""`agenthandoff stats`: what ran, from the logs in <home>/logs.

It also lists the questions that reached the model most often. A question that keeps coming
back is a candidate for a script (see README, Scripts): written once by you, it then answers
with no model, every time. agenthandoff never writes or runs such a script by itself.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from . import config

TOP = 5


def _records(path: Path) -> list[dict]:
    out = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


def _shape(question: str) -> str:
    """A question with its numbers and quoted parts blanked, so repeats group together."""
    q = re.sub(r"`[^`]*`|\"[^\"]*\"", "<x>", (question or "").strip().lower())
    return re.sub(r"\d+", "<n>", q)


def summary(home: Path) -> str:
    reads = _records(home / "logs" / "read.jsonl")
    digests = _records(home / "logs" / "digest.jsonl")
    out = [f"logs in {home / 'logs'}"]
    tools = Counter(t for r in reads for t in r.get("tools", []))
    out.append(f"read: {len(reads)} runs" + (": " + ", ".join(f"{t} x{c}" for t, c in tools.most_common())
                                             if tools else ""))
    if reads:
        out.append(f"   output: {sum(r.get('chars', 0) for r in reads) // len(reads):,} characters on average")
    failed = sum(1 for r in digests if r.get("failed"))
    hits = sum(1 for r in digests if r.get("cache") == "hit")
    out.append(f"digest (model): {len(digests)} runs, {failed} not done, {hits} answered from the cache")
    providers = Counter(r.get("provider") for r in digests if r.get("provider"))
    if providers:
        out.append("   providers: " + ", ".join(f"{p} x{c}" for p, c in providers.most_common()))
    fell = sum(1 for r in digests if r.get("fell_back"))
    if fell:
        out.append(f"   {fell} runs fell back to a later provider")
    asked = Counter(_shape(r.get("question", "")) for r in digests if not r.get("failed") and r.get("cache") != "hit")
    repeats = [(q, c) for q, c in asked.most_common(TOP) if c > 1]
    if repeats:
        out.append("questions that reached the model more than once (a script could answer them with no model):")
        out += [f"   x{c}  {q}" for q, c in repeats]
    return "\n".join(out)


def run_stats() -> int:
    print(summary(config.home_dir()))
    return 0
