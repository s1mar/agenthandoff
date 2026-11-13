"""What kind of file is this? By name first, then by looking at its first lines.

Kinds: code, json, jsonl (one JSON object per line, as many logs are), csv, log, text.
The router uses the kind to pick a code tool. Rules, not a model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CODE_EXT = {
    ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".go", ".rs", ".java", ".kt", ".kts", ".scala",
    ".cs", ".rb", ".php", ".swift", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".m", ".mm", ".sh",
    ".bash", ".zsh", ".ps1", ".psm1", ".lua", ".pl", ".r", ".dart", ".ex", ".exs", ".erl", ".clj", ".vue",
    ".svelte", ".sql",
}
LOG_EXT = {".log", ".out", ".err"}
JSONL_EXT = {".jsonl", ".ndjson"}
CSV_EXT = {".csv", ".tsv"}
SAMPLE = 200            # lines looked at
JSONL_SHARE = 0.6       # this share of lines are JSON objects -> jsonl
LOG_SHARE = 0.3         # this share look like log lines -> log

# A line looks like a log line if it starts with a time, holds an access-log time, or has a
# level word in capitals.
LOG_LINE = re.compile(
    r"^\s*\[?(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"           # ISO date and time
    r"|[A-Z][a-z]{2}\s+\d{1,2}\s\d{2}:\d{2}:\d{2}"                # syslog: Nov  1 03:04:05
    r"|\d{2}:\d{2}:\d{2})"                                        # time only
    r"|\[\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2}"             # access log: [11/Nov/2026:10:00:01
    r"|\b(INFO|WARN|WARNING|ERROR|DEBUG|TRACE|FATAL|CRITICAL|NOTICE)\b")


def _is_json_object(text: str) -> bool:
    text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return False
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


def kind_of(path: Path, lines: list[str]) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".json":
        return "json"
    if ext in CSV_EXT:
        return "csv"
    if ext in CODE_EXT:
        return "code"
    sample = [ln for ln in lines[:SAMPLE * 2] if ln.strip()][:SAMPLE]
    if ext in JSONL_EXT:
        return "jsonl"
    if sample and sum(_is_json_object(ln) for ln in sample) >= JSONL_SHARE * len(sample):
        return "jsonl"
    if ext in LOG_EXT:
        return "log"
    if sample and sum(bool(LOG_LINE.search(ln)) for ln in sample) >= LOG_SHARE * len(sample):
        return "log"
    return "text"
