"""shape: what is in a JSON, JSON-lines or CSV file, by counting. No model.

- JSON lines (one object per line, as many logs are): how many objects, lines that are not
  JSON, and for each key how often it is there and the spread of its values (as in the log
  profile: counts with any odd one out, or min/median/max with their lines).
- A JSON document: every path ($.a.b, $.list[].x) with its type and values.
- CSV or TSV: rows and columns, the spread of each column, and rows with the wrong number
  of columns.
Line numbers are real lines of the file (a JSON document has none: its paths are shown).
"""
from __future__ import annotations

import csv
import io
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from .profile import _cut, _describe_field, _value

MAX_EXAMPLE_CHARS = 60


def _flat(obj, prefix="") -> dict[str, str]:
    """A JSON object as {dotted.key: text}, nested objects flattened, lists summarized. The
    objects in a list are looked inside too, as name[].key, their values joined by ", "
    (e.g. an error flag in message.content[].is_error)."""
    out = {}
    for key, value in obj.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flat(value, name + "."))
        elif isinstance(value, list):
            out[name] = f"[{len(value)} items]"
            inner: dict[str, list[str]] = {}
            for item in value:
                if isinstance(item, dict):
                    for k, v in _flat(item, name + "[].").items():
                        inner.setdefault(k, []).append(v)
            for k, vs in inner.items():
                out[k] = ", ".join(vs)
        else:
            out[name] = json.dumps(value) if not isinstance(value, str) else value
    return out


FEW_ROWS = 3        # a key in this many rows or fewer has its values shown with their lines
QUESTION_WORD = re.compile(r"[A-Za-z][A-Za-z_]{3,}")


def _stem(word: str) -> str:
    word = word.lower()
    for end in ("ing", "ed", "es", "s"):
        if word.endswith(end) and len(word) - len(end) >= 4:
            return word[:-len(end)]
    return word


KEY_PART = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def _parts(key: str) -> set[str]:
    """The words of a key name: is_error, isError and message.content[].is_error all hold "error"."""
    return {_stem(p) for p in KEY_PART.findall(key)}


def _order(keys: list[str], question: str) -> list[str]:
    """Keys with a word of the question in their name first ("errors" puts is_error first), then
    the rest in the order they first appear. A word is a whole part of the key name, not any
    substring ("missing" is not in "submission"). The question's words take turns, rarest word
    first, so a word that names twenty keys ("tool" in a session log) cannot push down the one
    key another word names ("error"). Keys with spaces are data used as keys, not names, and
    are not matched."""
    stems = {_stem(w) for w in QUESTION_WORD.findall(question or "")}
    if not stems:
        return keys
    names = [k for k in keys if " " not in k]
    per_stem = {s: [k for k in names if s in _parts(k)] for s in stems}
    queues = sorted((q for q in per_stem.values() if q), key=len)
    named: list[str] = []
    for turn in range(max((len(q) for q in queues), default=0)):
        for q in queues:
            if turn < len(q) and q[turn] not in named:
                named.append(q[turn])
    chosen = set(named)
    return named + [k for k in keys if k not in chosen]


def _describe_key(key: str, rows) -> list[str]:
    """_describe_field, plus what it leaves out: a key with one value (said once, as "always"),
    and the values of a key that only a few rows have (often the one row that matters)."""
    lines = _describe_field(key, rows)
    if lines:
        return lines
    values = [(num, f[key]) for num, f, _ in rows if key in f]
    if not values:
        return []
    if len(values) <= FEW_ROWS:
        return [f"   {key}: " + ", ".join(f"{_value(v)} at L{num}" for num, v in values)]
    return [f"   {key}: always {_value(values[0][1])}"]


LAST_ROWS = 3           # a JSON-lines file's last rows, near the top: a session or run ends there
LAST_ROW_CHARS = 300
LAST_VALUE_CHARS = 120
NUMBER = re.compile(r"-?\d+(\.\d+)?([eE][-+]?\d+)?")
WORDS = re.compile(r"\s*[^\s\[{]\S*\s+\S+\s+\S")      # three words or more, not "[2 items]"


def _last_rows(rows) -> list[str]:
    """The last rows as key=value. Id-like keys (a new value on most rows, and no spaces in
    any, like a uuid or a timestamp) are left out, and values with spaces (words for people)
    come first, so the line shows what happened. Placed before the key summary so the output
    limit never cuts it. (A key summary alone never shows how a file ends.)"""
    if len(rows) <= LAST_ROWS:
        return []
    seen: dict[str, set] = {}
    count: Counter = Counter()
    for _, f, _ in rows:
        for k, v in f.items():
            seen.setdefault(k, set()).add(v)
            count[k] += 1
    id_like = {k for k in seen if count[k] > FEW_ROWS and len(seen[k]) * 2 > count[k]
               and not any(" " in v for v in seen[k])        # text has spaces; ids and times do not
               and not all(NUMBER.fullmatch(v) for v in seen[k])}       # a cost or a score is not an id
    out = ["   LAST ROWS:"]
    for num, f, _ in rows[-LAST_ROWS:]:
        kept = [(k, v) for k, v in f.items() if k not in id_like and v not in ("", "[0 items]", "null")]
        kept.sort(key=lambda kv: not WORDS.match(kv[1]))     # words for people first, then codes
        pairs = [f"{k}={_cut(' '.join(v.split()), LAST_VALUE_CHARS)}" for k, v in kept]
        text = ", ".join(pairs)
        out.append(f"   L{num}: " + (text if len(text) <= LAST_ROW_CHARS else text[:LAST_ROW_CHARS] + "..."))
    return out


def _jsonl(lines: list[str], name: str, max_keys: int, question: str = "") -> str:
    rows, bad = [], []
    for num, text in enumerate(lines, 1):
        if not text.strip():
            continue
        try:
            obj = json.loads(text)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            rows.append((num, _flat(obj), text))
        else:
            bad.append(num)
    keys = _order(list(dict.fromkeys(k for _, f, _ in rows for k in f)), question)
    out = [f"{name}: {len(rows)} JSON objects, {len(keys)} keys"]
    if bad:
        out.append(f"   {len(bad)} lines are not JSON (first: L{bad[0]})")
    out += _last_rows(rows)
    for key in keys[:max_keys]:
        missing = [num for num, f, _ in rows if key not in f]
        if missing:
            more = f" +{len(missing) - 3} more" if len(missing) > 3 else ""
            out.append(f"   {key}: in {len(rows) - len(missing)} of {len(rows)} (missing at "
                       + ", ".join(f"L{n}" for n in missing[:3]) + more + ")")
        out += _describe_key(key, rows)
    if len(keys) > max_keys:
        out.append(f"   ({len(keys) - max_keys} more keys not shown)")
    return "\n".join(out)


def _type(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return "list" if isinstance(value, list) else "object"


def _walk(value, path: str, seen: dict):
    if isinstance(value, dict):
        for key, child in value.items():
            _walk(child, f"{path}.{key}", seen)
    elif isinstance(value, list):
        seen.setdefault(path + "[]", {"items": 0, "values": []})["items"] += len(value)
        for child in value:
            _walk(child, path + "[]", seen)
    else:
        seen.setdefault(path, {"items": None, "values": []})["values"].append(value)


def _show(value) -> str:
    text = repr(value) if isinstance(value, str) else json.dumps(value)
    return text if len(text) <= MAX_EXAMPLE_CHARS else text[:MAX_EXAMPLE_CHARS] + "..."


def _json(lines: list[str], name: str, max_keys: int) -> str:
    try:
        doc = json.loads("\n".join(lines))
    except ValueError as e:
        return f"{name}: not valid JSON ({e})"
    seen: dict = {}
    _walk(doc, "$", seen)
    out = [f"{name}: JSON {_type(doc)}, {len(lines)} lines, {len(seen)} paths"]
    for path, info in list(seen.items())[:max_keys]:
        values = info["values"]
        if info["items"] is not None and not values:
            out.append(f"   {path}: {info['items']} items")
            continue
        if info["items"] is not None:
            out.append(f"   {path}: {info['items']} items")
        types = Counter(_type(v) for v in values)
        if len(types) > 1:
            out.append(f"   {path}: mixed: " + ", ".join(f"{t} x{c}" for t, c in types.most_common()))
            continue
        kind = next(iter(types))
        if len(values) == 1:
            out.append(f"   {path}: {kind} {_show(values[0])}")
        elif kind == "number":
            out.append(f"   {path}: number, {len(values)} values, min {min(values):g}, max {max(values):g}"
                       + (f", median {statistics.median(values):g}" if len(values) > 2 else ""))
        else:
            counts = Counter(_show(v) for v in values)
            out.append(f"   {path}: {kind}, {len(values)} values, {len(counts)} distinct, most common "
                       + ", ".join(f"{v} x{c}" for v, c in counts.most_common(3)))
    if len(seen) > max_keys:
        out.append(f"   ({len(seen) - max_keys} more paths not shown)")
    return "\n".join(out)


def _read_csv(text: str, lines: list[str], dialect):
    reader = csv.reader(io.StringIO(text), dialect)
    header, rows, bad = None, [], []
    for record in reader:
        num = reader.line_num
        if not any(cell.strip() for cell in record):
            continue
        if header is None:
            header = [h.strip() or f"column{i + 1}" for i, h in enumerate(record)]
            continue
        if len(record) != len(header):
            bad.append(num)
            continue
        rows.append((num, dict(zip(header, record)), lines[num - 1] if num <= len(lines) else ""))
    return header, rows, bad


def _csv(lines: list[str], name: str, max_keys: int, tab: bool) -> str:
    """The sniffer reads only the first 4 KB, so it can guess wrong for later rows (once it guessed no doubled quotes, and 83 rows with "" split). The standard dialect
    is tried too, and the reading with fewer bad rows is kept."""
    text = "\n".join(lines)
    plain = csv.excel_tab if tab else csv.excel
    try:
        dialects = [csv.Sniffer().sniff(text[:4096], delimiters="\t" if tab else ",;|\t"), plain]
    except csv.Error:
        dialects = [plain]
    header, rows, bad = None, [], []
    for dialect in dialects:
        h, r, b = _read_csv(text, lines, dialect)
        if header is None or (h is not None and len(b) < len(bad)):
            header, rows, bad = h, r, b
        if header is not None and not bad:
            break
    if header is None:
        return f"{name}: empty"
    out = [f"{name}: {len(rows) + len(bad)} rows, {len(header)} columns: {', '.join(header[:max_keys])}"]
    if bad:
        out.append("   rows with the wrong number of columns: " + ", ".join(f"L{n}" for n in bad[:10])
                   + (f" (+{len(bad) - 10} more)" if len(bad) > 10 else "")
                   + f"; counts below are of the {len(rows)} rows with {len(header)} columns")
    for key in header[:max_keys]:
        out += _describe_key(key, rows)
    return "\n".join(out)


def shape_file(path: Path, lines: list[str], name: str, kind: str, max_keys: int = 60, question: str = "") -> str:
    if kind == "jsonl":
        return _jsonl(lines, name, max_keys, question)
    if kind == "json":
        return _json(lines, name, max_keys)
    return _csv(lines, name, max_keys, tab=Path(path).suffix.lower() == ".tsv")
