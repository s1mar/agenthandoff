"""handoff_profile: a short summary of a log file, made by code alone. No model, no GPU.

Why: a small model reads a log one chunk at a time, so it cannot tell what is
unusual about the whole log (one host set differently, one tenant failing more).
Counting can. On the eval logs the profile was 0.4k to 4.7k characters and held
the answer, where an agent read 10k to 20k characters to find it.

How:
1. Each line becomes a pattern: the timestamp is dropped; numbers, hex IDs,
   paths, and the values of key=value pairs are masked. "web-4 config max_body_mb=1"
   becomes "web-<n> config max_body_mb=<max_body_mb>".
2. Each pattern seen more than RARE_COUNT times gets one block: how many lines,
   the first and last line number, one example, and the spread of each field:
   - few distinct values: each value with its count, plus "odd one out" when one
     value covers at least 80% of the lines and another appears only once or twice;
   - numbers: min, median and max with their lines, and the first value over ten
     times the early median (the first sign of a change);
   - many distinct values: how many, and the most common.
3. Patterns seen RARE_COUNT times or fewer are listed line by line, as RARE LINES:
   in a log, the line that happens once is often the answer.
Every line number is a real line of the file, so an agent can Read it to check.
"""
from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import chunk as ck
from .errors import NoInput

RARE_COUNT = 3            # a pattern seen this many times or fewer is listed line by line
MAX_PATTERNS = 40         # pattern blocks shown per file, in order of first appearance
MAX_RARE_LINES = 60       # rare lines shown per file
MAX_LINE_CHARS = 200      # long lines are cut when shown
FEW_VALUES = 8            # a field with this many distinct values or fewer lists them all
ODD_SHARE = 0.8           # "odd one out" needs one value on at least this share of lines
JUMP_FACTOR = 10          # "first value over 10x the early median"

TIMESTAMP = re.compile(
    r"^\s*\[?(?:\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"   # ISO
    r"|[A-Z][a-z]{2}\s+\d{1,2}\s\d{2}:\d{2}:\d{2}"                                       # syslog
    r"|\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)\]?,?\s*")                                           # time only
ACCESS_TIME = re.compile(r"\[\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4}\]\s*")   # [11/Nov/2026:10:00:01 +0000]
KEY_VALUE = re.compile(r"\b([A-Za-z_][\w.-]*)=(\"[^\"]*\"|\S+)")
GUID = re.compile(r"\{?\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b\}?")
HEX_CODE = re.compile(r"\b0[xX][0-9a-fA-F]+\b")                       # 0x80070002: kept as a field
# https://host/a?b=1: kept as a field. Closing punctuation at the end belongs to the sentence:
# "(see https://x.com/a)," keeps "https://x.com/a".
URL = re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]*[^\s\"'<>),.;:!?\]}]")
# After a date, time fields may be padded with spaces ("06/07/2026  2:18: 8.000", a Windows log).
PADDED_TIME = r"[ \d]?\d:[ \d]\d:[ \d]\d(?:[.,]\d+)?"
INNER_TIME = re.compile(rf"\d{{4}}[-/]\d{{2}}[-/]\d{{2}}(?:[ T:]+{PADDED_TIME})?"
                        rf"|\b\d{{1,2}}([/.])\d{{1,2}}\1(?:19|20)\d{{2}}\b(?:[ T]+{PADDED_TIME})?"   # 13/10/2026, 17.06.2026
                        r"|\b\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?")           # a date or time inside the line
HEX_ID = re.compile(r"\b(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{6,}\b")
WIN_PATH = re.compile(r"(?:\\\\\?\\)?[A-Za-z]:\\[^\s,;\"'<>|\]]*"     # C:\Windows\..., \\?\C:\...
                      r"|\\\\[\w.$-]+\\[^\s,;\"'<>|\]]*"              # \\server\share\...
                      r"|(?<![\w$-])\b[\w.$-]+(?:\\[\w.$-]+)+\\?")      # relative\with\backslashes
# The lookbehinds start a match only where a word starts: the same matches (checked on 182,385
# real log lines), without retrying from every letter of a long word. A 220 MB log took 204 s.
PATH = re.compile(r"(?<![\w.-])[\w.-]*(?:/|::)[\w./:-]*"
                  r"|\b[\w-]+\.(?:py|tsx?|jsx?|go|rs|java|rb|c|h|cpp|yaml|yml|json|log|dll|exe|sys|mum|cat|inf)\b")
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)*")    # not digits inside a word (amd64); versions as one
# Real case: a model-pull log split into 281 patterns, because of terminal codes, its
# timer ("1m11s") and the fill of its progress bar; a dashboard log split on its bar the same way.
ESCAPES = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?|\x1b[@-Z\\-_]")
# Only a timer of two or more parts ("1m11s", "2h3m5s"): a plain "12ms" or "0s" already masks
# as "<n>ms" and "<n>s". It masks to "<n>" plus its last unit, so "1m11s" and "0s" are one
# pattern, and its value counts in that unit ("1m11s" is 71, as "71s" would be).
DURATION = re.compile(r"(?<![\w.])(?:\d+(?:\.\d+)?[dhm])+\d+(?:\.\d+)?(ms|[smh])(?!\w)")
DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|[dhms])")
SECONDS = {"d": 86400, "h": 3600, "m": 60, "s": 1, "ms": 0.001}


def _in_last_unit(timer: str) -> str:
    parts = DURATION_PART.findall(timer)
    total = sum(float(n) * SECONDS[u] for n, u in parts)
    return f"{total / SECONDS[parts[-1][1]]:g}"
BAR = re.compile(r"[\[▕|][#=█-▎░-▓■□.\s>-]{3,}[\]▏|]")
PROBLEM = re.compile(r"\b(errors?|fail(?:ed|ure|ures|ing|s)?|fatal|exceptions?|traceback|critical|warn(?:ing|ings)?"
                     r"|denied|time[d ]?out|timed out|abort(?:ed)?|panic|unable|cannot|could not)\b", re.IGNORECASE)
PROBLEM_KIND = (("error", "err"), ("fail", "fail"), ("fatal", "fatal"), ("exception", "exc"), ("exception", "trace"),
                ("critical", "crit"), ("warning", "warn"), ("denied", "denied"), ("timeout", "time"),
                ("abort", "abort"), ("panic", "panic"), ("cannot", "unable"), ("cannot", "cannot"),
                ("cannot", "could"))


def clean(text: str) -> str:
    """The line as a terminal shows it: escape codes removed, and only what follows the last
    carriage return (a progress line redrawn in place)."""
    if "\r" in text:
        text = text.rstrip("\r").rsplit("\r", 1)[-1]
    return ESCAPES.sub("", text) if "\x1b" in text else text


def line_pattern(text: str) -> tuple[str, dict[str, str]]:
    """(pattern, fields) for one line. Fields are key=value values by key, paths as
    path1, path2, ... and other numbers as #1, #2, ... in the order they appear."""
    body = ACCESS_TIME.sub("", TIMESTAMP.sub("", clean(text), count=1), count=1)
    fields: dict[str, str] = {}
    found: dict[str, list[str]] = {"url": [], "path": [], "#": [], "hex": []}

    def key_value(m):
        fields[m.group(1)] = m.group(2)
        return f"{m.group(1)}=<{m.group(1)}>"

    def keep(kind, mask):
        def sub(m):
            found[kind].append(m.group(0))
            return mask
        return sub

    body = BAR.sub("[<bar>]", body)
    body = URL.sub(keep("url", "<url>"), body)
    body = KEY_VALUE.sub(key_value, body)
    body = GUID.sub("<guid>", body)
    body = HEX_CODE.sub(keep("hex", "<hex>"), body)
    body = INNER_TIME.sub("<time>", body)
    if "\\" in body:                    # every WIN_PATH form has a backslash
        body = WIN_PATH.sub(keep("path", "<path>"), body)
    body = HEX_ID.sub("<id>", body)
    body = PATH.sub(keep("path", "<path>"), body)
    def timer(m):
        found["#"].append(_in_last_unit(m.group(0)))
        return "<n>" + m.group(1)

    body = DURATION.sub(timer, body)
    body = NUMBER.sub(keep("#", "<n>"), body)
    for kind, name in (("url", "url"), ("path", "path"), ("#", "#"), ("hex", "hex")):
        for i, value in enumerate(found[kind], 1):
            fields.setdefault(f"{name}{i}", value)
    return " ".join(body.split()), fields


# A problem word that reports none of itself: "error = 0", "errors: none", "0 errors",
# "no errors", "without error". Not "error = 0x80004005" and not "Error: no space left".
# The value must end the clause (end of line, punctuation, or the next "key="): "FATAL: 0 bytes
# free" and "error: None of the hosts answered" are real problems.
NONE_AFTER = re.compile(r"(?:\(s\))?\s*[=:]\s*(?:0+|none|false)(?!\w|[.:/]\d)"
                        r"(?=\s*(?:$|[,;.)\]}|]|[A-Za-z_][\w.-]*(?:\(s\))?\s*[=:]))", re.IGNORECASE)
NONE_BEFORE = re.compile(r"\b(?:no|zero|without)\s+$", re.IGNORECASE)
# "0 errors" reports none at the start of a clause or after words like "with" and "found";
# "shard 0 errors: disk full" and "worker 0 fails" name item 0.
ZERO_BEFORE = re.compile(r"(\S*)\s*\b0\s+$")
ZERO_WORDS = {"with", "found", "had", "has", "have", "got", "reported", "returned", "total", "and", "or",
              "are", "were", "is", "was"}


def _zero_reports_none(before: str) -> bool:
    """Whether the "0" that ends `before` counts problems (see ZERO_BEFORE's comment)."""
    m = ZERO_BEFORE.search(before)
    if not m:
        return False
    prev = m.group(1)
    return not prev or not prev[-1].isalnum() or prev.lower() in ZERO_WORDS


def _scan(text: str) -> tuple[set[str], str]:
    """(kinds of problem word in the line, the first problem word that reports none, as
    text like "error = 0", or "")."""
    kinds, none = set(), ""
    for m in PROBLEM.finditer(text):
        after = NONE_AFTER.match(text, m.end())
        before = NONE_BEFORE.search(text, 0, m.start())
        plural = m.group(1).lower().endswith("s") or text.startswith("(s)", m.end())
        if not before and plural and _zero_reports_none(text[:m.start()]):
            before = re.search(r"\b0\s+$", text[:m.start()])   # "step 0 failed" is step zero; "0 errors" none
        if after or before:
            none = none or (text[m.start():after.end()] if after else text[before.start():m.end()]).strip()
            continue
        word = m.group(1).lower()
        kinds.add(next(kind for kind, start in PROBLEM_KIND if word.startswith(start)))
    return kinds, none


def problem_words(text: str) -> set[str]:
    """The kinds of problem word in a line: error, fail, warning, ... (empty for most lines,
    and for a line that only reports none, like "error = 0")."""
    return _scan(text)[0]


def _number(value: str) -> float | None:
    try:
        return float(value.strip('"').rstrip("ms").rstrip("%"))
    except ValueError:
        return None


def _cut(text: str, limit: int = MAX_LINE_CHARS) -> str:
    text = clean(text)
    return text if len(text) <= limit else text[:limit] + " [cut]"


def _cut_problem(text: str, limit: int = MAX_LINE_CHARS) -> str:
    """A problem line cut so its problem word and what follows it stay in view. Real case:
    a servicing log's date and level columns filled most of an 80-character cut,
    so "Failed to open pack [cut]" lost its reason."""
    text = clean(text)
    m = PROBLEM.search(text)
    if len(text) <= limit or not m or m.end() + limit // 3 <= limit:
        return _cut(text, limit)
    start = max(0, m.start() - limit // 3)
    piece = text[start:start + limit]
    return "[...] " + piece + (" [cut]" if start + limit < len(text) else "")


MAX_VALUE_CHARS = 80    # one value in a list of values; a log once listed a 4.9 KB prompt whole


def _value(value: str) -> str:
    return _cut(" ".join(str(value).split()), MAX_VALUE_CHARS)


@dataclass
class _Group:
    rows: list[tuple[int, dict[str, str], str]] = field(default_factory=list)   # (line number, fields, text)


def _describe_field(key: str, rows: list[tuple[int, dict[str, str], str]],
                    max_line_chars: int = MAX_LINE_CHARS) -> list[str]:
    values = [(num, f[key]) for num, f, _ in rows if key in f]
    counts = Counter(v for _, v in values)
    if len(counts) <= 1:
        return []                       # the same on every line: nothing to say
    if len(counts) <= FEW_VALUES:
        text = f"   {key}: " + ", ".join(f"{_value(v)} x{c}" for v, c in counts.most_common())
        top_share = counts.most_common(1)[0][1] / len(values)
        if top_share >= ODD_SHARE and len(values) >= 5:
            odd = [(num, v) for num, v in values if counts[v] <= 2]
            if odd:
                text += "  | odd one out: " + ", ".join(f"{_value(v)} at L{num}" for num, v in odd[:3])
        return [text]
    numbers = [_number(v) for _, v in values]
    if any(x is None for x in numbers):
        return [f"   {key}: {len(counts)} distinct values, most common "
                + ", ".join(f"{_value(v)} x{c}" for v, c in counts.most_common(3))]
    nums = [num for num, _ in values]
    hi_num, hi = max(zip(nums, numbers), key=lambda t: t[1])
    lo_num, lo = min(zip(nums, numbers), key=lambda t: t[1])
    early = sorted(numbers[:max(5, len(numbers) // 5)])
    base = early[len(early) // 2] or 1
    jump = next(((num, x) for num, x in zip(nums, numbers) if x > JUMP_FACTOR * base and x - base > 50), None)
    text = f"   {key}: min {lo:g} (L{lo_num}), median {statistics.median(numbers):g}, max {hi:g} (L{hi_num})"
    if jump:
        text += f"  | first value over {JUMP_FACTOR}x the early median ({base:g}): {jump[1]:g} at L{jump[0]}"
    raw = {num: t for num, _, t in rows}
    return [text, f"      max line L{hi_num}: {_cut(raw[hi_num], max_line_chars)}",
            f"      min line L{lo_num}: {_cut(raw[lo_num], max_line_chars)}"]


@dataclass
class Analysis:
    """The slow part of a profile (patterns and problem words), kept so the profile can be
    drawn again at less detail without reading every line again."""
    groups: dict[str, _Group]
    words: dict[str, list[int]]
    bad: set[int] = field(default_factory=set)    # lines with a problem word that counts
    none_lines: int = 0                           # lines whose only problem words report none
    none_example: str = ""
    filled: int = 0                               # lines that are not blank

    def format_kinds(self) -> set[str]:
        """Problem words on nearly every line: likely the log's own prefix, not a problem."""
        return {k for k, nums in self.words.items() if len(nums) >= 20 and len(nums) >= FORMAT_SHARE * self.filled}

    def counted_bad(self) -> set[int]:
        """The lines put first: those with a problem word that is not part of the line format."""
        fmt = self.format_kinds()
        if not fmt:
            return self.bad
        return {num for k, nums in self.words.items() if k not in fmt for num in nums}


def analyse(lines: list[ck.Line]) -> Analysis:
    a = Analysis({}, {})
    for line in lines:
        if line.text.strip():
            a.filled += 1
            pattern, fields = line_pattern(line.text)
            a.groups.setdefault(pattern, _Group()).rows.append((line.num, fields, line.text))
        kinds, none = _scan(line.text)
        for kind in kinds:
            a.words.setdefault(kind, []).append(line.num)
        if kinds:
            a.bad.add(line.num)
        elif none:
            a.none_lines += 1
            a.none_example = a.none_example or none
    return a


FORMAT_SHARE = 0.95     # a problem word on this share of lines or more is likely part of the line format


def _ranked_kinds(a: Analysis) -> list[tuple[str, list[int]]]:
    fmt = a.format_kinds()
    return sorted(a.words.items(), key=lambda kv: (kv[0] in fmt, -len(kv[1]), kv[1][0]))


MAX_FIRST_LINES = 12    # kinds listed under FIRST LINE OF EACH KIND


def _first_lines(a: Analysis, lines: list[ck.Line], max_line_chars: int) -> list[str]:
    """Each counted kind's first line, so a kind whose pattern is cut still shows its text
    (a count and a line number alone are not enough)."""
    fmt = a.format_kinds()
    kinds = [(k, nums[0]) for k, nums in _ranked_kinds(a) if k not in fmt][:MAX_FIRST_LINES]
    want = {num for _, num in kinds}
    text = {line.num: line.text for line in lines if line.num in want}
    return ["", "FIRST LINE OF EACH KIND:"] + [f"   {k} L{num}: {_cut_problem(text.get(num, ''), max_line_chars)}"
                                             for k, num in kinds]


LAST_LINES = 5          # the file's last lines, always shown: a log's outcome is usually there


def _last_lines(lines: list[ck.Line], max_line_chars: int) -> list[str]:
    """Real case: a build log whose tests failed inside it ended "Successfully built",
    a model pull ended "success"; both profiles hid that line, and one read as a failed build."""
    tail = [ln for ln in lines if ln.text.strip()][-LAST_LINES:]
    return ["", "LAST LINES:"] + [f"   L{ln.num}: {_cut(_message(ln.text), max_line_chars)}" for ln in tail] if tail else []


LEADING_TAGS = re.compile(r"^(?:\[[^\]]{1,40}\]\s*)+:?\s*")


def _message(text: str) -> str:
    """A line without its leading time and [tags], when that leaves something: at a small cap,
    "[2026-07-24 11:36:47,854] [docker_util.py] [INFO]: Successfully built" was cut before the
    words that matter."""
    rest = LEADING_TAGS.sub("", TIMESTAMP.sub("", clean(text), count=1), count=1)
    return rest if rest.strip() else text


def _problem_summary(a: Analysis) -> str:
    fmt = a.format_kinds()
    ranked = _ranked_kinds(a)
    parts = [f"{k} on {len(n)} line{'' if len(n) == 1 else 's'} (first L{n[0]}" + ("; on nearly every line, so likely part of the line format)"
                                                          if k in fmt else ")") for k, n in ranked]
    text = "PROBLEM WORDS (whole words, any case; failed counts as fail, unable and could not count as cannot): " + (", ".join(parts) or "none counted")
    if a.none_lines:
        text += f". {a.none_lines} lines that report none, like '{_cut(a.none_example, 40)}', are not counted"
    return text


def profile_lines(path: str, lines: list[ck.Line], rare_count: int = RARE_COUNT, max_patterns: int = MAX_PATTERNS,
                  max_rare_lines: int = MAX_RARE_LINES, max_line_chars: int = MAX_LINE_CHARS,
                  analysis: Analysis | None = None, problems_only: bool = False) -> str:
    """The profile of one file's lines. The limits come from the profile.* settings.
    Pass `analysis` (from analyse(lines)) to skip reading the lines again.
    `problems_only` (the question is about errors or problems) leaves out the lines
    with no problem word and says how many; a log with no problem lines is shown whole."""
    analysis = analysis or analyse(lines)
    groups, bad = analysis.groups, analysis.counted_bad()
    out = [f"LOG PROFILE: {path} - {len(lines)} lines, {len(groups)} line patterns"]
    if analysis.words or analysis.none_lines:
        out.append(_problem_summary(analysis))
    # The end of the file next, before anything long: an output cap cuts from the end, and at
    # a 4,000-character cap a build's "Successfully built" was lost.
    out += _last_lines(lines, max_line_chars)
    if problems_only and bad:
        out += _first_lines(analysis, lines, max_line_chars)

    # Lines with a problem word come first, so the most useful part survives any output cap.
    # Within each part, patterns keep the order they first appear in (dicts keep it).
    def is_bad(rows):
        return any(num in bad for num, _, _ in rows)

    common = [(p, g.rows) for p, g in groups.items() if len(g.rows) > rare_count]
    common.sort(key=lambda pr: not is_bad(pr[1]))
    rare = sorted(row for g in groups.values() if len(g.rows) <= rare_count for row in g.rows)
    rare_bad = [row for row in rare if row[0] in bad]
    rare_rest = [row for row in rare if row[0] not in bad]
    shown = common[:max_patterns]
    bad_blocks = [pr for pr in shown if is_bad(pr[1])]
    for pattern, rows in bad_blocks:
        out += _block(pattern, rows, max_line_chars)
    room = max_rare_lines
    if rare_bad:
        out.append(f"\nRARE LINES WITH PROBLEM WORDS (patterns seen {rare_count} times or fewer), {len(rare_bad)}:")
        out += [f"   L{num}: {_cut_problem(text, max_line_chars)}" for num, _, text in rare_bad[:room]]
        if len(rare_bad) > room:
            out.append(f"   ({len(rare_bad) - room} more)")
        room = max(0, room - len(rare_bad))
    if problems_only and bad:
        cut = sum(is_bad(rows) for _, rows in common[max_patterns:])
        if cut:
            out.append(f"\n({cut} more problem pattern{'s' if cut != 1 else ''} not shown)")
        other = sum(not is_bad(rows) for _, rows in common)
        out.append(f"\n({other} other line pattern{'s' if other != 1 else ''} and {len(rare_rest)} other rare "
                   f"line{'s' if len(rare_rest) != 1 else ''} not shown: the question is about problems. "
                   "Ask without naming problems for the whole profile.)")
        return "\n".join(out)
    for pattern, rows in shown[len(bad_blocks):]:
        out += _block(pattern, rows, max_line_chars)
    if len(common) > max_patterns:
        out.append(f"\n({len(common) - max_patterns} more patterns not shown)")
    if rare_rest:
        out.append(f"\nRARE LINES (patterns seen {rare_count} times or fewer), {len(rare_rest)}:")
        out += [f"   L{num}: {_cut(text, max_line_chars)}" for num, _, text in rare_rest[:room]]
        if len(rare_rest) > room:
            out.append(f"   ({len(rare_rest) - room} more)")
    return "\n".join(out)


def _block(pattern: str, rows, max_line_chars: int) -> list[str]:
    cut = _cut_problem if problem_words(rows[0][2]) else _cut
    out = ["", f"[{len(rows)}x, lines {rows[0][0]}-{rows[-1][0]}] {_cut(pattern, max_line_chars)}",
           f"   e.g. L{rows[0][0]}: {cut(rows[0][2], max_line_chars)}"]
    if len(rows) > 1 and clean(rows[-1][2]) != clean(rows[0][2]):
        # Real case: a dashboard log redrawn every 10 s showed its first frame only,
        # so the final "900/900 100%" was never seen.
        out.append(f"   last L{rows[-1][0]}: {_cut(rows[-1][2], max_line_chars)}")
    for key in dict.fromkeys(k for _, f, _ in rows for k in f):
        out += _describe_field(key, rows, max_line_chars)
    return out


def limits_from(settings) -> dict:
    """The profile.* settings as keyword arguments for profile_lines()."""
    return {"rare_count": settings.get("profile.rare_count"), "max_patterns": settings.get("profile.max_patterns"),
            "max_rare_lines": settings.get("profile.max_rare_lines"),
            "max_line_chars": settings.get("profile.max_line_chars")}


def run_profile(paths: list[str], include: list[str] | None = None, root: str | None = None,
                max_bytes: int = 64_000_000, **limits) -> str:
    """Profiles of every text file in `paths`, one after another. Raises NoInput if
    nothing is readable."""
    files, skipped = ck.collect_files(paths, include, max_bytes)
    lines, unreadable = ck.read_lines(files, Path(root) if root else None)
    skipped += unreadable
    if not lines:
        raise NoInput(f"no readable text in the given paths ({'; '.join(skipped[:5]) or 'the files are empty'}).")
    by_file: dict[str, list[ck.Line]] = {}
    for line in lines:
        by_file.setdefault(line.path, []).append(line)
    parts = [profile_lines(path, file_lines, **limits) for path, file_lines in by_file.items()]
    if skipped:
        parts.append(f"SKIPPED {len(skipped)}: " + "; ".join(skipped[:10]) + (" ..." if len(skipped) > 10 else ""))
    return "\n\n".join(parts)
