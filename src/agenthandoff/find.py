"""find: when a question names an exact thing, show the lines that hold it. No model.

Terms come from the question by rules only:
- text in double quotes or backticks ("max retries", `LIMIT`), or single quotes with no spaces;
- identifiers: snake_case or UPPER_CASE with an underscore, dotted names (session.max_redirects),
  camelCase and PascalCase with two or more capitals (getUserName, CartStore), --flags.
Plain words are never terms: "why did the build fail" has none, and goes to another tool.

Each term is searched as exact text; with no exact match, ignoring case. Matching lines are
shown like grep: "path:12: text" for a match, "path-11- text" for context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import chunk as ck

QUOTED = re.compile(r'"([^"\n]{2,80})"|`([^`\n]{2,80})`|(?<![\w])\'([^\'\s]{2,80})\'(?![\w])')
WORDY = re.compile(r"--[A-Za-z][\w-]+|[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*")


def _is_identifier(token: str) -> bool:
    if token.startswith("--"):
        return True
    if "." in token:
        parts = token.split(".")
        return all(len(p) >= 2 for p in parts) and not token.lower() in ("e.g", "i.e", "etc")
    if "_" in token.strip("_") and len(token) >= 3:
        return True
    if re.search(r"[a-z][A-Z]", token):                         # camelCase
        return True
    return len(re.findall(r"[A-Z][a-z]+", token)) >= 2 and token[0].isupper()   # PascalCase, two humps


FILE_ENDINGS = {"md", "txt", "log", "json", "jsonl", "csv", "yaml", "yml", "toml", "ini", "cfg", "js", "ts",
                "py", "html", "xml", "sh", "ps1", "net", "io", "com", "org"}


def weak(term: str) -> bool:
    """True for a name that is usually just a word in a question (GitHub, PowerShell, macOS,
    README.md, node.js), not a name in code. Such a term adds find hits after the usual tool
    instead of replacing it. Terms with an underscore, --flags and lowerCamelCase stay strong."""
    if "_" in term or term.startswith("--"):
        return False
    if "." in term:
        return term.rsplit(".", 1)[1].lower() in FILE_ENDINGS
    if re.fullmatch(r"[a-z]{1,3}[A-Z][A-Za-z]*", term) and not re.search(r"[A-Z][a-z]+[A-Z]", term):
        return True                                             # macOS, iOS, iPhone, eBay
    return term[0].isupper()                                    # GitHub, PowerShell, JavaScript


def terms(question: str) -> list[str]:
    """The exact things the question names, in the order they appear."""
    found: list[tuple[int, str]] = []
    for m in QUOTED.finditer(question or ""):
        found.append((m.start(), next(g for g in m.groups() if g)))
    quoted_spans = [m.span() for m in QUOTED.finditer(question or "")]
    for m in WORDY.finditer(question or ""):
        if any(a <= m.start() < b for a, b in quoted_spans):
            continue
        token = m.group(0).rstrip(".")
        if _is_identifier(token):
            found.append((m.start(), token))
    out: list[str] = []
    for _, term in sorted(found):
        if term not in out:
            out.append(term)
    return out


@dataclass
class TermResult:
    term: str
    matches: list[ck.Line] = field(default_factory=list)
    exact: bool = True
    files: int = 0


def search(lines: list[ck.Line], term_list: list[str]) -> list[TermResult]:
    results = []
    for term in term_list:
        r = TermResult(term, [ln for ln in lines if term in ln.text])
        if not r.matches:
            low = term.lower()
            r = TermResult(term, [ln for ln in lines if low in ln.text.lower()], exact=False)
        r.files = len({ln.path for ln in r.matches})
        results.append(r)
    return results


def render(results: list[TermResult], lines: list[ck.Line], context: int, max_matches: int) -> str:
    by_file: dict[str, list[ck.Line]] = {}
    for ln in lines:
        by_file.setdefault(ln.path, []).append(ln)
    out = []
    for r in results:
        n = len(r.matches)
        if not n:
            out.append(f"FIND '{r.term}': no match")
            continue
        plural = lambda k, word: f"{k} {word}{'' if k == 1 else 'es' if word == 'match' else 's'}"  # noqa: E731
        how = "" if r.exact else "no exact match; ignoring case: "
        out.append(f"FIND '{r.term}': {how}{plural(n, 'match')} in {plural(r.files, 'file')}")
        shown = r.matches[:max_matches]
        hit = {(ln.path, ln.num) for ln in shown}
        for path in dict.fromkeys(ln.path for ln in shown):
            file_lines = by_file[path]
            index = {ln.num: i for i, ln in enumerate(file_lines)}
            wanted = sorted({j for ln in shown if ln.path == path
                             for j in range(max(0, index[ln.num] - context),
                                            min(len(file_lines), index[ln.num] + context + 1))})
            previous = None
            for j in wanted:
                if previous is not None and j != previous + 1:
                    out.append("--")
                ln = file_lines[j]
                mark = ":" if (ln.path, ln.num) in hit else "-"
                out.append(f"{ln.path}{mark}{ln.num}{mark} {ln.text}")
                previous = j
        if n > len(shown):
            out.append(f"({n - len(shown)} more not shown; use a longer term or set find.max_matches)")
    return "\n".join(out)


def run_find(paths: list[str], term_list: list[str], context: int = 2, max_matches: int = 40,
             root: str | None = None, include: list[str] | None = None) -> str:
    from pathlib import Path
    from .errors import NoInput
    files, skipped = ck.collect_files(paths, include)
    lines, unreadable = ck.read_lines(files, Path(root) if root else None)
    if not lines:
        raise NoInput(f"no readable text in the given paths ({'; '.join((skipped + unreadable)[:5]) or 'empty'}).")
    return render(search(lines, term_list), lines, context, max_matches)
