"""read: one entry point that answers with code where it can and a model only where it must.

Rules, in order (code decides; a model never picks the route):
1. A user script with a `question` pattern that fits the question, and `match` patterns that
   fit every file. (A script with no `question` pattern is only run by name, never picked
   here: a script that matches "*" would otherwise answer every question.)
2. The question names an exact string or identifier: `find` shows the lines that hold it.
   If nothing matches, say so and go on to rule 3.
3. Each kind of file goes to its code tool: source code -> outline, logs -> profile, JSON,
   JSON lines and CSV -> shape.
4. Plain text (prose, notes, unknown formats) -> the model digest, if route.allow_model is on
   and a provider is set up. Otherwise "read it yourself" (exit 2) when nothing else ran.
route.policy changes this: code_only never uses a model; model_only always uses the digest.
`tool` forces one tool. The output starts with "ROUTE: <tool> (<why>)" per part and is cut
at tools.max_output_chars. Each read is logged to <home>/logs/read.jsonl.
"""
from __future__ import annotations

import fnmatch
import json
import re
import sys
import time
from pathlib import Path

from . import chunk as ck
from . import find, outline, shape, sniff
from .errors import NoInput, ToolFailed, Unavailable
from .profile import analyse, limits_from, profile_lines
from .scripts import run_script

TOOLS = ("find", "outline", "profile", "shape", "digest")
KIND_TOOL = {"code": "outline", "log": "profile", "json": "shape", "jsonl": "shape", "csv": "shape"}
DATA_KINDS = {"json", "jsonl", "csv"}
KIND_ORDER = ("code", "log", "jsonl", "json", "csv", "text")
KIND_WORDS = {"code": "source code", "log": "logs", "json": "JSON", "jsonl": "JSON lines", "csv": "CSV",
              "text": "plain text"}


class _File:
    def __init__(self, path: Path, lines: list[ck.Line]):
        self.path, self.lines = path, lines
        self.shown = lines[0].path if lines else str(path)
        self.kind = sniff.kind_of(path, [ln.text for ln in lines])
        self._analysis = None

    def analysis(self):
        """The log analysis, made once however many times the profile is drawn."""
        if self._analysis is None:
            self._analysis = analyse(self.lines)
        return self._analysis


def _read_files(paths, include, root, max_bytes) -> tuple[list[_File], list[str]]:
    files, skipped = ck.collect_files(paths, include, max_bytes)
    out = []
    for f in files:
        lines, bad = ck.read_lines([f], Path(root) if root else None)
        skipped += bad
        if lines:
            out.append(_File(f, lines))
    if not out:
        raise NoInput(f"no readable text in the given paths ({'; '.join(skipped[:5]) or 'the files are empty'}).")
    return out, skipped


def _script_for(files: list[_File], question: str, scripts: list[dict]) -> dict | None:
    for s in scripts:
        if not s.get("question") or not re.search(s["question"], question or "", re.IGNORECASE):
            continue
        if all(any(fnmatch.fnmatch(f.path.name.lower(), m.lower()) for m in s["match"]) for f in files):
            return s
    return None


def _model_ready(settings) -> bool:
    return bool(settings.get("route.allow_model")) and bool(settings.provider_chain or settings.backend)


def _digest(files: list[_File], question: str, settings, include, root) -> str:
    from .api import run_digest
    result = run_digest([str(f.path) for f in files], question or "Summarize what these files contain.",
                        include=include, root=root, settings=settings)
    return result.render()


# A question about problems gets the problem part of a log profile only (on long logs
# the full profile was longer than the reader needed).
PROBLEM_QUESTION = re.compile(r"\b(errors?|warn(?:ing|ings)?|fail\w*|problems?|issues?|crash\w*|exceptions?"
                              r"|fatal|denied|timeouts?|timed out|wrong|broken|bugs?)\b", re.IGNORECASE)


def _tool_on(tool: str, f: _File, settings, problems_only: bool = False, question: str = "") -> str:
    if tool == "outline":
        return outline.outline_file(f.path, [ln.text for ln in f.lines], f.shown, settings.get("outline.max_items"))
    if tool == "profile":
        return profile_lines(f.shown, f.lines, **limits_from(settings), analysis=f.analysis(),
                             problems_only=problems_only)
    return shape.shape_file(f.path, [ln.text for ln in f.lines], f.shown,
                            f.kind if f.kind in ("json", "jsonl", "csv") else "jsonl", settings.get("shape.max_keys"),
                            question=question)


def _code_tool(tool: str, group: list[_File], settings, question: str = "") -> str:
    return "\n\n".join(_tool_on(tool, f, settings, question=question) for f in group)


def _whole(f: _File) -> str:
    """The file with its name once and L-numbered lines (as profiles label them). The name on
    every line made a small file with a long name look bigger than its own profile."""
    return "\n".join([f"FILE: {f.shown} ({len(f.lines)} lines)"] + [f"L{ln.num}: {ln.text}" for ln in f.lines])


def _code_tool_or_whole(tool: str, group: list[_File], settings, problems_only: bool = False,
                        question: str = "") -> tuple[list[str], list[str], list[_File]]:
    """(summaries, whole files, the files summarised): a file whose summary would be no
    shorter than the file itself is shown whole, with line numbers, since the summary
    would save nothing."""
    summaries, whole, summarised = [], [], []
    for f in group:
        out = _tool_on(tool, f, settings, problems_only, question)
        text = _whole(f)
        if len(text) <= len(out):
            whole.append(text)
        else:
            summaries.append(out)
            summarised.append(f)
    return summaries, whole, summarised


def _find(files: list[_File], terms: list[str], settings) -> tuple[str, list[str]]:
    lines = [ln for f in files for ln in f.lines]
    results = find.search(lines, terms)
    found = [r for r in results if r.matches]
    missing = [r.term for r in results if not r.matches]
    text = find.render(found, lines, settings.get("find.context_lines"), settings.get("find.max_matches")) if found else ""
    return text, missing


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    note = f"\n[output cut at {limit} characters; set tools.max_output_chars to see more, or narrow the paths]"
    return text[:limit] + note


def read(paths: list[str], question: str = "", settings=None, include: list[str] | None = None,
         root: str | None = None, tool: str | None = None) -> str:
    """Answer about `paths` with the cheapest tool that fits. Raises Unavailable (exit 2)
    when no tool can give anything."""
    from . import config
    settings = settings or config.load(need_model=False)
    started = time.monotonic()
    if tool is not None and tool not in TOOLS:
        raise Unavailable(f"unknown tool {tool!r}; choose one of {', '.join(TOOLS)}")
    files, skipped = _read_files(paths, include, root, settings.get("tools.max_file_bytes"))
    policy = settings.get("route.policy")
    parts: list[str] = []
    used: list[str] = []
    notes: list[str] = []

    def add(name, why, body):
        parts.append(f"ROUTE: {name} ({why})\n{body}")
        used.append(name.split()[0] if not name.startswith("script") else name)

    if tool == "digest" or (tool is None and policy == "model_only"):
        if policy == "code_only":
            raise Unavailable("route.policy is code_only, so the model digest is off.")
        add("digest", "asked for" if tool else "route.policy is model_only", _digest(files, question, settings,
                                                                                    include, root))
    elif tool in ("outline", "profile", "shape"):
        add(tool, "asked for", _code_tool(tool, files, settings, question))
    elif tool == "find":
        terms = find.terms(question)
        if not terms:
            raise Unavailable("find needs an exact string or identifier in the question, in quotes or backticks.")
        text, missing = _find(files, terms, settings)
        if not text:
            raise ToolFailed("no match for " + ", ".join(f"'{t}'" for t in missing))
        add("find", "asked for", text)
    else:
        _route(files, question, settings, include, root, policy, add, notes)

    if not parts:
        raise Unavailable(" ".join(notes) or "no tool fits these files. Read them yourself.")
    body = "\n\n".join(notes + parts)
    if skipped:
        body += f"\n\nSKIPPED {len(skipped)}: " + "; ".join(skipped[:10]) + (" ..." if len(skipped) > 10 else "")
    out = _cap(body, settings.get("tools.max_output_chars"))
    _log(settings, {"question": question, "paths": paths, "tools": used, "chars": len(out),
                    "kinds": sorted({f.kind for f in files}), "seconds": round(time.monotonic() - started, 3),
                    "settings": settings.non_default()})
    return out


def _route(files, question, settings, include, root, policy, add, notes):
    script = _script_for(files, question, settings.scripts)
    if script:
        try:
            out = run_script(script, [str(f.path) for f in files], question, root,
                             settings.get("tools.max_output_chars"))
            add(f"script {script['name']}", "its question and file patterns fit", out)
            return
        except ToolFailed as e:
            notes.append(f"NOTE: {e}; using the built-in tools instead.")

    # A quoted term or a code name (MAX_RETRIES, session.max_redirects) is what the question is
    # about, so find answers alone. A name that is usually just a word (GitHub, macOS) only adds
    # its hits after the usual tool, so "what does it say about GitHub" still gets an outline.
    terms = find.terms(question)
    quoted = {next(g for g in m.groups() if g) for m in find.QUOTED.finditer(question or "")}
    strong = [t for t in terms if t in quoted or not find.weak(t)]
    # In a data file a name is usually a column or key, and find shows only the header line
    # (e.g. "which id has the highest score?"). The shape answers
    # that; find's hits come after it.
    if files and all(f.kind in DATA_KINDS for f in files):
        strong = [t for t in strong if t in quoted]
    extra = None
    if strong:
        text, missing = _find(files, terms, settings)
        if [t for t in missing if t in strong]:
            notes.append("NOTE: no match for " + ", ".join(f"'{t}'" for t in missing if t in strong)
                         + " (exact or ignoring case).")
        if text:
            add("find", "the question names " + ", ".join(f"'{t}'" for t in terms if t not in missing), text)
            return
    elif terms:
        text, missing = _find(files, terms, settings)
        if text:
            extra = ("the question also names " + ", ".join(f"'{t}'" for t in terms if t not in missing), text)
    _route_by_kind(files, question, settings, include, root, policy, add, notes)
    if extra:
        add("find", *extra)


def _route_by_kind(files, question, settings, include, root, policy, add, notes):

    groups: dict[str, list[_File]] = {}
    for f in files:
        groups.setdefault(f.kind, []).append(f)
    for kind in KIND_ORDER:
        group = groups.get(kind)
        if not group:
            continue
        count = f"{len(group)} file{'s' if len(group) != 1 else ''}"
        if kind in KIND_TOOL:
            tool = KIND_TOOL[kind]
            focus = tool == "profile" and bool(PROBLEM_QUESTION.search(question or ""))
            summaries, whole, summarised = _code_tool_or_whole(tool, group, settings, focus, question)
            note = ""
            if tool == "outline":
                summaries, note = _fit_outlines(summarised, summaries, whole, settings)
            elif tool == "profile":
                summaries, note = _fit_profiles(summarised, summaries, whole, settings, focus)
            if summaries:
                add(tool, f"{_files(len(summaries))} of {KIND_WORDS[kind]}", "\n\n".join(([note] if note else [])
                                                                                       + summaries))
            if whole:
                add("whole file", f"{_files(len(whole))} shorter than its {tool}", "\n\n".join(whole))
        elif policy == "code_first" and _model_ready(settings):
            try:
                add("digest", f"{count} of plain text; no code tool fits", _digest(group, question, settings,
                                                                                   include, root))
            except Unavailable as e:      # keep what the code tools found
                notes.append(f"NOTE: the model could not read {count} of plain text: {e} Read them yourself.")
        else:
            why = ("route.policy is code_only" if policy == "code_only" else
                   "no model is set up" if not (settings.provider_chain or settings.backend) else
                   "route.allow_model is off")
            names = ", ".join(f.shown for f in group[:5]) + (" ..." if len(group) > 5 else "")
            hint = (" To let a local model read plain text next time, the user can run: agenthandoff setup"
                    if why == "no model is set up" else
                    " To let the local model read plain text, the user can run: "
                    "agenthandoff config set route.allow_model true"
                    if why == "route.allow_model is off" else "")
            one = len(group) == 1
            notes.append(f"NOTE: {count} of plain text ({names}) need{'s' if one else ''} reading: no code tool "
                         f"fits and {why}. Read {'it' if one else 'them'} yourself.{hint}")


def _budget(summaries: list[str], whole: list[str], settings) -> int:
    """Characters left for the summaries, or 0 when they already fit."""
    budget = int(settings.get("tools.max_output_chars") * 0.9) - sum(len(w) for w in whole)
    return 0 if sum(len(s) + 2 for s in summaries) <= budget else budget


def _fit_outlines(files: list[_File], summaries: list[str], whole: list[str], settings) -> tuple[list[str], str]:
    """(outlines, note): outlines of many files at the most detail that fits
    tools.max_output_chars, so a question about every module never loses the last modules
    to the output cap. The note says which level was used ("" for full detail)."""
    budget = _budget(summaries, whole, settings)
    if not budget:
        return summaries, ""
    max_items = settings.get("outline.max_items")
    levels = [("names", n, f"names only (up to {n} per file)") for n in (40, 15, 5)] + [("head", 0, "first line only")]
    for detail, names, words in levels:
        out = [outline.outline_file(f.path, [ln.text for ln in f.lines], f.shown, max_items, detail=detail,
                                    max_names=names) for f in files]
        if sum(len(s) + 2 for s in out) <= budget or detail == "head":
            return out, (f"OUTLINE DETAIL: {words}, so all {len(files)} files fit in tools.max_output_chars. "
                         "Ask about fewer files, or one file, for full signatures and docstrings.")
    return summaries, ""


# (line chars, line kinds, rare lines) per file, most detail first. Kinds with a problem word
# are listed first, so they are the last to go.
PROFILE_LEVELS = [(120, 40, 60), (100, 30, 40), (80, 25, 30), (80, 20, 20), (80, 15, 15), (60, 10, 8)]


def _fit_profiles(files: list[_File], summaries: list[str], whole: list[str], settings,
                  problems_only: bool = False) -> tuple[list[str], str]:
    """(profiles, note): log profiles at the most detail that fits tools.max_output_chars,
    so the end of a long profile is not cut off. Never more detail than the profile.*
    settings ask for. The note says which level was used ("" for full detail)."""
    budget = _budget(summaries, whole, settings)
    if not budget:
        return summaries, ""
    limits = limits_from(settings)
    for chars, kinds, rare in PROFILE_LEVELS:
        lim = dict(limits, max_line_chars=min(chars, limits["max_line_chars"]),
                   max_patterns=min(kinds, limits["max_patterns"]), max_rare_lines=min(rare, limits["max_rare_lines"]))
        out = [profile_lines(f.shown, f.lines, **lim, analysis=f.analysis(), problems_only=problems_only)
               for f in files]
        if sum(len(s) + 2 for s in out) <= budget:
            break
    return out, (f"PROFILE DETAIL: lines cut to {lim['max_line_chars']} characters, up to {lim['max_patterns']} "
                 f"line kinds and {lim['max_rare_lines']} rare lines per file, so the profile fits "
                 "tools.max_output_chars. Profile one file, or raise tools.max_output_chars, for more.")


def _files(n: int) -> str:
    return f"{n} file{'s' if n != 1 else ''}"


def _log(settings, record: dict) -> None:
    try:
        folder = Path(settings.home) / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "read.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "kind": "read", **record}) + "\n")
    except OSError as e:
        print(f"agenthandoff: could not write the log: {e}", file=sys.stderr)
