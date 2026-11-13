"""Read files, label every line, pack lines into chunks, and find quotes.

The core rule: the model never gives a line number. It copies text (a "quote");
resolve() finds that text in the chunk the model was shown, and code works out
file:line. This came from a test where a 27b model said line 826 for a
line that was really at 819.

What the model sees for each chunk:
    === FILE: src/app.py ===
    L00001: import os
    L00002: ...
"""
from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
             ".tox", ".mypy_cache", ".pytest_cache", "dist", "build", ".idea", ".vscode"}
MAX_FILE_BYTES = 2_000_000   # the model digest's cap; code tools pass tools.max_file_bytes
MAX_LINE_CHARS = 1500  # longer lines are shown cut; quotes are still matched on the full text
MIN_QUOTE_CHARS = 6    # shorter quotes ("x = 1") match too much to point at anything
MAX_SPAN_LINES = 8     # a quote may run over at most this many (non-blank) lines of a file
LABEL_RE = re.compile(r"^\s*L\d{1,7}:\s?")   # a label the model copied into its quote
ELLIPSIS_RE = re.compile(r"\.\.\.|…")        # "start ... end" quotes skip the middle
# Comment marks at the start of a line. Models drop them when they join a wrapped
# comment into one sentence, so they are ignored on both sides.
COMMENT_RE = re.compile(r"^\s*(#+|//+|/\*+|\*+/?|--|;+|>+)\s?")
EDGE_CHARS = " \t\"'`.,;:"   # models add these around a quote ("...", a full stop)


WORD_RE = re.compile(r"[A-Za-z]+")
DIGIT_RE = re.compile(r"\d")
SYMBOL_RE = re.compile(r"[^\w\s]")


def est_tokens(text: str) -> int:
    """Estimated tokens: one per 5 letters of each word, one per digit, one per symbol
    and one per line. Tokenizers split numbers into single digits, so a log full of
    timestamps and IDs is far more tokens than its length suggests; the old "3
    characters a token" was 1.9x too low on such logs and 1.35x too high on code.
    Checked against Gemma 4's real counts on code, logs and prose (eval files and
    held-out files): from 11% under to 15% over, mostly over, so chunks stay in budget."""
    return (sum((len(w) + 4) // 5 for w in WORD_RE.findall(text)) + len(DIGIT_RE.findall(text))
            + len(SYMBOL_RE.findall(text)) + text.count("\n"))


@dataclass
class Line:
    path: str | None   # None marks the planted recall line (see digest.py), not a real line
    num: int           # 1-based line number in its file
    text: str

    @property
    def is_probe(self) -> bool:
        return self.path is None

    def label(self) -> str:
        """The line as the model sees it: 'L00042: text', long lines cut."""
        shown = self.text if len(self.text) <= MAX_LINE_CHARS else self.text[:MAX_LINE_CHARS] + " [line cut]"
        return f"L{self.num:05d}: {shown}"


def file_header(path: str) -> str:
    return f"=== FILE: {path} ==="


class Hit(NamedTuple):
    """Where a quote was found: lines start..end of one file. path None = the probe."""
    path: str | None
    start: int
    end: int


@dataclass
class Chunk:
    lines: list[Line] = field(default_factory=list)

    def render(self) -> str:
        out, current_file = [], None
        for line in self.lines:
            if not line.is_probe and line.path != current_file:
                out.append(file_header(line.path))
                current_file = line.path
            out.append(line.label())
        return "\n".join(out)


# ---------------------------------------------------------------- reading files

BOMS = ((b"\xef\xbb\xbf", "utf-8-sig"), (b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16"))


def _encoding_of(head: bytes) -> str | None:
    """The text encoding, or None for a binary file. UTF-16 files (Windows PowerShell 5.1
    writes them by default, e.g. `cmd > build.log`) are full of NUL bytes, so the byte
    order mark is checked before the NUL test."""
    for bom, encoding in BOMS:
        if head.startswith(bom):
            return encoding
    if b"\0" not in head:
        return "utf-8"
    pairs = len(head) // 2
    if pairs >= 8:      # UTF-16 with no byte order mark: every other byte is zero
        even = sum(1 for i in range(0, 2 * pairs, 2) if head[i] == 0)
        odd = sum(1 for i in range(1, 2 * pairs, 2) if head[i] == 0)
        if odd > 0.4 * pairs and even < 0.05 * pairs:
            return "utf-16-le"
        if even > 0.4 * pairs and odd < 0.05 * pairs:
            return "utf-16-be"
    # Zero bytes in a text log are common after a crash or power loss (the file was
    # extended but never written). It is text if what is left is text.
    rest = head.replace(b"\0", b"")
    if not rest:
        return None
    control = sum(1 for b in rest if b < 32 and b not in (9, 10, 12, 13, 27))
    broken = rest.decode("utf-8", errors="replace").count("�")
    return "utf-8" if control <= len(rest) * 0.01 and broken <= len(rest) * 0.01 + 4 else None


def _sniff(path: Path) -> str | None:
    with open(path, "rb") as f:
        return _encoding_of(f.read(8192))


def split_lines(text: str) -> list[str]:
    """Split on newlines only. str.splitlines() also splits on form feeds and other
    rare characters, which would shift every line number after them. read_text has
    already turned \\r\\n and \\r into \\n."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()          # a final newline does not start another line
    return lines


def collect_files(paths: list[str], include: list[str] | None = None,
                  max_bytes: int = MAX_FILE_BYTES) -> tuple[list[Path], list[str]]:
    """Expand files and folders into a list of text files.

    Folders are walked recursively, skipping SKIP_DIRS. `include` patterns (like
    "*.py") filter files found in folders; a file named directly is always read.
    Returns (files, skipped), where each skipped entry says why.
    """
    files, skipped, seen = [], [], set()

    def consider(path: Path):
        try:
            key = os.path.normcase(str(path.resolve()))
            if key in seen:
                return          # the same file reached twice (a folder and a file in it)
            seen.add(key)
            if path.stat().st_size > max_bytes:
                why = ("the most the model digest reads" if max_bytes == MAX_FILE_BYTES
                       else "raise tools.max_file_bytes to read it")
                skipped.append(f"{path}: larger than {max_bytes:,} bytes ({why})")
            elif _sniff(path) is None:
                skipped.append(f"{path}: binary")
            else:
                files.append(path)
        except OSError as e:
            skipped.append(f"{path}: {e.strerror or e}")

    for p in paths:
        root = Path(p)
        if root.is_file():
            consider(root)
        elif root.is_dir():
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
                for name in sorted(filenames):
                    if include and not any(fnmatch.fnmatch(name, pat) for pat in include):
                        continue
                    consider(Path(dirpath) / name)
        else:
            skipped.append(f"{p}: not found")
    return files, skipped


def read_lines(files: list[Path], display_root: Path | None = None) -> tuple[list[Line], list[str]]:
    """Read every file into Lines. Paths are shown relative to display_root when the
    file is inside it, always with forward slashes. Returns (lines, skipped)."""
    lines, skipped = [], []
    root = display_root.resolve() if display_root is not None else None
    for f in files:
        shown = str(f)
        if root is not None:
            try:
                shown = str(f.resolve().relative_to(root))
            except ValueError:
                pass            # outside the root: keep the path as given
        shown = shown.replace("\\", "/")
        try:
            encoding = _sniff(f) or "utf-8"
            text = f.read_text(encoding=encoding, errors="replace")
            if not encoding.startswith("utf-16"):
                text = text.replace("\0", "")     # zero bytes are padding, never text
        except OSError as e:    # permission denied, locked, or deleted since collect_files
            skipped.append(f"{f}: {e.strerror or e}")
            continue
        lines.extend(Line(shown, i, t) for i, t in enumerate(split_lines(text), start=1))
    return lines, skipped


# ---------------------------------------------------------------- packing

def line_cost(line: Line) -> int:
    return est_tokens(line.label() + "\n")


def header_cost(path: str) -> int:
    return est_tokens(file_header(path) + "\n")


def total_cost(lines: list[Line]) -> int:
    """Estimated tokens of all lines as rendered, headers included."""
    cost, current_file = 0, None
    for line in lines:
        if line.path != current_file:
            cost += header_cost(line.path)
            current_file = line.path
        cost += line_cost(line)
    return cost


def pack(lines: list[Line], budget_tokens: int) -> list[Chunk]:
    """Split lines into chunks of at most budget_tokens (estimated), never cutting a
    line. A chunk that starts mid-file repeats that file's header."""
    chunks, current, used, current_file = [], Chunk(), 0, None
    for line in lines:
        cost = line_cost(line)
        needs_header = line.path != current_file
        if current.lines and used + cost + (header_cost(line.path) if needs_header else 0) > budget_tokens:
            chunks.append(current)
            current, used, needs_header = Chunk(), 0, True
        if needs_header:
            used += header_cost(line.path)
            current_file = line.path
        current.lines.append(line)
        used += cost
    if current.lines:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------- finding quotes

def normalize(text: str) -> str:
    """Collapse all runs of whitespace to one space, so indentation never matters."""
    return " ".join(text.split())


def squash(text: str) -> str:
    """The form quotes are matched in: comment mark removed, no whitespace at all.
    Models re-space code (`if (a` for `if (\\n a`), so spaces carry no meaning here."""
    return "".join(COMMENT_RE.sub("", text).split())


def _pieces(quote: str) -> list[str]:
    """The parts of a quote that must appear in the file, in order. Lines are joined
    (the match may run over several file lines anyway) and split on "..."."""
    lines = [COMMENT_RE.sub("", LABEL_RE.sub("", raw)) for raw in quote.splitlines()
             if not raw.strip().startswith("=== FILE:")]
    joined = " ".join(lines).strip(EDGE_CHARS)
    return [p for p in (squash(x).strip(EDGE_CHARS) for x in ELLIPSIS_RE.split(joined)) if p]


def _contains_in_order(pieces: list[str], text: str) -> bool:
    pos = 0
    for piece in pieces:
        i = text.find(piece, pos)
        if i < 0:
            return False
        pos = i + len(piece)
    return True


@dataclass
class Resolution:
    ok: bool
    reason: str = ""                                 # why it was dropped, when not ok
    hits: list[Hit] = field(default_factory=list)


def resolve(quote: str, chunk: Chunk) -> Resolution:
    """Find where `quote` appears in `chunk`. Forgiving about form, strict about text.

    Ignored: labels the model copied ("L00012: "), file headers, all whitespace, comment
    marks at line starts, and quote marks or a full stop around the quote. "a ... b"
    matches a, then b. A quote may run over up to MAX_SPAN_LINES consecutive lines of
    one file, however the model broke (or joined) its lines; blank lines and the
    planted probe line never break a match.
    Every match is returned, smallest span first in file order; Claim.pointer() shows
    the first few. Dropped (ok False) when the quote is empty, too short, or not found.
    """
    pieces = _pieces(quote)
    if not pieces:
        return Resolution(False, "empty quote")
    if sum(len(p) for p in pieces) < MIN_QUOTE_CHARS:
        return Resolution(False, "quote too short")

    # The probe is matched on its own: a hit there is how we learn the model saw it.
    probe = [Hit(None, ln.num, ln.num) for ln in chunk.lines
             if ln.is_probe and _contains_in_order(pieces, squash(ln.text))]

    content = [ln for ln in chunk.lines if not ln.is_probe and squash(ln.text)]
    texts = [squash(ln.text) for ln in content]
    spans = []                       # (first, last) index into content, shortest per start
    for i in range(len(content)):
        joined = ""
        for j in range(i, min(i + MAX_SPAN_LINES, len(content))):
            if content[j].path != content[i].path:
                break
            joined += texts[j]
            if _contains_in_order(pieces, joined):
                spans.append((i, j))
                break
    # A span that starts before the text does (it also holds a shorter span) is not
    # where the quote is; keep only spans that contain no other span.
    spans = [(i, j) for i, j in spans if not any(i2 > i and j2 <= j for i2, j2 in spans)]
    hits = probe + [Hit(content[i].path, content[i].num, content[j].num) for i, j in spans]
    if not hits:
        return Resolution(False, "quote not found")
    return Resolution(True, hits=hits)
