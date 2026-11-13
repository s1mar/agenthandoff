"""outline: what is in a source file, by code. No model.

Python is read with the standard `ast` module: the module docstring's first sentence, the
imports, module constants (UPPER_CASE = value), classes and functions with their arguments
and the first sentence of their docstrings. Other languages are read by patterns that find
definition lines (functions, classes, types), plus the file's leading comment. Every line
number is the real line of the definition.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

CONSTANT = re.compile(r"^[A-Z][A-Z0-9_]*$")
MAX_VALUE_CHARS = 60
MAX_LINE_CHARS = 140

_JS = [r"^\s*(export\s+)?(default\s+)?(async\s+)?function\b", r"^\s*(export\s+)?(default\s+)?(abstract\s+)?class\s+\w",
       r"^\s*(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s*)?(\([^)]*\)|\w+)\s*=>",
       r"^\s*(export\s+)?(interface|type|enum)\s+\w+", r"^\s*(export\s+)?(const|let|var)\s+\w+\s*=\s*function\b"]
_C_LIKE = [r"^\s*(public|private|protected|internal|static|abstract|final|sealed|open|data|partial|export|\s)*"
           r"(class|interface|enum|record|struct|object|trait)\s+\w",
           r"^\s*(public|private|protected|internal|static|override|final|abstract|async|virtual|synchronized)"
           r"(\s+[\w<>\[\],.?]+)+\s*\("]
PATTERNS = {
    "js": _JS,
    "go": [r"^func\b", r"^type\s+\w+"],
    "rust": [r"^\s*(pub(\([^)]*\))?\s+)?(async\s+)?(unsafe\s+)?(fn|struct|enum|trait|impl|mod|type)\b"],
    "jvm": _C_LIKE + [r"^\s*(fun|func)\s+\w"],
    "ruby": [r"^\s*(def|class|module)\s+\S"],
    "c": [r"^\s*(typedef\s+)?(struct|class|enum|union|namespace)\s+\w+",
          r"^(?!\s*(if|for|while|switch|return|else|do|case)\b)[A-Za-z_][\w\s\*&:<>,~]*\b\w+\s*\([^;]*\)\s*(const\s*)?\{?\s*$"],
    "shell": [r"^\s*function\s+[\w-]+", r"^\s*[\w-]+\s*\(\)\s*\{?"],
    "php": [r"^\s*((public|private|protected|static|abstract|final)\s+)*function\s+\w",
            r"^\s*(abstract\s+|final\s+)?(class|interface|trait)\s+\w"],
    "python": [r"^\s*(async\s+)?def\s+\w", r"^\s*class\s+\w"],
}
EXT_LANG = {
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js", ".ts": "js", ".tsx": "js", ".vue": "js", ".svelte": "js",
    ".go": "go", ".rs": "rust", ".java": "jvm", ".kt": "jvm", ".kts": "jvm", ".scala": "jvm", ".cs": "jvm",
    ".swift": "jvm", ".dart": "jvm", ".rb": "ruby", ".c": "c", ".h": "c", ".cc": "c", ".cpp": "c", ".cxx": "c",
    ".hpp": "c", ".hh": "c", ".m": "c", ".mm": "c", ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".ps1": "shell", ".psm1": "shell", ".php": "php", ".py": "python", ".pyi": "python",
}
COMMENT = re.compile(r"^\s*(//+|#+|/\*+|\*+|--|;+)\s?")


def _first_sentence(text: str | None) -> str:
    if not text:
        return ""
    first = text.strip().splitlines()[0].strip()
    m = re.match(r"(.+?[.!?])(\s|$)", first)
    return (m.group(1) if m else first)[:MAX_LINE_CHARS]


def _cut(text: str, limit: int = MAX_LINE_CHARS) -> str:
    text = text.strip().rstrip("{").rstrip()
    return text if len(text) <= limit else text[:limit] + " ..."


def _python(lines: list[str], name: str, max_items: int) -> tuple[str, list[str]] | None:
    try:
        tree = ast.parse("\n".join(lines))
    except (SyntaxError, ValueError):
        return None
    source = "\n".join(lines)
    items: list[str] = []
    imports: list[str] = []

    def visit(body, depth):
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)) and depth == 0:
                if isinstance(node, ast.Import):
                    imports.extend(a.name for a in node.names)
                else:
                    imports.append("." * node.level + (node.module or ""))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and depth == 0:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [t.id for t in targets if isinstance(t, ast.Name) and CONSTANT.match(t.id)]
                if names and node.value is not None:
                    value = ast.get_source_segment(source, node.value) or "..."
                    value = " ".join(value.split())
                    if len(value) > MAX_VALUE_CHARS:
                        value = value[:MAX_VALUE_CHARS] + " ..."
                    items.append(f"L{node.lineno} {' = '.join(names)} = {value}")
            elif isinstance(node, ast.ClassDef):
                bases = ", ".join(ast.unparse(b) for b in node.bases)
                doc = _first_sentence(ast.get_docstring(node))
                items.append(f"L{node.lineno} {'  ' * depth}class {node.name}{f'({bases})' if bases else ''}"
                             + (f": {doc}" if doc else ""))
                visit(node.body, depth + 1)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                doc = _first_sentence(ast.get_docstring(node))
                sig = _cut(f"{node.name}({ast.unparse(node.args)})", MAX_LINE_CHARS)
                items.append(f"L{node.lineno} {'  ' * depth}{prefix} {sig}" + (f": {doc}" if doc else ""))

    visit(tree.body, 0)
    summary = _first_sentence(ast.get_docstring(tree))
    head = [f"{name} ({len(lines)} lines)" + (f": {summary}" if summary else "")]
    if imports:
        head.append("   imports: " + ", ".join(dict.fromkeys(imports)))
    return "\n".join(head), items


def _patterns(lines: list[str], lang: str | None) -> list[str]:
    regexes = [re.compile(p) for p in (PATTERNS.get(lang) if lang else [p for ps in PATTERNS.values() for p in ps])]
    return [f"L{i} {_cut(text)}" for i, text in enumerate(lines, 1)
            if text.strip() and any(r.search(text) for r in regexes)]


def _leading_comment(lines: list[str]) -> str:
    for text in lines[:15]:
        if not text.strip() or text.startswith("#!"):
            continue
        if COMMENT.match(text):
            body = COMMENT.sub("", text).strip().rstrip("*/").strip()
            if body:
                return _first_sentence(body)
            continue
        break
    return ""


ITEM_NAME = re.compile(r"^L(\d+) \s*(?:(?:export |pub |async |static |public |private )*"
                       r"(?:def|class|function|func|fn|var|const|let|interface|type|struct|enum|impl|trait|module)\s+)?"
                       r"([\w.$]+)")
MAX_NAMES = 40          # names shown per file at the "names" level


def outline_file(path: Path, lines: list[str], name: str, max_items: int = 200, detail: str = "full",
                 max_names: int = MAX_NAMES) -> str:
    """The outline of one source file, `name` being how its path is shown.
    detail: "full" (every definition with its signature and doc), "names" (one line of
    definition names with their lines) or "head" (the file's first line only). Many
    files use a lower level so that every file still fits in the output."""
    full = _outline(path, lines, name, max_items)
    if detail == "full":
        return full
    first, *rest = full.split("\n")
    items = [m for m in (ITEM_NAME.match(r.strip()) for r in rest) if m]
    if detail == "head":
        return f"{first} ({len(items)} definitions)"
    shown = ", ".join(f"{m.group(2)} L{m.group(1)}" for m in items[:max_names])
    more = f", +{len(items) - max_names} more" if len(items) > max_names else ""
    return first + (f"\n   {shown}{more}" if items else "")


def _outline(path: Path, lines: list[str], name: str, max_items: int) -> str:
    lang = EXT_LANG.get(Path(path).suffix.lower())
    note = ""
    parsed = _python(lines, name, max_items) if lang == "python" else None
    if parsed:
        head, items = parsed
    else:
        if lang == "python":
            note = "   (syntax error; outline by patterns)"
        summary = _leading_comment(lines)
        head = f"{name} ({len(lines)} lines)" + (f": {summary}" if summary else "")
        items = _patterns(lines, lang)
    out = [head] + ([note] if note else []) + [f"   {item}" for item in items[:max_items]]
    if len(items) > max_items:
        out.append(f"   ({len(items) - max_items} more definitions not shown)")
    if not items:
        out.append("   (no definitions found)")
    return "\n".join(out)
