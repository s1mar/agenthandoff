"""Guard for Windows .cmd and .bat programs.

Windows runs a .cmd or .bat file through cmd.exe, which reads the whole command line again
and treats " & | < > ^ % ! as commands, even inside Python's quoting. Many CLIs installed
by npm are .cmd files. So text from a file, a question or a file name must never reach such
a program's command line: `x" & del ...` would run. Text on stdin is safe.
"""
from __future__ import annotations

import os

SPECIAL = set('"&|<>^%!\r\n')


def is_batch(exe: str) -> bool:
    return os.name == "nt" and exe.lower().endswith((".cmd", ".bat"))


def unsafe_arg(exe: str, args: list[str]) -> str | None:
    """The first argument cmd.exe could run as a command, or None when all are safe."""
    if not is_batch(exe):
        return None
    return next((a for a in args if SPECIAL & set(a)), None)


def refusal(exe: str, what: str) -> str:
    return (f"{os.path.basename(exe)} is a .cmd or .bat file, which Windows runs through cmd.exe, so "
            f"{what} on its command line could run commands ({' '.join(sorted(SPECIAL - {chr(13), chr(10)}))} "
            "are special there).")
