"""The `agenthandoff` command.

    agenthandoff read PATH... [-q QUESTION]  an overview of files, by code (a model only if turned on)
    agenthandoff profile PATH...             summarize log files by counting (no model, no GPU)
    agenthandoff digest PATH... -q QUESTION  answer a question about files with the model
    agenthandoff script NAME [PATH...]       run one of your own scripts from config.json
    agenthandoff config show|get|set|unset|path   see and change settings
    agenthandoff stats                       what ran, and which questions keep reaching the model
    agenthandoff setup                       choose a model server and a model (asks questions)
    agenthandoff check                       check that everything works

`--set KEY=VALUE` (repeatable) changes a setting for one run, e.g. --set profile.rare_count=5.
Exit codes: 0 done; 2 not done, and stderr says why (the agent should do the job itself).
Set AGENTHANDOFF_DEBUG=1 to see full error tracebacks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from . import __version__
from .errors import BadSetting, Unavailable

FALLBACK_HINT = "Do this task yourself; the local model is not available for it."


def use_utf8_stdio():
    """Windows consoles default to a legacy code page; file text can be any language."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _paths_args(p, required=True, help_text="files or folders"):
    p.add_argument("paths", nargs="+" if required else "*", help=help_text)
    p.add_argument("--include", action="append", metavar="PATTERN",
                   help="only read files in folders matching this name pattern, e.g. '*.py' (repeatable)")
    p.add_argument("--root", help="show paths relative to this folder")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                   help="change a setting for this run (repeatable); `agenthandoff config show` lists them")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agenthandoff",
                                     description="Quick overviews of logs, data files and code folders, by plain code.")
    parser.add_argument("--version", action="version", version=f"agenthandoff {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    read = commands.add_parser("read", help="an overview of files, by code (a model only if turned on)")
    _paths_args(read)
    read.add_argument("-q", "--question", default="", help="what you want to know (optional)")
    read.add_argument("--tool", choices=["find", "outline", "profile", "shape", "digest"], help="force one tool")

    profile = commands.add_parser("profile", help="summarize log files by counting line patterns (no model)")
    _paths_args(profile, help_text="log files or folders")

    digest = commands.add_parser("digest", help="answer a question about files using the model")
    _paths_args(digest)
    digest.add_argument("-q", "--question", required=True)
    digest.add_argument("--no-recall-check", action="store_true", help="do not plant the recall-check line")
    digest.add_argument("--json", action="store_true", help="print the full result as JSON")

    script = commands.add_parser("script", help="run one of your own scripts from config.json")
    script.add_argument("name")
    _paths_args(script, required=False)
    script.add_argument("-q", "--question", default="")

    conf = commands.add_parser("config", help="see and change settings: show, get KEY, set KEY VALUE, unset KEY, path")
    conf.add_argument("args", nargs=argparse.REMAINDER)

    commands.add_parser("stats", help="what ran, and which questions keep reaching the model")

    setup = commands.add_parser("setup", help="choose a model server and a model")
    setup.add_argument("--backend", choices=["llamacpp", "ollama", "openai"],
                       help="skip the question: llamacpp (recommended), ollama, or openai (LM Studio)")
    setup.add_argument("--model", help="a .gguf path or catalog name (llamacpp), or a model name")
    setup.add_argument("--url", help="server address, if not the default (for llamacpp: a server you run)")
    setup.add_argument("--llama-server", metavar="PATH", help="path to the llama-server program")
    setup.add_argument("--yes", action="store_true",
                       help="accept every suggested answer, including installs and downloads, without asking")

    commands.add_parser("check", help="check that the server, the model and the GPU are ready")
    return parser


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise BadSetting(f"--set {pair!r}: use KEY=VALUE, e.g. --set profile.rare_count=5")
        out[key.strip()] = value
    return out


def main(argv=None) -> int:
    use_utf8_stdio()
    args = build_parser().parse_args(argv)
    if args.command == "setup":
        from .setup_wizard import run_setup
        return run_setup(args.backend, args.model, args.url, args.llama_server, args.yes)
    if args.command == "check":
        from .check import run_check
        return run_check()
    if args.command == "config":
        from .config_cli import run_config
        return run_config(args.args)
    if args.command == "stats":
        from .stats import run_stats
        return run_stats()
    hint = {"digest": FALLBACK_HINT, "profile": "Read the log yourself.",
            "script": "Do this task yourself."}.get(args.command, "Read the files yourself.")
    try:
        print(_run(args))
    except Unavailable as e:
        print(f"agenthandoff: {e}", file=sys.stderr)
        print(hint, file=sys.stderr)
        return 2
    except Exception as e:  # a bug of ours must still tell the agent to carry on without us
        if os.environ.get("AGENTHANDOFF_DEBUG"):
            traceback.print_exc()
        print(f"agenthandoff: unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        print(hint, file=sys.stderr)
        return 2
    return 0


def _run(args) -> str:
    from . import config
    overrides = parse_overrides(args.overrides)
    if args.command == "digest":
        from .api import run_digest
        settings = config.load(overrides=overrides)
        result = run_digest(args.paths, args.question, include=args.include, root=args.root,
                            recall_check=False if args.no_recall_check else None, settings=settings)
        return json.dumps(result.to_dict(), indent=1) if args.json else result.render()
    settings = config.load(need_model=False, overrides=overrides)
    if args.command == "read":
        from .router import read
        return read(args.paths, args.question, settings, include=args.include, root=args.root, tool=args.tool)
    if args.command == "profile":
        from .profile import limits_from, run_profile
        return run_profile(args.paths, include=args.include, root=args.root,
                           max_bytes=settings.get("tools.max_file_bytes"), **limits_from(settings))
    from .scripts import run_script
    script = next((s for s in settings.scripts if s["name"] == args.name), None)
    if script is None:
        known = ", ".join(s["name"] for s in settings.scripts) or "none yet"
        raise BadSetting(f"no script named {args.name!r} in your config.json (scripts: {known}).")
    return run_script(script, args.paths, args.question, args.root, settings.get("tools.max_output_chars"))


def run_profile_command(args) -> int:
    """Kept for callers of the old name."""
    return main(["profile", *args.paths])


if __name__ == "__main__":
    sys.exit(main())
