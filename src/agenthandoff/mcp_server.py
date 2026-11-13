"""MCP server over stdio: one JSON-RPC 2.0 message per line. Standard library only.

Tools:
- handoff_read: answers with code where it can (find, outline, profile, shape, your scripts)
  and the model only if the user turned it on. A one-call overview, not a token saver.
- handoff_profile: the log profile alone (no model, no GPU).
- handoff_digest: a model answers a question about files. Listed only when a model is set up.
- script_<name>: each of your own scripts from config.json.
An MCP client (Claude Code, Codex, Cursor, ...) starts this program and talks to it over
stdin/stdout, for example:
    claude mcp add agenthandoff -- agenthandoff-mcp

A digest can take minutes, so each tools/call runs on its own thread and the main
loop keeps answering (ping, a second call). A second digest while one runs gets
"unavailable" from the GPU lock, which tells the agent to do that one itself.

When the client closes stdin, the server exits within a few seconds, even mid-digest: nobody
is left to read the answer, and exiting closes the pipe that keeps a managed
llama-server alive (llama_server.py), so the GPU is freed straight away.
Cancel notifications are not supported yet; a cancelled digest runs to its end.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import traceback

from . import __version__, config
from .api import run_digest
from .cli import use_utf8_stdio
from .errors import Unavailable
from .profile import limits_from, run_profile
from .router import TOOLS as ROUTER_TOOLS, read
from .scripts import run_script

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")   # newest first
SHUTDOWN_GRACE_S = 5

_PATHS = {"type": "array", "items": {"type": "string"}, "description": "Absolute file or folder paths."}
_INCLUDE = {"type": "array", "items": {"type": "string"},
            "description": "Optional file name patterns for folders, e.g. [\"*.py\"]."}
_ROOT = {"type": "string", "description": "Optional folder to show paths relative to."}

READ_TOOL = {
    "name": "handoff_read",
    "description": (
        "A one-call overview of files or folders, in a fixed format with real line numbers. For one specific "
        "question, your own search and partial reads are usually cheaper. Code does the work, with no model: a named identifier or quoted string gets "
        "the matching lines; source code gets an outline (definitions, docstrings, constants); logs get a "
        "profile (each kind of line with counts, odd values, first big jump, rare lines in full); JSON and "
        "CSV get their shape. Plain text goes to the local model only if the user turned that on. The first line says "
        "which tool ran. Every line number is real: Read the lines that matter before acting on them."),
    "inputSchema": {"type": "object", "properties": {
        "paths": _PATHS,
        "question": {"type": "string", "description": "Optional: what you want to know. Put exact names in "
                                                      "backticks to get their lines."},
        "tool": {"type": "string", "enum": list(ROUTER_TOOLS), "description": "Optional: force one tool."},
        "include": _INCLUDE, "root": _ROOT},
        "required": ["paths"]},
}

TOOL = {
    "name": "handoff_digest",
    "description": (
        "Answer a question about files or folders using the local LLM on this machine. Slower than "
        "handoff_read and can miss things; use it for questions about many files of code or prose that "
        "handoff_read's outline does not answer. Returns a short answer plus claims, each with a file:line "
        "pointer whose quote was found in the file by code. If RECALL CHECK says FAILED, the digest may be "
        "incomplete. Read the pointed lines before acting on a claim that matters."),
    "inputSchema": {"type": "object", "properties": {
        "paths": _PATHS,
        "question": {"type": "string", "description": "What you want to know from these files."},
        "include": _INCLUDE, "root": _ROOT},
        "required": ["paths", "question"]},
}

PROFILE_TOOL = {
    "name": "handoff_profile",
    "description": (
        "Summarize log files by counting, with no model, so you do not read them into your own context. "
        "Each kind of line is shown once with its count, first and last line, and the spread of its values "
        "(with any odd one out and the first big jump in a number); lines that happen only a few times are "
        "listed in full. Line numbers are real: Read the lines that matter before acting on them."),
    "inputSchema": {"type": "object", "properties": {"paths": _PATHS, "include": _INCLUDE, "root": _ROOT},
                    "required": ["paths"]},
}
BUILT_IN = {t["name"]: t for t in (READ_TOOL, PROFILE_TOOL, TOOL)}
FALLBACK = {"handoff_digest": "Do this task yourself.", "handoff_read": "Read the files yourself.",
            "handoff_profile": "Read the log yourself."}

_write_lock = threading.Lock()   # replies come from several threads; one line must never mix with another


def send(msg_id, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    with _write_lock:
        try:
            sys.stdout.write(json.dumps(msg) + "\n")
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            pass  # the client has gone; nobody is left to read the reply


def _text_result(text: str, is_error: bool) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _settings():
    """Settings for tools that need no model (None if they cannot be read)."""
    try:
        return config.load(need_model=False)
    except Unavailable:
        return None


def _script_tool(script: dict) -> dict:
    return {"name": f"script_{script['name']}",
            "description": f"Your own script: {script['description']}. Runs `{' '.join(script['argv'])}`, "
                           "no model.",
            "inputSchema": {"type": "object", "properties": {
                "paths": {**_PATHS, "description": "Files or folders to pass to the script (may be empty)."},
                "question": {"type": "string"}, "root": _ROOT}}}


def list_tools() -> list[dict]:
    settings = _settings()
    tools = [READ_TOOL, PROFILE_TOOL]
    if settings is not None and (settings.provider_chain or settings.backend):
        tools.append(TOOL)
    if settings is not None:
        tools += [_script_tool(s) for s in settings.scripts]
    return tools


def _find_script(name: str) -> dict | None:
    if not name.startswith("script_"):
        return None
    settings = _settings()
    if settings is None:
        return None
    return next((s for s in settings.scripts if f"script_{s['name']}" == name), None)


def _check_arguments(args, needs_question: bool = True, needs_paths: bool = True) -> str | None:
    """A message naming the bad argument, or None if the arguments are usable."""
    if not isinstance(args, dict):
        return "arguments must be an object"
    paths = args.get("paths")
    if isinstance(paths, str):
        args["paths"] = paths = [paths]
    if paths is None and not needs_paths:
        args["paths"] = paths = []
    if not isinstance(paths, list) or (needs_paths and not paths) or not all(isinstance(p, str) for p in paths):
        return "'paths' must be a non-empty list of file or folder paths"
    if needs_question and (not isinstance(args.get("question"), str) or not args["question"].strip()):
        return "'question' must be a non-empty string"
    if args.get("question") is not None and not isinstance(args["question"], str):
        return "'question' must be a string"
    include = args.get("include")
    if include is not None and not (isinstance(include, list) and all(isinstance(p, str) for p in include)):
        return "'include' must be a list of strings"
    if args.get("root") is not None and not isinstance(args["root"], str):
        return "'root' must be a string"
    if args.get("tool") is not None and args["tool"] not in ROUTER_TOOLS:
        return f"'tool' must be one of {', '.join(ROUTER_TOOLS)}"
    return None


def _call(msg_id, name: str, args: dict, script: dict | None):
    fallback = FALLBACK.get(name, "Do this task yourself.")
    try:
        if name == "handoff_digest":
            text = run_digest(args["paths"], args["question"], args.get("include"), args.get("root")).render()
        elif name == "handoff_read":
            text = read(args["paths"], args.get("question") or "", include=args.get("include"),
                        root=args.get("root"), tool=args.get("tool"))
        elif name == "handoff_profile":
            settings = _settings()
            limits = ({**limits_from(settings), "max_bytes": settings.get("tools.max_file_bytes")}
                      if settings is not None else {})
            text = run_profile(args["paths"], args.get("include"), args.get("root"), **limits)
        else:
            settings = _settings()
            text = run_script(script, args["paths"], args.get("question") or "", args.get("root"),
                              settings.get("tools.max_output_chars") if settings else 12000)
        send(msg_id, _text_result(text, False))
    except Unavailable as e:
        label = "handoff_digest unavailable" if name == "handoff_digest" else name
        send(msg_id, _text_result(f"{label}: {e} {fallback}", True))
    except Exception as e:  # a bug of ours: report it and keep the server alive
        traceback.print_exc(file=sys.stderr)
        send(msg_id, _text_result(f"{name} failed: {type(e).__name__}: {e}. {fallback}", True))


def handle(msg: dict, workers: list[threading.Thread]):
    """Answer one message. Notifications (no id) get no reply."""
    method, msg_id = msg.get("method"), msg.get("id")
    if msg_id is None:
        return
    params = msg.get("params") if isinstance(msg.get("params"), dict) else {}
    if method == "initialize":
        asked = params.get("protocolVersion")
        version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        send(msg_id, {"protocolVersion": version, "capabilities": {"tools": {}},
                      "serverInfo": {"name": "agenthandoff", "version": __version__}})
    elif method == "ping":
        send(msg_id, {})
    elif method == "tools/list":
        send(msg_id, {"tools": list_tools()})
    elif method == "tools/call":
        name = params.get("name")
        script = _find_script(name) if isinstance(name, str) else None
        if name not in BUILT_IN and script is None:
            send(msg_id, error={"code": -32602, "message": f"unknown tool {name!r}"})
            return
        args = params.get("arguments") or {}
        problem = _check_arguments(args, needs_question=name == TOOL["name"], needs_paths=script is None)
        if problem:
            send(msg_id, _text_result(f"{name}: bad arguments: {problem}.", True))
            return
        worker = threading.Thread(target=_call, args=(msg_id, name, args, script), daemon=True)
        worker.start()
        workers[:] = [w for w in workers if w.is_alive()] + [worker]
    else:
        send(msg_id, error={"code": -32601, "message": f"method not found: {method}"})


def main() -> int:
    use_utf8_stdio()
    workers: list[threading.Thread] = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            send(None, error={"code": -32700, "message": "parse error"})
            continue
        if not isinstance(msg, dict):
            send(None, error={"code": -32600, "message": "each message must be a JSON object"})
            continue
        try:
            handle(msg, workers)
        except Exception as e:  # never die mid-session
            traceback.print_exc(file=sys.stderr)
            if msg.get("id") is not None:
                send(msg["id"], error={"code": -32603, "message": f"internal error: {e}"})
    # stdin closed. Give calls that are about to answer (quick refusals, code tools) a moment
    # to reply, then exit without waiting for long digests (see the module docstring).
    deadline = time.monotonic() + SHUTDOWN_GRACE_S
    for worker in workers:
        worker.join(max(0.0, deadline - time.monotonic()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
