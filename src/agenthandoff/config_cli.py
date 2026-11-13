"""`agenthandoff config`: see and change settings without editing JSON by hand.

    agenthandoff config show            every setting, its value, and where it came from
    agenthandoff config get KEY         one value
    agenthandoff config set KEY VALUE   save it in your config.json (checked first)
    agenthandoff config unset KEY       remove it from your config.json (back to the default)
    agenthandoff config path            where your config.json is

VALUE is read like an environment variable (true/false, numbers, comma-separated names);
for providers and scripts give JSON. A value is checked by loading all settings with it
before anything is written, so a bad value never reaches the file.
"""
from __future__ import annotations

import json
import sys

from . import config
from .errors import Unavailable

USAGE = "usage: agenthandoff config show | get KEY | set KEY VALUE | unset KEY | path"


def _show_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return "(not set)" if value is None else str(value)


def _parse(key: str, text: str):
    field = config.FIELD[key]
    if field.kind in ("object", "scripts") or (field.kind in ("args", "names") and text.lstrip().startswith("[")):
        try:
            return json.loads(text)
        except ValueError as e:
            raise config.BadSetting(f"{key}: not valid JSON: {e}") from None
    return config.convert(field, text, key)


def run_config(argv: list[str], project_dir=None) -> int:
    try:
        return _run(argv, project_dir)
    except Unavailable as e:
        print(f"agenthandoff: {e}", file=sys.stderr)
        return 2


def _run(argv, project_dir) -> int:
    if not argv or argv[0] not in ("show", "get", "set", "unset", "path"):
        print(USAGE, file=sys.stderr)
        return 2
    command, rest = argv[0], argv[1:]
    if command == "path":
        print(config.config_path())
        return 0
    if command == "show":
        s = config.load(need_model=False, project_dir=project_dir)
        for f in config.FIELDS:
            print(f"{f.key} = {_show_value(s.get(f.key))}  ({s.sources[f.key]})")
        return 0
    if not rest:
        print(USAGE, file=sys.stderr)
        return 2
    key = rest[0]
    if key not in config.FIELD:
        raise config.unknown_key(key, "config")
    if command == "get":
        print(_show_value(config.load(need_model=False, project_dir=project_dir).get(key)))
        return 0
    saved = config.read_config_file()
    if command == "unset":
        saved.pop(key, None)
    else:
        if len(rest) != 2:
            print(USAGE, file=sys.stderr)
            return 2
        saved[key] = _parse(key, rest[1])
    _check_before_writing(saved, project_dir)
    config.write_config_file(saved)
    if command == "unset":
        print(f"{key} removed from {config.config_path()}; the default applies: "
              f"{_show_value(config.FIELD[key].default)}")
    else:
        print(f"{key} = {_show_value(saved.get(key))}  (saved in {config.config_path()})")
    return 0


def _check_before_writing(values: dict, project_dir) -> None:
    """Load every setting as if `values` were saved; raises BadSetting if they would not
    work. Nothing is written."""
    for key in values:
        if key not in config.FIELD:
            raise config.unknown_key(key, str(config.config_path()))
    config.load(need_model=False, project_dir=project_dir, user_values=values)
