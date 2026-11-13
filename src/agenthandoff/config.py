"""Settings.

Every setting is listed once, in FIELDS below, with its type, default and allowed range.
Each is looked up in layers; a later layer wins:
  1. the built-in default
  2. <home>/config.json, the user's own settings (`agenthandoff setup` and
     `agenthandoff config set` write it)
  3. .agenthandoff.json in the project folder or a parent folder. Tuning values only:
     a project file comes with a cloned repo, so it may not name programs, servers,
     providers, scripts or paths (that would let a repo run code or send files away).
  4. an environment variable, e.g. AGENTHANDOFF_ROUTE_POLICY for route.policy
  5. a command-line `--set key=value`

<home> is AGENTHANDOFF_HOME if set, else ~/.agenthandoff. It also holds the lock
file, the logs, the cache and (for llama.cpp) downloaded models.

`providers` and `scripts` come from the user's config.json only. The old flat keys
(backend, model, url, llama_server, llama_args) still work: they describe one provider,
named "default". `agenthandoff config show` lists every setting and where it came from.
"""
from __future__ import annotations

import difflib
import json
import os
import shlex
from dataclasses import dataclass, field as dc_field
from pathlib import Path

from .errors import BadSetting, NotConfigured

BACKENDS = ("llamacpp", "ollama", "openai")      # the types the flat `backend` key accepts

# The model sees one chunk of file text plus our prompt and its own answer. These
# limits keep that inside the context window we ask for.
CONTEXT_TOKENS = 12288    # context window requested from the server
HARD_CHUNK_LIMIT = 8000   # max file text per chunk, in estimated tokens (from the recall test)

DEFAULT_URLS = {"ollama": "http://localhost:11434", "openai": "http://localhost:1234"}
DEFAULT_OLLAMA_MODEL = "gemma4:e4b-it-q4_K_M"
PROJECT_FILE = ".agenthandoff.json"
FLAT_PROVIDER_KEYS = ("backend", "model", "url", "llama_server", "llama_args")


@dataclass(frozen=True)
class Field:
    key: str                  # name in config.json; dots group related settings
    kind: str                 # text, path, args, int, float, bool, choice, names, object, scripts
    default: object
    help: str
    low: float | None = None  # allowed range for int and float
    high: float | None = None
    choices: tuple = ()       # for kind "choice"
    project: bool = False     # may a project file set it? Only harmless tuning values.

    @property
    def env(self) -> str:
        return "AGENTHANDOFF_" + self.key.upper().replace(".", "_")

    @property
    def user_file_only(self) -> bool:
        """Structured settings that only the user's own config.json may hold."""
        return self.kind in ("object", "scripts")


FIELDS = (
    # Which model to use. The flat keys describe one provider, named "default".
    Field("backend", "text", None, "llamacpp, ollama, or openai: the simple way to set one model provider"),
    Field("model", "text", None, "llamacpp: path to a .gguf file. ollama: model name. openai: model id"),
    Field("url", "text", None, "server address. Leave unset for llamacpp so agenthandoff starts the server itself"),
    Field("llama_server", "path", None, "path to the llama-server program (default: found on PATH)"),
    Field("llama_args", "args", [], "extra llama-server flags: a JSON list in config.json, or a string"),
    Field("providers", "object", {}, "named model providers (see README: Providers). config.json only"),
    Field("provider", "names", None, "which provider to use, or a list to try in order (default: the flat keys)"),
    Field("allow_remote", "bool", False, "allow providers that send file text off this machine (cloud CLIs, remote URLs)"),
    Field("scripts", "scripts", [], "your own scripts that answer some questions with no model (see README). config.json only"),
    Field("project_config", "bool", True, f"read {PROJECT_FILE} files in project folders (tuning values only)"),
    # Routing: which tool answers.
    Field("route.policy", "choice", "code_first", "code_first: code tools, then the model if needed. "
          "code_only: never a model. model_only: always the model digest", choices=("code_first", "code_only", "model_only"),
          project=True),
    Field("route.allow_model", "bool", False, "let `read` fall back to the model when no code tool fits (off by default: a small model can misread prose)", project=True),
    # Output size: the direct lever on how much the calling agent reads.
    Field("tools.max_output_chars", "int", 12000, "longest output any tool returns, in characters",
          low=500, high=1_000_000, project=True),
    # Code tools.
    Field("tools.max_file_bytes", "int", 64_000_000, "largest file the code tools read, in bytes "
          "(the model digest reads at most 2,000,000)", low=1000, high=4_000_000_000, project=True),
    Field("profile.rare_count", "int", 3, "a log line kind seen this many times or fewer is listed line by line",
          low=1, high=1000, project=True),
    Field("profile.max_patterns", "int", 40, "line kinds shown per log file", low=1, high=10_000, project=True),
    Field("profile.max_rare_lines", "int", 60, "rare lines shown per log file", low=1, high=100_000, project=True),
    Field("profile.max_line_chars", "int", 200, "long lines are cut to this many characters when shown",
          low=40, high=100_000, project=True),
    Field("outline.max_items", "int", 200, "definitions shown per source file", low=1, high=100_000, project=True),
    Field("find.context_lines", "int", 2, "lines shown before and after each match", low=0, high=50, project=True),
    Field("find.max_matches", "int", 40, "matches shown per search term", low=1, high=100_000, project=True),
    Field("shape.max_keys", "int", 60, "keys or columns shown per JSON or CSV file", low=1, high=100_000, project=True),
    # The model digest.
    Field("chunk_tokens", "int", 6000, f"file text per model call, in estimated tokens (max {HARD_CHUNK_LIMIT}, "
          "and at most half the model's context_tokens)", low=500, high=HARD_CHUNK_LIMIT, project=True),
    Field("context_tokens", "int", CONTEXT_TOKENS, "the model's context window, for the model the flat keys set up "
          "(a provider in providers takes its own)", low=4096, high=1_048_576),
    Field("timeout", "float", 300.0, "seconds to wait for one model call", low=1, high=86_400, project=True),
    Field("digest.recall_check", "bool", True, "plant one recall-check line in each digest"),
    Field("cache.enabled", "bool", True, "reuse a model answer when the same question is asked about unchanged files"),
    Field("cache.max_entries", "int", 500, "model answers kept in the cache", low=1, high=1_000_000),
    # The GPU.
    Field("external_lock", "path", None, "path to another program's JSON lock file (with a \"pid\") to respect"),
    Field("ignore_gpu_busy", "bool", False, "skip the check that refuses when another program is using the GPU"),
)
FIELD = {f.key: f for f in FIELDS}


@dataclass
class Settings:
    home: Path
    backend: str | None          # type of the first provider in the chain (None: no model set up)
    model: str                   # its model, for logs and the GPU lock
    url: str | None
    llama_server: str | None
    llama_args: list[str]
    chunk_tokens: int
    timeout: float
    external_lock: str | None
    ignore_gpu_busy: bool
    sources: dict                # key -> "default", "config.json", "project file", "env" or "--set"
    values: dict = dc_field(default_factory=dict)          # every setting by key
    providers: dict = dc_field(default_factory=dict)       # name -> provider spec
    provider_chain: list = dc_field(default_factory=list)  # names to try, in order
    scripts: list = dc_field(default_factory=list)
    allow_remote: bool = False

    def get(self, key: str):
        if key not in FIELD:
            raise KeyError(key)
        return self.values[key] if key in self.values else FIELD[key].default

    def non_default(self) -> dict:
        """Settings not at their default, for run logs (providers and scripts left out)."""
        return {k: v for k, v in self.values.items()
                if not FIELD[k].user_file_only and v != FIELD[k].default}


OLD_HOME = "~/.local-offload"     # this tool's folder before the rename to agenthandoff


def home_dir() -> Path:
    if os.environ.get("AGENTHANDOFF_HOME"):
        return Path(os.environ["AGENTHANDOFF_HOME"]).expanduser()
    home = Path("~/.agenthandoff").expanduser()
    move_old_home(Path(OLD_HOME).expanduser(), home)
    return home


def move_old_home(old: Path, new: Path) -> bool:
    """Move the folder from before the rename, once, and fix paths in its config.json.

    Only when the new folder does not exist yet, so it never overwrites anything. If the move
    fails, it says how to do it by hand and carries on with the new folder.
    """
    if new.exists() or not old.is_dir():
        return False
    import sys
    try:
        old.rename(new)
    except OSError as e:
        print(f"agenthandoff: could not move {old} to {new} ({e}). Move it by hand to keep your "
              "settings and models.", file=sys.stderr)
        return False
    cfg = new / "config.json"
    if cfg.is_file():
        text = cfg.read_text(encoding="utf-8")
        for a, b in ((str(old), str(new)), (old.as_posix(), new.as_posix()),
                     (json.dumps(str(old))[1:-1], json.dumps(str(new))[1:-1])):
            text = text.replace(a, b)
        cfg.write_text(text, encoding="utf-8")
    print(f"agenthandoff: moved your settings from {old} to {new}.", file=sys.stderr)
    return True


def config_path() -> Path:
    return home_dir() / "config.json"


def _read_json_object(path: Path, fix: str) -> dict:
    try:
        raw = path.read_bytes()
        # Windows PowerShell 5 writes UTF-16 (`>`, Out-File) or UTF-8 with a BOM (Set-Content).
        text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
        data = json.loads(text)
    except FileNotFoundError:
        return {}
    except UnicodeDecodeError as e:
        raise BadSetting(f"cannot read {path}: it is not saved as UTF-8 text ({e.reason} at byte {e.start}). "
                         f"Save it as UTF-8. {fix}") from e
    except (OSError, ValueError) as e:
        raise BadSetting(f"cannot read {path}: {e}. {fix}") from e
    if not isinstance(data, dict):
        raise BadSetting(f"{path} must hold a JSON object. {fix}")
    return data


def read_config_file() -> dict:
    """The user's saved settings, or {} if there are none. A broken file is an error, not
    silently ignored, so the user learns that their settings are not in effect."""
    return _read_json_object(config_path(), "Fix it, or run `agenthandoff setup` to replace it.")


def unknown_key(key: str, where: str) -> BadSetting:
    close = difflib.get_close_matches(key, list(FIELD), n=1)
    hint = f" Did you mean {close[0]!r}?" if close else " `agenthandoff config show` lists every setting."
    return BadSetting(f"{where}: unknown setting {key!r}.{hint}")


def write_config_file(values: dict) -> Path:
    """Write the user's config.json. Every key must be a known setting: an unknown one is
    refused, never silently dropped."""
    for key in values:
        if key not in FIELD:
            raise unknown_key(key, str(config_path()))
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in values.items() if v is not None}
    path.write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")
    return path


def find_project_file(start: Path | None) -> Path | None:
    here = Path(start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        candidate = folder / PROJECT_FILE
        if candidate.is_file():
            return candidate
    return None


def split_args(text: str) -> list[str]:
    """Split a flags string like a shell would. On Windows, backslashes in paths are
    kept (shlex would treat them as escapes)."""
    if os.name == "nt":
        text = text.replace("\\", "\\\\")
    return shlex.split(text)


def convert(f: Field, raw, where: str):
    """Turn a raw value (a string from the environment or the command line, or a JSON
    value) into the field's kind. Raises BadSetting naming `where` if it cannot."""
    if f.kind == "bool":
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off", ""):
            return False
        raise BadSetting(f"{where}: {raw!r} is not true or false")
    if f.kind in ("int", "float"):
        if isinstance(raw, bool):
            raise BadSetting(f"{where}: {raw!r} is not a number")
        try:
            value = int(raw) if f.kind == "int" else float(raw)
        except (TypeError, ValueError):
            raise BadSetting(f"{where}: {raw!r} is not a number") from None
        if (f.low is not None and value < f.low) or (f.high is not None and value > f.high):
            raise BadSetting(f"{where}: must be between {f.low:g} and {f.high:g}, got {raw!r}")
        return value
    if f.kind == "choice":
        text = str(raw).strip().lower()
        if text not in f.choices:
            raise BadSetting(f"{where}: must be one of {', '.join(f.choices)}; got {raw!r}")
        return text
    if f.kind == "args":
        if isinstance(raw, list) and all(isinstance(x, str) for x in raw):
            return raw
        if isinstance(raw, str):
            try:
                return split_args(raw)
            except ValueError as e:
                raise BadSetting(f"{where}: {e}") from None
        raise BadSetting(f"{where}: must be a list of strings or a string of flags")
    if f.kind == "names":
        if isinstance(raw, str):
            names = [n.strip() for n in raw.split(",") if n.strip()]
        elif isinstance(raw, list) and all(isinstance(x, str) and x.strip() for x in raw):
            names = [x.strip() for x in raw]
        else:
            raise BadSetting(f"{where}: must be a provider name or a list of names")
        if not names:
            raise BadSetting(f"{where}: no provider name given")
        return names
    if f.kind == "object":
        if not isinstance(raw, dict):
            raise BadSetting(f"{where}: must be a JSON object")
        return raw
    if f.kind == "scripts":
        if not isinstance(raw, list):
            raise BadSetting(f"{where}: must be a JSON list of scripts")
        return raw
    text = str(raw).strip()
    return str(Path(text).expanduser()) if f.kind == "path" else text


def _layer_values(overrides: dict | None, project_dir, user: dict | None = None) -> tuple[dict, dict]:
    """(values, sources) for every setting, after all layers. `user` stands in for the
    user's config.json (to check values before they are saved)."""
    values = {f.key: f.default for f in FIELDS}
    sources = {f.key: "default" for f in FIELDS}

    def apply(key, raw, where, source):
        if key not in FIELD:
            raise unknown_key(key, where)
        values[key] = convert(FIELD[key], raw, f"{where} key {key!r}" if source != "env" else where)
        sources[key] = source

    user = read_config_file() if user is None else user
    for key, raw in user.items():
        if raw is not None:
            apply(key, raw, str(config_path()), "config.json")

    if values["project_config"]:
        path = find_project_file(project_dir)
        if path:
            project = _read_json_object(path, f"Fix it, or delete it to stop using it.")
            for key, raw in project.items():
                if key not in FIELD:
                    raise unknown_key(key, str(path))
                way_out = (f"Set it in your own {config_path()} instead, or stop reading project files: "
                           "agenthandoff config set project_config false")
                if not FIELD[key].project:
                    raise BadSetting(f"{path}: {key!r} cannot be set in a project file, because a project file "
                                     f"comes with the repo. {way_out}")
                before = values[key]
                apply(key, raw, str(path), "project file")
                why = _riskier(FIELD[key], before, values[key])
                if why:
                    raise BadSetting(f"{path}: {key!r} {why}. A project file comes with the repo, so it may only "
                                     f"make things safer. {way_out}")

    for f in FIELDS:
        raw = os.environ.get(f.env, "")
        if raw == "":
            continue
        if f.user_file_only:
            raise BadSetting(f"{f.env}: {f.key} can only be set in your own config.json ({config_path()}).")
        apply(f.key, raw, f.env, "env")

    for key, raw in (overrides or {}).items():
        if key in FIELD and FIELD[key].user_file_only:
            raw = json.loads(raw) if isinstance(raw, str) else raw
        apply(key, raw, "--set", "--set")
    return values, sources


NOT_LIMITS = {"profile.rare_count"}     # a threshold that reshapes output; its size stays capped


def _riskier(f: Field, before, after) -> str:
    """Why a project file's value is riskier than the user's, or "" when it is not. A project
    file may lower a limit, turn the model off or pick code_only, never the reverse."""
    if f.kind in ("int", "float") and after > before and f.key not in NOT_LIMITS:
        return f"may lower your value ({before}), not raise it to {after}"
    if f.key == "route.policy" and after == "model_only" and before != "model_only":
        return "may not force the model digest (model_only)"
    if f.kind == "bool" and after and not before:
        return "may turn this off, not on"
    return ""


def _flat_provider(values: dict) -> dict | None:
    """The provider the flat keys describe, or None if `backend` is not set."""
    backend = (values["backend"] or "").lower()
    if not backend:
        return None
    if backend not in BACKENDS:
        raise BadSetting(f"backend must be one of {', '.join(BACKENDS)}; got {backend!r}")
    spec = {"type": backend}
    keys = ("model", "url", "llama_server", "llama_args") if backend == "llamacpp" else ("model", "url")
    for key in keys:
        if values[key] not in (None, []):
            spec[key] = values[key]
    if backend in DEFAULT_URLS:
        spec.setdefault("url", DEFAULT_URLS[backend])
    if values.get("context_tokens", CONTEXT_TOKENS) != CONTEXT_TOKENS:
        spec["context_tokens"] = values["context_tokens"]
    return spec


def load(need_model: bool = True, overrides: dict | None = None, project_dir=None,
         user_values: dict | None = None) -> Settings:
    """Read all settings. With need_model (a model job), raises NotConfigured if no model
    provider is set up. Raises BadSetting if any value is unusable. Both say what to do."""
    from . import providers as pv            # providers imports this module
    from .scripts import check_scripts

    values, sources = _layer_values(overrides, project_dir, user_values)

    specs = dict(values["providers"])
    flat = _flat_provider(values)
    if flat is not None:
        specs.setdefault("default", flat)
    for name, spec in specs.items():
        pv.check_spec(name, spec, f"{config_path()} providers.{name}" if name in values["providers"] else "settings")

    chain = list(values["provider"] or (["default"] if flat is not None else []))
    for name in chain:
        if name not in specs:
            known = ", ".join(sorted(specs)) or "none"
            raise BadSetting(f"provider {name!r} is not defined (known providers: {known}). "
                             "Add it under \"providers\" in config.json, or run `agenthandoff setup`.")
    if need_model and not chain:
        raise NotConfigured("no local model is set up yet. Run `agenthandoff setup`.")

    scripts = check_scripts(values["scripts"], str(config_path()))

    first = pv.resolved_spec(specs[chain[0]]) if chain else {}
    if need_model and chain:
        pv.require_model(chain[0], first)
    return Settings(
        home=home_dir(),
        backend=first.get("type"),
        model=first.get("model", ""),
        url=first.get("url"),
        llama_server=first.get("llama_server") or values["llama_server"],
        llama_args=first.get("llama_args") or values["llama_args"],
        chunk_tokens=values["chunk_tokens"],
        timeout=values["timeout"],
        external_lock=values["external_lock"],
        ignore_gpu_busy=values["ignore_gpu_busy"],
        sources=sources,
        values=values,
        providers=specs,
        provider_chain=chain,
        scripts=scripts,
        allow_remote=values["allow_remote"],
    )
