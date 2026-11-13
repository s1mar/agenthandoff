"""`agenthandoff check`: is everything in place?

Prints one line per step: "ok", "--" (skipped or unknown) or "FAIL" with what to do.
The code tools (read, profile, scripts) need no model and are checked first. Then each
provider in the chain is checked in order: its model, the GPU (only for a provider that
uses this machine's GPU) and a test prompt. A provider that fails stops its own checks;
the next one is still checked, since a digest falls back to it.
Exit code 0 when every step passed, 2 otherwise.
"""
from __future__ import annotations

import time

from . import config, hardware
from .errors import Unavailable
from .lock import GpuLock
from .providers import make_backend

TEST_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def _say(status: str, step: str, detail: str = ""):
    print(f"[{status:>4}] {step}{': ' + detail if detail else ''}")


def _kind(caps) -> str:
    if caps is None:
        return ""
    return ("remote, needs allow_remote" if caps.remote else "this machine") + \
        (", uses the GPU" if caps.uses_local_gpu else "") + \
        (", JSON schema" if caps.supports_schema else ", JSON by instruction")


def _check_provider(settings, name: str) -> bool:
    step = f"{name}: settings"
    try:
        backend = make_backend(settings, name)
        caps = getattr(backend, "capabilities", None)
        gpu = caps is None or caps.uses_local_gpu
        _say("ok", step, _kind(caps) or "ok")

        step = f"{name}: model"
        _say("ok", step, backend.describe_model())

        if gpu:
            step = f"{name}: GPU"
            reason = hardware.busy_reason(settings, backend.model_status())
            if reason:
                raise Unavailable(reason)
            gpus = hardware.nvidia_gpus()
            if gpus:
                best = max(gpus, key=lambda g: g.free_mb)
                _say("ok", step, f"{best.name}, {best.free_mb} MB of {best.total_mb} MB free")
            else:
                memory = hardware.memory_for_models()
                _say("--", step, "the busy check needs an NVIDIA GPU, so it is skipped here"
                     + (f" ({memory.description})" if memory else ""))

        step = f"{name}: test prompt"
        started = time.monotonic()
        if gpu:
            with GpuLock(settings.home, settings.model, "agenthandoff check", settings.external_lock):
                reply, start_s = _ask(backend)
        else:
            reply, start_s = _ask(backend)
        total_s = time.monotonic() - started
        if '"ok"' not in reply.text:
            raise Unavailable(f"unexpected reply {reply.text[:80]!r}; the model may not support JSON output")
        _say("ok", step, f"answered in {total_s:.1f} s"
             + (f" (of which {start_s:.1f} s starting the server)" if start_s > 0.5 else ""))
        return True
    except Unavailable as e:
        _say("FAIL", step, str(e))
        return False


def _ask(backend):
    started = time.monotonic()
    with backend:
        start_s = time.monotonic() - started
        return backend.chat("Reply with JSON only.", 'Return {"ok": true}.', TEST_SCHEMA, max_output=20), start_s


def run_check() -> int:
    try:
        settings = config.load(need_model=False)
    except Unavailable as e:
        _say("FAIL", "settings", str(e))
        return 2
    changed = ", ".join(f"{k} from {v}" for k, v in settings.sources.items() if v != "default")
    _say("ok", "settings", f"{config.config_path()}" + (f" ({changed})" if changed else ""))
    scripts = ", ".join(s["name"] for s in settings.scripts)
    _say("ok", "code tools", "find, outline, profile, shape" + (f"; your scripts: {scripts}" if scripts else ""))

    if not settings.provider_chain:
        _say("--", "model", "none set up (optional). handoff_read and handoff_profile still work; plain "
                            "text needs a model. Run `agenthandoff setup` to add one.")
        print("\nReady for the code tools. Next: connect your agent (README, 'Connect your agent').")
        return 0
    results = [_check_provider(settings, name) for name in settings.provider_chain]
    if all(results):
        print("\nReady. Next: connect your agent (README, 'Connect your agent').")
        return 0
    if any(results):
        print("\nAt least one provider works, so a digest can still run (it falls back in order).")
    return 2
