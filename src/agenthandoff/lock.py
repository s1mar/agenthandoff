"""A lock file so two agenthandoff jobs never use the GPU at the same time.

Why: two jobs at once make the model server swap models in and out of memory, and
both become very slow. The lock is cooperative: programs that do not know about it
(a game, a training run) are not stopped. hardware.py checks the GPU for those.

The lock is <home>/gpu.lock.json, holding {"pid", "model", "who", "ts", "lease_s"}.
It is created with O_EXCL, so of two jobs starting together exactly one gets it.
A lock is stale, and is taken over, when its process is gone or its lease ran out
(the lease guards against the pid being reused by another program, which Windows
does quickly). A file we cannot parse (a crash between creating and writing it) is
stale once it is UNREADABLE_GRACE_S old.

`external` may point at another program's JSON lock file with a "pid" field. We
only read it: while a live process holds it, we refuse.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from .errors import LockBusy

DEFAULT_LEASE_S = 3 * 3600   # longer than any digest (even CPU-only); the pid check usually frees it first
UNREADABLE_GRACE_S = 30      # a lock being written right now is unreadable for a moment


def pid_alive(pid) -> bool:
    """True if a process with this pid is running. On any doubt, True, so a flaky
    check can never steal a live lock."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)          # signal 0 only checks; on POSIX it sends nothing
        return True
    except PermissionError:
        return True              # exists, owned by another user
    except OSError:
        return False


def _pid_alive_windows(pid: int) -> bool:
    # Never os.kill(pid, 0) on Windows: there it calls TerminateProcess.
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        SYNCHRONIZE, PROCESS_QUERY_LIMITED_INFORMATION, ERROR_ACCESS_DENIED = 0x00100000, 0x1000, 5
        WAIT_TIMEOUT = 0x102
        handle = kernel32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ctypes.get_last_error() == ERROR_ACCESS_DENIED   # exists but not ours to open
        try:
            # A process handle is "signalled" once the process has ended. Waiting 0 ms
            # tells us which, without the exit-code-259 ambiguity of GetExitCodeProcess.
            return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        # ctypes unavailable or an unexpected API failure: fall back to tasklist.
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True, timeout=15).stdout
            return f'"{pid}"' in out
        except Exception:
            return True


def read_lock(path: Path) -> dict | None:
    """The lock's contents; None if there is no lock file. A file we cannot parse
    comes back as {"unreadable": True}, which callers treat as held."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {"unreadable": True}
    if not isinstance(data, dict):
        return {"unreadable": True}
    return data


def _number(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_stale(lock: dict, path: Path) -> bool:
    if lock.get("unreadable"):
        try:
            return time.time() - path.stat().st_mtime > UNREADABLE_GRACE_S
        except OSError:
            return False
    expired = time.time() - _number(lock.get("ts")) > _number(lock.get("lease_s"), DEFAULT_LEASE_S)
    return expired or not pid_alive(lock.get("pid"))


class GpuLock:
    """Use as `with GpuLock(home, model, who):`. Raises LockBusy if another job holds it."""

    def __init__(self, home: Path, model: str, who: str, external: str | None = None,
                 lease_s: int = DEFAULT_LEASE_S):
        self.path = Path(home) / "gpu.lock.json"
        self.model, self.who, self.lease_s = model, who, lease_s
        self.external = Path(external) if external else None
        self.held = False

    def _check_external(self):
        if self.external is None:
            return
        lock = read_lock(self.external)
        if lock is None:
            return
        if lock.get("unreadable"):
            raise LockBusy(f"the lock file {self.external} cannot be read, so we do not know who holds it")
        pid = lock.get("pid")
        # A parent program (for example an eval script) may take the external lock and
        # then run us, so its pid, or ours, counts as "us" here.
        if pid in (os.getpid(), os.getppid()):
            return
        if pid_alive(pid):
            raise LockBusy(f"the GPU is reserved by '{lock.get('who')}' (pid {pid}) in {self.external}")

    def _try_create(self) -> bool:
        """Create the lock file only if it does not exist. True if we created it."""
        record = json.dumps({"pid": os.getpid(), "model": self.model, "who": self.who,
                             "ts": time.time(), "lease_s": self.lease_s})
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(record)
        return True

    def acquire(self):
        self._check_external()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            if self._try_create():
                self.held = True
                return self
            current = read_lock(self.path)
            if current is None:
                continue            # released between our two steps; try again
            if not _is_stale(current, self.path):
                who = "unknown (unreadable lock file)" if current.get("unreadable") else current.get("who")
                raise LockBusy(f"another agenthandoff job is using the GPU: '{who}' "
                               f"(pid {current.get('pid')}). It frees the lock when it finishes. "
                               f"Lock file: {self.path}")
            # Stale. Remove it only if it is still the same stale lock we looked at.
            if read_lock(self.path) == current:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
        raise LockBusy(f"could not take the lock {self.path}; another job keeps taking it")

    def release(self):
        if not self.held:
            return
        self.held = False
        lock = read_lock(self.path)
        if lock is not None and lock.get("pid") == os.getpid():
            try:
                self.path.unlink()
            except OSError:
                pass  # already gone, or briefly opened by a virus scanner; it is stale now anyway

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
