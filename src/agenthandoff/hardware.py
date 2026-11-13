"""What memory this machine has, and whether the GPU is free right now.

Two jobs:
- memory_for_models(): how much memory a model can use, so setup suggests a model
  that fits (NVIDIA GPU memory, or a Mac's unified memory, or plain RAM).
- busy_reason(): before a job, refuse if another program is using the GPU. Found on
  Real case: a training run held the GPU while our lock showed it free, because the
  lock only covers our own jobs. The model server then ran very slowly and once
  crashed. NVIDIA only (nvidia-smi); on other machines the answer is "unknown" and
  the job goes ahead.
"""
from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass

from .backend import ModelStatus
from .config import Settings

BUSY_UTIL_PERCENT = 50      # above this on two samples a second apart = someone else is working
HEADROOM_MB = 1024          # room for the context cache on top of the model's own size


@dataclass
class GpuStatus:
    index: int
    name: str
    util_percent: int
    free_mb: int
    total_mb: int


def nvidia_gpus() -> list[GpuStatus]:
    """One entry per NVIDIA GPU. Empty if there is no nvidia-smi or it fails."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=index,utilization.gpu,memory.free,memory.total,name",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        gpus = []
        for row in out.strip().splitlines():
            # name is last and split at most 4 times, so a comma in a GPU name is harmless
            index, util, free, total, name = (x.strip() for x in row.split(",", 4))
            gpus.append(GpuStatus(int(index), name, int(util), int(free), int(total)))
        return gpus
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []


def _best_gpu() -> GpuStatus | None:
    """The GPU with the most free memory; that is where the server would put the model."""
    gpus = nvidia_gpus()
    return max(gpus, key=lambda g: g.free_mb) if gpus else None


def system_ram_mb() -> int | None:
    try:
        if os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                            ("avail_ext", ctypes.c_ulonglong)]
            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.total_phys // 2**20
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip()) // 2**20
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2**20
    except (OSError, ValueError, AttributeError):
        return None


@dataclass
class ModelMemory:
    kind: str          # "nvidia", "apple" or "ram"
    usable_mb: int     # memory a model can reasonably use
    description: str   # for people, e.g. "NVIDIA GeForce RTX 4070, 12282 MB"


def memory_for_models() -> ModelMemory | None:
    gpu = _best_gpu()
    if gpu is not None:
        return ModelMemory("nvidia", gpu.total_mb, f"{gpu.name}, {gpu.total_mb} MB GPU memory")
    ram = system_ram_mb()
    if ram is None:
        return None
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # macOS lets the GPU use about two thirds to three quarters of unified memory.
        return ModelMemory("apple", ram * 2 // 3, f"Apple Silicon, {ram} MB unified memory")
    return ModelMemory("ram", ram // 2, f"no NVIDIA GPU found; {ram} MB RAM (runs on the CPU or another GPU)")


def busy_reason(settings: Settings, model: ModelStatus) -> str | None:
    """Why the GPU should not be used now, or None if it looks fine or we cannot tell."""
    if settings.ignore_gpu_busy:
        return None
    first = _best_gpu()
    if first is None:
        return None
    if first.util_percent >= BUSY_UTIL_PERCENT:
        time.sleep(1.0)              # one busy sample can be a blip; look again at the same GPU
        second = next((g for g in nvidia_gpus() if g.index == first.index), first)
        if second.util_percent >= BUSY_UTIL_PERCENT:
            return (f"the GPU is busy with another program ({first.util_percent}% then "
                    f"{second.util_percent}% used, {second.free_mb} MB of {second.total_mb} MB free). "
                    "Try again when it is done, or set AGENTHANDOFF_IGNORE_GPU_BUSY=1 to run anyway.")
    if not model.loaded and model.size_mb is not None and first.free_mb < model.size_mb + HEADROOM_MB:
        return (f"the GPU has {first.free_mb} MB free but the model needs about "
                f"{model.size_mb + HEADROOM_MB} MB, so it would run very slowly or fail. "
                "Close other programs using the GPU, or set AGENTHANDOFF_IGNORE_GPU_BUSY=1.")
    return None
