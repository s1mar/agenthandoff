"""Models that setup can download for the llama.cpp backend, and the download itself.

Each entry is an official ggml-org GGUF file on Hugging Face, with its exact size
and SHA-256 (from the Hugging Face API on 2026-11-06), so a broken or tampered
download is caught before it is used. Setup suggests the largest model that fits
the machine's memory; users can also point `model` at any .gguf file they have.

Downloads resume: the data goes to <file>.part, and running setup again continues
from where it stopped. Only a complete file is checksummed; a file that fails the
check is deleted. A checked file gets a small "<file>.verified" marker so it is not
hashed again every time setup runs.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .errors import BackendError

CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class CatalogModel:
    key: str
    title: str
    repo: str
    file: str
    size_bytes: int
    sha256: str
    min_memory_mb: int     # file size plus room for the context cache (an estimate)

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.file}"

    @property
    def size_gb(self) -> str:
        return f"{self.size_bytes / 1e9:.1f} GB"


CATALOG = (  # biggest first
    CatalogModel("gemma4-e4b", "Gemma 4 E4B (recommended, needs about 6 GB)",
                 "ggml-org/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_0.gguf", 4_590_807_392,
                 "a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2", 6_000),
    CatalogModel("gemma4-e2b", "Gemma 4 E2B (smaller and faster, less accurate, needs about 4 GB)",
                 "ggml-org/gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q4_0.gguf", 2_841_481_184,
                 "8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52", 4_000),
)


def suggest(usable_mb: int | None) -> CatalogModel:
    """The biggest catalog model that fits, else the smallest one."""
    if usable_mb is not None:
        for model in CATALOG:
            if usable_mb >= model.min_memory_mb:
                return model
    return CATALOG[-1]


def by_key(key: str) -> CatalogModel | None:
    return next((m for m in CATALOG if m.key == key), None)


def models_dir(home: Path) -> Path:
    return Path(home) / "models"


def target_path(model: CatalogModel, home: Path) -> Path:
    return models_dir(home) / model.file


def is_ready(model: CatalogModel, home: Path) -> bool:
    """True if the file is there and has passed the checksum before."""
    target = target_path(model, home)
    return target.is_file() and target.with_name(target.name + ".verified").is_file()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(model: CatalogModel, part: Path, progress: bool):
    """Append the missing bytes of the file to `part`."""
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    try:
        with urllib.request.urlopen(urllib.request.Request(model.url, headers=headers), timeout=60) as resp:
            if have and resp.status != 206:
                have = 0                      # the server ignored the range: start again
            with open(part, "ab" if have else "wb") as out:
                done, last_shown = have, -1
                while block := resp.read(CHUNK_BYTES):
                    out.write(block)
                    done += len(block)
                    percent = done * 100 // model.size_bytes
                    if progress and percent != last_shown:
                        print(f"\r  downloading {model.file}: {percent}% of {model.size_gb}",
                              end="", file=sys.stderr, flush=True)
                        last_shown = percent
    except urllib.error.HTTPError as e:
        if e.code != 416:                     # 416: we asked for bytes past the end, i.e. it is complete
            raise BackendError(f"download of {model.url} failed: HTTP {e.code}. Run setup again.") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise BackendError(f"the download was interrupted ({e}). Run setup again to continue it.") from e
    finally:
        if progress:
            print(file=sys.stderr)


def download(model: CatalogModel, home: Path, progress: bool = True) -> Path:
    """Make sure the model is in <home>/models and passes its checksum. Downloads or
    resumes as needed. Returns the file path; raises BackendError on failure."""
    target = target_path(model, home)
    marker = target.with_name(target.name + ".verified")
    if target.is_file() and marker.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target if target.is_file() else target.with_name(target.name + ".part")

    if not part.exists() or part.stat().st_size < model.size_bytes:
        _fetch(model, part, progress)
    size = part.stat().st_size
    if size < model.size_bytes:
        raise BackendError(f"the download stopped early ({size:,} of {model.size_bytes:,} bytes). "
                           "Run setup again to continue it.")
    if progress:
        print(f"  checking {model.file} ...", file=sys.stderr)
    if size != model.size_bytes or sha256_of(part) != model.sha256:
        part.unlink()
        raise BackendError(f"{model.file} did not match its checksum and was deleted. Run setup again.")
    if part != target:
        part.replace(target)
    marker.write_text(model.sha256 + "\n", encoding="utf-8")
    return target
