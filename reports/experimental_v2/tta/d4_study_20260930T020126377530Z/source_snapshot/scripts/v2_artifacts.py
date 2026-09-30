"""Small filesystem/provenance helpers shared by the experimental notebooks."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import re
import shutil
import subprocess

from scripts.experimental_validity import sha256_file


def repository_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for parent in [start, *start.parents]:
        if (parent / "scripts/experimental_protocol.py").is_file() and (parent / "notebooks").is_dir():
            return parent
    raise FileNotFoundError("Run inside the repository or pass its root explicitly")


def relative_path(root: Path, value: str | Path) -> Path:
    """Reject absolute Windows paths even on Linux; never guess missing paths."""
    value = str(value)
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute() or PureWindowsPath(value).drive:
        raise ValueError("Artifact paths must be relative to their root")
    path = (root / value.replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes root")
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", value):
        raise ValueError("Invalid artifact identifier")
    return value


def run_id(prefix: str) -> str:
    return identifier(f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}")


def content_id(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    """Published JSON is append-only; a new result gets a new directory/identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def environment(include_torch: bool = False) -> dict:
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    result = {"python": platform.python_version(), "platform": platform.platform(), "dependencies": dict(sorted(packages.items()))}
    if include_torch:
        import torch
        import torchvision
        result.update(torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                      numpy=packages.get("numpy"), cuda=torch.version.cuda,
                      cudnn=torch.backends.cudnn.version(),
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      cuda_compiled_architectures=torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
                      cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                      cpu_threads=torch.get_num_threads())
    return result


def provenance(root: Path, output: Path, dataset: dict) -> dict:
    sources = sorted((root / "scripts").glob("*.py")) + sorted((root / "notebooks").glob("*.ipynb"))
    sources += [root / "docs/experimental_protocol_v2.md"]
    digests = {}
    for source in sources:
        if source.is_file():
            rel = source.relative_to(root)
            dest = output / "source_snapshot" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            digests[rel.as_posix()] = sha256_file(dest)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True))
    return {"created_at_utc": utc_now(), "git_commit": commit, "working_tree_dirty": dirty,
            "dataset": dataset, "source_hashes": digests, "source_snapshot_sha256": content_id(digests)}


def seal_bundle(directory: Path) -> dict:
    files = {p.relative_to(directory).as_posix(): sha256_file(p) for p in sorted(directory.rglob("*")) if p.is_file()}
    if "artifacts.json" in files:
        raise FileExistsError("Bundle is already sealed")
    manifest = {"files": files, "content_sha256": content_id(files)}
    write_json(directory / "artifacts.json", manifest)
    return manifest


def verify_bundle(directory: Path) -> dict:
    manifest = read_json(directory / "artifacts.json")
    if content_id(manifest["files"]) != manifest["content_sha256"]:
        raise ValueError("Artifact manifest identity changed")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(manifest["files"]) | {"artifacts.json"}:
        raise ValueError("Sealed bundle file set changed")
    for name, digest in manifest["files"].items():
        if sha256_file(relative_path(directory, name)) != digest:
            raise ValueError(f"Artifact content changed: {name}")
    return manifest
