"""Atomic checkpoints and complete-model publication using the local Gaussian backend."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from gs_merger.gaussian_io import merge_gaussian_files


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".json_", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def same_schema(first, second):
    return set(first.dtype.names) == set(second.dtype.names) and all(
        first.dtype[name] == second.dtype[name] for name in first.dtype.names
    )


def merge_streaming(source, target, transform, destination, *, log=None):
    """Use the bounded-memory Gaussian merge, publishing only a complete result.

    The historical helper name is retained for callers. Computation and PLY
    serialization use the Gaussian implementation shipped inside this package.
    """
    source, target, destination = map(lambda p: Path(p).resolve(), (source, target, destination))
    if destination in (source, target) or (
        destination.exists() and (destination.samefile(source) or destination.samefile(target))
    ):
        raise ValueError("Merged output would overwrite an input model")
    started = time.monotonic()

    destination.parent.mkdir(parents=True, exist_ok=True)
    if log is not None:
        log("[merge] 合并后端：gs_merger.gaussian_io.merge_gaussian_files。")
    with tempfile.TemporaryDirectory(prefix=".merge_", dir=destination.parent) as temporary:
        report = merge_gaussian_files(
            source, target, transform, temporary, minimum_opacity=0, log=log
        )
        Path(report["merged"]).replace(destination)
    if log is not None:
        log(f"[merge] 合并完成：{destination}；耗时 {time.monotonic() - started:.1f} 秒")
    return {
        "target_gaussians": report["target_output_count"],
        "source_gaussians": report["source_output_count"],
        "merged_gaussians": report["merged_count"],
    }
