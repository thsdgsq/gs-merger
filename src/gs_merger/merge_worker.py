"""Isolate numeric merge work from Open3D and its native thread pools."""

from __future__ import annotations

import argparse
from collections import deque
import faulthandler
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

import numpy as np

from gs_merger.io_utils import atomic_json, merge_streaming, read_json


def merge_isolated(source, target, transform, destination, *, log=None):
    """Run a fresh interpreter, relay progress and return only a completed merge."""
    with tempfile.TemporaryDirectory(prefix="gs_merge_job_") as temporary:
        request, result = [Path(temporary) / name for name in ("request.json", "result.json")]
        atomic_json(request, {
            "source": str(Path(source).resolve()),
            "target": str(Path(target).resolve()),
            "transform": np.asarray(transform).tolist(),
            "destination": str(Path(destination).resolve()),
        })
        environment = os.environ.copy()
        # Set these BEFORE NumPy is imported by the fresh interpreter. The
        # parent GUI's libraries and environment are deliberately left intact.
        environment.update(dict.fromkeys((
            "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        ), "1"))
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "gs_merger.merge_worker",
             "--request", str(request), "--result", str(result)],
            env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        tail = deque(maxlen=12)
        try:
            if log is not None:
                log(f"[merge] 已启动独立计算进程 PID={process.pid}，数值库线程数=1。")
            for line in process.stdout:
                message = line.rstrip("\r\n")
                tail.append(message)
                if log is not None:
                    log(message)
            status = process.wait()
            data = read_json(result) if result.exists() else {}
            if status != 0 or "error" in data or not isinstance(data.get("report"), dict):
                detail = data.get("error") or "\n".join(tail) or "no result returned"
                raise RuntimeError(f"Merge worker failed (exit {status}): {detail}")
            return data["report"]
        finally:
            if process.poll() is None:
                # Give the writer a chance to clean up its temporary PLY first.
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    faulthandler.enable()

    def progress(message):
        # Reset the watchdog on progress. A native operation that stops making
        # progress then exposes its Python call site without attaching a debugger.
        faulthandler.dump_traceback_later(60, repeat=True)
        print(message, flush=True)

    try:
        request = read_json(args.request)
        progress("[merge] 无进度超过 60 秒时，将自动输出调用栈用于定位。")
        report = merge_streaming(**request, log=progress)
        atomic_json(args.result, {"report": report})
        return 0
    except KeyboardInterrupt:
        print("[merge] 合并已中断，完整点对和原始模型保留。", flush=True)
        return 130
    except Exception as exc:
        atomic_json(args.result, {"error": str(exc)})
        print(f"[merge] Error: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    raise SystemExit(main())
