#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"
ARGS=()
while (($#)); do
  case "$1" in
    --python)
      if (($# < 2)); then
        echo "--python 需要指定 Python 解释器路径" >&2
        exit 2
      fi
      PYTHON_BIN="$2"
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

# Source checkout and child processes use only this project's own package.
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -u -m gs_merger "${ARGS[@]}"
