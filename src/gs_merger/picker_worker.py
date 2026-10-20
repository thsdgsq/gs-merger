"""One Open3D native window per process; used internally by the pair picker."""

import argparse
import sys
from pathlib import Path

from gs_merger.interaction import Open3DInteraction
from gs_merger.io_utils import atomic_json, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        request = read_json(args.request)
        ui = Open3DInteraction(request["settings"])
        if request["saved"] is not None:
            ui.show_saved(request["path"], request["role"], request["saved"], request["slot"])
            result = {"comparison_only": True}
        else:
            selected = ui.pick_once(
                request["path"], request["role"], request["count"], request["slot"]
            )
            if ui.valid_selection(selected, request["count"]):
                result = {"selection": selected}
            else:
                # Even the temporary IPC result carries no coordinates when
                # there are fewer/more points than the requested complete set.
                result = {"incomplete": True, "picked_count": len(selected["indices"])}
        atomic_json(args.result, result)
        return 0
    except Exception as exc:
        atomic_json(args.result, {"error": str(exc)})
        print(f"Picker error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
