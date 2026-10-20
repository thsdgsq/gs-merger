"""Run ordered, interactive multi-model 3DGS stitching from one shell command."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import logging
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from gs_merger import __version__
from gs_merger.interaction import Open3DInteraction, UserQuit
from gs_merger.io_utils import (
    atomic_json,
    fingerprint,
    read_json,
    same_schema,
)
from gs_merger.merge_worker import merge_isolated
from gs_merger.gaussian_io import (
    export_view_cloud,
    read_gaussians,
    split_similarity,
)
from gs_merger.similarity import fit_control_points


DEFAULTS = {
    "pairs_per_step": 6,
    "control_threshold": 0.1,
    "min_inlier_ratio": 0.8,
    "view_max_points": 0,
    "preview_max_points": 300000,
    "min_opacity": 0.05,
    "point_size": 2.0,
    "marker_point_size": 6.0,
    "marker_size": 0.0001,
    "pick_window_width": 800,
    "pick_window_height": 700,
    "seed": 123,
}

DISPLAY_SETTINGS = {
    "point_size",
    "marker_point_size",
    "marker_size",
    "pick_window_width",
    "pick_window_height",
    "view_max_points",
    "preview_max_points",
    "min_opacity",
}


def load_settings(args):
    base = Path.cwd()
    supplied = {}
    if args.config:
        config = args.config.expanduser().resolve()
        supplied = read_json(config)
        base = config.parent
        if not isinstance(supplied, dict):
            raise ValueError("Configuration must be a JSON object")
    unknown = set(supplied) - set(DEFAULTS) - {"models", "output_dir"}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    settings = {**DEFAULTS, **supplied}

    def path(value, root):
        value = Path(value).expanduser()
        return str((root / value).resolve())

    models = args.models if args.models is not None else settings.get("models", [])
    if (
        not isinstance(models, list)
        or len(models) < 2
        or not all(isinstance(p, str) for p in models)
    ):
        raise ValueError("Specify at least two PLY paths in models, in stitching order")
    settings["models"] = [path(p, Path.cwd() if args.models is not None else base) for p in models]
    if len(set(settings["models"])) != len(models):
        raise ValueError("The model list contains duplicate input paths")
    output = args.output_dir if args.output_dir is not None else settings.get("output_dir")
    if output is None:
        raise ValueError("Specify output_dir in the configuration or --output-dir")
    settings["output_dir"] = path(output, Path.cwd() if args.output_dir is not None else base)
    out = Path(settings["output_dir"])
    if any(out in Path(p).parents for p in settings["models"]):
        raise ValueError("Choose an output directory that does not contain input models")
    for argument, key in (
        (args.pairs, "pairs_per_step"),
        (args.max_view_points, "view_max_points"),
        (args.control_threshold, "control_threshold"),
        (args.marker_size, "marker_size"),
        (args.marker_point_size, "marker_point_size"),
        (args.point_size, "point_size"),
        (args.pick_window_width, "pick_window_width"),
        (args.pick_window_height, "pick_window_height"),
    ):
        if argument is not None:
            settings[key] = argument
    pairs = settings["pairs_per_step"]
    if type(pairs) is int:
        pairs = [pairs] * (len(models) - 1)
    if (
        not isinstance(pairs, list)
        or len(pairs) != len(models) - 1
        or any(type(n) is not int or n < 3 for n in pairs)
    ):
        raise ValueError("pairs_per_step must be >= 3 or a list of N-1 integers >= 3")
    settings["pairs_per_step"] = pairs
    for key in (
        "view_max_points",
        "preview_max_points",
        "seed",
        "pick_window_width",
        "pick_window_height",
    ):
        if type(settings[key]) is not int or settings[key] < 0:
            raise ValueError(f"{key} must be an integer >= 0")
    for key in (
        "control_threshold",
        "point_size",
        "marker_point_size",
        "marker_size",
        "min_inlier_ratio",
        "min_opacity",
    ):
        if isinstance(settings[key], bool):
            raise ValueError(f"{key} must be numeric")
        settings[key] = float(settings[key])
        if not np.isfinite(settings[key]):
            raise ValueError(f"{key} must be finite")
    if settings["control_threshold"] <= 0 or settings["point_size"] <= 0:
        raise ValueError("control_threshold and point_size must be positive")
    if not 1 <= settings["point_size"] <= 20:
        raise ValueError("point_size must be between 1 and 20 pixels")
    if not 1 <= settings["marker_point_size"] <= 30:
        raise ValueError("marker_point_size must be between 1 and 30 pixels")
    if settings["marker_size"] <= 0:
        raise ValueError("marker_size must be positive")
    if settings["pick_window_width"] < 240 or settings["pick_window_height"] < 200:
        raise ValueError("Picking windows must be at least 240 x 200 pixels")
    if not 0 < settings["min_inlier_ratio"] <= 1 or not 0 <= settings["min_opacity"] < 1:
        raise ValueError("min_inlier_ratio must be in (0,1]; min_opacity must be in [0,1)")
    return settings


def inspect_models(settings):
    infos, reference = [], None
    for path in settings["models"]:
        vertices = read_gaussians(path)
        if reference is None:
            reference = vertices
        elif not same_schema(reference, vertices):
            raise ValueError(f"Gaussian schema / SH degree differs: {path}")
        infos.append({"signature": fingerprint(path), "gaussians": len(vertices)})
    accumulated, merged_bytes = infos[0]["gaussians"], 0
    for info in infos[1:]:
        accumulated += info["gaussians"]
        merged_bytes += accumulated * reference.dtype.itemsize
    return infos, merged_bytes


@contextmanager
def output_lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".workflow.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another workflow is already using this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def session_state(settings, infos):
    out = Path(settings["output_dir"])
    path = out / "session.json"
    identity = {"version": 1, "settings": settings, "inputs": infos}
    if path.exists():
        state = read_json(path)
        # Display changes do not change model coordinates, saved XYZ pairs or
        # committed results. Also accept sessions created before these UI keys
        # existed, so this improvement can resume the user's existing work.
        previous_settings = {
            key: value
            for key, value in state.get("settings", {}).items()
            if key not in DISPLAY_SETTINGS
        }
        current_settings = {
            key: value for key, value in settings.items() if key not in DISPLAY_SETTINGS
        }
        if (
            state.get("version") != identity["version"]
            or state.get("inputs") != infos
            or previous_settings != current_settings
        ):
            raise ValueError(
                "模型、顺序或配置与此目录的已有会话不同。请恢复原配置，或指定新的输出目录。"
            )
        completed = state.get("completed", [])
        if len(completed) > len(settings["models"]) - 1:
            raise ValueError("Invalid completed-round count in session.json")
        for number, record in enumerate(completed, 1):
            expected = out / "steps" / f"{number:03d}" / "merged.ply"
            if record["round"] != number or record["merged"] != str(expected):
                raise ValueError("Invalid completed-round paths in session.json")
            if not expected.is_file() or fingerprint(expected) != record["merged_signature"]:
                raise ValueError(f"已完成的第 {number} 轮输出缺失或被修改，不能继续复用。")
        if state["settings"] != settings:
            state["settings"] = dict(settings)
            atomic_json(path, state)
        return state
    unexpected = set(p.name for p in out.iterdir()) - {".workflow.lock"}
    if unexpected:
        raise ValueError(
            "Output directory is not empty and has no session.json; choose another one"
        )
    state = {**identity, "completed": []}
    atomic_json(path, state)
    return state


def view_cloud(model, path, settings):
    metadata = path.with_suffix(".json")
    identity = {
        "model": fingerprint(model),
        "min_opacity": settings["min_opacity"],
        "max_points": settings["view_max_points"],
        "seed": settings["seed"],
    }
    if metadata.is_file() and path.is_file():
        cached = read_json(metadata)
        if cached.get("identity") == identity and cached.get("output_signature") == fingerprint(
            path
        ):
            return path
    report = export_view_cloud(
        model, path, settings["min_opacity"], settings["view_max_points"], settings["seed"]
    )
    atomic_json(metadata, {"identity": identity, "output_signature": fingerprint(path), **report})
    return path


def validate_selection(selection, count):
    points = np.asarray(selection["points"], dtype=np.float64)
    indices = selection["indices"]
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise ValueError("Saved selections have invalid XYZ coordinates")
    if (
        len(indices) != count
        or len(set(indices)) != count
        or len(np.unique(points, axis=0)) != count
    ):
        raise ValueError("Please pick distinct points for distinct landmarks")


def save_alignment(step, controls, settings):
    source, target = np.asarray(controls["source"]), np.asarray(controls["target"])
    transform, mask, residuals = fit_control_points(
        source, target, settings["control_threshold"], seed=settings["seed"]
    )
    required = max(3, math.ceil(settings["min_inlier_ratio"] * len(source)))
    qualified = int(mask.sum()) >= required
    report = {
        "scale": split_similarity(transform)[0],
        "transform_source_to_target": transform.tolist(),
        "inlier_count": int(mask.sum()),
        "required_inliers": required,
        "inlier_mask": mask.tolist(),
        "residuals_in_reference_units": residuals.tolist(),
        "control_threshold": settings["control_threshold"],
        "qualified": qualified,
    }
    atomic_json(step / "alignment.json", report)
    np.save(step / "transform_source_to_target.npy", transform)
    np.savetxt(step / "transform_source_to_target.txt", transform, fmt="%.12g")
    return transform, report


def run_round(settings, number, source, target, ui, log):
    step = Path(settings["output_dir"]) / "steps" / f"{number:03d}"
    step.mkdir(parents=True, exist_ok=True)
    count = settings["pairs_per_step"][number - 1]
    context = {
        "round": number,
        "source": fingerprint(source),
        "target": fingerprint(target),
        "pairs": count,
    }
    context_path = step / "context.json"
    if context_path.exists() and read_json(context_path) != context:
        raise ValueError(
            "This pending round refers to different inputs; use a new output directory"
        )
    atomic_json(context_path, context)
    log(f"\n===== 第 {number}/{len(settings['models']) - 1} 轮，选择 {count} 对点 =====")
    log(f"已有模型 target: {target}\n新模型 source: {source}")
    target_view = view_cloud(target, step / "target_view.ply", settings)
    source_view = view_cloud(source, step / "source_view.ply", settings)
    selection_path = step / "selections.json"
    selections = (
        read_json(selection_path) if selection_path.exists() else {"source": None, "target": None}
    )
    if selection_path.exists():
        try:
            for role in ("source", "target"):
                if selections.get(role) is None:
                    raise ValueError("Incomplete saved pair")
                validate_selection(selections[role], count)
        except (ValueError, KeyError, TypeError):
            log(f"本轮旧选点未凑齐 {count} 对，已丢弃；将重新打开两侧选取。")
            selections = {"source": None, "target": None}
            selection_path.unlink()
    while True:

        def save_selections(updated):
            for role in ("source", "target"):
                if updated.get(role) is None:
                    raise ValueError("Cannot save an incomplete set of correspondence pairs")
                validate_selection(updated[role], count)
            atomic_json(selection_path, updated)

        selections = ui.select_pair(
            {"target": target_view, "source": source_view},
            count,
            number,
            selections,
            on_save=save_selections,
        )
        for selected in selections.values():
            validate_selection(selected, count)
        controls = {
            "source": selections["source"]["points"],
            "target": selections["target"]["points"],
            "source_model": str(source),
            "target_model": str(target),
            "source_selection_indices": selections["source"]["indices"],
            "target_selection_indices": selections["target"]["indices"],
            "coordinate_frame": "original incoming model -> accumulated first-model frame",
        }
        atomic_json(step / "correspondences.json", controls)
        try:
            transform, report = save_alignment(step, controls, settings)
        except ValueError as exc:
            atomic_json(step / "alignment.json", {"qualified": False, "error": str(exc)})
            for name in ("transform_source_to_target.npy", "transform_source_to_target.txt"):
                (step / name).unlink(missing_ok=True)
            log(f"控制点求解失败：{exc}。请检查对应顺序并避免共线点。")
            action = ui.choice(
                "[t] 重选已有模型 / [s] 重选新模型 / [b] 两侧重选 / [q] 退出：",
                {"t", "s", "b", "q"},
            )
        else:
            log(
                f"Sim(3) scale={report['scale']:.8g}; "
                f"inliers={report['inlier_count']}/{count}, required={report['required_inliers']}"
            )
            for i, (residual, inlier) in enumerate(
                zip(report["residuals_in_reference_units"], report["inlier_mask"]), 1
            ):
                log(f"  对应点 {i}: residual={residual:.8g}; {'内点' if inlier else '外点'}")
            if not report["qualified"]:
                log("一致点不足，本轮禁止合并。阈值使用第一个模型的坐标单位，不一定是米。")
            while True:
                ui.preview(source_view, target_view, transform)
                action = ui.review(report["qualified"])
                if action != "v":
                    break
            if action == "a":
                if not report["qualified"]:
                    raise ValueError("Cannot merge a rejected alignment")
                # Confirm input files did not change while the user was picking.
                if (
                    fingerprint(source) != context["source"]
                    or fingerprint(target) != context["target"]
                ):
                    raise ValueError(
                        "Input model changed during picking; stop and start a new session"
                    )
                destination = step / "merged.ply"
                log("正在分块变换完整高斯属性并写入 merged.ply …")
                merge_report = merge_isolated(source, target, transform, destination, log=log)
                record = {
                    **context,
                    **merge_report,
                    "merged": str(destination),
                    "merged_signature": fingerprint(destination),
                    "transform_source_to_reference": transform.tolist(),
                }
                atomic_json(step / "step.json", record)
                return record
        if action == "q":
            raise UserQuit()
        if action in ("t", "b"):
            selections["target"] = None
        if action in ("s", "b"):
            selections["source"] = None
        # Rejected pairs must not be restored after a restart. Retain the other
        # side only in memory during this run, never as a partial checkpoint.
        selection_path.unlink(missing_ok=True)
        for name in (
            "correspondences.json",
            "alignment.json",
            "transform_source_to_target.npy",
            "transform_source_to_target.txt",
        ):
            (step / name).unlink(missing_ok=True)


def publish_final(settings, state):
    out = Path(settings["output_dir"])
    result = Path(state["completed"][-1]["merged"]).resolve()
    transforms = [{"model": settings["models"][0], "transform_to_reference": np.eye(4).tolist()}]
    transforms.extend(
        {
            "model": settings["models"][i],
            "transform_to_reference": record["transform_source_to_reference"],
        }
        for i, record in enumerate(state["completed"], 1)
    )
    atomic_json(
        out / "final.json",
        {"reference": settings["models"][0], "merged": str(result), "models": transforms},
    )
    # The last round already wrote the complete model. Report that actual file.
    return result


def run_workflow(settings, interaction_factory=None, log=print):
    infos, merged_bytes = inspect_models(settings)
    log(
        f"按顺序拼接 {len(infos)} 个模型；"
        f"所有中间 merged.ply 合计约 {merged_bytes / 1e9:.2f} GB（另有查看点云）。"
    )
    out = Path(settings["output_dir"])
    with output_lock(out):
        state = session_state(settings, infos)
        logger = logging.getLogger(f"gs_merger.{out}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(out / "workflow.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)

        def session_log(message):
            log(message)
            logger.info(message)

        try:
            ui = None
            for number in range(len(state["completed"]) + 1, len(settings["models"])):
                if ui is None:
                    ui = (interaction_factory or Open3DInteraction)(settings, log=session_log)
                target = (
                    Path(state["completed"][-1]["merged"])
                    if state["completed"]
                    else Path(settings["models"][0])
                )
                record = run_round(
                    settings, number, Path(settings["models"][number]), target, ui, session_log
                )
                state["completed"].append(record)
                atomic_json(out / "session.json", state)
                session_log(f"第 {number} 轮完成：{record['merged']}")
            final = publish_final(settings, state)
            session_log(f"全部完成：{final}")
            return final
        except (UserQuit, KeyboardInterrupt) as exc:
            session_log(
                "已退出。已完成轮次和完整点对已保存；未凑齐整组的选点不保存，下次重新选取。"
            )
            raise UserQuit() from exc
        finally:
            logger.removeHandler(handler)
            handler.close()


def build_parser():
    parser = argparse.ArgumentParser(prog="gs-merger", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--config", type=Path, help="JSON configuration; paths are relative to this file"
    )
    parser.add_argument(
        "--models", nargs="+", help="PLYs in stitching order; overrides config models"
    )
    parser.add_argument(
        "--output-dir", help="Output directory; relative to current working directory"
    )
    parser.add_argument(
        "--pairs", type=int, help="Correspondence pairs per round; overrides config"
    )
    parser.add_argument("--max-view-points", type=int, help="Viewing points per model; 0 means all")
    parser.add_argument(
        "--point-size", type=float, help="Initial RGB cloud point size, 1-20 pixels"
    )
    parser.add_argument(
        "--marker-point-size", type=float, help="Initial selection marker size, 1-30 pixels"
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        help="Legacy compatibility option (ignored); use --marker-point-size for pixel size",
    )
    parser.add_argument(
        "--pick-window-width", type=int, help="Width of each picking window, default 800"
    )
    parser.add_argument(
        "--pick-window-height", type=int, help="Height of each picking window, default 700"
    )
    parser.add_argument(
        "--control-threshold", type=float, help="Control residual threshold in reference units"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check configuration, input schemas and dependencies without windows or model output",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args)
        if args.check:
            import open3d  # noqa: F401

            infos, merged_bytes = inspect_models(settings)
            print(f"Python: {sys.executable}; Open3D: {importlib.metadata.version('open3d')}")
            for number, info in enumerate(infos, 1):
                print(f"{number}: {info['signature']['path']} ({info['gaussians']:,} Gaussians)")
            print(f"Pairs per round: {settings['pairs_per_step']}; reference: model 1")
            print(
                f"Intermediate merged PLYs: about {merged_bytes / 1e9:.2f} GB, plus viewing clouds"
            )
            print(
                "Check passed. GUI / OpenGL has not been tested; "
                "run normally from a graphical desktop terminal."
            )
            return 0
        if not sys.stdin.isatty():
            raise ValueError(
                "交互工作流需要终端输入，请在图形桌面的终端运行；仅检查时使用 --check。"
            )
        run_workflow(settings)
        return 0
    except (UserQuit, KeyboardInterrupt):
        return 130
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
