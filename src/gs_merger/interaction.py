"""Open3D picking and review. Window closure always returns to an explicit choice."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData


class UserQuit(Exception):
    """The user stopped the session; committed rounds and selections remain on disk."""


class Open3DInteraction:
    def __init__(self, settings, ask=input, log=print):
        import open3d as o3d

        self.o3d = o3d
        self.settings, self.ask, self.log = settings, ask, log

    def choice(self, prompt, allowed):
        while True:
            try:
                answer = self.ask(prompt).strip().lower()
            except (EOFError, KeyboardInterrupt) as exc:
                raise UserQuit() from exc
            if answer in allowed:
                self.log(f"[choice] {answer}")
                return answer
            self.log("请输入括号中列出的选项。")

    def load_cloud(self, path, maximum=0):
        if maximum:
            # Read only sampled vertices into Open3D's double-precision arrays.
            # Reading the full accumulated cloud first defeats the preview cap.
            vertices = PlyData.read(str(path))["vertex"].data
            if len(vertices) > maximum:
                indices = np.random.default_rng(self.settings["seed"]).choice(
                    len(vertices), maximum, replace=False
                )
                vertices = vertices[indices]
            cloud = self.o3d.geometry.PointCloud()
            cloud.points = self.o3d.utility.Vector3dVector(
                np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float64)
            )
            cloud.colors = self.o3d.utility.Vector3dVector(
                np.column_stack([vertices[name] for name in ("red", "green", "blue")]) / 255.0
            )
        else:
            cloud = self.o3d.io.read_point_cloud(str(path))
        if not len(cloud.points):
            raise ValueError(f"Open3D could not read viewing points: {path}")
        return cloud

    def window(self, visualizer, title, *, slot=None):
        width = self.settings.get("pick_window_width", 800) if slot is not None else 1280
        height = self.settings.get("pick_window_height", 700) if slot is not None else 800
        left = 20 + slot * (width + 20) if slot is not None else 50
        if not visualizer.create_window(
            window_name=title, width=width, height=height, left=left, top=60
        ):
            visualizer.destroy_window()
            raise RuntimeError(
                "Open3D 无法创建交互窗口。请在本机图形桌面的终端运行，并检查 DISPLAY/OpenGL。"
            )
        options = visualizer.get_render_option()
        options.point_size = self.settings["point_size"]
        options.background_color = np.array([0.12, 0.12, 0.12])
        options.point_color_option = self.o3d.visualization.PointColorOption.Color

    def pick_once(self, path, role, count, slot=None):
        cloud = self.load_cloud(path)
        if len(cloud.points) < count:
            raise ValueError(
                f"Only {len(cloud.points)} viewing centers; cannot pick {count} points"
            )
        from gs_merger.point_picker import run_picker

        return run_picker(self.o3d, self.settings, cloud, role, count, slot, log=self.log)

    @staticmethod
    def valid_selection(selected, count):
        indices, points = selected["indices"], np.asarray(selected["points"])
        return (
            points.shape == (count, 3)
            and np.isfinite(points).all()
            and len(indices) == count
            and len(set(indices)) == count
            and len(np.unique(points, axis=0)) == count
        )

    def show_saved(self, path, role, selection, slot):
        """Keep an already confirmed side visible while the other side is picked."""
        from gs_merger.point_picker import run_picker

        run_picker(
            self.o3d,
            self.settings,
            self.load_cloud(path),
            role,
            len(selection["points"]),
            slot,
            saved=selection,
            log=self.log,
        )

    def start_picker(self, directory, role, path, count, number, slot, saved, attempt):
        request = directory / f"{role}_{attempt}_request.json"
        result = directory / f"{role}_{attempt}_result.json"
        request.write_text(
            json.dumps(
                {
                    "settings": self.settings,
                    "path": str(path),
                    "count": count,
                    "role": f"Round {number}: {role}",
                    "slot": slot,
                    "saved": saved,
                }
            ),
            encoding="utf-8",
        )
        # Each GUI picker owns its event loop on its process's main thread.
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "gs_merger.picker_worker",
                "--request",
                str(request),
                "--result",
                str(result),
            ],
            stdin=subprocess.DEVNULL,
        )
        return {"process": process, "result": result, "readonly": saved is not None}

    def select_pair(self, paths, count, number, selections, on_save):
        selections = dict(selections)
        for role, selected in selections.items():
            if selected is not None and not self.valid_selection(selected, count):
                selections[role] = None
        if all(
            selection is not None and self.valid_selection(selection, count)
            for selection in selections.values()
        ):
            return selections
        self.log(
            "将同时打开两个选点窗口：左侧 target，右侧 source。\n"
            "可在两窗间交替选 A、B、C…；两侧各自的选点顺序必须对应。\n"
            "必须凑齐两侧全部点后才保存；两侧都关闭且有点数不足时，"
            "输入一次 r 会同时重开两侧。"
        )
        active, pending, attempt = {}, {}, 0
        with tempfile.TemporaryDirectory(prefix="gs_pair_") as temporary:

            def launch(role):
                nonlocal attempt
                attempt += 1
                active[role] = self.start_picker(
                    Path(temporary),
                    role,
                    paths[role],
                    count,
                    number,
                    0 if role == "target" else 1,
                    selections[role],
                    attempt,
                )

            def collect_closed():
                # Read every finished window before asking anything. Otherwise
                # input() can block reopening the second already-closed side.
                for role in list(active):
                    worker = active[role]
                    status = worker["process"].poll()
                    if status is None:
                        continue
                    worker["process"].wait()
                    del active[role]
                    result = worker["result"]
                    data = json.loads(result.read_text(encoding="utf-8")) if result.exists() else {}
                    if status != 0 or "error" in data:
                        raise RuntimeError(
                            f"{role} 选点窗口失败：{data.get('error', f'exit {status}')}。"
                        )
                    if worker["readonly"]:
                        continue
                    selected = data.get("selection")
                    if selected is None and not data.get("incomplete"):
                        raise RuntimeError(f"{role} picker exited without selection data")
                    pending[role] = {
                        "selection": selected,
                        "valid": selected is not None and self.valid_selection(selected, count),
                        "count": data.get(
                            "picked_count", len(selected["indices"]) if selected else 0
                        ),
                    }

            def both_need_review():
                return len(pending) == 2 and any(not item["valid"] for item in pending.values())

            def restart_both():
                pending.clear()
                for role in ("target", "source"):
                    selections[role] = None
                    launch(role)
                self.log("已同时重新打开 target 和 source；本轮两侧从零选取。")

            def reopen_comparison():
                # Never reopen a reference while the other closed window is
                # merely waiting for terminal confirmation.
                if pending:
                    return
                for role, selected in selections.items():
                    if selected is not None and role not in active and any(
                        selections[other] is None and worker["process"].poll() is None
                        and not worker["result"].exists()
                        for other, worker in active.items()
                    ):
                        self.log(f"[{role}] 已确认；重新打开只读对照窗口，无需再次选点。")
                        launch(role)

            try:
                # Launch BOTH before waiting for either window or terminal input.
                launch("target")
                launch("source")
                while any(value is None for value in selections.values()):
                    collect_closed()
                    if not pending:
                        time.sleep(0.05)
                        continue
                    if both_need_review():
                        self.log(
                            f"两侧窗口都已关闭：target {pending['target']['count']}/{count}，"
                            f"source {pending['source']['count']}/{count}；"
                            "未凑齐完整点对，不保存。"
                        )
                        action = self.choice("[r] 同时重选两侧 / [q] 退出：", {"r", "q"})
                        if action == "q":
                            raise UserQuit()
                        restart_both()
                        continue
                    role = next(iter(pending))
                    item = pending[role]
                    selected = item["selection"]
                    for i, xyz in enumerate(selected["points"] if selected else [], 1):
                        self.log(f"[{role}] 地物 {i}: {xyz}")
                    if item["valid"]:
                        action = self.choice(
                            f"[{role}] [y] 确认本侧（满 {count} 对后保存）"
                            " / [r] 重选 / [q] 退出：",
                            {"y", "r", "q"},
                        )
                    else:
                        self.log(
                            f"[{role}] 需要 {count} 个不同坐标的点，当前选了 {item['count']} 个，"
                            "已丢弃且不会保存。"
                        )
                        action = self.choice(f"[{role}] [r] 重选 / [q] 退出：", {"r", "q"})
                    if action == "q":
                        raise UserQuit()
                    if action == "r":
                        # The other window may have closed while input() waited.
                        collect_closed()
                        if both_need_review():
                            restart_both()
                            continue
                        del pending[role]
                        launch(role)
                    else:
                        selections[role] = selected
                        del pending[role]
                    reopen_comparison()
            finally:
                for worker in active.values():
                    process = worker["process"]
                    if process.poll() is None:
                        process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        if not all(self.valid_selection(selected, count) for selected in selections.values()):
            raise ValueError("Incomplete correspondence pairs must never be saved")
        on_save(dict(selections))
        return selections

    def select(self, path, role, count):
        while True:
            selected = self.pick_once(path, role, count)
            indices, points = selected["indices"], selected["points"]
            for i, xyz in enumerate(points, 1):
                self.log(f"  地物 {i}: {xyz}")
            valid = self.valid_selection(selected, count)
            if not valid:
                self.log(f"需要 {count} 个不同的点，当前选了 {len(indices)} 个。")
                action = self.choice("[r] 重选本侧 / [q] 保存进度并退出：", {"r", "q"})
            else:
                action = self.choice(
                    "[y] 保存这些点 / [r] 重选本侧 / [q] 保存进度并退出：", {"y", "r", "q"}
                )
            if action == "q":
                raise UserQuit()
            if action == "y":
                return selected

    def preview(self, source_path, target_path, transform):
        maximum = self.settings["preview_max_points"]
        source = self.load_cloud(source_path, maximum)
        target = self.load_cloud(target_path, maximum)
        source.transform(transform)
        source.paint_uniform_color([1.0, 0.55, 0.05])
        target.paint_uniform_color([0.1, 0.65, 1.0])
        self.log("对齐预览：橙色=本轮新模型，蓝色=已有合并模型；关闭窗口后在终端选择操作。")
        visualizer = self.o3d.visualization.Visualizer()
        self.window(visualizer, "Alignment preview | orange: incoming | blue: accumulated")
        try:
            if not visualizer.add_geometry(target) or not visualizer.add_geometry(source):
                raise RuntimeError("Open3D failed to add alignment preview geometry")
            visualizer.run()
        finally:
            visualizer.destroy_window()

    def review(self, qualified):
        prompt = "[a] 确认并合并 / " if qualified else ""
        prompt += "[t] 重选已有模型 / [s] 重选新模型 / [b] 两侧重选 / [v] 再预览 / [q] 退出："
        return self.choice(
            prompt, {"a", "t", "s", "b", "v", "q"} if qualified else {"t", "s", "b", "v", "q"}
        )
