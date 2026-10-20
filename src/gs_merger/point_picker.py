"""Open3D picker with original RGB, matched pair colors and readable point numbers."""

from __future__ import annotations

import colorsys

import numpy as np


def pair_colors(count):
    """Stable bright colors by pair number, independent of window or current count."""
    palette = [
        (1.0, 0.15, 0.15), (0.05, 1.0, 1.0), (1.0, 0.9, 0.05),
        (0.95, 0.25, 1.0), (0.2, 1.0, 0.2), (1.0, 0.5, 0.05),
        (0.2, 0.55, 1.0), (1.0, 0.4, 0.65), (0.65, 0.45, 1.0), (0.1, 1.0, 0.65),
    ]
    return np.asarray([
        palette[i] if i < len(palette) else colorsys.hsv_to_rgb(
            ((i - len(palette) + 1) * 0.618033988749895) % 1, 0.8, 1.0
        )
        for i in range(count)
    ], dtype=float).reshape(-1, 3)


class OrderedPoints:
    """Retain click order and the exact viewing-cloud coordinates, never depth estimates."""

    def __init__(self, points, required):
        self.points = np.asarray(points)
        self.required = required
        self.indices = []

    def add(self, index):
        if len(self.indices) >= self.required or index in self.indices:
            return False
        if not 0 <= index < len(self.points):
            return False
        if any(np.array_equal(self.points[index], self.points[i]) for i in self.indices):
            return False
        self.indices.append(int(index))
        return True

    def undo(self):
        if self.indices:
            self.indices.pop()

    def clear(self):
        self.indices.clear()

    def result(self):
        return {"indices": list(self.indices), "points": self.points[self.indices].tolist()}


def point_material(o3d, size):
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = float(size)
    # With post-processing disabled, pass the stored RGB through unchanged.
    # True would linearize vertex colors while building the rendering buffers.
    material.sRGB_color = False
    return material


def configure_raw_rgb(scene):
    """Bypass color grading/tone mapping and sample blending; keep vertex RGB intact."""
    scene.view.set_post_processing(False)
    scene.view.set_sample_count(1)


def marker_positions(points, camera_position):
    """Tiny rendering-only depth bias along each view ray; picked XYZ is unchanged."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    camera = np.asarray(camera_position, dtype=np.float64)
    return camera + (points - camera) * (1.0 - 1e-5)


class PointPicker:
    def __init__(self, o3d, settings, cloud, role, count, slot, saved=None, log=print):
        self.o3d, self.settings, self.cloud = o3d, settings, cloud
        self.gui = o3d.visualization.gui
        self.rendering = o3d.visualization.rendering
        self.app = self.gui.Application.instance
        self.role, self.count, self.saved, self.log = role, count, saved, log
        self.ordered = OrderedPoints(np.asarray(cloud.points), count)
        self.size = float(
            saved.get("point_size", settings["point_size"]) if saved else settings["point_size"]
        )
        self.marker_size = float(
            saved.get("marker_point_size", settings.get("marker_point_size", 6))
            if saved else settings.get("marker_point_size", 6)
        )
        self.closed, self.pending_pick, self.has_markers = False, False, False
        self.camera_state = None
        self.point_labels = []
        self.label_points = np.empty((0, 3))
        self.tree = None
        self.selection_revision = 0
        if saved is None:
            self.log(f"[{role}] 正在建立选点索引 …")
            self.tree = o3d.geometry.KDTreeFlann(cloud)
        width = settings.get("pick_window_width", 800) if slot is not None else 1280
        height = settings.get("pick_window_height", 700) if slot is not None else 800
        left = 20 + slot * (width + 20) if slot is not None else 50
        title = f"{role} | {'comparison only' if saved else f'pick {count} IN ORDER'}"
        self.window = self.app.create_window(title, width, height, left, 60)
        self.widget = self.gui.SceneWidget()
        self.widget.scene = self.rendering.Open3DScene(self.window.renderer)
        configure_raw_rgb(self.widget.scene)
        self.widget.scene.set_background([0.12, 0.12, 0.12, 1])
        self.widget.scene.add_geometry("cloud", cloud, point_material(o3d, self.size), False)
        self.widget.setup_camera(60, cloud.get_axis_aligned_bounding_box(), cloud.get_center())
        self.toolbar = self.gui.Vert(4, self.gui.Margins(8, 4, 8, 4))
        size_row = self.gui.Horiz(8)
        size_row.add_child(self.gui.Label("Point size"))
        self.slider = self.gui.Slider(self.gui.Slider.DOUBLE)
        self.slider.set_limits(1, 20)
        self.slider.double_value = self.size
        self.slider.set_on_value_changed(self.set_size)
        size_row.add_child(self.slider)
        self.size_label = self.gui.Label(f"{self.size:g} px")
        size_row.add_child(self.size_label)
        self.toolbar.add_child(size_row)
        marker_row = self.gui.Horiz(8)
        marker_row.add_child(self.gui.Label("Marker size"))
        self.marker_slider = self.gui.Slider(self.gui.Slider.DOUBLE)
        self.marker_slider.set_limits(1, 30)
        self.marker_slider.double_value = self.marker_size
        self.marker_slider.set_on_value_changed(self.set_marker_size)
        marker_row.add_child(self.marker_slider)
        self.marker_size_label = self.gui.Label(f"{self.marker_size:g} px")
        marker_row.add_child(self.marker_size_label)
        self.toolbar.add_child(marker_row)
        action_row = self.gui.Horiz(8)
        self.status = self.gui.Label("")
        action_row.add_child(self.status)
        action_row.add_stretch()
        for text, callback in (("Undo", self.undo), ("Reset", self.reset)):
            button = self.gui.Button(text)
            button.enabled = saved is None
            button.set_on_clicked(callback)
            action_row.add_child(button)
        self.finish = self.gui.Button("Finish" if saved is None else "Close")
        self.finish.set_on_clicked(self.window.close)
        action_row.add_child(self.finish)
        self.toolbar.add_child(action_row)
        self.window.add_child(self.widget)
        self.window.add_child(self.toolbar)
        self.window.set_on_layout(self.layout)
        self.window.set_on_close(self.close)
        self.window.set_on_tick_event(self.tick)
        self.window.set_on_key(self.key)
        self.widget.set_on_mouse(self.mouse)
        self.refresh()
        if saved is None:
            self.log(
                f"[{role}] Shift+左键选点，Shift+右键撤销；"
                "Point size 调点云，Marker size 调标记。\n"
                f"[{role}] 对应点按编号同色高亮，数字使用深色底；未满 {count} 点关闭不会保存。"
            )
        else:
            self.log(f"[{role}] 只读对照窗口：保留已确认的彩色编号，不需要再次选点。")

    def layout(self, context):
        rect = self.window.content_rect
        height = int(max(96, 6 * self.window.theme.font_size))
        self.toolbar.frame = self.gui.Rect(rect.x, rect.y, rect.width, height)
        self.widget.frame = self.gui.Rect(
            rect.x, rect.y + height, rect.width, max(1, rect.height - height)
        )
        self.position_labels()

    def close(self):
        self.closed = True
        return True

    def key(self, event):
        if event.type != self.gui.KeyEvent.Type.DOWN:
            return False
        if event.key in (self.gui.KeyName.PLUS, self.gui.KeyName.EQUALS):
            self.set_size(min(20, self.size + 1))
            return True
        if event.key == self.gui.KeyName.MINUS:
            self.set_size(max(1, self.size - 1))
            return True
        if event.key in (self.gui.KeyName.Q, self.gui.KeyName.ESCAPE):
            self.window.close()
            return True
        return False

    def set_size(self, size):
        self.size = float(np.clip(size, 1, 20))
        if abs(self.slider.double_value - self.size) > 1e-8:
            self.slider.double_value = self.size
        self.size_label.text = f"{self.size:g} px"
        self.widget.scene.modify_geometry_material("cloud", point_material(self.o3d, self.size))
        self.window.set_needs_layout()
        self.widget.force_redraw()

    def set_marker_size(self, size):
        self.marker_size = float(np.clip(size, 1, 30))
        if abs(self.marker_slider.double_value - self.marker_size) > 1e-8:
            self.marker_slider.double_value = self.marker_size
        self.marker_size_label.text = f"{self.marker_size:g} px"
        if self.has_markers:
            self.widget.scene.modify_geometry_material(
                "picked", point_material(self.o3d, self.marker_size)
            )
        self.position_labels()
        self.window.set_needs_layout()
        self.widget.force_redraw()

    def undo(self):
        if self.saved is None:
            self.selection_revision += 1
            self.ordered.undo()
            self.refresh()

    def reset(self):
        if self.saved is None:
            self.selection_revision += 1
            self.ordered.clear()
            self.refresh()

    def state(self):
        camera = self.widget.scene.camera
        frame = self.widget.frame
        return (
            camera.get_view_matrix().tobytes(),
            camera.get_projection_matrix().tobytes(),
            frame.width,
            frame.height,
        )

    def tick(self):
        if self.closed:
            return False
        current = self.state()
        if current != self.camera_state:
            self.camera_state = current
            self.refresh_markers()
            return True
        return False

    def refresh(self):
        n = len(self.saved["points"]) if self.saved is not None else len(self.ordered.indices)
        suffix = " (read-only)" if self.saved else (
            f" | next #{n + 1}" if n < self.count else " | ready"
        )
        self.status.text = f"{n}/{self.count} selected" + suffix
        self.finish.enabled = self.saved is not None or n == self.count
        self.window.set_needs_layout()
        self.refresh_markers()

    def refresh_markers(self):
        if self.has_markers:
            self.widget.scene.remove_geometry("picked")
            self.has_markers = False
        points = self.saved["points"] if self.saved is not None else self.ordered.result()["points"]
        if len(points):
            marker = self.o3d.geometry.PointCloud()
            camera = self.widget.scene.camera.get_model_matrix()[:3, 3]
            marker.points = self.o3d.utility.Vector3dVector(marker_positions(points, camera))
            marker.colors = self.o3d.utility.Vector3dVector(pair_colors(len(points)))
            self.widget.scene.add_geometry(
                "picked", marker, point_material(self.o3d, self.marker_size), False
            )
            self.widget.scene.scene.set_geometry_priority("picked", 7)
            self.has_markers = True
        self.refresh_labels(points)
        self.widget.force_redraw()

    def refresh_labels(self, points):
        # Screen-space labels add contrast without tinting RGB or adding large
        # geometry to the depth image used for picking.
        self.label_points = np.asarray(points, dtype=float).reshape(-1, 3)
        colors = pair_colors(len(self.label_points))
        added = False
        while len(self.point_labels) < len(self.label_points):
            index = len(self.point_labels)
            label = self.gui.Label(str(index + 1))
            label.text_color = self.gui.Color(*colors[index])
            label.background_color = self.gui.Color(0.015, 0.015, 0.015, 1)
            self.window.add_child(label)
            self.point_labels.append(label)
            added = True
        self.position_labels()
        if added:
            self.window.set_needs_layout()

    def position_labels(self):
        for label in self.point_labels:
            label.visible = False
        if not len(self.label_points):
            return
        frame = self.widget.frame
        if frame.width <= 0 or frame.height <= 0:
            return
        camera = self.widget.scene.camera
        points = np.column_stack((self.label_points, np.ones(len(self.label_points))))
        clip = points @ (camera.get_projection_matrix() @ camera.get_view_matrix()).T
        valid = np.isfinite(clip).all(axis=1) & (clip[:, 3] > 0)
        ndc = np.zeros((len(clip), 3))
        np.divide(clip[:, :3], clip[:, 3:], out=ndc, where=valid[:, None])
        visible = valid & (np.abs(ndc) <= 1).all(axis=1)
        em = self.window.theme.font_size
        gap = max(6, self.marker_size / 2 + 3)
        for i, label in enumerate(self.point_labels[:len(self.label_points)]):
            if not visible[i]:
                continue
            width, height = int(max(26, (len(label.text) + 0.8) * em)), int(max(22, 1.6 * em))
            x = frame.x + (ndc[i, 0] + 1) * frame.width / 2
            y = frame.y + (1 - ndc[i, 1]) * frame.height / 2
            left = int(np.clip(x + gap, frame.x, frame.x + max(0, frame.width - width)))
            top = int(np.clip(y - height - gap, frame.y, frame.y + max(0, frame.height - height)))
            label.frame = self.gui.Rect(left, top, width, height)
            label.visible = True

    def mouse(self, event):
        ignored = self.gui.Widget.EventCallbackResult.IGNORED
        consumed = self.gui.Widget.EventCallbackResult.CONSUMED
        if event.type != self.gui.MouseEvent.Type.BUTTON_DOWN or not event.is_modifier_down(
            self.gui.KeyModifier.SHIFT
        ):
            return ignored
        if self.saved is not None:
            return consumed
        if event.is_button_down(self.gui.MouseButton.RIGHT):
            self.undo()
            return consumed
        if not event.is_button_down(self.gui.MouseButton.LEFT):
            return ignored
        if self.pending_pick or len(self.ordered.indices) == self.count:
            return consumed
        frame = self.widget.frame
        x, y = int(event.x - frame.x), int(event.y - frame.y)
        if not (0 <= x < frame.width and 0 <= y < frame.height):
            return consumed
        captured = self.state()
        revision = self.selection_revision
        self.pending_pick = True

        def depth_ready(image):
            if self.closed:
                return
            depth = np.asarray(image)
            value = float(depth[y, x]) if y < depth.shape[0] and x < depth.shape[1] else 1.0

            def apply():
                self.pending_pick = False
                if self.closed or revision != self.selection_revision:
                    return
                if captured != self.state():
                    self.log(f"[{self.role}] 视角已变化，本次选点取消，请稳定视角后重新点击。")
                    return
                if not np.isfinite(value) or value >= 1:
                    self.log(f"[{self.role}] 此处没有可见点，请点击点云上的像素。")
                    return
                world = self.widget.scene.camera.unproject(x, y, value, frame.width, frame.height)
                _, indices, _ = self.tree.search_knn_vector_3d(np.asarray(world).reshape(3), 1)
                if indices and self.ordered.add(indices[0]):
                    xyz = self.ordered.points[indices[0]].tolist()
                    self.log(f"[{self.role}] {len(self.ordered.indices)}/{self.count}: {xyz}")
                    self.refresh()

            self.app.post_to_main_thread(self.window, apply)

        self.widget.scene.scene.render_to_depth_image(depth_ready)
        return consumed


def run_picker(o3d, settings, cloud, role, count, slot=None, saved=None, log=print):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    picker = PointPicker(o3d, settings, cloud, role, count, slot, saved, log)
    app.run()
    return {
        **picker.ordered.result(),
        "point_size": picker.size,
        "marker_point_size": picker.marker_size,
    }
