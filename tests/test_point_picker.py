"""Picking math, ordering and materials without creating a GUI/rendering engine."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from gs_merger.point_picker import (
    OrderedPoints,
    PointPicker,
    marker_positions,
    pair_colors,
    point_material,
)


class PointPickerTests(unittest.TestCase):
    def test_click_order_limit_undo_and_reset_keep_original_xyz(self):
        xyz = np.array([[2, 3, 4], [1, 0, 0], [9, 8, 7], [2, 3, 4]], dtype=float)
        selected = OrderedPoints(xyz, 3)
        self.assertTrue(selected.add(2))
        self.assertTrue(selected.add(0))
        self.assertFalse(selected.add(0))
        self.assertFalse(selected.add(3))  # Different index, same physical coordinate.
        self.assertTrue(selected.add(1))
        self.assertEqual(selected.result()["indices"], [2, 0, 1])
        np.testing.assert_array_equal(selected.result()["points"], xyz[[2, 0, 1]])
        self.assertFalse(selected.add(3))
        selected.undo()
        self.assertEqual(selected.indices, [2, 0])
        selected.clear()
        self.assertEqual(selected.result(), {"indices": [], "points": []})

    def test_marker_bias_preserves_projection_and_never_changes_control_coordinates(self):
        points = np.array([[1, 2, 8], [3, 4, 15]], dtype=float)
        original = points.copy()
        camera = np.array([-2, 1, 0])
        rendered = marker_positions(points, camera)
        before, after = points - camera, rendered - camera
        np.testing.assert_allclose(before[:, :2] / before[:, 2:], after[:, :2] / after[:, 2:])
        np.testing.assert_array_equal(points, original)

    def test_cloud_and_marker_sliders_change_only_their_own_material(self):
        import open3d as o3d

        self.assertEqual(point_material(o3d, 4).point_size, 4)
        picker = PointPicker.__new__(PointPicker)
        picker.o3d = o3d
        picker.slider = SimpleNamespace(double_value=2.0)
        picker.size_label = SimpleNamespace(text="")
        picker.marker_size = 6
        picker.marker_slider = SimpleNamespace(double_value=6.0)
        picker.marker_size_label = SimpleNamespace(text="")
        picker.widget = Mock()
        picker.window = Mock()
        picker.position_labels = Mock()
        picker.has_markers = True
        picker.set_size(7)
        changes = picker.widget.scene.modify_geometry_material.call_args_list
        self.assertEqual([call.args[0] for call in changes], ["cloud"])
        self.assertEqual([call.args[1].point_size for call in changes], [7])
        self.assertTrue(all(call.args[1].shader == "defaultUnlit" for call in changes))
        self.assertTrue(all(not call.args[1].sRGB_color for call in changes))
        for call in changes:
            np.testing.assert_array_equal(call.args[1].base_color, [1, 1, 1, 1])
        self.assertEqual(picker.size_label.text, "7 px")
        self.assertEqual(picker.marker_size, 6)
        picker.widget.scene.modify_geometry_material.reset_mock()
        picker.set_marker_size(12)
        name, material = picker.widget.scene.modify_geometry_material.call_args.args
        self.assertEqual(name, "picked")
        self.assertEqual(material.point_size, 12)
        self.assertEqual(picker.size, 7)
        self.assertEqual(picker.slider.double_value, 7)
        self.assertEqual(picker.marker_size_label.text, "12 px")

    def test_saved_markers_are_point_clouds_with_an_independent_material_size(self):
        import open3d as o3d

        picker = PointPicker.__new__(PointPicker)
        picker.o3d = o3d
        picker.size = 5
        picker.marker_size = 9
        picker.saved = {"points": [[1, 2, 3], [4, 5, 6]]}
        picker.has_markers = False
        picker.widget = Mock()
        picker.widget.scene.camera.get_model_matrix.return_value = np.eye(4)
        picker.refresh_labels = Mock()
        picker.refresh_markers()
        name, cloud, material, _ = picker.widget.scene.add_geometry.call_args.args
        self.assertEqual(name, "picked")
        self.assertIsInstance(cloud, o3d.geometry.PointCloud)
        self.assertEqual(material.point_size, 9)
        np.testing.assert_array_equal(np.asarray(cloud.colors), pair_colors(2))
        np.testing.assert_array_equal(picker.saved["points"], [[1, 2, 3], [4, 5, 6]])

    def test_pair_colors_are_distinct_bright_and_stable_after_undo_or_in_the_other_window(self):
        colors = pair_colors(30)
        self.assertEqual(len(np.unique(colors, axis=0)), 30)
        self.assertTrue(np.all((colors >= 0) & (colors <= 1)))
        self.assertTrue(np.all(colors.max(axis=1) >= 0.95))
        np.testing.assert_array_equal(pair_colors(6), colors[:6])

    def test_numbered_badges_follow_camera_and_reset_without_changing_model_rgb(self):
        picker = PointPicker.__new__(PointPicker)
        picker.marker_size = 6
        picker.point_labels = []
        picker.gui = SimpleNamespace(
            Label=lambda text: SimpleNamespace(text=text, visible=True),
            Color=lambda *values: tuple(values),
            Rect=lambda x, y, width, height: SimpleNamespace(x=x, y=y, width=width, height=height),
        )
        picker.window, picker.widget = Mock(), Mock()
        picker.window.theme.font_size = 14
        picker.widget.frame = SimpleNamespace(x=20, y=64, width=400, height=300)
        picker.widget.scene.camera.get_projection_matrix.return_value = np.eye(4)
        picker.widget.scene.camera.get_view_matrix.return_value = np.eye(4)
        points = np.array([[0, 0, 0], [0.5, 0.5, 0], [10, 0, 0]], dtype=float)
        original = points.copy()
        picker.refresh_labels(points)
        self.assertEqual([label.text for label in picker.point_labels], ["1", "2", "3"])
        np.testing.assert_array_equal(picker.point_labels[0].text_color, pair_colors(1)[0])
        self.assertEqual(picker.point_labels[0].background_color, (0.015, 0.015, 0.015, 1))
        self.assertEqual([label.visible for label in picker.point_labels], [True, True, False])
        before = picker.point_labels[0].frame.x
        view = np.eye(4)
        view[0, 3] = 0.25
        picker.widget.scene.camera.get_view_matrix.return_value = view
        picker.position_labels()
        self.assertGreater(picker.point_labels[0].frame.x, before)
        picker.refresh_labels(points[:1])
        self.assertEqual([label.visible for label in picker.point_labels], [True, False, False])
        picker.refresh_labels([])
        self.assertTrue(all(not label.visible for label in picker.point_labels))
        np.testing.assert_array_equal(points, original)

    def depth_picker(self):
        import open3d as o3d

        picker = PointPicker.__new__(PointPicker)
        picker.o3d, picker.gui = o3d, o3d.visualization.gui
        picker.saved = None
        picker.role, picker.count = "source", 3
        picker.closed, picker.pending_pick, picker.selection_revision = False, False, 0
        points = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0]], dtype=float)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        picker.cloud = cloud  # Keep the indexed geometry alive, as the real picker does.
        picker.ordered = OrderedPoints(points, 3)
        picker.tree = o3d.geometry.KDTreeFlann(cloud)
        picker.widget, picker.app = Mock(), Mock()
        picker.widget.frame = SimpleNamespace(x=0, y=64, width=100, height=100)
        picker.widget.scene.camera.get_view_matrix.return_value = np.eye(4)
        picker.widget.scene.camera.get_projection_matrix.return_value = np.eye(4)
        picker.widget.scene.camera.unproject.return_value = [1.001, 2.001, 3.001]
        picker.window = object()
        picker.app.post_to_main_thread.side_effect = lambda window, callback: callback()
        picker.refresh, picker.log = Mock(), Mock()
        event = SimpleNamespace(
            type=picker.gui.MouseEvent.Type.BUTTON_DOWN,
            x=10,
            y=74,
            is_modifier_down=lambda modifier: True,
            is_button_down=lambda button: button == picker.gui.MouseButton.LEFT,
        )
        return picker, event

    def test_depth_pick_snaps_to_exact_original_vertex(self):
        picker, event = self.depth_picker()
        picker.widget.scene.scene.render_to_depth_image.side_effect = lambda callback: callback(
            np.full((100, 100), 0.5)
        )
        picker.mouse(event)
        self.assertEqual(picker.ordered.result(), {"indices": [0], "points": [[1, 2, 3]]})
        self.assertFalse(picker.pending_pick)

    def test_background_and_camera_movement_never_add_a_point(self):
        for camera_moved in (False, True):
            picker, event = self.depth_picker()

            def depth(callback):
                if camera_moved:
                    picker.widget.scene.camera.get_view_matrix.return_value = np.eye(4) * 2
                callback(np.full((100, 100), 0.5 if camera_moved else 1.0))

            picker.widget.scene.scene.render_to_depth_image.side_effect = depth
            picker.mouse(event)
            self.assertEqual(picker.ordered.indices, [])

    def test_undo_while_depth_request_is_pending_cancels_the_late_result(self):
        picker, event = self.depth_picker()
        pending = []
        picker.widget.scene.scene.render_to_depth_image.side_effect = pending.append
        picker.mouse(event)
        picker.undo()
        pending[0](np.full((100, 100), 0.5))
        self.assertEqual(picker.ordered.indices, [])


if __name__ == "__main__":
    unittest.main()
