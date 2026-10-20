"""Opt-in EGL pixel check; ordinary tests never initialize a rendering engine."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from gs_merger.point_picker import PointPicker, configure_raw_rgb, point_material


@unittest.skipUnless(os.environ.get("GS_MERGER_TEST_RENDERING") == "1", "opt-in EGL test")
class RGBRenderingTests(unittest.TestCase):
    def test_all_gray_levels_and_color_samples_render_without_recoloring(self):
        import open3d as o3d

        # Every 8-bit gray value, primary/secondary colors and mixed RGB samples.
        grays = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
        colors = np.array(
            [[r, g, b] for r in (0, 85, 170, 255) for g in (0, 255) for b in (0, 255)],
            dtype=np.uint8,
        )
        mixed = np.random.default_rng(123).integers(0, 256, (16, 3), dtype=np.uint8)
        expected = np.vstack((grays, colors, mixed))
        width, height = 256, 288
        xs = 8 + 16 * (np.arange(len(expected)) % 16)
        ys = 8 + 16 * (np.arange(len(expected)) // 16)
        xyz = np.column_stack((xs, height - ys, np.zeros(len(expected))))
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        cloud.colors = o3d.utility.Vector3dVector(expected.astype(float) / 255)
        original_xyz = np.asarray(cloud.points).copy()
        original_rgb = np.asarray(cloud.colors).copy()
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        scene = renderer.scene
        configure_raw_rgb(scene)
        scene.set_background([0.12, 0.12, 0.12, 1])
        scene.add_geometry("cloud", cloud, point_material(o3d, 7), False)
        scene.camera.look_at([128, 144, 0], [128, 144, 1], [0, 1, 0])
        scene.camera.set_projection(
            o3d.visualization.rendering.Camera.Projection.Ortho,
            -width / 2, width / 2, -height / 2, height / 2, 0.1, 10,
        )
        # Check both the initial material and the path used by the size slider.
        for size in (7, 3):
            scene.modify_geometry_material("cloud", point_material(o3d, size))
            rendered = np.asarray(renderer.render_to_image())
            actual = rendered[ys, xs, :3]
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(np.asarray(cloud.points), original_xyz)
            np.testing.assert_array_equal(np.asarray(cloud.colors), original_rgb)

        # The numbered badges use the real renderer's projection, but create no
        # native UI. They stay next to the same screen points after projection.
        picker = PointPicker.__new__(PointPicker)
        picker.marker_size = 6
        picker.point_labels = []
        picker.gui = SimpleNamespace(
            Label=lambda text: SimpleNamespace(text=text, visible=False),
            Color=lambda *values: tuple(values),
            Rect=lambda x, y, width, height: SimpleNamespace(x=x, y=y, width=width, height=height),
        )
        picker.window = Mock()
        picker.window.theme.font_size = 14
        picker.widget = SimpleNamespace(
            scene=scene, frame=SimpleNamespace(x=0, y=0, width=width, height=height)
        )
        picker.refresh_labels(xyz[[0, 32]])
        self.assertTrue(all(label.visible for label in picker.point_labels))
        for label, index in zip(picker.point_labels, (0, 32)):
            self.assertAlmostEqual(label.frame.x, int(xs[index]) + 6, delta=1)
            expected_y = max(0, int(ys[index]) - label.frame.height - 6)
            self.assertAlmostEqual(label.frame.y, expected_y, delta=1)


if __name__ == "__main__":
    unittest.main()
