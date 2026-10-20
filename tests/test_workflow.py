"""Small synthetic models and scripted interactions; no display or large user data."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from plyfile import PlyData

from gs_merger import pipeline
from gs_merger.interaction import Open3DInteraction, UserQuit
from gs_merger.io_utils import fingerprint, merge_streaming
from gs_merger.merge_worker import merge_isolated
from gs_merger.gaussian_io import (
    DC,
    NORMALS,
    ROT,
    SCALE,
    XYZ,
    columns,
    merge_gaussian_files,
    read_gaussians,
    set_columns,
    transform_gaussians,
    write_point_cloud,
    write_vertices,
)


def gaussian_fixture(count=9):
    names = XYZ + NORMALS + DC + [f"f_rest_{i}" for i in range(45)] + ["opacity"] + SCALE + ROT
    vertices = np.zeros(count, dtype=[(name, "f4") for name in names] + [("tag", "i4")])
    rng = np.random.default_rng(872)
    for name in XYZ + NORMALS + DC + [f"f_rest_{i}" for i in range(45)]:
        vertices[name] = rng.normal(size=count)
    set_columns(
        vertices[:5], XYZ, np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 1, 0.5], [-1, 0, 1]])
    )
    set_columns(vertices, SCALE, rng.uniform(-4, -2, size=(count, 3)))
    vertices["rot_0"] = 1
    vertices["opacity"] = 4
    vertices["tag"] = np.arange(count)
    return vertices


class ScriptedInteraction:
    def __init__(self, settings, log=print, actions=None, quit_role=None, wrong_once=False):
        self.settings, self.log = settings, log
        self.actions = iter(actions or ["a"] * 10)
        self.quit_role, self.wrong_once = quit_role, wrong_once
        self.selected, self.previews, self.qualified = [], [], []

    def select(self, path, role, count):
        self.selected.append(role)
        if role == self.quit_role:
            raise UserQuit()
        points = np.asarray(PlyData.read(path)["vertex"].data[XYZ].tolist())
        indices = list(range(count))
        if self.wrong_once and "source" in role:
            indices[-1] = count
            self.wrong_once = False
        return {"indices": indices, "points": points[indices].tolist()}

    def preview(self, source, target, transform):
        self.previews.append((source, target, transform.copy()))

    def select_pair(self, paths, count, number, selections, on_save):
        selections = dict(selections)
        for role in ("target", "source"):
            if selections[role] is None:
                selections[role] = self.select(paths[role], f"Round {number}: {role}", count)
        on_save(dict(selections))
        return selections

    def review(self, qualified):
        self.qualified.append(qualified)
        return next(self.actions)

    def choice(self, prompt, allowed):
        action = next(self.actions)
        if action not in allowed:
            raise AssertionError(f"Unexpected action {action} for {allowed}")
        return action


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.models, self.transforms, self.originals = [], [], []
        anchor = gaussian_fixture()
        for number in range(4):
            angle = number * 0.23
            transform = np.eye(4)
            transform[:3, :3] = (1 + number * 0.4) * np.array(
                [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
            )
            transform[:3, 3] = [number * 0.4, -number * 0.3, number * 0.15]
            vertices = transform_gaussians(anchor[: 6 + number], np.linalg.inv(transform))
            vertices["tag"] += number * 100
            path = self.root / f"model{number + 1}.ply"
            write_vertices(path, vertices)
            self.models.append(str(path))
            self.transforms.append(transform)
            self.originals.append(vertices)
        self.settings = {
            **pipeline.DEFAULTS,
            "models": self.models,
            "output_dir": str(self.root / "output"),
            "pairs_per_step": [4, 5, 4],
            "control_threshold": 1e-5,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def run_with(self, ui, settings=None):
        return pipeline.run_workflow(
            settings or self.settings, interaction_factory=lambda *a, **k: ui, log=lambda text: None
        )

    def test_four_models_follow_order_preserve_anchor_and_full_gaussian_attributes(self):
        ui = ScriptedInteraction(self.settings)
        final = self.run_with(ui)
        out = Path(self.settings["output_dir"])
        result = read_gaussians(final)
        self.assertEqual(len(result), 6 + 7 + 8 + 9)
        self.assertEqual(final, out / "steps/003/merged.ply")
        self.assertTrue(final.is_file())
        self.assertFalse(final.is_symlink())
        self.assertFalse((out / "final.ply").is_symlink())
        self.assertFalse((out / "final.ply").exists())
        np.testing.assert_array_equal(result[:6], self.originals[0])
        ledger = json.loads((out / "final.json").read_text())
        self.assertEqual(ledger["merged"], str(final))
        self.assertEqual([r["model"] for r in ledger["models"]], self.models)
        for entry, expected in zip(ledger["models"], self.transforms):
            np.testing.assert_allclose(entry["transform_to_reference"], expected, atol=1e-6)
        offset = 0
        for original, transform in zip(self.originals, self.transforms):
            expected = transform_gaussians(original, transform)
            section = result[offset : offset + len(original)]
            for name in original.dtype.names:
                np.testing.assert_allclose(section[name], expected[name], atol=2e-6)
            offset += len(original)
        for number in (2, 3):
            context = json.loads((out / f"steps/{number:03d}/context.json").read_text())
            self.assertEqual(
                context["target"]["path"],
                str(out / f"steps/{number - 1:03d}/merged.ply"),
            )
        for original, path in zip(self.originals, self.models):
            np.testing.assert_array_equal(read_gaussians(path), original)

    def test_resume_skips_completed_round_and_reselects_unfinished_round(self):
        first_ui = ScriptedInteraction(self.settings, quit_role="Round 2: source")
        with self.assertRaises(UserQuit):
            self.run_with(first_ui)
        out = Path(self.settings["output_dir"])
        state = json.loads((out / "session.json").read_text())
        self.assertEqual(len(state["completed"]), 1)
        self.assertFalse((out / "steps/002/selections.json").exists())
        merged_before = fingerprint(out / "steps/001/merged.ply")
        second_ui = ScriptedInteraction(self.settings)
        final = self.run_with(second_ui)
        self.assertEqual(fingerprint(out / "steps/001/merged.ply"), merged_before)
        self.assertNotIn("Round 1: target", second_ui.selected)
        self.assertIn("Round 2: target", second_ui.selected)
        self.assertIn("Round 2: source", second_ui.selected)
        self.assertEqual(len(read_gaussians(final)), 30)
        # A third run on a completed session needs no interaction at all.
        factory = Mock(side_effect=AssertionError("GUI should not open"))
        pipeline.run_workflow(self.settings, interaction_factory=factory, log=lambda text: None)
        factory.assert_not_called()

    def test_reselect_both_after_bad_pair_does_not_merge_rejected_fit(self):
        settings = {
            **self.settings,
            "models": self.models[:2],
            "pairs_per_step": [4],
            "min_inlier_ratio": 1.0,
        }
        ui = ScriptedInteraction(settings, actions=["b", "a"], wrong_once=True)
        with patch("gs_merger.pipeline.merge_isolated", wraps=merge_isolated) as merge:
            final = self.run_with(ui, settings)
        self.assertEqual(ui.qualified, [False, True])
        self.assertEqual(ui.selected, ["Round 1: target", "Round 1: source"] * 2)
        self.assertEqual(merge.call_count, 1)
        self.assertEqual(len(read_gaussians(final)), 13)

    def test_resume_refuses_changed_config_or_input(self):
        self.run_with(ScriptedInteraction(self.settings))
        changed = {**self.settings, "pairs_per_step": [5, 5, 5]}
        with self.assertRaisesRegex(ValueError, "配置"):
            self.run_with(ScriptedInteraction(changed), changed)
        model = Path(self.models[1])
        with model.open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "模型"):
            self.run_with(ScriptedInteraction(self.settings))

    def test_resume_refuses_missing_completed_output(self):
        self.run_with(ScriptedInteraction(self.settings))
        (Path(self.settings["output_dir"]) / "steps/001/merged.ply").unlink()
        with self.assertRaisesRegex(ValueError, "输出缺失"):
            self.run_with(ScriptedInteraction(self.settings))

    def test_output_lock_prevents_two_writers(self):
        with pipeline.output_lock(Path(self.settings["output_dir"])):
            with self.assertRaisesRegex(RuntimeError, "Another workflow"):
                with pipeline.output_lock(Path(self.settings["output_dir"])):
                    pass

    def test_pipeline_refuses_a_partial_checkpoint_even_if_ui_requests_it(self):
        ui = ScriptedInteraction(self.settings)

        def partial(paths, count, number, selections, on_save):
            target = ui.select(paths["target"], "target", count)
            on_save({"target": target, "source": None})

        ui.select_pair = partial
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.run_with(ui)
        step = Path(self.settings["output_dir"]) / "steps/001"
        self.assertFalse((step / "selections.json").exists())
        self.assertFalse((step / "merged.ply").exists())

    def test_point_size_command_line_sets_both_windows_initial_size(self):
        args = pipeline.build_parser().parse_args(
            [
                "--models",
                *self.models,
                "--output-dir",
                str(self.root / "out"),
                "--pairs",
                "4",
                "--point-size",
                "6",
                "--marker-point-size",
                "12",
            ]
        )
        settings = pipeline.load_settings(args)
        self.assertEqual(settings["point_size"], 6)
        self.assertEqual(settings["marker_point_size"], 12)

    def test_old_session_accepts_new_marker_settings_without_repeating_merges(self):
        self.run_with(ScriptedInteraction(self.settings))
        out = Path(self.settings["output_dir"])
        path = out / "session.json"
        state = json.loads(path.read_text())
        for key in ("marker_size", "marker_point_size", "pick_window_width", "pick_window_height"):
            state["settings"].pop(key)
        path.write_text(json.dumps(state))
        updated = {
            **self.settings, "marker_size": 0.00005, "point_size": 3.0, "marker_point_size": 12,
        }
        factory = Mock(side_effect=AssertionError("Completed rounds must not reopen GUI"))
        pipeline.run_workflow(updated, interaction_factory=factory, log=lambda text: None)
        factory.assert_not_called()
        self.assertEqual(json.loads(path.read_text())["settings"]["marker_size"], 0.00005)
        self.assertEqual(json.loads(path.read_text())["settings"]["marker_point_size"], 12)

    def test_old_pending_session_discards_partial_pairs_after_display_update(self):
        with self.assertRaises(UserQuit):
            self.run_with(ScriptedInteraction(self.settings, quit_role="Round 2: source"))
        out = Path(self.settings["output_dir"])
        completed = fingerprint(out / "steps/001/merged.ply")
        path = out / "session.json"
        state = json.loads(path.read_text())
        target = PlyData.read(out / "steps/002/target_view.ply")["vertex"].data
        (out / "steps/002/selections.json").write_text(
            json.dumps(
                {
                    "target": {
                        "indices": list(range(5)),
                        "points": columns(target[:5], XYZ).tolist(),
                    },
                    "source": None,
                }
            )
        )
        for key in ("marker_size", "marker_point_size", "pick_window_width", "pick_window_height"):
            state["settings"].pop(key)
        path.write_text(json.dumps(state))
        updated = {**self.settings, "marker_size": 0.00002, "pick_window_width": 640}
        ui = ScriptedInteraction(updated)
        final = self.run_with(ui, updated)
        self.assertEqual(fingerprint(out / "steps/001/merged.ply"), completed)
        self.assertIn("Round 2: target", ui.selected)
        self.assertIn("Round 2: source", ui.selected)
        self.assertEqual(len(read_gaussians(final)), 30)

    def test_streaming_merge_reorders_fields_and_never_replaces_input(self):
        source = self.originals[1]
        reordered = np.empty(
            len(source), dtype=[(name, source.dtype[name]) for name in source.dtype.names[::-1]]
        )
        for name in source.dtype.names:
            reordered[name] = source[name]
        source_path, target = self.root / "reordered.ply", Path(self.models[0])
        write_vertices(source_path, reordered)
        output = self.root / "merged.ply"
        with patch("gs_merger.gaussian_io.CHUNK_SIZE", 2):
            merge_streaming(source_path, target, self.transforms[1], output)
        merged = read_gaussians(output)
        self.assertEqual(merged.dtype.names, self.originals[0].dtype.names)
        np.testing.assert_array_equal(merged[:6], self.originals[0])
        np.testing.assert_allclose(
            columns(merged[6:], XYZ), columns(gaussian_fixture()[:7], XYZ), atol=1e-6
        )
        with self.assertRaises(ValueError):
            merge_streaming(source_path, target, self.transforms[1], target)

    def test_failed_streaming_merge_keeps_previous_complete_output(self):
        output = self.root / "merged.ply"
        output.write_bytes(b"previous completed result")
        calls = 0

        def fail_after_one_chunk(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated disk / transform failure")
            return transform_gaussians(*args, **kwargs)

        with patch("gs_merger.gaussian_io.CHUNK_SIZE", 2), patch(
            "gs_merger.gaussian_io.transform_gaussians", side_effect=fail_after_one_chunk
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                merge_streaming(self.models[1], self.models[0], self.transforms[1], output)
        self.assertEqual(output.read_bytes(), b"previous completed result")
        self.assertEqual(list(self.root.glob(".merge_*")), [])

    def test_merge_progress_finishes_only_after_committing_complete_valid_ply(self):
        output = self.root / "merged.ply"
        messages = []

        def progress(message):
            messages.append(message)
            if "合并完成：" in message:
                result = read_gaussians(output)
                self.assertEqual(len(result), 13)
                np.testing.assert_array_equal(result[:6], self.originals[0])
                expected = transform_gaussians(self.originals[1], self.transforms[1])
                for name in result.dtype.names:
                    np.testing.assert_array_equal(result[6:][name], expected[name])

        with patch("gs_merger.gaussian_io.CHUNK_SIZE", 2):
            merge_streaming(
                self.models[1], self.models[0], self.transforms[1], output, log=progress
            )
        self.assertTrue(any("已有模型分块复制" in message for message in messages))
        self.assertTrue(any("新模型分块变换" in message for message in messages))
        self.assertTrue(any("高斯准备进度 100.0%" in message for message in messages))
        self.assertIn("合并完成：", messages[-1])

    def test_invalid_attribute_in_either_stream_preserves_previous_output(self):
        for invalid_role, invalid_field in ((0, "f_rest_44"), (1, "rot_0")):
            output = self.root / "merged.ply"
            output.write_bytes(b"previous completed result")
            paths = list(self.models[:2])
            broken = self.originals[invalid_role].copy()
            broken[invalid_field][-1] = np.nan
            path = self.root / "invalid.ply"
            write_vertices(path, broken)
            paths[invalid_role] = str(path)
            with patch("gs_merger.gaussian_io.CHUNK_SIZE", 2):
                with self.assertRaisesRegex(ValueError, "Non-finite Gaussian attribute"):
                    merge_streaming(paths[1], paths[0], self.transforms[1], output)
            self.assertEqual(output.read_bytes(), b"previous completed result")
            self.assertEqual(list(self.root.glob(".merge_*")), [])

    def test_atomic_adapter_matches_local_gaussian_backend_byte_for_byte(self):
        reference = merge_gaussian_files(
            self.models[1], self.models[0], self.transforms[1], self.root / "old_backend"
        )
        output = self.root / "adapter.ply"
        report = merge_streaming(self.models[1], self.models[0], self.transforms[1], output)
        self.assertEqual(output.read_bytes(), Path(reference["merged"]).read_bytes())
        self.assertEqual(report["merged_gaussians"], reference["merged_count"])
        self.assertEqual(list(self.root.glob(".merge_*")), [])

    def test_config_paths_and_shell_wrapper_work_outside_repo_with_spaces(self):
        folder = self.root / "配置 with spaces"
        folder.mkdir()
        config = folder / "models.json"
        config.write_text(
            json.dumps(
                {
                    "models": ["../model1.ply", "../model2.ply"],
                    "output_dir": "../checked output",
                    "pairs_per_step": 4,
                }
            ),
            encoding="utf-8",
        )
        script = Path(__file__).resolve().parents[1] / "stitch.sh"
        result = subprocess.run(
            ["bash", str(script), "--python", sys.executable, "--config", str(config), "--check"],
            cwd=folder,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Check passed", result.stdout)
        self.assertFalse((self.root / "checked output").exists())

    def test_config_rejects_wrong_round_counts_and_mixed_sh_before_gui(self):
        args = pipeline.build_parser().parse_args(
            ["--models", *self.models, "--output-dir", str(self.root / "other"), "--pairs", "2"]
        )
        with self.assertRaises(ValueError):
            pipeline.load_settings(args)
        vertices = self.originals[1]
        degree_zero = np.empty(
            len(vertices),
            dtype=[
                (name, vertices.dtype[name])
                for name in vertices.dtype.names
                if not name.startswith("f_rest_")
            ],
        )
        for name in degree_zero.dtype.names:
            degree_zero[name] = vertices[name]
        write_vertices(self.models[1], degree_zero)
        factory = Mock()
        with self.assertRaisesRegex(ValueError, "SH degree"):
            pipeline.run_workflow(self.settings, interaction_factory=factory, log=lambda text: None)
        factory.assert_not_called()
        self.assertFalse(Path(self.settings["output_dir"]).exists())


class InteractionTests(unittest.TestCase):
    def interaction(self, answers):
        ui = Open3DInteraction.__new__(Open3DInteraction)
        ui.ask = Mock(side_effect=answers)
        ui.log = lambda text: None
        return ui

    def test_close_incomplete_and_reject_valid_selection_can_reopen(self):
        ui = self.interaction(["r", "r", "y"])
        valid = {"indices": [2, 3, 4], "points": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}
        ui.pick_once = Mock(
            side_effect=[{"indices": [], "points": []}, valid, copy.deepcopy(valid)]
        )
        result = ui.select("unused.ply", "source", 3)
        self.assertEqual(result, valid)
        self.assertEqual(ui.pick_once.call_count, 3)

    def test_close_window_can_quit_and_rejected_fit_has_no_accept_option(self):
        ui = self.interaction(["q"])
        ui.pick_once = Mock(return_value={"indices": [], "points": []})
        with self.assertRaises(UserQuit):
            ui.select("unused.ply", "target", 3)
        ui = self.interaction(["a", "t"])
        self.assertEqual(ui.review(False), "t")

    def test_picker_delegates_to_marker_gui(self):
        ui = self.interaction([])
        ui.o3d = SimpleNamespace()
        ui.settings = dict(pipeline.DEFAULTS)
        ui.load_cloud = Mock(return_value=SimpleNamespace(points=np.zeros((5, 3))))
        expected = {"indices": [0, 1, 2], "points": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}
        with patch("gs_merger.point_picker.run_picker", return_value=expected) as picker:
            selected = ui.pick_once("unused.ply", "source", 3)
        self.assertEqual(selected, expected)
        self.assertEqual(picker.call_args.args[4:6], (3, None))

    def test_duplicate_coordinates_can_be_reselected_without_leaving_workflow(self):
        ui = self.interaction(["r", "y"])
        ui.pick_once = Mock(
            side_effect=[
                {"indices": [1, 2, 3], "points": [[0, 0, 0], [0, 0, 0], [1, 2, 3]]},
                {"indices": [1, 2, 3], "points": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]},
            ]
        )
        selected = ui.select("unused.ply", "target", 3)
        self.assertEqual(len(np.unique(selected["points"], axis=0)), 3)

    def test_real_open3d_loads_rgb_and_caps_preview_without_gui(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "view.ply"
            points = np.arange(150, dtype=float).reshape(-1, 3)
            write_point_cloud(path, points, np.tile([1, 0, 0], (50, 1)))
            ui = Open3DInteraction({"seed": 123})
            cloud = ui.load_cloud(path, maximum=7)
            self.assertEqual(len(cloud.points), 7)
            self.assertTrue(all(any(np.array_equal(p, q) for q in points) for p in cloud.points))
            np.testing.assert_array_equal(np.asarray(cloud.colors), np.tile([1, 0, 0], (7, 1)))


if __name__ == "__main__":
    unittest.main()
