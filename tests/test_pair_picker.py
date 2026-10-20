"""Two native windows are coordinated through isolated workers; no real GUI opens."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from gs_merger import picker_worker
from gs_merger.interaction import Open3DInteraction, UserQuit
from gs_merger.pipeline import DEFAULTS


PICKS = {"indices": [0, 2, 4], "points": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}


class FakeProcess:
    def __init__(self, status=0):
        self.status = status
        self.terminate = Mock(side_effect=lambda: self.set_status(-15))
        self.kill = Mock(side_effect=lambda: self.set_status(-9))

    def set_status(self, status):
        self.status = status

    def poll(self):
        return self.status

    def wait(self, timeout=None):
        return self.status


class PairPickerTests(unittest.TestCase):
    def ui(self, answers):
        ui = Open3DInteraction.__new__(Open3DInteraction)
        ui.settings = dict(DEFAULTS)
        ui.ask = Mock(side_effect=answers)
        ui.log = lambda text: None
        return ui

    def launcher(self, requests, processes, target_empty_once=False, fail=False):
        empty_sent = False

        def launch(command, **kwargs):
            nonlocal empty_sent
            request = json.loads(Path(command[command.index("--request") + 1]).read_text())
            result = Path(command[command.index("--result") + 1])
            requests.append(request)
            if fail and len(requests) == 1:
                result.write_text(json.dumps({"error": "simulated OpenGL failure"}))
                process = FakeProcess(1)
            elif fail or request["saved"] is not None:
                process = FakeProcess(None)
            else:
                picks = PICKS
                if target_empty_once and "target" in request["role"] and not empty_sent:
                    empty_sent = True
                    result.write_text(json.dumps({"incomplete": True, "picked_count": 0}))
                else:
                    result.write_text(json.dumps({"selection": picks}))
                process = FakeProcess(0)
            processes.append(process)
            return process

        return launch

    def test_both_windows_launch_before_confirmation_and_only_complete_pair_is_saved(self):
        requests, processes = [], []
        ui = self.ui([])
        answers = iter(["y", "y"])

        def ask(prompt):
            self.assertGreaterEqual(len(requests), 2)
            return next(answers)

        ui.ask = ask
        saved = []
        with patch(
            "gs_merger.interaction.subprocess.Popen",
            side_effect=self.launcher(requests, processes),
        ):
            selected = ui.select_pair(
                {"target": Path("left.ply"), "source": Path("right.ply")},
                3,
                1,
                {"target": None, "source": None},
                on_save=saved.append,
            )
        self.assertEqual([r["slot"] for r in requests[:2]], [0, 1])
        self.assertEqual([r["saved"] for r in requests[:2]], [None, None])
        self.assertEqual(selected, {"target": PICKS, "source": PICKS})
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0], selected)
        # Both windows were already closed: no redundant comparison window.
        self.assertEqual(len(requests), 2)

    def test_closing_incomplete_side_reopens_it_without_restarting_other_picking_side(self):
        requests, processes = [], []
        ui = self.ui([])
        answers = iter(["r", "y", "y"])
        original_launcher = self.launcher(requests, processes, target_empty_once=True)

        def launch(command, **kwargs):
            process = original_launcher(command, **kwargs)
            if len(requests) == 2:
                process.status = None  # source is still being picked during target's retry.
            return process

        def ask(prompt):
            answer = next(answers)
            if answer == "y":
                processes[1].status = 0
            return answer

        ui.ask = ask
        with patch(
            "gs_merger.interaction.subprocess.Popen",
            side_effect=launch,
        ):
            selected = ui.select_pair(
                {"target": Path("left.ply"), "source": Path("right.ply")},
                3,
                1,
                {"target": None, "source": None},
                on_save=lambda data: None,
            )
        target_picks = [r for r in requests if "target" in r["role"] and r["saved"] is None]
        source_picks = [r for r in requests if "source" in r["role"] and r["saved"] is None]
        self.assertEqual(len(target_picks), 2)
        self.assertEqual(len(source_picks), 1)
        self.assertEqual(selected["target"], PICKS)

    def test_one_retry_reopens_both_closed_incomplete_windows_including_close_during_prompt(self):
        for close_during_prompt in (False, True):
            with self.subTest(close_during_prompt=close_during_prompt):
                requests, processes, saved = [], [], []
                ui = self.ui([])
                answers = iter(["r", "y", "y"])
                ten = {"indices": list(range(10)), "points": [[i, i % 3, 0] for i in range(10)]}

                def launch(command, **kwargs):
                    request = json.loads(Path(command[command.index("--request") + 1]).read_text())
                    result = Path(command[command.index("--result") + 1])
                    requests.append(request)
                    if len(requests) <= 2:
                        result.write_text(json.dumps({
                            "incomplete": True, "picked_count": 5 if request["slot"] == 0 else 6,
                        }))
                        status = None if close_during_prompt and request["slot"] == 1 else 0
                    else:
                        result.write_text(json.dumps({"selection": ten}))
                        status = 0
                    process = FakeProcess(status)
                    processes.append(process)
                    return process

                def ask(prompt):
                    answer = next(answers)
                    if answer == "r":
                        self.assertEqual(len(requests), 2)
                        processes[1].status = 0
                    else:
                        self.assertEqual(len(requests), 4)  # Both reopened before any more input.
                    return answer

                ui.ask = Mock(side_effect=ask)
                with patch("gs_merger.interaction.subprocess.Popen", side_effect=launch):
                    selected = ui.select_pair(
                        {"target": Path("merge1.ply"), "source": Path("road7.ply")},
                        10, 2, {"target": None, "source": None}, on_save=saved.append,
                    )
                self.assertEqual(ui.ask.call_count, 3)
                self.assertEqual([r["slot"] for r in requests], [0, 1, 0, 1])
                self.assertEqual([r["path"] for r in requests], [
                    "merge1.ply", "road7.ply", "merge1.ply", "road7.ply",
                ])
                self.assertTrue(all(r["saved"] is None for r in requests))
                self.assertEqual(selected, {"target": ten, "source": ten})
                self.assertEqual(saved, [selected])

    def test_saved_side_is_displayed_for_comparison_without_overwriting_its_points(self):
        requests, processes = [], []
        ui = self.ui(["y"])
        with patch(
            "gs_merger.interaction.subprocess.Popen",
            side_effect=self.launcher(requests, processes),
        ):
            selected = ui.select_pair(
                {"target": Path("left.ply"), "source": Path("right.ply")},
                3,
                2,
                {"target": PICKS, "source": None},
                on_save=lambda data: None,
            )
        self.assertEqual(requests[0]["saved"], PICKS)
        self.assertIs(selected["target"], PICKS)
        processes[0].terminate.assert_called_once()
        self.assertEqual(ui.ask.call_count, 1)

    def test_accepted_side_reopens_as_comparison_while_other_window_is_still_open(self):
        requests, processes, results = [], [], []
        ui = self.ui(["y", "y"])

        def launch(command, **kwargs):
            request = json.loads(Path(command[command.index("--request") + 1]).read_text())
            result = Path(command[command.index("--result") + 1])
            requests.append(request)
            results.append(result)
            if request["saved"] is not None:
                self.assertEqual(request["saved"], PICKS)
                # Finish source only after the accepted target is visible again.
                results[1].write_text(json.dumps({"selection": PICKS}))
                processes[1].status = 0
                process = FakeProcess(None)
            elif request["slot"] == 0:
                result.write_text(json.dumps({"selection": PICKS}))
                process = FakeProcess(0)
            else:
                process = FakeProcess(None)
            processes.append(process)
            return process

        with patch("gs_merger.interaction.subprocess.Popen", side_effect=launch):
            selected = ui.select_pair(
                {"target": Path("left.ply"), "source": Path("right.ply")},
                3, 1, {"target": None, "source": None}, on_save=lambda data: None,
            )
        self.assertEqual([r["slot"] for r in requests], [0, 1, 0])
        self.assertEqual(selected, {"target": PICKS, "source": PICKS})
        processes[-1].terminate.assert_called_once()

    def test_worker_failure_or_user_quit_cleans_up_other_window(self):
        for fail in (True, False):
            requests, processes = [], []
            ui = self.ui(["q"])
            error = RuntimeError if fail else UserQuit
            with self.subTest(fail=fail), patch(
                "gs_merger.interaction.subprocess.Popen",
                side_effect=self.launcher(requests, processes, fail=fail),
            ):
                with self.assertRaises(error):
                    ui.select_pair(
                        {"target": Path("left.ply"), "source": Path("right.ply")},
                        3,
                        1,
                        {"target": None, "source": None},
                        on_save=lambda data: None,
                    )
            self.assertTrue(all(p.poll() is not None for p in processes))

    def test_windows_have_distinct_positions_and_preview_uses_point_size(self):
        ui = self.ui([])
        ui.o3d = SimpleNamespace(
            visualization=SimpleNamespace(PointColorOption=SimpleNamespace(Color=1))
        )
        options = SimpleNamespace()
        visualizers = [Mock(), Mock()]
        for slot, vis in enumerate(visualizers):
            vis.get_render_option.return_value = options
            ui.window(vis, "pick", slot=slot)
        self.assertEqual(options.point_size, 2.0)
        left = visualizers[0].create_window.call_args.kwargs
        right = visualizers[1].create_window.call_args.kwargs
        self.assertEqual(left["width"], 800)
        self.assertGreaterEqual(right["left"], left["left"] + left["width"])

    def test_worker_dispatches_both_modes_and_saves_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            request, result = Path(temporary) / "in.json", Path(temporary) / "out.json"
            data = {
                "settings": dict(DEFAULTS),
                "path": "cloud.ply",
                "count": 3,
                "role": "target",
                "slot": 0,
                "saved": None,
            }
            arguments = ["--request", str(request), "--result", str(result)]
            with patch.object(picker_worker, "Open3DInteraction") as factory:
                factory.return_value.valid_selection.side_effect = Open3DInteraction.valid_selection
                factory.return_value.pick_once.return_value = PICKS
                request.write_text(json.dumps(data))
                self.assertEqual(picker_worker.main(arguments), 0)
                self.assertEqual(json.loads(result.read_text())["selection"], PICKS)
                data["saved"] = PICKS
                request.write_text(json.dumps(data))
                self.assertEqual(picker_worker.main(arguments), 0)
                factory.return_value.show_saved.assert_called_once()
                factory.return_value.show_saved.side_effect = ValueError("window could not open")
                self.assertEqual(picker_worker.main(arguments), 1)
                self.assertIn("window could not open", json.loads(result.read_text())["error"])

    def test_nine_of_ten_pairs_do_not_checkpoint_even_when_other_side_is_complete(self):
        ui = self.ui(["q"])
        saved = Mock()
        ten = {"indices": list(range(10)), "points": [[i, i % 3, i % 5] for i in range(10)]}

        def launch(command, **kwargs):
            request = json.loads(Path(command[command.index("--request") + 1]).read_text())
            result = Path(command[command.index("--result") + 1])
            if request["saved"] is not None:
                return FakeProcess(None)
            data = (
                {"selection": ten}
                if "target" in request["role"]
                else {"incomplete": True, "picked_count": 9}
            )
            result.write_text(json.dumps(data))
            return FakeProcess(0)

        with patch("gs_merger.interaction.subprocess.Popen", side_effect=launch):
            with self.assertRaises(UserQuit):
                ui.select_pair(
                    {"target": Path("left.ply"), "source": Path("right.ply")},
                    10,
                    1,
                    {"target": None, "source": None},
                    on_save=saved,
                )
        saved.assert_not_called()

    def test_worker_does_not_write_incomplete_coordinates_even_to_ipc_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            request, result = Path(temporary) / "in.json", Path(temporary) / "out.json"
            request.write_text(
                json.dumps(
                    {
                        "settings": dict(DEFAULTS),
                        "path": "cloud.ply",
                        "count": 10,
                        "role": "target",
                        "slot": 0,
                        "saved": None,
                    }
                )
            )
            with patch.object(picker_worker, "Open3DInteraction") as factory:
                factory.return_value.valid_selection.side_effect = Open3DInteraction.valid_selection
                factory.return_value.pick_once.return_value = {
                    "indices": list(range(9)),
                    "points": [[i, i, i] for i in range(9)],
                }
                self.assertEqual(
                    picker_worker.main(["--request", str(request), "--result", str(result)]), 0
                )
            self.assertEqual(
                json.loads(result.read_text()), {"incomplete": True, "picked_count": 9}
            )


if __name__ == "__main__":
    unittest.main()
