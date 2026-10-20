"""Fresh-process merging, failure propagation and cancellation without user models."""

import io
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from gs_merger.io_utils import merge_streaming
from gs_merger.merge_worker import merge_isolated
from tests.test_workflow import gaussian_fixture
from gs_merger.gaussian_io import read_gaussians, write_vertices


class MergeWorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source, self.target, self.output = [
            self.root / name for name in ("新模型.ply", "已有 model.ply", "merged.ply")
        ]
        self.vertices = gaussian_fixture()
        write_vertices(self.source, self.vertices)
        write_vertices(self.target, self.vertices[:6])
        self.transform = np.eye(4)
        angle = 0.37
        self.transform[:3, :3] = 0.39897248 * np.array(
            [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        )
        self.transform[:3, 3] = [2, 0.5, -4]

    def test_fresh_worker_preserves_every_attribute_and_parent_thread_environment(self):
        baseline = self.root / "baseline.ply"
        expected = merge_streaming(self.source, self.target, self.transform, baseline)
        source_bytes, target_bytes = self.source.read_bytes(), self.target.read_bytes()
        messages = []
        popen = subprocess.Popen
        with patch.dict(os.environ, {"OPENBLAS_NUM_THREADS": "8", "OMP_NUM_THREADS": "9"}):
            with patch("gs_merger.merge_worker.subprocess.Popen", wraps=popen) as launch:
                report = merge_isolated(
                    self.source, self.target, self.transform, self.output, log=messages.append
                )
            child_env = launch.call_args.kwargs["env"]
            self.assertEqual(child_env["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(child_env["OMP_NUM_THREADS"], "1")
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "8")
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "9")
        self.assertEqual(report, expected)
        actual, reference = read_gaussians(self.output), read_gaussians(baseline)
        self.assertEqual(actual.dtype, reference.dtype)
        for name in actual.dtype.names:
            np.testing.assert_allclose(actual[name], reference[name], rtol=1e-6, atol=1e-7)
        self.assertEqual(self.source.read_bytes(), source_bytes)
        self.assertEqual(self.target.read_bytes(), target_bytes)
        self.assertTrue(any("高斯准备进度 100.0%" in message for message in messages))

    def test_real_worker_validation_failure_reaches_parent_without_committing_output(self):
        broken = self.vertices.copy()
        broken["rot_0"][-1] = 0
        write_vertices(self.source, broken)
        self.output.write_bytes(b"previous completed result")
        with self.assertRaisesRegex(RuntimeError, "Zero-length Gaussian quaternion"):
            merge_isolated(self.source, self.target, self.transform, self.output)
        self.assertEqual(self.output.read_bytes(), b"previous completed result")
        self.assertEqual(list(self.root.glob(".merge_*")), [])

    def test_native_crash_without_result_is_never_treated_as_a_completed_merge(self):
        process = Mock(pid=123, stdout=io.StringIO("native stopped\n"))
        process.wait.return_value = process.poll.return_value = -11
        with patch("gs_merger.merge_worker.subprocess.Popen", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "exit -11.*native stopped"):
                merge_isolated(self.source, self.target, self.transform, self.output)
        self.assertFalse(self.output.exists())
        self.assertTrue(process.stdout.closed)

    def test_interrupted_parent_interrupts_and_reaps_a_nonresponsive_child(self):
        class InterruptedStream:
            def __iter__(self):
                raise KeyboardInterrupt()

            close = Mock()

        process = Mock(pid=123, stdout=InterruptedStream())
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("merge", 5), 0]
        with patch("gs_merger.merge_worker.subprocess.Popen", return_value=process):
            with self.assertRaises(KeyboardInterrupt):
                merge_isolated(self.source, self.target, self.transform, self.output)
        process.send_signal.assert_called_once_with(signal.SIGINT)
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_count, 2)
        process.stdout.close.assert_called_once_with()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
