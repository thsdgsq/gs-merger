"""Independent covariance, SH, Sim(3), and dense-view checks; no parent project."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from plyfile import PlyData
from gs_merger.gaussian_io import (
    DC, NORMALS, ROT, SCALE, XYZ, columns, export_view_cloud, merge_gaussian_files,
    opacity_mask, read_gaussians, real_sh_basis, rotation_quaternion, set_columns,
    sh_degree, split_similarity, transform_gaussians, write_vertices,
)
from gs_merger.similarity import estimate_similarity, fit_control_points



def quaternion_matrix(q):
    """Independent covariance reference using the quaternion vector formula."""
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    w, vector = q[0], q[1:]
    x, y, z = vector
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (w * w - vector @ vector) * np.eye(3) + 2 * np.outer(vector, vector) + 2 * w * cross


def example_gaussians(count=32, degree=3):
    rng = np.random.default_rng(123)
    rest = [f"f_rest_{i}" for i in range(3 * ((degree + 1) ** 2 - 1))]
    names = XYZ + NORMALS + DC + rest + ["opacity"] + SCALE + ROT
    vertices = np.zeros(count, dtype=[(name, "f4") for name in names] + [("tag", "i4")])
    set_columns(vertices, XYZ, rng.normal(size=(count, 3)))
    set_columns(vertices, NORMALS, rng.normal(size=(count, 3)))
    set_columns(vertices, DC, rng.normal(size=(count, 3)))
    if rest:
        set_columns(vertices, rest, rng.normal(size=(count, len(rest))))
    q = rng.normal(size=(count, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    set_columns(vertices, ROT, q)
    set_columns(vertices, SCALE, rng.uniform(-5, -1, size=(count, 3)))
    vertices["opacity"] = rng.uniform(-8, 8, count)
    vertices["tag"] = np.arange(count)
    return vertices


def example_transform(scale=1.0, quaternion=(0.3, 0.4, -0.1, 0.8)):
    transform = np.eye(4)
    transform[:3, :3] = scale * quaternion_matrix(quaternion)
    transform[:3, 3] = [12, -8, 3]
    return transform


class GaussianMathTests(unittest.TestCase):
    def test_quaternion_conversion_near_half_turns(self):
        quaternions = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0.00000001, 0.2, 0.7, 0.3],
            [0.3, -0.1, 0.6, 0.2],
        ]
        for q in quaternions:
            rotation = quaternion_matrix(q)
            np.testing.assert_allclose(
                quaternion_matrix(rotation_quaternion(rotation)), rotation, atol=1e-12
            )

    def test_identity_preserves_attributes(self):
        original = example_gaussians()
        result = transform_gaussians(original, np.eye(4))
        for name in original.dtype.names:
            np.testing.assert_allclose(result[name], original[name], atol=1e-7)

    def test_similarity_preserves_full_covariance(self):
        original = example_gaussians()
        transform = example_transform(2.73)
        result = transform_gaussians(original, transform)
        np.testing.assert_allclose(
            columns(result, XYZ),
            columns(original, XYZ) @ transform[:3, :3].T + transform[:3, 3],
            atol=1e-6,
        )
        scale, rotation, _ = split_similarity(transform)
        for old, new in zip(original, result):
            old_q = np.array([old[name] for name in ROT])
            new_q = np.array([new[name] for name in ROT])
            old_r, new_r = quaternion_matrix(old_q), quaternion_matrix(new_q)
            old_cov = old_r @ np.diag(np.exp(2 * np.array([old[n] for n in SCALE]))) @ old_r.T
            new_cov = new_r @ np.diag(np.exp(2 * np.array([new[n] for n in SCALE]))) @ new_r.T
            np.testing.assert_allclose(
                new_cov, scale**2 * rotation @ old_cov @ rotation.T, atol=3e-7, rtol=2e-6
            )
        np.testing.assert_allclose(
            columns(result, NORMALS), columns(original, NORMALS) @ rotation.T, atol=2e-7
        )
        for name in DC + ["opacity", "tag"]:
            np.testing.assert_array_equal(result[name], original[name])

    def test_sh_convention_matches_xyz_polynomials(self):
        directions = np.array([[1.0, 2, 3], [-2.0, 3, -1], [0.0, 0, 1]])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        x, y, z = directions.T
        c1 = np.sqrt(3 / (4 * np.pi))
        c2 = np.sqrt(15 / (4 * np.pi))
        expected = np.column_stack(
            (
                np.full(len(x), 1 / np.sqrt(4 * np.pi)),
                -c1 * y,
                c1 * z,
                -c1 * x,
                c2 * x * y,
                -c2 * y * z,
                np.sqrt(5 / (16 * np.pi)) * (3 * z * z - 1),
                -c2 * x * z,
                c2 / 2 * (x * x - y * y),
            )
        )
        np.testing.assert_allclose(real_sh_basis(2, directions), expected, atol=1e-14)

    def test_view_dependent_colors_follow_rotation(self):
        rng = np.random.default_rng(777)
        directions = rng.normal(size=(200, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        for degree in range(5):
            original = example_gaussians(7, degree)
            for quaternion in ([0, 0, 1, 0], [0.3, 0.4, -0.1, 0.8]):
                transform = example_transform(1.63, quaternion)
                result = transform_gaussians(original, transform)
                _, rotation, _ = split_similarity(transform)
                names = [f"f_rest_{i}" for i in range(3 * ((degree + 1) ** 2 - 1))]

                def coefficients(vertices):
                    rest = (
                        columns(vertices, names).reshape(len(vertices), 3, -1)
                        if names
                        else (np.empty((len(vertices), 3, 0)))
                    )
                    return np.concatenate((columns(vertices, DC)[..., None], rest), axis=2)

                before = coefficients(original) @ real_sh_basis(degree, directions).T
                after = coefficients(result) @ real_sh_basis(degree, directions @ rotation.T).T
                np.testing.assert_allclose(after, before, atol=4e-7, rtol=2e-6)

    def test_unsupported_affine_and_bad_attributes_fail(self):
        bad = np.eye(4)
        for linear in (
            np.diag([-1, 1, 1]),
            np.diag([1, 2, 1]),
            np.array([[1, 0.1, 0], [0, 1, 0], [0, 0, 1]]),
        ):
            bad[:3, :3] = linear
            with self.assertRaises(ValueError):
                split_similarity(bad)
        vertices = example_gaussians()
        vertices["rot_0"] = vertices["rot_1"] = vertices["rot_2"] = vertices["rot_3"] = 0
        with self.assertRaises(ValueError):
            transform_gaussians(vertices, np.eye(4))
        vertices = example_gaussians()
        vertices["f_rest_3"][0] = np.nan
        with self.assertRaises(ValueError):
            transform_gaussians(vertices, np.eye(4))

    def test_opacity_is_a_logit(self):
        vertices = example_gaussians(3)
        vertices["opacity"] = [-4, 0, 4]
        np.testing.assert_array_equal(opacity_mask(vertices, 0.5), [False, True, True])
        np.testing.assert_array_equal(opacity_mask(vertices, 0), [True, True, True])


class AlignmentTests(unittest.TestCase):
    def test_planar_identified_controls_determine_similarity(self):
        source = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0], [3, 4, 0]], dtype=float)
        expected = example_transform(2.37)
        target = source @ expected[:3, :3].T + expected[:3, 3]
        np.testing.assert_allclose(estimate_similarity(source, target), expected, atol=1e-12)

    def test_control_ransac_rejects_wrong_landmarks(self):
        rng = np.random.default_rng(1234)
        source = rng.normal(size=(36, 3))
        expected = example_transform(2.37)
        target = source @ expected[:3, :3].T + expected[:3, 3]
        target[:7] += rng.uniform(20, 50, (7, 3))
        result, mask, _ = fit_control_points(source, target, 1e-4)
        np.testing.assert_allclose(result, expected, atol=1e-10)
        self.assertEqual(int(mask.sum()), 29)
        self.assertFalse(mask[:7].any())

    def test_collinear_controls_fail(self):
        source = np.array([[0.0, 0, 0], [1, 0, 0], [3, 0, 0]])
        with self.assertRaises(ValueError):
            estimate_similarity(source, source * 2)


class FileWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_dense_view_exports_all_valid_centers_in_original_coordinates(self):
        vertices = example_gaussians(23)
        vertices["opacity"] = 4
        vertices["opacity"][3] = -20
        vertices["x"][7] = np.nan
        vertices["f_dc_0"][9] = np.inf
        vertices["opacity"][12] = np.nan
        set_columns(vertices[:1], DC, np.array([[-4, 0, 4]]))
        source, output = self.root / "gaussians.ply", self.root / "view.ply"
        write_vertices(source, vertices)
        with patch("gs_merger.gaussian_io.CHUNK_SIZE", 5):
            report = export_view_cloud(source, output)
        cloud = PlyData.read(output)["vertex"].data
        expected = np.delete(vertices, [3, 7, 9, 12])
        self.assertEqual(report["exported_centers"], 19)
        self.assertEqual(cloud.dtype.names, ("x", "y", "z", "red", "green", "blue"))
        np.testing.assert_array_equal(columns(cloud, XYZ), columns(expected, XYZ))
        np.testing.assert_array_equal(columns(cloud[:1], ["red", "green", "blue"]), [[0, 128, 255]])
        with self.assertRaises(ValueError):
            read_gaussians(output)

    def test_dense_view_optional_limit_is_reproducible_across_chunks(self):
        source = self.root / "gaussians.ply"
        vertices = example_gaussians(57)
        vertices["opacity"] = 4
        vertices["opacity"][::4] = -20
        write_vertices(source, vertices)
        outputs = [self.root / f"view_{i}.ply" for i in range(2)]
        for chunk_size, output in zip((4, 13), outputs):
            with patch("gs_merger.gaussian_io.CHUNK_SIZE", chunk_size):
                report = export_view_cloud(source, output, max_points=17, seed=98)
            self.assertEqual(report["exported_centers"], 17)
        first = PlyData.read(outputs[0])["vertex"].data
        second = PlyData.read(outputs[1])["vertex"].data
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(columns(first, XYZ), axis=0)), 17)
        usable = columns(vertices[opacity_mask(vertices, 0.05)], XYZ)
        self.assertTrue(
            all(np.any(np.all(usable == point, axis=1)) for point in columns(first, XYZ))
        )

    def test_dense_view_has_no_registration_sampling_cap(self):
        vertices = example_gaussians(200137, degree=0)
        vertices["opacity"] = 4
        source, output = self.root / "gaussians.ply", self.root / "view.ply"
        write_vertices(source, vertices)
        report = export_view_cloud(source, output)
        cloud = PlyData.read(output)["vertex"].data
        self.assertEqual(report["exported_centers"], len(vertices))
        self.assertEqual(len(cloud), len(vertices))
        np.testing.assert_array_equal(columns(cloud, XYZ), columns(vertices, XYZ))

    def test_dense_view_empty_filter_preserves_existing_output(self):
        source, output = self.root / "gaussians.ply", self.root / "view.ply"
        vertices = example_gaussians(5)
        vertices["opacity"] = -20
        write_vertices(source, vertices)
        output.write_bytes(b"existing output")
        with self.assertRaisesRegex(ValueError, "No usable viewing centers"):
            export_view_cloud(source, output)
        self.assertEqual(output.read_bytes(), b"existing output")

    def test_merge_preserves_reference_and_reorders_properties_by_name(self):
        source, target = example_gaussians(23), example_gaussians(17)
        reordered = np.zeros(
            len(target), dtype=[(n, target.dtype[n]) for n in target.dtype.names[::-1]]
        )
        for name in target.dtype.names:
            reordered[name] = target[name]
        a, b = self.root / "a.ply", self.root / "b.ply"
        write_vertices(a, source)
        write_vertices(b, reordered)
        transform = example_transform(1.85)
        # Force several chunks to exercise disk-backed concatenation boundaries.
        with patch("gs_merger.gaussian_io.CHUNK_SIZE", 8):
            report = merge_gaussian_files(a, b, transform, self.root / "out")
        result = read_gaussians(report["merged"])
        self.assertEqual(len(result), 40)
        expected = transform_gaussians(source, transform)
        for name in target.dtype.names:
            np.testing.assert_array_equal(result[:17][name], target[name])
            np.testing.assert_array_equal(result[17:][name], expected[name])
        self.assertEqual(sh_degree(result), 3)
        np.testing.assert_array_equal(read_gaussians(a), source)

    def test_input_collision_and_schema_mismatch_fail(self):
        a, b = self.root / "merged.ply", self.root / "b.ply"
        write_vertices(a, example_gaussians())
        write_vertices(b, example_gaussians(degree=4))
        with self.assertRaises(ValueError):
            merge_gaussian_files(a, b, np.eye(4), self.root)
        with self.assertRaises(ValueError):
            merge_gaussian_files(a, b, np.eye(4), self.root / "out")
        self.assertEqual(sh_degree(read_gaussians(a)), 3)
