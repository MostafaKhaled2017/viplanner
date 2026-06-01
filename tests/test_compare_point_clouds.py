import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np


@unittest.skipUnless(
    importlib.util.find_spec("open3d") is not None,
    "open3d is required for point-cloud comparison tests",
)
class TestComparePointClouds(unittest.TestCase):
    def write_cloud(self, path: Path, points: np.ndarray) -> None:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
        self.assertTrue(o3d.io.write_point_cloud(str(path), pcd))

    def run_main(self, args):
        from viplanner import compare_point_clouds

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = compare_point_clouds.main(args)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_identical_environment_clouds_print_metadata_and_score(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])

        with tempfile.TemporaryDirectory() as tmp_dir:
            env_dir = Path(tmp_dir) / "forest"
            env_dir.mkdir()
            self.write_cloud(env_dir / "cloud.ply", points)
            self.write_cloud(env_dir / "cloud_original.ply", points)

            code, stdout, stderr = self.run_main(
                [
                    "--env-dir",
                    str(env_dir),
                    "--similarity-distance-scale",
                    "0.01",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("Point cloud A", stdout)
        self.assertIn("point_count: 3", stdout)
        self.assertIn("min_xyz: [0, 0, 0]", stdout)
        self.assertIn("max_xyz: [2, 3, 4]", stdout)
        self.assertIn("similarity_score: 100.000000", stdout)
        self.assertIn("similarity_label: nearly_identical", stdout)

    def test_perturbed_cloud_produces_lower_score_and_distance_metrics(self):
        points_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        points_b = points_a + np.array([0.001, 0.0, 0.0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cloud_a = root / "a.ply"
            cloud_b = root / "b.ply"
            self.write_cloud(cloud_a, points_a)
            self.write_cloud(cloud_b, points_b)

            code, stdout, stderr = self.run_main(
                [
                    "--cloud-a",
                    str(cloud_a),
                    "--cloud-b",
                    str(cloud_b),
                    "--similarity-distance-scale",
                    "0.01",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("a_to_b_mean: 0.001", stdout)
        self.assertIn("b_to_a_mean: 0.001", stdout)
        self.assertIn("similarity_score: 90.000000", stdout)
        self.assertIn("similarity_label: similar", stdout)

    def test_shuffled_cloud_keeps_geometry_score_and_reports_ordered_mismatch(self):
        points_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        points_b = points_a[::-1]

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cloud_a = root / "a.ply"
            cloud_b = root / "b.ply"
            self.write_cloud(cloud_a, points_a)
            self.write_cloud(cloud_b, points_b)

            code, stdout, stderr = self.run_main(
                [
                    "--cloud-a",
                    str(cloud_a),
                    "--cloud-b",
                    str(cloud_b),
                    "--similarity-distance-scale",
                    "0.01",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("ordered_abs_error_max: 2", stdout)
        self.assertIn("symmetric_max: 0", stdout)
        self.assertIn("similarity_score: 100.000000", stdout)

    def test_large_offset_produces_low_score_without_failure(self):
        points_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        points_b = points_a + np.array([10.0, 0.0, 0.0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cloud_a = root / "a.ply"
            cloud_b = root / "b.ply"
            self.write_cloud(cloud_a, points_a)
            self.write_cloud(cloud_b, points_b)

            code, stdout, stderr = self.run_main(
                [
                    "--cloud-a",
                    str(cloud_a),
                    "--cloud-b",
                    str(cloud_b),
                    "--similarity-distance-scale",
                    "0.01",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("similarity_score: 0.000000", stdout)
        self.assertIn("similarity_label: different", stdout)

    def test_missing_and_empty_inputs_return_invalid_input_failure(self):
        points = np.array([[0.0, 0.0, 0.0]])

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            valid_cloud = root / "valid.ply"
            missing_cloud = root / "missing.ply"
            self.write_cloud(valid_cloud, points)

            code, _stdout, stderr = self.run_main(
                [
                    "--cloud-a",
                    str(valid_cloud),
                    "--cloud-b",
                    str(missing_cloud),
                    "--similarity-distance-scale",
                    "0.01",
                ]
            )

            self.assertEqual(code, 2)
            self.assertIn("file does not exist", stderr)

            empty_cloud = root / "empty.ply"
            empty_cloud.write_text(
                "ply\n"
                "format ascii 1.0\n"
                "element vertex 0\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "end_header\n"
            )
            code, _stdout, stderr = self.run_main(
                [
                    "--cloud-a",
                    str(valid_cloud),
                    "--cloud-b",
                    str(empty_cloud),
                    "--similarity-distance-scale",
                    "0.01",
                ]
            )

        self.assertEqual(code, 2)
        self.assertIn("empty or unreadable", stderr)

