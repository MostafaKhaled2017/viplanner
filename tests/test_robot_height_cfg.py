import tempfile
import unittest
from pathlib import Path

import numpy as np

from viplanner.config import (
    ReconstructionCfg,
    compute_robot_height_from_dataset,
    evenly_spaced_sample_indices,
)


class TestRobotHeightConfig(unittest.TestCase):
    def test_evenly_spaced_sample_indices_match_expected_stride(self):
        np.testing.assert_array_equal(
            evenly_spaced_sample_indices(num_frames=6, sample_count=3),
            np.array([0, 2, 4]),
        )

    def test_sample_count_larger_than_dataset_uses_all_frames(self):
        np.testing.assert_array_equal(
            evenly_spaced_sample_indices(num_frames=3, sample_count=50),
            np.array([0, 1, 2]),
        )

    def test_compute_robot_height_from_constant_altitude(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_extrinsics(tmp_dir, "forest", [2.0, 2.0, 2.0])
            cfg = ReconstructionCfg(
                data_dir=tmp_dir,
                env_list=["forest"],
                depth_suffix="_cam0",
                robot_height_margin=0.3,
                robot_height_sample_count=2,
            )

            info = compute_robot_height_from_dataset(cfg)

        self.assertAlmostEqual(info.altitude, 2.0)
        self.assertAlmostEqual(info.robot_height, 2.3)
        self.assertAlmostEqual(info.z_range, 0.0)

    def test_variation_below_threshold_passes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_extrinsics(tmp_dir, "forest", [2.0, 2.005, 2.009])
            cfg = ReconstructionCfg(
                data_dir=tmp_dir,
                env_list=["forest"],
                depth_suffix="_cam0",
                robot_height_variation_threshold=0.01,
                robot_height_sample_count=50,
            )

            info = compute_robot_height_from_dataset(cfg)

        self.assertAlmostEqual(info.z_range, 0.009)

    def test_variation_above_threshold_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_extrinsics(tmp_dir, "forest", [2.0, 2.02, 2.0])
            cfg = ReconstructionCfg(
                data_dir=tmp_dir,
                env_list=["forest"],
                depth_suffix="_cam0",
                robot_height_variation_threshold=0.01,
                robot_height_sample_count=50,
            )

            with self.assertRaisesRegex(ValueError, "exceeds robot_height_variation_threshold"):
                compute_robot_height_from_dataset(cfg)

    def test_invalid_extrinsics_fail(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_dir = Path(tmp_dir) / "forest"
            env_dir.mkdir()
            np.savetxt(env_dir / "camera_extrinsic_cam0.txt", np.array([[1.0, 2.0, 3.0]]), delimiter=",")
            cfg = ReconstructionCfg(data_dir=tmp_dir, env_list=["forest"], depth_suffix="_cam0")

            with self.assertRaisesRegex(ValueError, "Expected camera extrinsics"):
                compute_robot_height_from_dataset(cfg)

    def test_empty_extrinsics_fail(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_dir = Path(tmp_dir) / "forest"
            env_dir.mkdir()
            (env_dir / "camera_extrinsic_cam0.txt").write_text("")
            cfg = ReconstructionCfg(data_dir=tmp_dir, env_list=["forest"], depth_suffix="_cam0")

            with self.assertRaisesRegex(ValueError, "Camera extrinsic file is empty"):
                compute_robot_height_from_dataset(cfg)

    @staticmethod
    def _write_extrinsics(tmp_dir, env, z_values):
        env_dir = Path(tmp_dir) / env
        env_dir.mkdir()
        rows = np.array([[idx, 0.0, z, 0.0, 0.0, 0.0, 1.0] for idx, z in enumerate(z_values)])
        np.savetxt(env_dir / "camera_extrinsic_cam0.txt", rows, delimiter=",")


if __name__ == "__main__":
    unittest.main()
