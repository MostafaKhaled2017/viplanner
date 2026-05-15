import importlib.util
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from viplanner.config import ReconstructionCfg


class TestReconstructionCfg(unittest.TestCase):
    def test_from_yaml_loads_reconstruction_section(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "costmap.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    reconstruction:
                      data_dir: /tmp/dataset
                      env: forest
                      depth_suffix: _cam0
                      sem_suffix: _cam1
                      start_idx: 3
                      semantics: false
                    config:
                      geometry: true
                    """
                )
            )

            cfg = ReconstructionCfg.from_yaml(str(config_path))

            self.assertEqual(cfg.data_dir, "/tmp/dataset")
            self.assertEqual(cfg.env, "forest")
            self.assertEqual(cfg.depth_suffix, "_cam0")
            self.assertEqual(cfg.sem_suffix, "_cam1")
            self.assertEqual(cfg.start_idx, 3)
            self.assertFalse(cfg.semantics)
            self.assertEqual(cfg.voxel_size, 0.05)
            self.assertEqual(cfg.point_cloud_batch_size, 200)

    def test_from_yaml_accepts_reconstruction_only_mapping(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "reconstruction.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    data_dir: /tmp/dataset
                    env: warehouse
                    max_images: null
                    depth_scale: 500
                    """
                )
            )

            cfg = ReconstructionCfg.from_yaml(str(config_path))

            self.assertEqual(cfg.data_dir, "/tmp/dataset")
            self.assertEqual(cfg.env, "warehouse")
            self.assertIsNone(cfg.max_images)
            self.assertEqual(cfg.depth_scale, 500)

    def test_from_yaml_rejects_shared_file_without_reconstruction_section(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "costmap.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    config:
                      geometry: true
                      visualize: false
                    """
                )
            )

            with self.assertRaisesRegex(ValueError, "top-level 'reconstruction' section"):
                ReconstructionCfg.from_yaml(str(config_path))


@unittest.skipUnless(importlib.util.find_spec("open3d") is not None, "open3d is required for CLI tests")
class TestDepthReconstructCli(unittest.TestCase):
    def test_depth_only_intrinsics_use_first_projection_row(self):
        from viplanner import depth_reconstruct

        with tempfile.TemporaryDirectory() as tmp_dir:
            env_dir = Path(tmp_dir) / "forest"
            env_dir.mkdir()
            projection_rows = np.array(
                [
                    [1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [5.0, 0.0, 6.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ]
            )
            np.savetxt(env_dir / "intrinsics.txt", projection_rows, delimiter=",")

            cfg = ReconstructionCfg(data_dir=tmp_dir, env="forest", semantics=False)
            reconstruction = depth_reconstruct.DepthReconstruction.__new__(depth_reconstruct.DepthReconstruction)
            reconstruction._cfg = cfg

            reconstruction._read_intrinsic()

        np.testing.assert_array_equal(
            reconstruction.K_depth,
            np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 4.0], [0.0, 0.0, 1.0]]),
        )

    def test_main_requires_config_argument(self):
        from viplanner import depth_reconstruct

        with self.assertRaises(SystemExit) as exc:
            depth_reconstruct.main([])

        self.assertEqual(exc.exception.code, 2)

    def test_main_loads_yaml_config_and_runs_reconstruction(self):
        from viplanner import depth_reconstruct

        cfg = ReconstructionCfg(data_dir="/tmp/dataset", env="forest")
        constructor = mock.MagicMock()

        with mock.patch.object(depth_reconstruct.ReconstructionCfg, "from_yaml", return_value=cfg) as from_yaml:
            with mock.patch.object(depth_reconstruct, "DepthReconstruction", return_value=constructor) as cls_mock:
                result = depth_reconstruct.main(["--config", "/tmp/costmap.yaml"])

        self.assertEqual(result, 0)
        from_yaml.assert_called_once_with("/tmp/costmap.yaml")
        cls_mock.assert_called_once_with(cfg)
        constructor.depth_reconstruction.assert_called_once_with()
        constructor.save_pcd.assert_called_once_with()
        constructor.show_pcd.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
