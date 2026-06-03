import importlib.util
import tempfile
import textwrap
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import numpy as np

from viplanner.config import CostMapConfig, ReconstructionCfg


class TestReconstructionCfg(unittest.TestCase):
    def test_from_yaml_loads_reconstruction_section(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "costmap.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    reconstruction:
                      data_dir: /tmp/dataset
                      env_list:
                        - forest
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
            self.assertEqual(cfg.env_list, ["forest"])
            self.assertEqual(cfg.get_data_path(), "/tmp/dataset/forest")
            self.assertEqual(cfg.depth_suffix, "_cam0")
            self.assertEqual(cfg.sem_suffix, "_cam1")
            self.assertEqual(cfg.start_idx, 3)
            self.assertFalse(cfg.semantics)
            self.assertEqual(cfg.voxel_size, 0.05)
            self.assertEqual(cfg.robot_height_margin, 0.3)
            self.assertEqual(cfg.robot_height_variation_threshold, 0.01)
            self.assertEqual(cfg.robot_height_sample_count, 50)
            self.assertIsNone(cfg.semantic_ignore_classes)
            self.assertEqual(cfg.point_cloud_batch_size, 200)

    def test_from_yaml_accepts_reconstruction_only_mapping(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "reconstruction.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    data_dir: /tmp/dataset
                    env_list:
                      - warehouse
                    max_images: null
                    depth_scale: 500
                    robot_height_margin: 0.4
                    robot_height_variation_threshold: 0.02
                    robot_height_sample_count: 10
                    """
                )
            )

            cfg = ReconstructionCfg.from_yaml(str(config_path))

            self.assertEqual(cfg.data_dir, "/tmp/dataset")
            self.assertEqual(cfg.env_list, ["warehouse"])
            self.assertEqual(cfg.get_data_path(), "/tmp/dataset/warehouse")
            self.assertIsNone(cfg.max_images)
            self.assertEqual(cfg.depth_scale, 500)
            self.assertEqual(cfg.robot_height_margin, 0.4)
            self.assertEqual(cfg.robot_height_variation_threshold, 0.02)
            self.assertEqual(cfg.robot_height_sample_count, 10)

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

    def test_from_yaml_accepts_empty_semantic_ignore_classes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "costmap.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    reconstruction:
                      data_dir: /tmp/dataset
                      env_list:
                        - forest
                      semantic_ignore_classes: []
                    config:
                      semantics: true
                    """
                )
            )

            cfg = ReconstructionCfg.from_yaml(str(config_path))

            self.assertEqual(cfg.semantic_ignore_classes, [])

    def test_from_yaml_rejects_old_env_field(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "costmap.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    reconstruction:
                      data_dir: /tmp/dataset
                      env: forest
                    config:
                      semantics: true
                    """
                )
            )

            with self.assertRaisesRegex(ValueError, "Use reconstruction.env_list"):
                ReconstructionCfg.from_yaml(str(config_path))

    def test_from_yaml_rejects_empty_env_list(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "costmap.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    reconstruction:
                      data_dir: /tmp/dataset
                      env_list: []
                    config:
                      semantics: true
                    """
                )
            )

            with self.assertRaisesRegex(ValueError, "non-empty list"):
                ReconstructionCfg.from_yaml(str(config_path))

    def test_for_env_resolves_selected_environment_path(self):
        cfg = ReconstructionCfg(data_dir="/tmp/dataset", env_list=["forest", "desert"])

        env_cfg = cfg.for_env("desert")

        self.assertEqual(env_cfg.get_data_path(), "/tmp/dataset/desert")
        with self.assertRaisesRegex(ValueError, "before resolving"):
            cfg.get_data_path()


class TestCostMapRootPathResolution(unittest.TestCase):
    def test_shared_config_with_null_root_path_derives_environment_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "costmap.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    reconstruction:
                      data_dir: {tmp_dir}
                      env_list:
                        - forest
                    config:
                      geometry: true
                      general:
                        root_path: null
                        ply_file: cloud.ply
                    """
                )
            )

            cfg = CostMapConfig.from_yaml(str(config_path))

            self.assertEqual(cfg.general.root_path, str(Path(tmp_dir, "forest").resolve()))

    def test_shared_config_with_omitted_root_path_derives_environment_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "costmap.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    reconstruction:
                      data_dir: {tmp_dir}
                      env_list:
                        - forest
                    config:
                      geometry: true
                      general:
                        ply_file: cloud.ply
                    """
                )
            )

            cfg = CostMapConfig.from_yaml(str(config_path))

            self.assertEqual(cfg.general.root_path, str(Path(tmp_dir, "forest").resolve()))

    def test_shared_config_with_matching_root_path_passes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "costmap.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    reconstruction:
                      data_dir: {tmp_dir}
                      env_list:
                        - forest
                    config:
                      geometry: true
                      general:
                        root_path: {tmp_dir}
                    """
                )
            )

            cfg = CostMapConfig.from_yaml(str(config_path))

            self.assertEqual(cfg.general.root_path, str(Path(tmp_dir, "forest").resolve()))

    def test_shared_config_with_environment_root_path_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "costmap.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    reconstruction:
                      data_dir: {tmp_dir}
                      env_list:
                        - forest
                    config:
                      geometry: true
                      general:
                        root_path: {Path(tmp_dir, "forest")}
                    """
                )
            )

            with self.assertRaisesRegex(ValueError, "parent directory"):
                CostMapConfig.from_yaml(str(config_path))

    def test_standalone_costmap_config_preserves_default_root_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "costmap.yaml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    config:
                      geometry: true
                      general:
                        ply_file: cloud.ply
                    """
                )
            )

            cfg = CostMapConfig.from_yaml(str(config_path))

            self.assertEqual(cfg.general.root_path, "<path-to-data>/<env-name>")


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

            cfg = ReconstructionCfg(data_dir=tmp_dir, env_list=["forest"], semantics=False)
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

        cfg = ReconstructionCfg(data_dir="/tmp/dataset", env_list=["forest", "desert"])
        robot_height_info = SimpleNamespace(
            altitude=2.0,
            margin=0.3,
            robot_height=2.3,
            sample_indices=np.array([0]),
            z_range=0.0,
        )
        constructors = [mock.MagicMock(), mock.MagicMock()]

        with mock.patch.object(depth_reconstruct.ReconstructionCfg, "from_yaml", return_value=cfg) as from_yaml:
            with mock.patch.object(
                depth_reconstruct,
                "compute_robot_height_from_dataset",
                return_value=robot_height_info,
            ) as compute_height:
                with mock.patch.object(
                    depth_reconstruct,
                    "DepthReconstruction",
                    side_effect=constructors,
                ) as cls_mock:
                    result = depth_reconstruct.main(["--config", "/tmp/costmap.yaml"])

        self.assertEqual(result, 0)
        from_yaml.assert_called_once_with("/tmp/costmap.yaml")
        self.assertEqual(compute_height.call_count, 2)
        self.assertEqual([call.args[0].get_data_path() for call in compute_height.call_args_list], [
            "/tmp/dataset/forest",
            "/tmp/dataset/desert",
        ])
        self.assertEqual(cls_mock.call_count, 2)
        self.assertEqual([call.args[0].get_data_path() for call in cls_mock.call_args_list], [
            "/tmp/dataset/forest",
            "/tmp/dataset/desert",
        ])
        for constructor in constructors:
            constructor.depth_reconstruction.assert_called_once_with()
            constructor.save_pcd.assert_called_once_with()
            constructor.show_pcd.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
