import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("open3d") is not None, "open3d is required for cost builder tests")
class TestCostBuilderCli(unittest.TestCase):
    def test_run_from_config_computes_height_and_passes_it_to_builder(self):
        from viplanner import cost_builder

        reconstruction_cfg = cost_builder.ReconstructionCfg(data_dir="/tmp/dataset", env_list=["forest", "desert"])
        costmap_cfgs = [mock.MagicMock(), mock.MagicMock()]
        robot_height_info = SimpleNamespace(
            altitude=2.0,
            margin=0.3,
            robot_height=2.3,
            sample_indices=np.array([0, 2, 4]),
            z_range=0.0,
        )

        with mock.patch.object(cost_builder.ReconstructionCfg, "from_yaml", return_value=reconstruction_cfg) as load_reconstruction:
            with mock.patch.object(cost_builder.CostMapConfig, "from_yaml", side_effect=costmap_cfgs) as load_costmap:
                with mock.patch.object(
                    cost_builder,
                    "compute_robot_height_from_dataset",
                    return_value=robot_height_info,
                ) as compute_height:
                    with mock.patch.object(cost_builder, "main") as build_costmap:
                        cost_builder.run_from_config("/tmp/costmap.yaml", final_viz=False)

        load_reconstruction.assert_called_once_with("/tmp/costmap.yaml")
        self.assertEqual(compute_height.call_count, 2)
        self.assertEqual([call.args[0].get_data_path() for call in compute_height.call_args_list], [
            "/tmp/dataset/forest",
            "/tmp/dataset/desert",
        ])
        self.assertEqual(load_costmap.call_count, 2)
        self.assertEqual([call.kwargs["reconstruction_cfg"].get_data_path() for call in load_costmap.call_args_list], [
            "/tmp/dataset/forest",
            "/tmp/dataset/desert",
        ])
        self.assertEqual(build_costmap.call_count, 2)
        build_costmap.assert_has_calls(
            [
                mock.call(costmap_cfgs[0], robot_height=2.3, final_viz=False),
                mock.call(costmap_cfgs[1], robot_height=2.3, final_viz=False),
            ]
        )

    def test_run_from_config_stops_on_first_environment_failure(self):
        from viplanner import cost_builder

        reconstruction_cfg = cost_builder.ReconstructionCfg(data_dir="/tmp/dataset", env_list=["forest", "desert"])

        with mock.patch.object(cost_builder.ReconstructionCfg, "from_yaml", return_value=reconstruction_cfg):
            with mock.patch.object(
                cost_builder,
                "compute_robot_height_from_dataset",
                side_effect=RuntimeError("height failed"),
            ) as compute_height:
                with mock.patch.object(cost_builder.CostMapConfig, "from_yaml") as load_costmap:
                    with mock.patch.object(cost_builder, "main") as build_costmap:
                        with self.assertRaisesRegex(RuntimeError, "height failed"):
                            cost_builder.run_from_config("/tmp/costmap.yaml", final_viz=False)

        compute_height.assert_called_once()
        self.assertEqual(compute_height.call_args.args[0].get_data_path(), "/tmp/dataset/forest")
        load_costmap.assert_not_called()
        build_costmap.assert_not_called()


if __name__ == "__main__":
    unittest.main()
