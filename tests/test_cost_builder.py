import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("open3d") is not None, "open3d is required for cost builder tests")
class TestCostBuilderCli(unittest.TestCase):
    def test_run_from_config_computes_height_and_passes_it_to_builder(self):
        from viplanner import cost_builder

        reconstruction_cfg = mock.MagicMock()
        costmap_cfg = mock.MagicMock()
        robot_height_info = SimpleNamespace(
            altitude=2.0,
            margin=0.3,
            robot_height=2.3,
            sample_indices=np.array([0, 2, 4]),
            z_range=0.0,
        )

        with mock.patch.object(cost_builder.ReconstructionCfg, "from_yaml", return_value=reconstruction_cfg) as load_reconstruction:
            with mock.patch.object(cost_builder.CostMapConfig, "from_yaml", return_value=costmap_cfg) as load_costmap:
                with mock.patch.object(
                    cost_builder,
                    "compute_robot_height_from_dataset",
                    return_value=robot_height_info,
                ) as compute_height:
                    with mock.patch.object(cost_builder, "main") as build_costmap:
                        cost_builder.run_from_config("/tmp/costmap.yaml", final_viz=False)

        load_reconstruction.assert_called_once_with("/tmp/costmap.yaml")
        load_costmap.assert_called_once_with("/tmp/costmap.yaml", reconstruction_cfg=reconstruction_cfg)
        compute_height.assert_called_once_with(reconstruction_cfg)
        build_costmap.assert_called_once_with(costmap_cfg, robot_height=2.3, final_viz=False)


if __name__ == "__main__":
    unittest.main()
