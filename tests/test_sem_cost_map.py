import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("open3d") is not None, "open3d is required for semantic cost map tests")
class TestSemCostMap(unittest.TestCase):
    def _make_cost_map(self):
        from viplanner.config import GeneralCostMapConfig, SemCostMapConfig
        from viplanner.cost_maps.sem_cost_map import SemCostMap

        cost_map = SemCostMap(
            GeneralCostMapConfig(resolution=1.0),
            SemCostMapConfig(),
            robot_height=1.0,
            visualize=False,
        )
        cost_map._num_x = 3
        cost_map._num_y = 5
        return cost_map

    def test_unique_grid_idx_uses_y_dimension_as_flattening_stride(self):
        cost_map = self._make_cost_map()
        pts = np.array(
            [
                [1.0, 0.0, 2.0],
                [1.0, 0.0, 10.0],
                [0.0, 3.0, 4.0],
            ]
        )

        grid_idx, pts_idx = cost_map._get_unqiue_grid_idx(pts)

        self.assertEqual({tuple(idx) for idx in grid_idx.tolist()}, {(1, 0), (0, 3)})
        self.assertEqual(set(pts_idx.tolist()), {1, 2})

    def test_distance_based_gradient_returns_empty_array_for_empty_index(self):
        cost_map = self._make_cost_map()

        gradient = cost_map._distance_based_gradient(
            (np.array([], dtype=int), np.array([], dtype=int)),
            0.0,
            0.5,
            False,
        )

        self.assertEqual(gradient.shape, (0,))

    def test_class_loss_overrides_update_semantic_metadata(self):
        from viplanner.config import GeneralCostMapConfig, SemCostMapConfig
        from viplanner.cost_maps.sem_cost_map import SemCostMap

        cost_map = SemCostMap(
            GeneralCostMapConfig(resolution=1.0),
            SemCostMapConfig(class_loss_overrides={"static": 0.0}),
            robot_height=1.0,
            visualize=False,
        )

        self.assertEqual(cost_map.sem_meta.class_loss["static"], 0.0)

    def test_unknown_class_loss_override_fails(self):
        from viplanner.config import GeneralCostMapConfig, SemCostMapConfig
        from viplanner.cost_maps.sem_cost_map import SemCostMap

        with self.assertRaisesRegex(ValueError, "Unknown class_loss_overrides"):
            SemCostMap(
                GeneralCostMapConfig(resolution=1.0),
                SemCostMapConfig(class_loss_overrides={"forest_floor": 0.0}),
                robot_height=1.0,
                visualize=False,
            )


if __name__ == "__main__":
    unittest.main()
