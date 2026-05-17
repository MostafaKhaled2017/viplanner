import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(
    importlib.util.find_spec("open3d") is not None and importlib.util.find_spec("pypose") is not None,
    "open3d and pypose are required for dataset tests",
)
class TestPlannerDataGeneratorGraph(unittest.TestCase):
    def test_edge_collisions_treat_out_of_bounds_as_collision(self):
        from viplanner.utils.dataset import PlannerDataGenerator

        occupancy_map = np.zeros((2, 2), dtype=np.uint8)
        occupancy_map[1, 0] = 1
        occupancy_idx = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 2.0],
                [1.0, 1.0],
            ]
        )

        collision = PlannerDataGenerator._edge_collisions_from_occupancy(
            occupancy_map,
            occupancy_idx,
            num_intermediate=3,
        )

        np.testing.assert_array_equal(collision, np.array([False, True, True]))


if __name__ == "__main__":
    unittest.main()
