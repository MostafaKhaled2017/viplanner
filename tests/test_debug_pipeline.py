import unittest

import numpy as np
import torch

from viplanner.utils.debug_summary import array_stats, loss_frame_tensors, tensor_to_list, xyz_summary


class DebugPipelineHelpersTest(unittest.TestCase):
    def test_array_stats_reports_finite_nan_and_inf_counts(self):
        stats = array_stats(np.array([1.0, 2.0, np.nan, np.inf], dtype=np.float32))
        self.assertEqual(stats["shape"], [4])
        self.assertEqual(stats["finite_count"], 2)
        self.assertEqual(stats["nan_count"], 1)
        self.assertEqual(stats["inf_count"], 1)
        self.assertAlmostEqual(stats["mean"], 1.5)

    def test_xyz_summary_reports_z_range(self):
        points = torch.tensor([[[1.0, 2.0, 0.0], [3.0, 4.0, -0.2], [5.0, 6.0, 0.3]]])
        summary = xyz_summary(points)
        self.assertEqual(summary["shape"], [3, 3])
        self.assertAlmostEqual(summary["z"]["min"], -0.2, places=6)
        self.assertAlmostEqual(summary["z"]["max"], 0.3, places=6)

    def test_loss_frame_tensors_mirror_augmented_y_axis_only(self):
        preds = torch.tensor([[[1.0, 2.0, 3.0], [4.0, -5.0, 6.0]]])
        goal = torch.tensor([[7.0, 8.0, 9.0]])
        preds_loss, goal_loss = loss_frame_tensors(preds, goal, augment=True)

        expected_preds = torch.tensor([[[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]]])
        expected_goal = torch.tensor([[7.0, -8.0, 9.0]])
        self.assertTrue(torch.equal(preds_loss, expected_preds))
        self.assertTrue(torch.equal(goal_loss, expected_goal))
        self.assertTrue(torch.equal(preds, torch.tensor([[[1.0, 2.0, 3.0], [4.0, -5.0, 6.0]]])))

    def test_tensor_to_list_rounds_for_json_readability(self):
        values = tensor_to_list(torch.tensor([1.1234567]))
        self.assertEqual(values, [1.123457])


if __name__ == "__main__":
    unittest.main()

