import tempfile
import unittest
from pathlib import Path

from viplanner.utils.dataset import PlannerDataGenerator


class DatasetGeneratedDirsTest(unittest.TestCase):
    def test_cleanup_only_removes_generator_owned_directories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_warp_dir = root / "img_warp" / "pid_1"
            other_img_warp_dir = root / "img_warp" / "pid_2"
            depth_noise_dir = root / "depth_noise_edges" / "pid_1"
            other_depth_noise_dir = root / "depth_noise_edges" / "pid_2"

            for path in (img_warp_dir, other_img_warp_dir, depth_noise_dir, other_depth_noise_dir):
                path.mkdir(parents=True)
                (path / "sample.png").touch()

            generator = PlannerDataGenerator.__new__(PlannerDataGenerator)
            generator.root = str(root)
            generator.img_warp_dir = str(img_warp_dir)
            generator.depth_noise_edge_dir = str(depth_noise_dir)

            generator.cleanup()

            self.assertFalse(img_warp_dir.exists())
            self.assertFalse(depth_noise_dir.exists())
            self.assertTrue(other_img_warp_dir.is_dir())
            self.assertTrue(other_depth_noise_dir.is_dir())

    def test_cleanup_removes_empty_generated_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            img_warp_dir = root / "img_warp" / "pid_1"
            depth_noise_dir = root / "depth_noise_edges" / "pid_1"

            img_warp_dir.mkdir(parents=True)
            depth_noise_dir.mkdir(parents=True)

            generator = PlannerDataGenerator.__new__(PlannerDataGenerator)
            generator.root = str(root)
            generator.img_warp_dir = str(img_warp_dir)
            generator.depth_noise_edge_dir = str(depth_noise_dir)

            generator.cleanup()

            self.assertFalse((root / "img_warp").exists())
            self.assertFalse((root / "depth_noise_edges").exists())


if __name__ == "__main__":
    unittest.main()
