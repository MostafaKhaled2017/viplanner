import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(
    importlib.util.find_spec("open3d") is not None and importlib.util.find_spec("pypose") is not None,
    "open3d and pypose are required for cost map tests",
)
class TestCostMapPCD(unittest.TestCase):
    def test_read_tsdf_map_reports_missing_files_with_depth_only_hint(self):
        from viplanner.cost_maps.cost_to_pcd import CostMapPCD

        with tempfile.TemporaryDirectory() as tmpdir:
            root_path = Path(tmpdir)

            with self.assertRaises(FileNotFoundError) as ctx:
                CostMapPCD.ReadTSDFMap(str(root_path), "cost_map_sem")

        message = str(ctx.exception)
        self.assertIn("Missing cost map files for 'cost_map_sem'", message)
        self.assertIn("cost_map_name: cost_map_geom", message)


if __name__ == "__main__":
    unittest.main()
