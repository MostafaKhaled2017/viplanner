import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "planner"))

from utils.data_collection_paths import resolve_output_dir


class TestResolveOutputDir(unittest.TestCase):
    def test_uses_base_directory_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            output_dir = resolve_output_dir(root, "warehouse")

            self.assertEqual(output_dir, root / "warehouse")

    def test_reuses_base_directory_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "warehouse").mkdir()

            output_dir = resolve_output_dir(root, "warehouse")

            self.assertEqual(output_dir, root / "warehouse")

    def test_versions_directory_when_base_contains_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base_dir = root / "warehouse"
            base_dir.mkdir()
            (base_dir / "depth").mkdir()

            output_dir = resolve_output_dir(root, "warehouse")

            self.assertEqual(output_dir, root / "warehouse_v2")

    def test_increments_to_next_available_version(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for name in ("warehouse", "warehouse_v2"):
                env_dir = root / name
                env_dir.mkdir()
                (env_dir / "rgb").mkdir()

            output_dir = resolve_output_dir(root, "warehouse")

            self.assertEqual(output_dir, root / "warehouse_v3")


if __name__ == "__main__":
    unittest.main()
