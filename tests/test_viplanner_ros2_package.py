import ast
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

PKG_ROOT = Path(__file__).resolve().parents[1] / "src" / "viplanner_ros2"
sys.path.insert(0, str(PKG_ROOT))

from viplanner_ros2.image_utils import depth_msg_to_numpy, prepare_depth_image  # noqa: E402
from viplanner_ros2.inference import extract_state_dict, resolve_model_files  # noqa: E402
from viplanner_ros2.learning_cfg import TrainCfg  # noqa: E402
from viplanner_ros2.planning_utils import FearState, clip_goal_xy, is_forward_tracking  # noqa: E402


class VIPlannerRos2PackageTest(unittest.TestCase):
    def test_model_directory_requires_checkpoint_and_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir)
            with self.assertRaises(FileNotFoundError):
                resolve_model_files(str(model_dir))

            (model_dir / "model.pt").touch()
            with self.assertRaises(FileNotFoundError):
                resolve_model_files(str(model_dir))

            (model_dir / "model.yaml").write_text("config: {}\n")
            model_path, config_path = resolve_model_files(str(model_dir))
            self.assertEqual(model_path, model_dir / "model.pt")
            self.assertEqual(config_path, model_dir / "model.yaml")

    def test_train_config_parses_model_modes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "model.yaml"
            for sem, rgb in [(False, False), (True, False), (False, True)]:
                path.write_text(yaml.safe_dump({"config": {"sem": sem, "rgb": rgb, "img_input_size": [360, 640]}}))
                cfg = TrainCfg.from_yaml(str(path))
                self.assertEqual(cfg.sem, sem)
                self.assertEqual(cfg.rgb, rgb)
                self.assertEqual(cfg.img_input_size, [360, 640])

    def test_extract_state_dict_accepts_tuple_raw_and_model_key(self):
        raw = {"layer.weight": torch.tensor([1.0])}
        self.assertIs(extract_state_dict(raw), raw)
        self.assertIs(extract_state_dict((raw, 1.25)), raw)
        wrapped = {"model": raw, "loss": 1.25}
        self.assertIs(extract_state_dict(wrapped), raw)

    def test_depth_image_conversion(self):
        msg_32f = SimpleNamespace(height=1, width=2, encoding="32FC1", data=np.array([1.0, np.inf], dtype=np.float32).tobytes())
        np.testing.assert_allclose(depth_msg_to_numpy(msg_32f), np.array([[1.0, np.inf]], dtype=np.float32))
        np.testing.assert_allclose(prepare_depth_image(msg_32f, False, 10.0, False), np.array([[1.0, 0.0]], dtype=np.float32))

        msg_16u = SimpleNamespace(height=1, width=2, encoding="16UC1", data=np.array([1000, 20000], dtype=np.uint16).tobytes())
        np.testing.assert_allclose(prepare_depth_image(msg_16u, True, 15.0, False), np.array([[1.0, 0.0]], dtype=np.float32))

        msg_bad = SimpleNamespace(height=1, width=1, encoding="mono8", data=b"\0")
        with self.assertRaises(RuntimeError):
            depth_msg_to_numpy(msg_bad)

    def test_fear_forward_tracking_and_goal_clipping(self):
        fear_state = FearState(buffer_size=1, threshold=0.7)
        self.assertFalse(fear_state.update(0.8, True))
        self.assertTrue(fear_state.update(0.8, True))
        self.assertTrue(fear_state.update(0.1, False))
        self.assertFalse(fear_state.update(0.1, False))

        forward_waypoints = np.array([[0.0, 0.0, 0.0], [1.0, 0.05, 0.0]])
        turning_waypoints = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertTrue(is_forward_tracking(forward_waypoints, 0.5, 0.3))
        self.assertFalse(is_forward_tracking(turning_waypoints, 0.5, 0.3))

        clipped = clip_goal_xy([6.0, 8.0, 2.0], 5.0)
        np.testing.assert_allclose(clipped, (3.0, 4.0, 2.0))

    def test_package_imports_are_isolated_from_repo_modules(self):
        package_dir = PKG_ROOT / "viplanner_ros2"
        banned = ("viplanner", "src.planner", "ref.iplanner")
        for file_path in package_dir.glob("*.py"):
            tree = ast.parse(file_path.read_text(), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imports = [node.module]
                else:
                    continue
                for import_name in imports:
                    self.assertFalse(
                        any(import_name == item or import_name.startswith(item + ".") for item in banned),
                        f"{file_path} imports forbidden module {import_name}",
                    )


if __name__ == "__main__":
    unittest.main()
