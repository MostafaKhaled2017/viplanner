import ast
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

PKG_ROOT = Path(__file__).resolve().parents[1] / "src" / "viplanner_ros1"
sys.path.insert(0, str(PKG_ROOT / "src"))

from viplanner_ros1.image_utils import (  # noqa: E402
    depth_msg_to_numpy,
    prepare_depth_image,
    rgb_msg_to_numpy,
    validate_array_dimensions,
    validate_message_dimensions,
)
from viplanner_ros1.inference import VIPlannerInference, extract_state_dict, resolve_model_files  # noqa: E402
from viplanner_ros1.learning_cfg import TrainCfg  # noqa: E402
from viplanner_ros1.planning_utils import FearState, clip_goal_xy, is_forward_tracking  # noqa: E402
from viplanner_ros1.debug_utils import array_stats, xyz_summary  # noqa: E402


class VIPlannerRos1PackageTest(unittest.TestCase):
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

    def test_train_config_accepts_checkpoint_interval(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "model.yaml"
            path.write_text(yaml.safe_dump({"config": {"checkpoint_interval": 3}}))

            cfg = TrainCfg.from_yaml(str(path))

            self.assertEqual(cfg.checkpoint_interval, 3)

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

    def test_rgb_image_conversion_returns_training_rgb_order(self):
        rgb_pixel = np.array([[[10, 20, 30]]], dtype=np.uint8)
        msg_rgb = SimpleNamespace(height=1, width=1, encoding="rgb8", data=rgb_pixel.tobytes())
        np.testing.assert_array_equal(rgb_msg_to_numpy(msg_rgb), rgb_pixel)

        bgr_pixel = np.array([[[30, 20, 10]]], dtype=np.uint8)
        msg_bgr = SimpleNamespace(height=1, width=1, encoding="bgr8", data=bgr_pixel.tobytes())
        np.testing.assert_array_equal(rgb_msg_to_numpy(msg_bgr), rgb_pixel)

    def test_rgb_normalization_matches_training_float_tensor_path(self):
        inference = VIPlannerInference.__new__(VIPlannerInference)
        inference.train_cfg = SimpleNamespace(rgb=True)
        inference.pixel_mean = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
        inference.pixel_std = np.asarray([2.0, 4.0, 5.0], dtype=np.float32)
        inference.device = torch.device("cpu")
        inference.transforms = __import__("torchvision").transforms.ToTensor()

        image = np.array([[[12, 24, 40]]], dtype=np.uint8)
        tensor = inference.sem_rgb_converter(image)

        expected = torch.tensor([[[[1.0]], [[1.0]], [[2.0]]]], dtype=torch.float32)
        self.assertTrue(torch.equal(tensor, expected))

    def test_image_dimension_validation_hard_fails_on_mismatch(self):
        msg = SimpleNamespace(width=640, height=360)
        validate_message_dimensions(msg, 640, 360, "Depth")
        with self.assertRaises(RuntimeError):
            validate_message_dimensions(msg, 320, 360, "Depth")

        image = np.zeros((360, 640, 3), dtype=np.uint8)
        validate_array_dimensions(image, 640, 360, "RGB")
        with self.assertRaises(RuntimeError):
            validate_array_dimensions(image, 640, 180, "RGB")

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

        same_height_goal = clip_goal_xy([1.0, 1.0, 0.0], 5.0)
        self.assertEqual(same_height_goal[2], 0.0)

    def test_debug_helpers_summarize_arrays_and_paths(self):
        stats = array_stats(np.array([[0.0, 1.0], [np.nan, np.inf]], dtype=np.float32))
        self.assertEqual(stats["finite_count"], 2)
        self.assertEqual(stats["nan_count"], 1)
        self.assertEqual(stats["inf_count"], 1)

        path_summary = xyz_summary(np.array([[0.0, 0.0, -0.1], [1.0, 2.0, 0.2]], dtype=np.float32))
        self.assertAlmostEqual(path_summary["z"]["min"], -0.1, places=6)
        self.assertAlmostEqual(path_summary["z"]["max"], 0.2, places=6)

    def test_ros1_debug_config_defaults_are_disabled(self):
        config_path = PKG_ROOT / "config" / "viplanner.yaml"
        data = yaml.safe_load(config_path.read_text())
        self.assertFalse(data["debug_enabled"])
        self.assertFalse(data["debug_save_input_tensors"])
        self.assertEqual(data["debug_dump_every_n"], 1)
        self.assertIn("debug_dump_dir", data)
        self.assertIn("debug_height_warn_threshold", data)

    def test_ros1_fear_topic_config_and_message_generation(self):
        config_path = PKG_ROOT / "config" / "viplanner.yaml"
        data = yaml.safe_load(config_path.read_text())
        self.assertEqual(data["fear_topic"], "/viplanner/fear")

        msg_path = PKG_ROOT / "msg" / "Fear.msg"
        msg_lines = [line.strip() for line in msg_path.read_text().splitlines() if line.strip()]
        self.assertEqual(msg_lines, ["std_msgs/Header header", "float64 fear"])

        cmake_text = (PKG_ROOT / "CMakeLists.txt").read_text()
        self.assertIn("message_generation", cmake_text)
        self.assertIn("add_message_files", cmake_text)
        self.assertIn("Fear.msg", cmake_text)
        self.assertIn("generate_messages", cmake_text)
        self.assertIn("message_runtime", cmake_text)

        package_text = (PKG_ROOT / "package.xml").read_text()
        self.assertIn("<build_depend>message_generation</build_depend>", package_text)
        self.assertIn("<exec_depend>message_runtime</exec_depend>", package_text)

    def test_package_imports_are_isolated_from_repo_modules(self):
        package_dir = PKG_ROOT / "src" / "viplanner_ros1"
        banned_exact = {"viplanner"}
        banned_prefixes = ("viplanner.", "src.planner", "ref.iplanner")
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
                        import_name in banned_exact or any(import_name.startswith(prefix) for prefix in banned_prefixes),
                        f"{file_path} imports forbidden module {import_name}",
                    )


if __name__ == "__main__":
    unittest.main()
