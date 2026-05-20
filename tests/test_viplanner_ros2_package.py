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

from viplanner_ros2.image_utils import (  # noqa: E402
    depth_msg_to_numpy,
    prepare_depth_image,
    rgb_msg_to_numpy,
    validate_array_dimensions,
    validate_message_dimensions,
)
from viplanner_ros2.inference import VIPlannerInference, extract_state_dict, resolve_model_files  # noqa: E402
from viplanner_ros2.learning_cfg import TrainCfg  # noqa: E402
from viplanner_ros2.planning_utils import (  # noqa: E402
    FearState,
    clip_goal_xy,
    is_forward_tracking,
    stamp_is_after,
    stamp_to_seconds,
)


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

    def test_stamp_helpers_compare_ros_time_messages(self):
        stamp = SimpleNamespace(sec=10, nanosec=500_000_000)
        reference = SimpleNamespace(sec=10, nanosec=250_000_000)

        self.assertAlmostEqual(stamp_to_seconds(stamp), 10.5)
        self.assertTrue(stamp_is_after(stamp, reference, tolerance=0.1))
        self.assertFalse(stamp_is_after(stamp, reference, tolerance=0.25))

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
