# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import viplanner.utils.semantic_inference as semantic_inference
from viplanner.generate_semantics import discover_environment_dirs, process_environment
from viplanner.utils.semantic_inference import Mask2FormerPredictor


class _FakePredictor:
    def predict(self, image: np.ndarray) -> np.ndarray:
        assert image.ndim == 3
        output = np.zeros_like(image)
        output[..., 0] = 10
        output[..., 1] = 20
        output[..., 2] = 30
        return output


class TestSemanticGeneration(unittest.TestCase):
    def test_colorize_panoptic_segmentation_falls_back_to_static(self):
        semantic_ids = np.array([[0, 9999]], dtype=np.int32)
        classes = ["person", "background"]
        color_mapping = {0: [1, 2, 3]}
        static_color = [9, 8, 7]

        output = Mask2FormerPredictor.colorize_panoptic_segmentation(
            semantic_ids=semantic_ids,
            dataset_classes=classes,
            coco_viplanner_color_mapping=color_mapping,
            static_color=static_color,
        )

        self.assertEqual(output.shape, (1, 2, 3))
        np.testing.assert_array_equal(output[0, 0], np.array([1, 2, 3], dtype=np.uint8))
        np.testing.assert_array_equal(output[0, 1], np.array([9, 8, 7], dtype=np.uint8))

    def test_colorize_detectron2_panoptic_segmentation_uses_segments_info(self):
        panoptic_ids = np.array([[0, 1], [2, 2]], dtype=np.int32)
        segments_info = [
            {"id": 1, "category_id": 4},
            {"id": 2, "category_id": 7},
        ]
        color_mapping = {4: [1, 2, 3]}
        static_color = [9, 8, 7]

        output = Mask2FormerPredictor.colorize_detectron2_panoptic_segmentation(
            panoptic_ids=panoptic_ids,
            segments_info=segments_info,
            coco_viplanner_color_mapping=color_mapping,
            static_color=static_color,
        )

        self.assertEqual(output.shape, (2, 2, 3))
        np.testing.assert_array_equal(output[0, 0], np.array([9, 8, 7], dtype=np.uint8))
        np.testing.assert_array_equal(output[0, 1], np.array([1, 2, 3], dtype=np.uint8))
        np.testing.assert_array_equal(output[1, 0], np.array([9, 8, 7], dtype=np.uint8))
        np.testing.assert_array_equal(output[1, 1], np.array([9, 8, 7], dtype=np.uint8))

    def test_yaml_config_selects_detectron2_backend(self):
        self.assertEqual(Mask2FormerPredictor._infer_backend(Path("config.yaml")), "detectron2")
        self.assertEqual(Mask2FormerPredictor._infer_backend(Path("config.py")), "mmdet")

    def test_predictor_import_error_includes_original_failure(self):
        original_init_detector = semantic_inference.init_detector
        original_inference_detector = semantic_inference.inference_detector
        original_import_error = semantic_inference.MMDET_IMPORT_ERROR

        try:
            semantic_inference.init_detector = None
            semantic_inference.inference_detector = None
            semantic_inference.MMDET_IMPORT_ERROR = ImportError("missing shared library")

            with tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                config_file = root / "config.py"
                checkpoint_file = root / "checkpoint.pth"
                config_file.write_text("model = {}")
                checkpoint_file.write_bytes(b"checkpoint")

                with self.assertRaisesRegex(ImportError, "missing shared library"):
                    Mask2FormerPredictor(str(config_file), str(checkpoint_file))
        finally:
            semantic_inference.init_detector = original_init_detector
            semantic_inference.inference_detector = original_inference_detector
            semantic_inference.MMDET_IMPORT_ERROR = original_import_error

    def test_discover_environment_dirs_recursive(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "env_a" / "rgb").mkdir(parents=True)
            (root / "env_b" / "rgb").mkdir(parents=True)

            environments = discover_environment_dirs(root, recursive=True)

            self.assertEqual(environments, [root / "env_a", root / "env_b"])

    def test_process_environment_skips_existing_and_non_images(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_dir = Path(tmp_dir) / "env"
            rgb_dir = env_dir / "rgb"
            semantics_dir = env_dir / "semantics"
            rgb_dir.mkdir(parents=True)
            semantics_dir.mkdir()

            image_a = np.full((6, 8, 3), 50, dtype=np.uint8)
            image_b = np.full((6, 8, 3), 100, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(rgb_dir / "0000.png"), image_a))
            self.assertTrue(cv2.imwrite(str(rgb_dir / "0001.png"), image_b))
            self.assertTrue(cv2.imwrite(str(semantics_dir / "0001.png"), image_b))
            (rgb_dir / "notes.txt").write_text("ignore me")

            stats = process_environment(env_dir, predictor=_FakePredictor(), force=False)

            self.assertEqual(stats.processed, 1)
            self.assertEqual(stats.skipped, 1)
            self.assertEqual(stats.failed, 0)
            self.assertEqual(stats.non_image_files, 1)
            self.assertTrue((semantics_dir / "0000.png").is_file())


if __name__ == "__main__":
    unittest.main()
