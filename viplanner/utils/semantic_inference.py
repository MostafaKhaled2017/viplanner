# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from viplanner.config.coco_sem_meta import get_class_for_id, get_class_for_id_mmdet
from viplanner.config.viplanner_sem_meta import VIPlannerSemMetaHandler

try:
    from mmdet.apis import inference_detector, init_detector
    from mmdet.evaluation import INSTANCE_OFFSET as MMDET_INSTANCE_OFFSET
except ImportError as exc:  # pragma: no cover - exercised when optional dependency is missing
    inference_detector = None
    init_detector = None
    MMDET_INSTANCE_OFFSET = 1000
    MMDET_IMPORT_ERROR = exc
else:
    MMDET_IMPORT_ERROR = None


class Mask2FormerPredictor:
    """Run Mask2Former inference and map predictions to VIPlanner semantic colors."""

    def __init__(
        self,
        config_file: str,
        checkpoint_file: str,
        device: str = "cuda:0",
        warn_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.warn_fn = warn_fn

        config_path = Path(config_file)
        checkpoint_path = Path(checkpoint_file)
        if not config_path.is_file():
            raise FileNotFoundError(f"Mask2Former config file not found: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Mask2Former checkpoint file not found: {checkpoint_path}")

        viplanner_meta = VIPlannerSemMetaHandler()
        self.viplanner_sem_class_color_map = viplanner_meta.class_color
        self.backend = self._infer_backend(config_path)

        if self.backend == "detectron2":
            self.model = self._init_detectron2_predictor(config_path, checkpoint_path, device)
            coco_viplanner_cls_mapping = get_class_for_id()
            self.coco_viplanner_color_mapping = {
                coco_id: self.viplanner_sem_class_color_map[viplanner_cls_name]
                for coco_id, viplanner_cls_name in coco_viplanner_cls_mapping.items()
            }
            return

        if init_detector is None or inference_detector is None:
            message = (
                "mmdetection is required for semantic inference. Install compatible OpenMMLab packages "
                "with the 'inference' extras."
            )
            if MMDET_IMPORT_ERROR is not None:
                message = f"{message} Original import error: {MMDET_IMPORT_ERROR}"
            raise ImportError(message)

        self.model = init_detector(str(config_path), str(checkpoint_path), device=device)

        coco_viplanner_cls_mapping = get_class_for_id_mmdet(self.model.dataset_meta["classes"])
        self.coco_viplanner_color_mapping = {
            coco_id: self.viplanner_sem_class_color_map[viplanner_cls_name]
            for coco_id, viplanner_cls_name in coco_viplanner_cls_mapping.items()
        }

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict a colorized semantic image from a BGR input image."""

        if self.backend == "detectron2":
            predictions = self.model(image)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                return self.colorize_detectron2_panoptic_segmentation(
                    panoptic_seg.detach().cpu().numpy(),
                    segments_info,
                    self.coco_viplanner_color_mapping,
                    self.viplanner_sem_class_color_map["static"],
                    warn_fn=self.warn_fn,
                )
            if "sem_seg" in predictions:
                semantic_ids = predictions["sem_seg"].argmax(dim=0).detach().cpu().numpy()
                return self.colorize_semantic_segmentation(
                    semantic_ids,
                    self.coco_viplanner_color_mapping,
                    self.viplanner_sem_class_color_map["static"],
                    warn_fn=self.warn_fn,
                )
            raise RuntimeError("Detectron2 Mask2Former output did not contain 'panoptic_seg' or 'sem_seg'.")

        result = inference_detector(self.model, image)
        semantic_ids = result.pred_panoptic_seg.sem_seg.detach().cpu().numpy()[0]
        return self.colorize_panoptic_segmentation(
            semantic_ids,
            self.model.dataset_meta["classes"],
            self.coco_viplanner_color_mapping,
            self.viplanner_sem_class_color_map["static"],
            warn_fn=self.warn_fn,
        )

    @staticmethod
    def _infer_backend(config_path: Path) -> str:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            return "detectron2"
        return "mmdet"

    @staticmethod
    def _init_detectron2_predictor(config_path: Path, checkpoint_path: Path, device: str):
        mask2former_root = Path(__file__).resolve().parents[1] / "third_party" / "mask2former"
        if mask2former_root.is_dir():
            ops_build_dir = mask2former_root / "mask2former" / "modeling" / "pixel_decoder" / "ops" / "build"
            for path in (mask2former_root, *sorted(ops_build_dir.glob("lib.*"))):
                path_str = str(path)
                if path_str not in sys.path:
                    sys.path.insert(0, path_str)

        try:
            from detectron2.config import get_cfg
            from detectron2.engine.defaults import DefaultPredictor
            from detectron2.projects.deeplab import add_deeplab_config
            from mask2former import add_maskformer2_config
        except ImportError as exc:
            raise ImportError(
                "Detectron2 and Facebook Mask2Former are required for YAML Mask2Former configs. "
                "Install Detectron2, initialize the Mask2Former submodule, and build the "
                "MultiScaleDeformableAttention CUDA op."
                f" Original import error: {exc}"
            ) from exc

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(str(config_path))
        cfg.MODEL.WEIGHTS = str(checkpoint_path)
        cfg.MODEL.DEVICE = device
        cfg.freeze()
        return DefaultPredictor(cfg)

    @staticmethod
    def colorize_panoptic_segmentation(
        semantic_ids: np.ndarray,
        dataset_classes: Sequence[str],
        coco_viplanner_color_mapping: dict[int, Sequence[int]],
        static_color: Sequence[int],
        warn_fn: Optional[Callable[[str], None]] = None,
    ) -> np.ndarray:
        """Convert panoptic class ids to the VIPlanner semantic color palette."""

        panoptic_mask = np.zeros((semantic_ids.shape[0], semantic_ids.shape[1], 3), dtype=np.uint8)
        for curr_sem_class in np.unique(semantic_ids):
            curr_label = int(curr_sem_class % MMDET_INSTANCE_OFFSET)
            color = coco_viplanner_color_mapping.get(curr_label)
            if color is None:
                if warn_fn is not None and curr_sem_class != len(dataset_classes):
                    warn_fn(f"Category {curr_label} not found in coco_viplanner_cls_mapping.")
                color = static_color
            panoptic_mask[semantic_ids == curr_sem_class] = np.asarray(color, dtype=np.uint8)
        return panoptic_mask

    @staticmethod
    def colorize_detectron2_panoptic_segmentation(
        panoptic_ids: np.ndarray,
        segments_info: Sequence[dict],
        coco_viplanner_color_mapping: dict[int, Sequence[int]],
        static_color: Sequence[int],
        warn_fn: Optional[Callable[[str], None]] = None,
    ) -> np.ndarray:
        """Convert Detectron2 panoptic segment ids to the VIPlanner semantic color palette."""

        panoptic_mask = np.zeros((panoptic_ids.shape[0], panoptic_ids.shape[1], 3), dtype=np.uint8)
        panoptic_mask[:] = np.asarray(static_color, dtype=np.uint8)
        for segment in segments_info:
            segment_id = segment["id"]
            category_id = int(segment["category_id"])
            color = coco_viplanner_color_mapping.get(category_id)
            if color is None:
                if warn_fn is not None:
                    warn_fn(f"Category {category_id} not found in coco_viplanner_cls_mapping.")
                color = static_color
            panoptic_mask[panoptic_ids == segment_id] = np.asarray(color, dtype=np.uint8)
        return panoptic_mask

    @staticmethod
    def colorize_semantic_segmentation(
        semantic_ids: np.ndarray,
        coco_viplanner_color_mapping: dict[int, Sequence[int]],
        static_color: Sequence[int],
        warn_fn: Optional[Callable[[str], None]] = None,
    ) -> np.ndarray:
        """Convert contiguous semantic class ids to the VIPlanner semantic color palette."""

        semantic_mask = np.zeros((semantic_ids.shape[0], semantic_ids.shape[1], 3), dtype=np.uint8)
        for category_id in np.unique(semantic_ids):
            category_id = int(category_id)
            color = coco_viplanner_color_mapping.get(category_id)
            if color is None:
                if warn_fn is not None:
                    warn_fn(f"Category {category_id} not found in coco_viplanner_cls_mapping.")
                color = static_color
            semantic_mask[semantic_ids == category_id] = np.asarray(color, dtype=np.uint8)
        return semantic_mask
