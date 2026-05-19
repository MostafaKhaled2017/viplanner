from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .semantic_meta import COCO_PANOPTIC_NAMES, VIPLANNER_CLASS_COLORS, map_dataset_classes

try:
    from mmdet.apis import inference_detector, init_detector
    from mmdet.evaluation import INSTANCE_OFFSET as MMDET_INSTANCE_OFFSET
except ImportError as exc:
    inference_detector = None
    init_detector = None
    MMDET_INSTANCE_OFFSET = 1000
    MMDET_IMPORT_ERROR = exc
else:
    MMDET_IMPORT_ERROR = None


class Mask2FormerPredictor:
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

        self.backend = "detectron2" if config_path.suffix.lower() in {".yaml", ".yml"} else "mmdet"
        if self.backend == "detectron2":
            self.model = self._init_detectron2_predictor(config_path, checkpoint_path, device)
            self.id_to_class = map_dataset_classes(COCO_PANOPTIC_NAMES)
            return

        if init_detector is None or inference_detector is None:
            message = "mmdetection is required for semantic inference with non-YAML configs."
            if MMDET_IMPORT_ERROR is not None:
                message = f"{message} Original import error: {MMDET_IMPORT_ERROR}"
            raise ImportError(message)

        self.model = init_detector(str(config_path), str(checkpoint_path), device=device)
        self.id_to_class = map_dataset_classes(self.model.dataset_meta["classes"])

    def predict(self, image: np.ndarray) -> np.ndarray:
        if self.backend == "detectron2":
            predictions = self.model(image)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                return self.colorize_detectron2_panoptic_segmentation(
                    panoptic_seg.detach().cpu().numpy(),
                    segments_info,
                    self.id_to_class,
                    warn_fn=self.warn_fn,
                )
            if "sem_seg" in predictions:
                semantic_ids = predictions["sem_seg"].argmax(dim=0).detach().cpu().numpy()
                return self.colorize_semantic_segmentation(semantic_ids, self.id_to_class, warn_fn=self.warn_fn)
            raise RuntimeError("Detectron2 Mask2Former output did not contain panoptic or semantic predictions.")

        result = inference_detector(self.model, image)
        semantic_ids = result.pred_panoptic_seg.sem_seg.detach().cpu().numpy()[0]
        return self.colorize_panoptic_segmentation(
            semantic_ids,
            self.model.dataset_meta["classes"],
            self.id_to_class,
            warn_fn=self.warn_fn,
        )

    @staticmethod
    def _init_detectron2_predictor(config_path: Path, checkpoint_path: Path, device: str):
        try:
            from detectron2.config import get_cfg
            from detectron2.engine.defaults import DefaultPredictor
            from detectron2.projects.deeplab import add_deeplab_config
            from mask2former import add_maskformer2_config
        except ImportError as exc:
            raise ImportError(
                "Detectron2 semantic inference requires detectron2 and an installed mask2former package. "
                f"Original import error: {exc}"
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
    def _class_color(category_id: int, id_to_class: dict[int, str], warn_fn=None):
        class_name = id_to_class.get(category_id, "static")
        if class_name == "static" and category_id not in id_to_class and warn_fn is not None:
            warn_fn(f"Category {category_id} not found in semantic mapping.")
        return np.asarray(VIPLANNER_CLASS_COLORS.get(class_name, VIPLANNER_CLASS_COLORS["static"]), dtype=np.uint8)

    @classmethod
    def colorize_panoptic_segmentation(
        cls,
        semantic_ids: np.ndarray,
        dataset_classes: Sequence[str],
        id_to_class: dict[int, str],
        warn_fn=None,
    ) -> np.ndarray:
        panoptic_mask = np.zeros((semantic_ids.shape[0], semantic_ids.shape[1], 3), dtype=np.uint8)
        for curr_sem_class in np.unique(semantic_ids):
            curr_label = int(curr_sem_class % MMDET_INSTANCE_OFFSET)
            if curr_sem_class == len(dataset_classes):
                color = np.asarray(VIPLANNER_CLASS_COLORS["static"], dtype=np.uint8)
            else:
                color = cls._class_color(curr_label, id_to_class, warn_fn=warn_fn)
            panoptic_mask[semantic_ids == curr_sem_class] = color
        return panoptic_mask

    @classmethod
    def colorize_detectron2_panoptic_segmentation(cls, panoptic_ids, segments_info, id_to_class, warn_fn=None):
        panoptic_mask = np.zeros((panoptic_ids.shape[0], panoptic_ids.shape[1], 3), dtype=np.uint8)
        panoptic_mask[:] = np.asarray(VIPLANNER_CLASS_COLORS["static"], dtype=np.uint8)
        for segment in segments_info:
            color = cls._class_color(int(segment["category_id"]), id_to_class, warn_fn=warn_fn)
            panoptic_mask[panoptic_ids == segment["id"]] = color
        return panoptic_mask

    @classmethod
    def colorize_semantic_segmentation(cls, semantic_ids, id_to_class, warn_fn=None):
        semantic_mask = np.zeros((semantic_ids.shape[0], semantic_ids.shape[1], 3), dtype=np.uint8)
        for category_id in np.unique(semantic_ids):
            color = cls._class_color(int(category_id), id_to_class, warn_fn=warn_fn)
            semantic_mask[semantic_ids == category_id] = color
        return semantic_mask
