from __future__ import annotations

import numpy as np
from PIL import Image as PILImage


def depth_msg_to_numpy(msg) -> np.ndarray:
    height = msg.height
    width = msg.width
    encoding = msg.encoding.upper()

    if encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(height, width)
    if encoding == "16UC1":
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(height, width)
    if encoding == "16SC1":
        return np.frombuffer(msg.data, dtype=np.int16).reshape(height, width)

    raise RuntimeError(f"Unsupported depth image encoding: {msg.encoding}")


def prepare_depth_image(msg, depth_uint_type: bool, max_depth: float, image_flip: bool) -> np.ndarray:
    image = depth_msg_to_numpy(msg).astype(np.float32, copy=True)
    image[~np.isfinite(image)] = 0.0
    if depth_uint_type or msg.encoding.upper() == "16UC1":
        image = image / 1000.0
    image[image > max_depth] = 0.0
    if image_flip:
        return np.asarray(PILImage.fromarray(image).transpose(PILImage.Transpose.ROTATE_180))
    return image


def rgb_msg_to_numpy(msg) -> np.ndarray:
    encoding = msg.encoding.lower()
    channels = 3
    if encoding in {"rgb8", "bgr8"}:
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
        if encoding == "rgb8":
            return image[:, :, ::-1]
        return image
    if encoding in {"rgba8", "bgra8"}:
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)[:, :, :3]
        if encoding == "rgba8":
            return image[:, :, ::-1]
        return image
    raise RuntimeError(f"Unsupported RGB image encoding: {msg.encoding}")
