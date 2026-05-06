"""
Extracción de descriptores cromáticos (histogramas HSV).

Para cada imagen o crop, calcula un histograma del canal H, otro de S y
otro de V, y los concatena en un único vector.
"""

from typing import Tuple

import cv2
import numpy as np


def hsv_histogram(
    image: np.ndarray,
    bins: Tuple[int, int, int] = (16, 8, 8),
    is_rgb: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Calcula el histograma HSV concatenado de una imagen."""
    if image.dtype != np.uint8:
        raise ValueError(f"Esperaba uint8, recibí {image.dtype}")

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if is_rgb else image
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [bins[0]], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [bins[1]], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [bins[2]], [0, 256]).flatten()

    feature = np.concatenate([hist_h, hist_s, hist_v])

    if normalize:
        total = feature.sum()
        if total > 0:
            feature = feature / total

    return feature.astype(np.float32)


def hsv_histogram_dim(bins: Tuple[int, int, int] = (16, 8, 8)) -> int:
    return sum(bins)
