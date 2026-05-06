"""
Extracción de descriptores de forma (HOG).
"""

import cv2
import numpy as np
from skimage.feature import hog


DEFAULT_RESIZE = (128, 128)
DEFAULT_ORIENTATIONS = 9
DEFAULT_PIXELS_PER_CELL = (16, 16)
DEFAULT_CELLS_PER_BLOCK = (2, 2)


def hog_features(
    image: np.ndarray,
    resize: tuple = DEFAULT_RESIZE,
    orientations: int = DEFAULT_ORIENTATIONS,
    pixels_per_cell: tuple = DEFAULT_PIXELS_PER_CELL,
    cells_per_block: tuple = DEFAULT_CELLS_PER_BLOCK,
    is_rgb: bool = True,
) -> np.ndarray:
    """Calcula el descriptor HOG."""
    if image.dtype != np.uint8:
        raise ValueError(f"Esperaba uint8, recibí {image.dtype}")

    img_resized = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
    if is_rgb:
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    feature = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feature.astype(np.float32)


def hog_dim(
    resize: tuple = DEFAULT_RESIZE,
    orientations: int = DEFAULT_ORIENTATIONS,
    pixels_per_cell: tuple = DEFAULT_PIXELS_PER_CELL,
    cells_per_block: tuple = DEFAULT_CELLS_PER_BLOCK,
) -> int:
    h, w = resize
    cells_y = h // pixels_per_cell[0]
    cells_x = w // pixels_per_cell[1]
    blocks_y = cells_y - cells_per_block[0] + 1
    blocks_x = cells_x - cells_per_block[1] + 1
    return blocks_y * blocks_x * cells_per_block[0] * cells_per_block[1] * orientations
