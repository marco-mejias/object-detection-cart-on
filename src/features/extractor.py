"""
Extractor de features combinado: HSV + HOG + LBP.

Esta es la función principal que se usará en el resto del proyecto.
Concatena tres descriptores complementarios:

    [histograma HSV (32 dims)]   ← color
    [HOG (1764 dims)]            ← forma / silueta
    [LBP (10 dims)]              ← textura

Total: 1806 dimensiones por imagen.

Por qué los tres:
- HSV solo no distingue una manzana roja de un tomate (mismo color).
- HOG solo no distingue una manzana roja de una manzana verde (misma forma).
- LBP solo es muy débil aislado, pero ayuda a romper empates: una piel de
  fruta tiene microtextura distinta a un envase plástico o metálico.
"""

import numpy as np

from .color_histogram import hsv_histogram, hsv_histogram_dim
from .hog_features import hog_features, hog_dim
from .lbp_features import lbp_features, lbp_dim


def extract_features(
    image: np.ndarray,
    is_rgb: bool = True,
    # Parámetros de cada descriptor
    hsv_bins: tuple = (16, 8, 8),
    hog_resize: tuple = (128, 128),
    hog_orientations: int = 9,
    hog_pixels_per_cell: tuple = (16, 16),
    hog_cells_per_block: tuple = (2, 2),
    lbp_resize: tuple = (128, 128),
    lbp_radius: int = 1,
    lbp_n_points: int = 8,
) -> np.ndarray:
    """
    Extrae el vector de features combinado (HSV + HOG + LBP) de una imagen.

    Returns
    -------
    np.ndarray
        Vector concatenado [hist_HSV, HOG, LBP]. Con los defaults: 1806 dims.
    """
    color = hsv_histogram(image, bins=hsv_bins, is_rgb=is_rgb, normalize=True)
    shape = hog_features(
        image,
        resize=hog_resize,
        orientations=hog_orientations,
        pixels_per_cell=hog_pixels_per_cell,
        cells_per_block=hog_cells_per_block,
        is_rgb=is_rgb,
    )
    texture = lbp_features(
        image,
        resize=lbp_resize,
        radius=lbp_radius,
        n_points=lbp_n_points,
        method="uniform",
        is_rgb=is_rgb,
        normalize=True,
    )
    return np.concatenate([color, shape, texture]).astype(np.float32)


def feature_dim(
    hsv_bins: tuple = (16, 8, 8),
    hog_resize: tuple = (128, 128),
    hog_orientations: int = 9,
    hog_pixels_per_cell: tuple = (16, 16),
    hog_cells_per_block: tuple = (2, 2),
    lbp_radius: int = 1,
    lbp_n_points: int = 8,
) -> int:
    """Devuelve la dimensión total del vector de features."""
    return (
        hsv_histogram_dim(hsv_bins)
        + hog_dim(hog_resize, hog_orientations, hog_pixels_per_cell, hog_cells_per_block)
        + lbp_dim(lbp_radius, lbp_n_points, method="uniform")
    )
