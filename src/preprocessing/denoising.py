"""
Reducción de ruido (denoising).

Usamos el filtro bilateral porque suaviza el ruido SIN borrar los bordes,
lo cual es crítico para el region proposal que viene después (los bordes
son la base de la segmentación).

Comparación rápida:
- Filtro gaussiano: simple, rápido, pero borra bordes.
- Filtro bilateral: respeta bordes, pero es más lento.
- Non-Local Means: muy buena calidad pero lentísimo (no escala bien).

Para nuestro caso el bilateral es el equilibrio correcto.
"""

import cv2
import numpy as np


def bilateral_denoise(
    image: np.ndarray,
    diameter: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
    is_rgb: bool = True,
) -> np.ndarray:
    """
    Aplica filtro bilateral para reducir ruido preservando bordes.

    Parameters
    ----------
    image : np.ndarray
        Imagen (H, W, 3) uint8.
    diameter : int
        Tamaño del vecindario que se considera para cada píxel. Más
        grande = más suavizado, más lento. 5-9 es habitual.
    sigma_color : float
        Cuánto se mezclan colores distintos. Más alto = colores más
        diferentes se promedian.
    sigma_space : float
        Cuánto influye la distancia espacial. Más alto = vecinos lejanos
        cuentan más.
    is_rgb : bool
        Si True, RGB; si False, BGR.

    Returns
    -------
    np.ndarray
        Imagen con ruido reducido, mismo formato y dtype.
    """
    if image.dtype != np.uint8:
        raise ValueError(f"Esperaba uint8, recibí {image.dtype}")

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if is_rgb else image
    denoised = cv2.bilateralFilter(bgr, diameter, sigma_color, sigma_space)
    return cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB) if is_rgb else denoised
