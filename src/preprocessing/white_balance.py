"""
Balance de blancos automático.

Implementa el algoritmo Gray World, que asume que la media de cada canal
RGB en una escena variada debería ser gris (R = G = B). Si alguna media
está desviada, es porque la luz de la escena tiene un tinte (focos
cálidos = más rojo, neones = más azul, etc.).

Por qué lo necesitamos en este proyecto:
- En supermercados la iluminación cambia mucho de un sitio a otro.
- Si no normalizamos el color, los rangos HSV del region proposal fallan
  cuando una misma lata se ve "más roja" o "más azul" según la tienda.
- Aplicar white balance ANTES de CLAHE da resultados muy estables.
"""

import cv2
import numpy as np


def gray_world_white_balance(
    image: np.ndarray,
    is_rgb: bool = True,
) -> np.ndarray:
    """
    Aplica balance de blancos por el método Gray World.

    Calcula la media de cada canal y la reescala para que las tres
    medias coincidan con la media global. Es un algoritmo simple, rápido
    y robusto para condiciones normales.

    Parameters
    ----------
    image : np.ndarray
        Imagen (H, W, 3) uint8.
    is_rgb : bool
        Si True, asume RGB; si False, BGR.

    Returns
    -------
    np.ndarray
        Imagen con balance de blancos corregido, mismo formato y dtype.
    """
    if image.dtype != np.uint8:
        raise ValueError(f"Esperaba uint8, recibí {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Esperaba imagen (H, W, 3), recibí {image.shape}")

    # Trabajamos en float para evitar overflow al multiplicar
    img_f = image.astype(np.float32)

    # Media por canal
    mean_per_channel = img_f.reshape(-1, 3).mean(axis=0)
    mean_global = mean_per_channel.mean()

    # Factor de escala para cada canal (evitar dividir por 0)
    scale = mean_global / np.maximum(mean_per_channel, 1e-6)

    # Aplicar y recortar al rango válido
    img_balanced = img_f * scale
    img_balanced = np.clip(img_balanced, 0, 255).astype(np.uint8)

    return img_balanced
