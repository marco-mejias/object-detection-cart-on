"""
Preprocesado clásico de imágenes para el pipeline de visión.

Pipeline completo en este orden:

    1. White balance (Gray World)  → corrige el tinte de la iluminación
    2. CLAHE sobre canal L de Lab  → iguala el contraste local
    3. Denoising bilateral         → reduce ruido conservando bordes

Por qué este orden:
- White balance primero porque trabaja sobre la imagen "cruda" tal como
  viene de la cámara. Si lo hacemos después de CLAHE, el contraste
  artificial puede confundir la estimación de medias.
- CLAHE en segundo lugar para igualar contraste sobre colores ya neutros.
- Denoising al final, después de CLAHE, porque CLAHE puede amplificar
  ruido en zonas oscuras y queremos limpiarlo antes de pasar al region
  proposal.
"""

from typing import Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. White balance (Gray World)
# ---------------------------------------------------------------------------

def gray_world_white_balance(
    image: np.ndarray,
    is_rgb: bool = True,
) -> np.ndarray:
    """
    Balance de blancos automático por el método Gray World.

    Asume que la media de cada canal RGB debería ser gris en una escena
    variada. Si alguna media está desviada, es porque la luz tiene tinte
    (focos cálidos, neones, etc.) y reescala para compensar.
    """
    if image.dtype != np.uint8:
        raise ValueError(f"Esperaba uint8, recibí {image.dtype}")

    img_f = image.astype(np.float32)
    mean_per_channel = img_f.reshape(-1, 3).mean(axis=0)
    mean_global = mean_per_channel.mean()
    scale = mean_global / np.maximum(mean_per_channel, 1e-6)
    return np.clip(img_f * scale, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. CLAHE sobre canal L de Lab
# ---------------------------------------------------------------------------

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    is_rgb: bool = True,
) -> np.ndarray:
    """
    Aplica CLAHE sobre el canal L del espacio Lab, conservando los colores.
    """
    if image.dtype != np.uint8:
        raise ValueError(f"Esperaba uint8, recibí {image.dtype}")

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if is_rgb else image
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    bgr_out = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB) if is_rgb else bgr_out


# ---------------------------------------------------------------------------
# 3. Denoising (filtro bilateral)
# ---------------------------------------------------------------------------

def bilateral_denoise(
    image: np.ndarray,
    diameter: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
    is_rgb: bool = True,
) -> np.ndarray:
    """
    Filtro bilateral: suaviza ruido preservando bordes.
    """
    if image.dtype != np.uint8:
        raise ValueError(f"Esperaba uint8, recibí {image.dtype}")

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if is_rgb else image
    denoised = cv2.bilateralFilter(bgr, diameter, sigma_color, sigma_space)
    return cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB) if is_rgb else denoised


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def preprocess(
    image: np.ndarray,
    is_rgb: bool = True,
    apply_wb: bool = True,
    apply_clahe_step: bool = True,
    apply_denoise: bool = True,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    denoise_diameter: int = 7,
    denoise_sigma_color: float = 50.0,
    denoise_sigma_space: float = 50.0,
) -> np.ndarray:
    """
    Pipeline de preprocesado completo: White Balance → CLAHE → Denoising.

    Cada paso se puede desactivar con su flag correspondiente, útil para
    experimentar y aislar el efecto de cada técnica.

    Parameters
    ----------
    image : np.ndarray
        Imagen (H, W, 3) uint8.
    is_rgb : bool
        Si True, RGB; si False, BGR.
    apply_wb, apply_clahe_step, apply_denoise : bool
        Permiten activar/desactivar cada paso individualmente.
    clahe_clip_limit, clahe_tile_grid_size : ver `apply_clahe`.
    denoise_diameter, denoise_sigma_color, denoise_sigma_space : ver
        `bilateral_denoise`.

    Returns
    -------
    np.ndarray
        Imagen preprocesada, mismo formato y dtype que la entrada.
    """
    out = image
    if apply_wb:
        out = gray_world_white_balance(out, is_rgb=is_rgb)
    if apply_clahe_step:
        out = apply_clahe(out,
                          clip_limit=clahe_clip_limit,
                          tile_grid_size=clahe_tile_grid_size,
                          is_rgb=is_rgb)
    if apply_denoise:
        out = bilateral_denoise(out,
                                diameter=denoise_diameter,
                                sigma_color=denoise_sigma_color,
                                sigma_space=denoise_sigma_space,
                                is_rgb=is_rgb)
    return out
