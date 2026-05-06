"""
Extracción de descriptores de textura (LBP).

Local Binary Patterns: para cada píxel, mira a sus N vecinos en un círculo
de radio R. Si el vecino es más claro que el píxel central, marca un 1; si
es más oscuro, un 0. Eso genera un código binario por píxel. Después se
hace un histograma de esos códigos sobre toda la región.

Por qué LBP en este proyecto:
- Captura TEXTURA, que es lo que falta a HSV (color) y HOG (forma):
  * Brillo metálico de una lata Monster vs.
  * Plástico mate de una botella vs.
  * Cartón rugoso de una caja vs.
  * Piel rugosa de una fruta.
- HSV los confunde si el color es similar; HOG los confunde si la silueta
  es similar; LBP los distingue por la microestructura.
- Es muy rápido y produce un vector pequeño (10 dims con uniform LBP).

Usamos la variante 'uniform' de skimage, que es la más robusta y
estándar (Ojala et al. 2002).
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


# Parámetros estándar de LBP uniform
DEFAULT_RESIZE = (128, 128)
DEFAULT_RADIUS = 1
DEFAULT_N_POINTS = 8 * DEFAULT_RADIUS    # 8 vecinos para R=1
DEFAULT_METHOD = "uniform"


def lbp_features(
    image: np.ndarray,
    resize: tuple = DEFAULT_RESIZE,
    radius: int = DEFAULT_RADIUS,
    n_points: int = DEFAULT_N_POINTS,
    method: str = DEFAULT_METHOD,
    is_rgb: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Calcula el descriptor LBP de una imagen.

    La imagen se convierte a escala de grises y se redimensiona antes de
    aplicar LBP.

    Parameters
    ----------
    image : np.ndarray
        Imagen (H, W, 3) uint8.
    resize : tuple
        Tamaño al que se redimensiona la imagen antes de aplicar LBP.
    radius : int
        Radio del círculo de vecinos. Más grande = patrones más globales.
    n_points : int
        Número de vecinos a considerar. Estándar: 8 * radius.
    method : str
        Variante de LBP. 'uniform' es la más usada y robusta.
    is_rgb : bool
        Si True, RGB; si False, BGR.
    normalize : bool
        Si True, normaliza el histograma para que sume 1.

    Returns
    -------
    np.ndarray
        Vector LBP. Para uniform con n_points=8: 10 dimensiones.
    """
    if image.dtype != np.uint8:
        raise ValueError(f"Esperaba uint8, recibí {image.dtype}")

    img_resized = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
    if is_rgb:
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray, n_points, radius, method=method)

    # Para uniform: los códigos posibles son n_points + 2 (n_points uniformes +
    # 1 categoría "no uniform" + extras según skimage). Hacemos histograma con
    # ese número de bins.
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)

    if normalize:
        total = hist.sum()
        if total > 0:
            hist = hist / total

    return hist


def lbp_dim(
    radius: int = DEFAULT_RADIUS,
    n_points: int = DEFAULT_N_POINTS,
    method: str = DEFAULT_METHOD,
) -> int:
    """
    Devuelve la dimensión aproximada del vector LBP uniform.

    Para 'uniform' con n_points vecinos: n_points + 2 bins.
    """
    if method == "uniform":
        return n_points + 2
    # Para otros métodos (default, ror, var) la dim varía
    return 2 ** n_points
