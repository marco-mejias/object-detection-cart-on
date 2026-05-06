"""
Utilidades de entrada/salida para cargar y guardar imágenes.

Centraliza la carga de imágenes para que todo el proyecto use el mismo
criterio (BGR vs RGB, redimensionado, etc.) y nadie se equivoque.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def load_image(path: str | Path, as_rgb: bool = True) -> np.ndarray:
    """
    Carga una imagen desde disco.

    Parameters
    ----------
    path : str o Path
        Ruta a la imagen.
    as_rgb : bool
        Si True (por defecto), devuelve la imagen en RGB.
        Si False, en BGR (formato nativo de OpenCV).

    Returns
    -------
    np.ndarray
        Imagen como array NumPy (H, W, 3) en uint8.

    Raises
    ------
    FileNotFoundError
        Si la imagen no existe o no se puede leer.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe: {path}")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")

    if as_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(img: np.ndarray, path: str | Path, is_rgb: bool = True) -> None:
    """
    Guarda una imagen en disco.

    Parameters
    ----------
    img : np.ndarray
        Imagen en (H, W, 3) uint8.
    path : str o Path
        Destino. La carpeta se crea si no existe.
    is_rgb : bool
        Si True, asume que la imagen está en RGB y la convierte a BGR
        antes de guardar (cv2.imwrite espera BGR).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if is_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise IOError(f"No se pudo guardar la imagen en: {path}")


def resize_keeping_aspect(
    img: np.ndarray,
    max_dim: int = 1024,
) -> Tuple[np.ndarray, float]:
    """
    Redimensiona una imagen manteniendo el aspect ratio, de forma que el
    lado más largo no supere `max_dim`.

    Útil para acelerar el preprocesado sin distorsionar las proporciones.

    Returns
    -------
    img_resized : np.ndarray
        Imagen redimensionada.
    scale : float
        Factor de escala aplicado (necesario si luego se quieren mapear
        bounding boxes al tamaño original).
    """
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img, 1.0

    scale = max_dim / longest
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_resized, scale


def list_images(folder: str | Path, recursive: bool = False) -> list[Path]:
    """
    Lista todas las imágenes (.jpg, .jpeg, .png) en una carpeta.

    Parameters
    ----------
    folder : str o Path
        Carpeta a explorar.
    recursive : bool
        Si True, busca también en subcarpetas.

    Returns
    -------
    list[Path]
        Rutas ordenadas alfabéticamente.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"No es una carpeta: {folder}")

    extensions = {".jpg", ".jpeg", ".png"}
    pattern = "**/*" if recursive else "*"
    images = [
        p for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return sorted(images)
