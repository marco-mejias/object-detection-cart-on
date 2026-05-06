"""
Generación de regiones candidatas (region proposal) por segmentación HSV.

Este módulo es el núcleo de la parte clásica de detección. Dada una imagen,
genera una lista de "cajas candidatas" (bounding boxes) donde podría haber
un producto, sin usar Deep Learning.

Pipeline interno:
1. Convertir a HSV.
2. Para cada categoría definida en `hsv_ranges.py`, crear una máscara
   binaria con los píxeles que caen en su rango.
3. Limpiar las máscaras con morfología (closing rellena agujeros, opening
   elimina puntitos).
4. Extraer contornos de cada máscara y calcular sus bounding boxes.
5. Filtrar las cajas demasiado pequeñas, demasiado grandes, o con aspect
   ratio raro.
6. Aplicar NMS para fusionar cajas que se solapen mucho.

El resultado es una lista de cajas (x, y, w, h) con su categoría tentativa.
"""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .hsv_ranges import HSV_RANGES, LOW_SATURATION_RANGE


@dataclass
class Proposal:
    """Una región candidata."""
    x: int
    y: int
    w: int
    h: int
    category: str       # categoría HSV que generó la propuesta
    score: float        # área relativa (proxy de confianza)

    @property
    def box(self) -> tuple:
        return (self.x, self.y, self.w, self.h)

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def aspect_ratio(self) -> float:
        return self.w / self.h if self.h > 0 else 0.0


def build_mask(
    hsv: np.ndarray,
    ranges: list,
) -> np.ndarray:
    """
    Construye una máscara binaria combinando varios rangos HSV.

    Parameters
    ----------
    hsv : np.ndarray
        Imagen ya convertida a HSV (H, W, 3) uint8.
    ranges : list of (lower, upper)
        Lista de pares (lower_hsv, upper_hsv).

    Returns
    -------
    np.ndarray
        Máscara binaria (H, W) uint8 con valores 0 o 255.
    """
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        partial = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_or(mask, partial)
    return mask


def clean_mask(
    mask: np.ndarray,
    kernel_size: int = 5,
    closing_iters: int = 2,
    opening_iters: int = 1,
) -> np.ndarray:
    """
    Limpia una máscara binaria con operaciones morfológicas.

    - Closing (dilation→erosion): rellena pequeños agujeros dentro de los
      objetos detectados.
    - Opening (erosion→dilation): elimina pequeñas motas de ruido fuera.

    Parameters
    ----------
    mask : np.ndarray
        Máscara binaria.
    kernel_size : int
        Tamaño del kernel cuadrado. Más grande = más agresivo.
    closing_iters, opening_iters : int
        Número de iteraciones para cada operación.

    Returns
    -------
    np.ndarray
        Máscara limpia.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=closing_iters)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=opening_iters)
    return opened


def boxes_from_mask(
    mask: np.ndarray,
    category: str,
    image_area: int,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.5,
    min_aspect: float = 0.2,
    max_aspect: float = 5.0,
) -> List[Proposal]:
    """
    Extrae bounding boxes de una máscara binaria.

    Filtra las cajas según criterios de tamaño y forma para descartar
    ruido (cajas microscópicas) y falsos positivos masivos (cajas que
    cubren casi toda la imagen, normalmente el fondo).

    Parameters
    ----------
    mask : np.ndarray
        Máscara binaria limpia.
    category : str
        Categoría que vamos a asociar a cada propuesta extraída.
    image_area : int
        Área total de la imagen (W * H), para los ratios.
    min_area_ratio, max_area_ratio : float
        Rango aceptable de área respecto al total de la imagen.
        Por defecto: entre 0.1% y 50% de la imagen.
    min_aspect, max_aspect : float
        Rango aceptable de aspect ratio (w/h). Por defecto entre 0.2 (muy
        alto) y 5.0 (muy ancho).

    Returns
    -------
    list of Proposal
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = image_area * min_area_ratio
    max_area = image_area * max_area_ratio

    proposals: List[Proposal] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area < min_area or area > max_area:
            continue

        aspect = w / h if h > 0 else 0
        if aspect < min_aspect or aspect > max_aspect:
            continue

        proposals.append(Proposal(
            x=x, y=y, w=w, h=h,
            category=category,
            score=area / image_area,
        ))

    return proposals


def non_max_suppression(
    proposals: List[Proposal],
    iou_threshold: float = 0.4,
) -> List[Proposal]:
    """
    Non-Maximum Suppression: elimina cajas redundantes.

    Cuando dos categorías HSV detectan la misma región (por ejemplo, una
    lata roja con etiqueta naranja), nos queda con la propuesta de mayor
    score y descarta las que se solapen demasiado con ella.

    Parameters
    ----------
    proposals : list of Proposal
    iou_threshold : float
        Si dos cajas tienen IoU mayor que este umbral, se considera que
        son la misma y se descarta la de menor score.

    Returns
    -------
    list of Proposal
        Lista filtrada, ordenada por score descendente.
    """
    if not proposals:
        return []

    # Convertir a formato (x1, y1, x2, y2) para calcular IoU fácilmente
    boxes = np.array([[p.x, p.y, p.x + p.w, p.y + p.h] for p in proposals],
                     dtype=np.float32)
    scores = np.array([p.score for p in proposals], dtype=np.float32)

    # Ordenar por score descendente
    order = scores.argsort()[::-1]
    keep_indices: List[int] = []

    while order.size > 0:
        i = order[0]
        keep_indices.append(int(i))

        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        intersection = inter_w * inter_h

        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_i + area_rest - intersection

        iou = np.where(union > 0, intersection / union, 0.0)
        order = rest[iou < iou_threshold]

    return [proposals[i] for i in keep_indices]


def propose_regions(
    image: np.ndarray,
    is_rgb: bool = True,
    kernel_size: int = 7,                # ← antes 5: morfología más agresiva
    closing_iters: int = 2,
    opening_iters: int = 2,              # ← antes 1: limpiar más motas
    min_area_ratio: float = 0.005,       # ← antes 0.001: descarta cajas microscópicas
    max_area_ratio: float = 0.5,
    min_aspect: float = 0.2,
    max_aspect: float = 5.0,
    iou_threshold: float = 0.3,          # ← antes 0.4: NMS más agresivo
    use_low_saturation: bool = True,
) -> List[Proposal]:
    """
    Función principal: dada una imagen, devuelve la lista de regiones
    candidatas.

    Pipeline completo: HSV → máscaras por categoría → morfología →
    bounding boxes filtradas → NMS.

    Parameters
    ----------
    image : np.ndarray
        Imagen (H, W, 3) uint8.
    is_rgb : bool
        Si True, la imagen está en RGB; si False, en BGR.
    kernel_size, closing_iters, opening_iters
        Parámetros morfológicos (ver `clean_mask`).
    min_area_ratio, max_area_ratio, min_aspect, max_aspect
        Filtros de tamaño/forma (ver `boxes_from_mask`).
    iou_threshold : float
        Umbral de NMS.
    use_low_saturation : bool
        Si True, también busca regiones poco saturadas (blancos, grises).

    Returns
    -------
    list of Proposal
        Cajas candidatas, ordenadas por score descendente.
    """
    # OpenCV trabaja en BGR
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if is_rgb else image
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    image_area = h * w

    all_proposals: List[Proposal] = []

    # Una máscara y un set de cajas por categoría de color
    for category, ranges in HSV_RANGES.items():
        mask = build_mask(hsv, ranges)
        mask = clean_mask(mask, kernel_size, closing_iters, opening_iters)
        boxes = boxes_from_mask(
            mask, category, image_area,
            min_area_ratio, max_area_ratio,
            min_aspect, max_aspect,
        )
        all_proposals.extend(boxes)

    # Categoría especial: poca saturación (blancos/grises/plateados)
    if use_low_saturation:
        mask = build_mask(hsv, [LOW_SATURATION_RANGE])
        mask = clean_mask(mask, kernel_size, closing_iters, opening_iters)
        boxes = boxes_from_mask(
            mask, "neutro", image_area,
            min_area_ratio, max_area_ratio,
            min_aspect, max_aspect,
        )
        all_proposals.extend(boxes)

    # NMS final para fusionar cajas redundantes entre categorías
    return non_max_suppression(all_proposals, iou_threshold)


def draw_proposals(
    image: np.ndarray,
    proposals: List[Proposal],
    is_rgb: bool = True,
    thickness: int = 2,
    show_labels: bool = True,
) -> np.ndarray:
    """
    Dibuja las cajas candidatas sobre una copia de la imagen.

    Útil para visualización y depuración. Cada categoría usa un color
    distinto.
    """
    # Paleta de colores en BGR (uno por categoría)
    palette = {
        "rojo":     (0,   0, 255),
        "naranja":  (0, 140, 255),
        "amarillo": (0, 255, 255),
        "verde":    (0, 200,   0),
        "azul":     (255, 0,   0),
        "morado":   (200, 0, 200),
        "rosa":     (200, 100, 255),
        "neutro":   (200, 200, 200),
    }

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if is_rgb else image.copy()

    for p in proposals:
        color = palette.get(p.category, (255, 255, 255))
        cv2.rectangle(bgr, (p.x, p.y), (p.x + p.w, p.y + p.h), color, thickness)
        if show_labels:
            label = p.category
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                   0.5, 1)
            cv2.rectangle(bgr,
                          (p.x, p.y - text_h - 6),
                          (p.x + text_w + 4, p.y),
                          color, -1)
            cv2.putText(bgr, label, (p.x + 2, p.y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if is_rgb else bgr
