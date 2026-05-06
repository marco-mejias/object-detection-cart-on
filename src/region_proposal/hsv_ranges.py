"""
Rangos HSV por categoría de producto.

Define los rangos de color en HSV (Hue 0-179, Sat 0-255, Val 0-255 en
OpenCV) que vamos a usar para detectar regiones candidatas de cada
categoría de producto.

¡IMPORTANTE! Estos rangos son un PUNTO DE PARTIDA. Hay que afinarlos
empíricamente sobre las imágenes reales del dataset usando el notebook
`03_region_proposal.ipynb`. Los valores actuales son razonables pero no
están optimizados.

Notas sobre HSV en OpenCV:
- Hue va de 0 a 179 (no 0-360). Rojo ≈ 0 o 179, verde ≈ 60, azul ≈ 120.
- El rojo cruza el 0, así que necesita DOS rangos (uno cerca de 0 y otro
  cerca de 179) que luego se combinan.
- Para colores poco saturados (blancos, grises) los rangos por color no
  funcionan bien; mejor usar saturación baja como criterio.
"""

from typing import Dict, List, Tuple

import numpy as np

# Tipo: cada categoría tiene una lista de (lower_hsv, upper_hsv).
# La lista permite definir múltiples rangos para una misma categoría
# (necesario para el rojo, que cruza Hue=0).
HSVRange = Tuple[np.ndarray, np.ndarray]
CategoryRanges = Dict[str, List[HSVRange]]


def _r(h_lo: int, s_lo: int, v_lo: int,
       h_hi: int, s_hi: int, v_hi: int) -> HSVRange:
    """Helper para crear rangos sin repetir np.array() todo el rato."""
    return (np.array([h_lo, s_lo, v_lo], dtype=np.uint8),
            np.array([h_hi, s_hi, v_hi], dtype=np.uint8))


# Rangos iniciales por categoría. Pensar en términos de:
# - Frutas frescas: colores saturados naturales (verdes, rojos, naranjas, amarillos)
# - Latas: muchas son rojas, azules, plateadas (saturación media-alta)
# - Botellas: a menudo transparentes (saturación baja) o coloreadas
# - Cajas: cartón (marrón) o packaging colorido
HSV_RANGES: CategoryRanges = {
    "rojo": [
        _r(0,   80, 50,   10, 255, 255),    # Rojo "bajo"
        _r(170, 80, 50,  179, 255, 255),    # Rojo "alto" (cruza el 0)
    ],
    "naranja": [
        _r(11,  80, 80,   25, 255, 255),
    ],
    "amarillo": [
        _r(26,  80, 80,   34, 255, 255),
    ],
    "verde": [
        _r(35,  60, 50,   85, 255, 255),
    ],
    "azul": [
        _r(86,  80, 50,  130, 255, 255),
    ],
    "morado": [
        _r(131, 60, 50,  160, 255, 255),
    ],
    "rosa": [
        _r(161, 60, 80,  169, 255, 255),
    ],
}

# Categoría especial: regiones poco saturadas (blancos, grises, plateados).
# Útil para detectar latas metálicas o packaging blanco.
LOW_SATURATION_RANGE: HSVRange = _r(0, 0, 100,   179, 60, 255)
