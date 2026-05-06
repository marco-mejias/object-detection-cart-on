"""Generación de regiones candidatas mediante visión clásica."""

from .hsv_proposal import (
    Proposal,
    build_mask,
    clean_mask,
    boxes_from_mask,
    non_max_suppression,
    propose_regions,
    draw_proposals,
)
from .hsv_ranges import HSV_RANGES, LOW_SATURATION_RANGE

__all__ = [
    "Proposal",
    "build_mask",
    "clean_mask",
    "boxes_from_mask",
    "non_max_suppression",
    "propose_regions",
    "draw_proposals",
    "HSV_RANGES",
    "LOW_SATURATION_RANGE",
]
