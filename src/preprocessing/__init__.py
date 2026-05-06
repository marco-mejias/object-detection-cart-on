"""Preprocesado clásico de imágenes."""

from .clahe import (
    preprocess,
    apply_clahe,
    gray_world_white_balance,
    bilateral_denoise,
)

__all__ = [
    "preprocess",
    "apply_clahe",
    "gray_world_white_balance",
    "bilateral_denoise",
]
