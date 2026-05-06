"""Clasificación de regiones por categoría con técnicas clásicas de ML."""

from .class_mapping import (
    map_klasson_class,
    PROJECT_CATEGORIES,
    KLASSON_TO_CATEGORY,
    get_unmapped_classes,
)
from .svm_trainer import (
    split_dataset,
    build_svm_pipeline,
    evaluate,
)
from .rf_trainer import build_rf_pipeline

__all__ = [
    "map_klasson_class",
    "PROJECT_CATEGORIES",
    "KLASSON_TO_CATEGORY",
    "get_unmapped_classes",
    "split_dataset",
    "build_svm_pipeline",
    "build_rf_pipeline",
    "evaluate",
]
