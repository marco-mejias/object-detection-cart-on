"""
Entrenamiento de un clasificador SVM con kernel RBF para categorías de
producto.

Pipeline:
1. Cargar X, y desde results/dataset/.
2. Split train/val/test 70/15/15 estratificado.
3. Estandarizar features (StandardScaler) — crítico para SVM.
4. Entrenar SVM con kernel RBF y parámetros razonables.
5. Evaluar en val y test.
6. Guardar modelo + scaler en un Pipeline en results/models/.
"""

from typing import Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split en train/val/test estratificado.

    Estratificado = mantiene la proporción de clases en cada partición.
    Esto es importante si las clases están desbalanceadas (ej: muchas
    frutas, pocas verduras).

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Primero separar test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )
    # Después separar val del resto
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_relative,
        stratify=y_temp,
        random_state=random_state,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_svm_pipeline(
    C: float = 10.0,
    gamma: str | float = "scale",
    class_weight: str = "balanced",
) -> Pipeline:
    """
    Construye un Pipeline con StandardScaler + SVM RBF.

    Parameters
    ----------
    C : float
        Regularización inversa. Más alto = más complejidad, más riesgo de overfitting.
        Valores típicos a probar en grid search: 0.1, 1, 10, 100.
    gamma : str o float
        Anchura del kernel RBF. 'scale' = 1 / (n_features * X.var()), buen default.
    class_weight : str
        'balanced' compensa clases desbalanceadas (recomendado).

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            class_weight=class_weight,
            probability=True,  # para poder predecir probabilidades luego
        )),
    ])


def evaluate(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    label: str = "evaluación",
) -> dict:
    """
    Evalúa un modelo y devuelve métricas (también las imprime).
    """
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")

    print(f"\n--- {label} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 macro:  {f1:.4f}")
    print()
    print(classification_report(y, y_pred, zero_division=0))

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "y_true": y,
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y, y_pred, labels=sorted(set(y))),
        "labels": sorted(set(y)),
    }
