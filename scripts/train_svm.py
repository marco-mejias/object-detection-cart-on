"""
Entrena el clasificador SVM sobre el dataset de features.

Carga X.npy, y.npy de results/dataset/, hace el split train/val/test,
entrena el SVM, evalúa, y guarda el modelo en results/models/.

Uso:
    python scripts/train_svm.py
    python scripts/train_svm.py --C 1.0 --gamma scale
"""

import argparse
import sys
import time
from pathlib import Path

# Permitir imports desde src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np

from src.classification import build_svm_pipeline, evaluate, split_dataset


DATASET_DIR = Path("results/dataset")
MODELS_DIR = Path("results/models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=10.0,
                        help="Regularización del SVM (default 10.0)")
    parser.add_argument("--gamma", default="scale",
                        help="Gamma del kernel RBF (default 'scale')")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    # Cargar dataset
    if not (DATASET_DIR / "X.npy").is_file():
        print(f"❌  No encuentro {DATASET_DIR / 'X.npy'}")
        print("Ejecuta primero: python scripts/build_dataset.py")
        return 1

    print("Cargando dataset...")
    X = np.load(DATASET_DIR / "X.npy")
    y = np.load(DATASET_DIR / "y.npy", allow_pickle=True)
    print(f"X: {X.shape}, y: {y.shape}")
    print(f"Clases presentes: {sorted(set(y))}")
    print()

    # Split
    print("Haciendo split train/val/test (70/15/15)...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X, y, test_size=0.15, val_size=0.15, random_state=args.random_state,
    )
    print(f"  Train: {X_train.shape[0]} muestras")
    print(f"  Val:   {X_val.shape[0]} muestras")
    print(f"  Test:  {X_test.shape[0]} muestras")
    print()

    # Pipeline
    print(f"Construyendo SVM (C={args.C}, gamma={args.gamma})...")
    model = build_svm_pipeline(C=args.C, gamma=args.gamma)

    # Entrenar
    print("Entrenando... (esto puede tardar varios minutos)")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"Entrenado en {elapsed:.1f}s")

    # Evaluar
    val_metrics = evaluate(model, X_val, y_val, label="Validación")
    test_metrics = evaluate(model, X_test, y_test, label="Test")

    # Guardar
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "svm_categoria.joblib"
    joblib.dump({
        "model": model,
        "config": {"C": args.C, "gamma": args.gamma, "random_state": args.random_state},
        "metrics": {
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
        },
    }, out_path)

    print(f"\n✓  Modelo guardado en: {out_path}")
    print(f"   Tamaño: {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Guardar también las predicciones de test para el notebook
    np.save(DATASET_DIR / "y_test.npy", test_metrics["y_true"])
    np.save(DATASET_DIR / "y_pred.npy", test_metrics["y_pred"])
    print(f"   Predicciones de test guardadas en {DATASET_DIR}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
