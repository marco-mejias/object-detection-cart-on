"""
Compara SVM (con grid search) y Random Forest sobre el dataset de features.

Pipeline:
1. Carga X, y desde results/dataset/.
2. Split train/val/test 70/15/15.
3. Grid search del SVM sobre (C, gamma) con cross-validation, eligiendo
   el mejor por F1 macro en val.
4. Entrena Random Forest con hiperparámetros razonables.
5. Evalúa ambos en test.
6. Genera una imagen comparativa con:
   - Tabla de métricas (accuracy, F1 por clase) lado a lado.
   - Matriz de confusión de cada modelo.
   - Importancia de features del RF (top 30).
7. Guarda ambos modelos en results/models/.

Uso:
    python scripts/compare_svm_rf.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.classification import (
    build_rf_pipeline,
    split_dataset,
)


DATASET_DIR = Path("results/dataset")
MODELS_DIR = Path("results/models")
FIGURES_DIR = Path("results/figures")


def grid_search_svm(X_train, y_train, X_val, y_val, verbose=True):
    """
    Hace grid search sobre el SVM probando varios (C, gamma) y devuelve
    el mejor pipeline ajustado a train, junto con todas las puntuaciones.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True)),
    ])

    param_grid = {
        "svm__C": [1.0, 10.0, 100.0],
        "svm__gamma": ["scale", 0.01, 0.001],
    }

    if verbose:
        print(f"Probando {len(param_grid['svm__C']) * len(param_grid['svm__gamma'])} combinaciones de SVM...")

    # 3-fold CV sobre train. Es razonable para nuestro tamaño (1848 muestras).
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=1 if verbose else 0,
        refit=True,
    )
    grid.fit(X_train, y_train)

    return grid


def evaluate_model(model, X_test, y_test, name):
    """Evalúa un modelo y devuelve métricas completas."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "name": name,
        "accuracy": acc,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "labels": labels,
        "y_true": y_test,
        "y_pred": y_pred,
        "report": report_dict,
    }


def plot_comparison(svm_results, rf_results, rf_pipeline, n_features, output_path):
    """
    Genera una figura comparativa con:
    - Matriz de confusión SVM (izquierda)
    - Matriz de confusión RF (centro)
    - Tabla con métricas resumen (derecha arriba)
    - Top features importantes del RF (derecha abajo)
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.1, 1], hspace=0.35, wspace=0.3)

    # --- Matrices de confusión ---
    for col, results in enumerate([svm_results, rf_results]):
        ax = fig.add_subplot(gs[0, col])
        cm = results["confusion_matrix"]
        labels = results["labels"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(f"{results['name']} — acc={results['accuracy']:.3f}, F1={results['f1_macro']:.3f}",
                      fontsize=12)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # --- Tabla de métricas resumen ---
    ax_table = fig.add_subplot(gs[0, 2])
    ax_table.axis("off")
    ax_table.set_title("Resumen comparativo", fontsize=12, pad=15)

    rows = [["Métrica", "SVM", "Random Forest"]]
    rows.append(["Accuracy", f"{svm_results['accuracy']:.3f}", f"{rf_results['accuracy']:.3f}"])
    rows.append(["F1 macro", f"{svm_results['f1_macro']:.3f}", f"{rf_results['f1_macro']:.3f}"])

    # F1 por clase
    for label in svm_results["labels"]:
        svm_f1 = svm_results["report"][label]["f1-score"]
        rf_f1 = rf_results["report"][label]["f1-score"]
        rows.append([f"F1 ({label})", f"{svm_f1:.3f}", f"{rf_f1:.3f}"])

    table = ax_table.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="center",
        loc="center",
        colWidths=[0.4, 0.3, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Resaltar el ganador en verde claro en cada fila numérica
    for i, row in enumerate(rows[1:], start=1):
        try:
            v_svm = float(row[1])
            v_rf = float(row[2])
            if v_svm > v_rf:
                table[(i, 1)].set_facecolor("#d4edda")
            elif v_rf > v_svm:
                table[(i, 2)].set_facecolor("#d4edda")
        except ValueError:
            pass

    # --- Importancia de features del RF (top 30) ---
    ax_imp = fig.add_subplot(gs[1, :])
    rf_estimator = rf_pipeline.named_steps["rf"]
    importances = rf_estimator.feature_importances_

    # Indicar a qué bloque pertenece cada feature
    HSV_DIM, HOG_DIM, LBP_DIM = 32, 1764, 10
    block_names = (["HSV"] * HSV_DIM) + (["HOG"] * HOG_DIM) + (["LBP"] * LBP_DIM)
    block_colors = {"HSV": "#3498db", "HOG": "#e67e22", "LBP": "#27ae60"}

    top_idx = np.argsort(importances)[::-1][:30]
    top_imp = importances[top_idx]
    top_blocks = [block_names[i] for i in top_idx]
    top_colors = [block_colors[b] for b in top_blocks]

    ax_imp.bar(range(len(top_imp)), top_imp, color=top_colors, edgecolor="black", linewidth=0.5)
    ax_imp.set_xticks(range(len(top_imp)))
    ax_imp.set_xticklabels([f"{idx}" for idx in top_idx], rotation=90, fontsize=8)
    ax_imp.set_xlabel("Índice de feature")
    ax_imp.set_ylabel("Importancia")
    ax_imp.set_title("Top 30 features más importantes para Random Forest", fontsize=12)

    # Leyenda con los bloques
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor="black", label=n)
                        for n, c in block_colors.items()]
    ax_imp.legend(handles=legend_elements, loc="upper right")

    # Resumen de cuánta importancia total acumula cada bloque
    total_by_block = {}
    for name in ["HSV", "HOG", "LBP"]:
        mask = np.array([b == name for b in block_names])
        total_by_block[name] = importances[mask].sum()
    summary = " | ".join(f"{n}: {v:.2%}" for n, v in total_by_block.items())
    ax_imp.text(0.5, -0.30, f"Importancia total acumulada por bloque  →  {summary}",
                 transform=ax_imp.transAxes, ha="center", fontsize=10,
                 style="italic")

    fig.suptitle("Comparativa SVM vs Random Forest sobre el dataset de categorías",
                  fontsize=14, fontweight="bold", y=0.995)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n✓  Comparativa visual guardada en: {output_path}")


def main():
    # Cargar dataset
    if not (DATASET_DIR / "X.npy").is_file():
        print(f"❌  No encuentro {DATASET_DIR / 'X.npy'}")
        print("Ejecuta primero: python scripts/build_dataset.py")
        return 1

    print("Cargando dataset...")
    X = np.load(DATASET_DIR / "X.npy")
    y = np.load(DATASET_DIR / "y.npy", allow_pickle=True)
    print(f"X: {X.shape}, y: {y.shape}")
    print(f"Clases: {sorted(set(y))}")
    print()

    print("Split train/val/test 70/15/15...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X, y, test_size=0.15, val_size=0.15, random_state=42,
    )
    print(f"  Train: {X_train.shape[0]}")
    print(f"  Val:   {X_val.shape[0]}")
    print(f"  Test:  {X_test.shape[0]}")
    print()

    # ------- SVM con grid search -------
    print("=" * 60)
    print("SVM con grid search")
    print("=" * 60)
    t0 = time.time()
    svm_grid = grid_search_svm(X_train, y_train, X_val, y_val, verbose=True)
    print(f"Mejor combinación: {svm_grid.best_params_}")
    print(f"Mejor F1 macro (CV): {svm_grid.best_score_:.4f}")
    print(f"Tiempo SVM: {time.time() - t0:.1f}s")

    svm_results = evaluate_model(svm_grid.best_estimator_, X_test, y_test,
                                   name=f"SVM (C={svm_grid.best_params_['svm__C']}, "
                                        f"γ={svm_grid.best_params_['svm__gamma']})")
    print(f"\nTest accuracy: {svm_results['accuracy']:.4f}")
    print(f"Test F1 macro: {svm_results['f1_macro']:.4f}")

    # ------- Random Forest -------
    print()
    print("=" * 60)
    print("Random Forest")
    print("=" * 60)
    t0 = time.time()
    rf_pipeline = build_rf_pipeline(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    rf_pipeline.fit(X_train, y_train)
    print(f"Tiempo RF: {time.time() - t0:.1f}s")

    rf_results = evaluate_model(rf_pipeline, X_test, y_test,
                                 name="Random Forest (300 árboles)")
    print(f"\nTest accuracy: {rf_results['accuracy']:.4f}")
    print(f"Test F1 macro: {rf_results['f1_macro']:.4f}")

    # ------- Comparativa visual -------
    print()
    print("=" * 60)
    print("Generando comparativa visual...")
    plot_comparison(
        svm_results, rf_results, rf_pipeline,
        n_features=X.shape[1],
        output_path=FIGURES_DIR / "comparacion_svm_rf.png",
    )

    # ------- Imprimir reports completos -------
    print()
    print("=" * 60)
    print("Classification report SVM")
    print("=" * 60)
    print(classification_report(svm_results["y_true"], svm_results["y_pred"],
                                  zero_division=0))
    print("=" * 60)
    print("Classification report Random Forest")
    print("=" * 60)
    print(classification_report(rf_results["y_true"], rf_results["y_pred"],
                                  zero_division=0))

    # ------- Guardar modelos -------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    svm_path = MODELS_DIR / "svm_categoria_gridsearch.joblib"
    rf_path = MODELS_DIR / "rf_categoria.joblib"

    joblib.dump({
        "model": svm_grid.best_estimator_,
        "best_params": svm_grid.best_params_,
        "metrics": {
            "test_accuracy": svm_results["accuracy"],
            "test_f1_macro": svm_results["f1_macro"],
        },
    }, svm_path)

    joblib.dump({
        "model": rf_pipeline,
        "metrics": {
            "test_accuracy": rf_results["accuracy"],
            "test_f1_macro": rf_results["f1_macro"],
        },
    }, rf_path)

    print(f"\n✓  Modelos guardados:")
    print(f"   - {svm_path}")
    print(f"   - {rf_path}")

    # ------- Veredicto final -------
    print()
    print("=" * 60)
    print("VEREDICTO")
    print("=" * 60)
    if svm_results["f1_macro"] > rf_results["f1_macro"]:
        diff = svm_results["f1_macro"] - rf_results["f1_macro"]
        print(f"Gana SVM por {diff:.4f} en F1 macro.")
    elif rf_results["f1_macro"] > svm_results["f1_macro"]:
        diff = rf_results["f1_macro"] - svm_results["f1_macro"]
        print(f"Gana Random Forest por {diff:.4f} en F1 macro.")
    else:
        print("Empate técnico.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
