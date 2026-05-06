"""
Construcción del dataset de features a partir de Klasson aplanado.

Recorre las imágenes de data/external/klasson_flat/, mapea cada clase a
una categoría del proyecto, extrae el vector de features (HSV + HOG) y
guarda dos arrays NumPy:

    results/dataset/X.npy        # features (n_samples, n_features)
    results/dataset/y.npy        # labels (n_samples,) como strings de categoría
    results/dataset/paths.npy    # rutas originales de cada imagen (para debug)

Se hace una única vez (la extracción HOG es lenta) y luego el script de
entrenamiento las carga rápidamente.

Uso:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --max-per-class 50   # para pruebas rápidas
"""

import argparse
import sys
import time
from pathlib import Path

# Permitir imports desde src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm

from src.features import extract_features, feature_dim
from src.classification.class_mapping import (
    map_klasson_class,
    PROJECT_CATEGORIES,
)
from src.utils.io_utils import load_image, list_images


DATA_ROOT = Path("data/external/combined")
OUT_DIR = Path("results/dataset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Máximo de imágenes por clase (para pruebas rápidas)")
    parser.add_argument("--data-root", default=str(DATA_ROOT),
                        help="Carpeta raíz del dataset aplanado")
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help="Carpeta donde guardar X.npy / y.npy")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)

    if not data_root.is_dir():
        print(f"❌  No encuentro {data_root}")
        print("Ejecuta primero: python scripts/flatten_klasson.py")
        return 1

    # Listar todas las clases (carpetas)
    class_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"❌  No hay clases en {data_root}")
        return 1

    print(f"Encontradas {len(class_dirs)} clases en {data_root}")
    print()

    # Construir lista de (path, categoria) antes de procesar nada
    samples = []
    unmapped = set()
    for class_dir in class_dirs:
        klasson_class = class_dir.name
        category = map_klasson_class(klasson_class)
        if category == "otros" and klasson_class not in (
            # listas de "otros" intencionales irían aquí; si no, avisamos
        ):
            unmapped.add(klasson_class)

        images = list_images(class_dir)
        if args.max_per_class:
            images = images[:args.max_per_class]
        for img_path in images:
            samples.append((img_path, category))

    if unmapped:
        print(f"⚠  {len(unmapped)} clases de Klasson no están mapeadas y van a 'otros':")
        for c in sorted(unmapped):
            print(f"   - {c}")
        print()
        print("Si alguna debería ir a otra categoría, edita")
        print("   src/classification/class_mapping.py")
        print()

    print(f"Total de imágenes a procesar: {len(samples)}")
    print()

    # Distribución por categoría
    print("Distribución por categoría del proyecto:")
    counter = {c: 0 for c in PROJECT_CATEGORIES}
    for _, cat in samples:
        counter[cat] = counter.get(cat, 0) + 1
    for cat in PROJECT_CATEGORIES:
        n = counter.get(cat, 0)
        bar = "█" * min(n // 10, 50)
        print(f"   {cat:10s} {n:5d}  {bar}")
    print()

    # Reservar memoria
    n_samples = len(samples)
    n_features = feature_dim()
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    y = np.empty(n_samples, dtype=object)
    paths = np.empty(n_samples, dtype=object)

    # Extraer features
    print("Extrayendo features (HSV + HOG)...")
    t0 = time.time()
    failed = 0
    for i, (img_path, category) in enumerate(tqdm(samples)):
        try:
            img = load_image(img_path)
            X[i] = extract_features(img)
            y[i] = category
            paths[i] = str(img_path)
        except Exception as e:
            print(f"\n⚠  Error con {img_path}: {e}")
            failed += 1
            y[i] = None  # marcar como fallida

    # Filtrar las que fallaron
    if failed > 0:
        keep = np.array([yi is not None for yi in y])
        X = X[keep]
        y = y[keep]
        paths = paths[keep]
        print(f"\n⚠  {failed} imágenes fallaron y se han descartado.")

    elapsed = time.time() - t0
    print(f"\nExtracción completa en {elapsed:.1f}s ({elapsed/len(samples)*1000:.1f} ms/img)")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Guardar
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "paths.npy", paths)
    print(f"\n✓  Guardado en {out_dir}/")
    print(f"   - X.npy        ({X.nbytes / 1024 / 1024:.1f} MB)")
    print(f"   - y.npy")
    print(f"   - paths.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
