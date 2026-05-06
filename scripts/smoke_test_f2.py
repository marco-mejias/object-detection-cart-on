"""
Test rápido del pipeline F2 (preprocesado + region proposal).

Coge una imagen real del dataset Klasson, le aplica el pipeline entero,
y guarda el resultado en results/figures/smoke_test.png para que lo
visualices.

Uso:
    python scripts/smoke_test_f2.py
"""

import sys
from pathlib import Path

# Permite importar desde src/ aunque ejecutemos desde scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from src.preprocessing import preprocess
from src.region_proposal import propose_regions, draw_proposals
from src.utils.io_utils import load_image, list_images


DATA_ROOT = Path("data/external/klasson_flat")
OUTPUT = Path("results/figures/smoke_test_f2.png")


def main():
    if not DATA_ROOT.is_dir():
        print(f"❌  No encuentro {DATA_ROOT}")
        print("Ejecuta primero: python scripts/flatten_klasson.py")
        return 1

    # Coger 4 imágenes de 4 categorías distintas
    classes = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
    if len(classes) < 4:
        print(f"❌  Solo encuentro {len(classes)} clases en {DATA_ROOT}, necesito al menos 4.")
        return 1

    # Elegir 4 que tengan al menos una imagen
    samples = []
    for class_dir in classes:
        images = list_images(class_dir)
        if images:
            samples.append((class_dir.name, images[0]))
        if len(samples) == 4:
            break

    print(f"Probando con {len(samples)} imágenes:")
    for name, path in samples:
        print(f"  - {name}: {path.name}")

    # Aplicar pipeline a cada una
    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))
    if len(samples) == 1:
        axes = axes.reshape(1, -1)

    for i, (name, path) in enumerate(samples):
        img = load_image(path)
        img_pre = preprocess(img)
        proposals = propose_regions(img_pre)
        annotated = draw_proposals(img_pre, proposals)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"{name} — original", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_pre)
        axes[i, 1].set_title(f"{name} — preprocesada (CLAHE)", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(annotated)
        axes[i, 2].set_title(f"{name} — {len(proposals)} propuestas", fontsize=10)
        axes[i, 2].axis("off")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=120, bbox_inches="tight")
    plt.close()

    print()
    print(f"✓  Resultado guardado en: {OUTPUT}")
    print("Ábrelo con tu visor de imágenes para ver si funciona correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())