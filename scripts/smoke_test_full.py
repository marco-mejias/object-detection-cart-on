"""
Smoke test ampliado del pipeline F2+F3.

Muestra para 4 imágenes reales del dataset Klasson:

    Col 1: Original
    Col 2: + White Balance
    Col 3: + CLAHE
    Col 4: + Denoise (preprocesado completo)
    Col 5: Region proposal sobre el preprocesado completo

Útil para ver visualmente el efecto de cada técnica clásica que ahora
tenemos implementada (WB + CLAHE + Denoising) y comprobar que el region
proposal sigue funcionando bien sobre las imágenes ya preprocesadas.

Uso:
    python scripts/smoke_test_full.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from src.preprocessing import (
    gray_world_white_balance,
    apply_clahe,
    bilateral_denoise,
    preprocess,
)
from src.region_proposal import propose_regions, draw_proposals
from src.utils.io_utils import load_image, list_images


DATA_ROOT = Path("data/external/klasson_flat")
OUTPUT = Path("results/figures/smoke_test_full.png")
N_IMAGES = 4


def main():
    if not DATA_ROOT.is_dir():
        print(f"❌  No encuentro {DATA_ROOT}")
        return 1

    classes = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])
    samples = []
    for class_dir in classes:
        images = list_images(class_dir)
        if images:
            samples.append((class_dir.name, images[0]))
        if len(samples) == N_IMAGES:
            break

    print(f"Probando con {len(samples)} imágenes:")
    for name, path in samples:
        print(f"  - {name}: {path.name}")
    print()

    fig, axes = plt.subplots(len(samples), 5, figsize=(20, 4 * len(samples)))
    if len(samples) == 1:
        axes = axes.reshape(1, -1)

    for i, (name, path) in enumerate(samples):
        img = load_image(path)
        wb = gray_world_white_balance(img)
        wb_clahe = apply_clahe(wb)
        full_pre = bilateral_denoise(wb_clahe)
        proposals = propose_regions(full_pre)
        annotated = draw_proposals(full_pre, proposals)

        steps = [
            (img,       f"{name} — original"),
            (wb,        "+ White Balance"),
            (wb_clahe,  "+ CLAHE"),
            (full_pre,  "+ Denoise (final)"),
            (annotated, f"Region proposal: {len(proposals)} cajas"),
        ]
        for ax, (im, title) in zip(axes[i], steps):
            ax.imshow(im)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=110, bbox_inches="tight")
    plt.close()

    print(f"✓  Resultado guardado en: {OUTPUT}")
    print("Ábrelo con tu visor de imágenes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())