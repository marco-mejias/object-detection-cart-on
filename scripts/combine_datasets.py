"""
Combina los datasets Klasson + Freiburg en un único directorio plano.

El resultado es:
    data/external/combined/<klasson_or_freiburg_class>/*.{jpg,png}

Donde <klasson_or_freiburg_class> es el nombre de carpeta original de
cada dataset. Usamos enlaces simbólicos para no duplicar espacio.

Después de combinar, build_dataset.py se aplica sobre esta carpeta y
el mapeo a categorías del proyecto se hace en class_mapping.py.

Uso:
    python scripts/combine_datasets.py
"""

from pathlib import Path

KLASSON_FLAT = Path("data/external/klasson_flat")
FREIBURG_IMAGES = Path("data/external/freiburg_groceries/images")
COMBINED = Path("data/external/combined")


def main():
    if not KLASSON_FLAT.is_dir():
        print(f"❌  No encuentro {KLASSON_FLAT}")
        print("Ejecuta antes: python scripts/flatten_klasson.py")
        return 1

    if not FREIBURG_IMAGES.is_dir():
        print(f"❌  No encuentro {FREIBURG_IMAGES}")
        print("Ejecuta antes: python scripts/download_freiburg_v2.py")
        return 1

    if COMBINED.exists():
        print(f"Ya existe {COMBINED}. Bórralo si quieres regenerar:")
        print(f"  rm -rf {COMBINED}")
        return 0

    COMBINED.mkdir(parents=True)
    counter = {}

    # ---- Klasson (.jpg) ----
    print("Enlazando imágenes de Klasson...")
    for class_dir in KLASSON_FLAT.iterdir():
        if not class_dir.is_dir():
            continue
        target = COMBINED / class_dir.name
        target.mkdir(exist_ok=True)
        n = 0
        for img in class_dir.glob("*.jpg"):
            link = target / f"klasson_{img.name}"
            if not link.exists():
                # Si el original ya es un symlink, resolver a la ruta real
                source = img.resolve()
                link.symlink_to(source)
                n += 1
        counter[class_dir.name] = n

    # ---- Freiburg (.png) ----
    print("Enlazando imágenes de Freiburg...")
    for class_dir in FREIBURG_IMAGES.iterdir():
        if not class_dir.is_dir():
            continue
        target = COMBINED / class_dir.name
        target.mkdir(exist_ok=True)
        n = 0
        for img in class_dir.glob("*.png"):
            link = target / f"freiburg_{img.name}"
            if not link.exists():
                link.symlink_to(img.resolve())
                n += 1
        counter[class_dir.name] = counter.get(class_dir.name, 0) + n

    # Resumen
    total = sum(counter.values())
    print()
    print(f"✓  Dataset combinado en: {COMBINED}")
    print(f"   Total clases: {len(counter)}")
    print(f"   Total imágenes: {total}")
    print()
    print("Top 15 clases por número de imágenes:")
    for name, n in sorted(counter.items(), key=lambda x: -x[1])[:15]:
        print(f"   {name:30s} {n:5d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
