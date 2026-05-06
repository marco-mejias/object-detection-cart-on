"""
Crea una vista 'plana' del dataset Klasson en data/external/klasson_flat/.

El dataset Klasson original tiene jerarquía:
    train/Fruit/Apple/Golden-Delicious/*.jpg
    train/Fruit/Apple/Granny-Smith/*.jpg
    train/Packages/Juice/Bravo-Apple-Juice/*.jpg
    ...

Esta jerarquía dificulta usar el dataset con los notebooks de F2, que
esperan una estructura plana del estilo:
    klasson_flat/<clase>/*.jpg

Este script crea esa estructura usando ENLACES SIMBÓLICOS (no copias),
así que no ocupa espacio extra y es trivial revertir borrando la carpeta.

Por defecto agrupa al nivel medio (Apple, Banana, Juice, Milk...) que es
el más útil para clasificación por categoría. Si quieres otro nivel,
pasa --level fine (último nivel, más detallado) o --level coarse (raíz:
solo Fruit/Packages/Vegetables).

Uso:
    python scripts/flatten_klasson.py
    python scripts/flatten_klasson.py --level fine
"""

import argparse
from pathlib import Path

SRC_ROOT = Path("data/external/grocery_store/dataset/train")
DST_ROOT = Path("data/external/klasson_flat")


def main():
    parser = argparse.ArgumentParser(description="Aplana el dataset Klasson.")
    parser.add_argument(
        "--level", default="mid",
        choices=["coarse", "mid", "fine"],
        help=(
            "Nivel de agrupación: "
            "coarse=Fruit/Packages/Vegetables, "
            "mid=Apple/Juice/Milk/... (recomendado), "
            "fine=Golden-Delicious/Bravo-Juice/... (subclase)"
        ),
    )
    parser.add_argument(
        "--src", default=str(SRC_ROOT),
        help="Carpeta train original",
    )
    parser.add_argument(
        "--dst", default=str(DST_ROOT),
        help="Carpeta de la vista plana",
    )
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.is_dir():
        print(f"❌  No existe: {src}")
        print("Descarga primero el dataset con: python scripts/download_klasson.py")
        return 1

    if dst.exists():
        print(f"Ya existe {dst}. Bórralo si quieres regenerar:")
        print(f"  rm -rf {dst}")
        return 0

    dst.mkdir(parents=True)

    # Recorrer según el nivel pedido
    counter: dict[str, int] = {}

    if args.level == "coarse":
        # train/<clase>/  → la clase es el primer nivel (Fruit, Packages, Vegetables)
        for class_dir in src.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            target_dir = dst / class_name
            target_dir.mkdir(exist_ok=True)
            for img in class_dir.rglob("*.jpg"):
                link = target_dir / f"{img.parent.parent.name}_{img.parent.name}_{img.name}"
                if not link.exists():
                    link.symlink_to(img.resolve())
                    counter[class_name] = counter.get(class_name, 0) + 1

    elif args.level == "mid":
        # train/<grupo>/<clase>/  → la clase es el segundo nivel
        for group_dir in src.iterdir():
            if not group_dir.is_dir():
                continue
            for class_dir in group_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                target_dir = dst / class_name
                target_dir.mkdir(exist_ok=True)
                for img in class_dir.rglob("*.jpg"):
                    # Renombramos para evitar colisiones entre subclases
                    link = target_dir / f"{img.parent.name}_{img.name}"
                    if not link.exists():
                        link.symlink_to(img.resolve())
                        counter[class_name] = counter.get(class_name, 0) + 1

    elif args.level == "fine":
        # Último nivel (subclase concreta)
        for img in src.rglob("*.jpg"):
            class_name = img.parent.name
            target_dir = dst / class_name
            target_dir.mkdir(exist_ok=True)
            link = target_dir / img.name
            if not link.exists():
                link.symlink_to(img.resolve())
                counter[class_name] = counter.get(class_name, 0) + 1

    # Resumen
    print()
    print(f"✓  Vista plana creada en: {dst}")
    print(f"   Nivel: {args.level}")
    print(f"   Clases: {len(counter)}")
    print(f"   Total imágenes: {sum(counter.values())}")
    print()
    print("Top 10 clases por número de imágenes:")
    for name, n in sorted(counter.items(), key=lambda x: -x[1])[:10]:
        print(f"   {name:30s} {n:4d}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())