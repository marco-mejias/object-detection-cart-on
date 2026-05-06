"""
Estadísticas del dataset Monster.

Cuenta cuántas fotos hay por variante y por tienda en data/raw/, y muestra
un resumen para ver cómo va la captura.

Uso:
    python scripts/dataset_stats.py
    python scripts/dataset_stats.py --root ./data/raw
"""

import argparse
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Estadísticas del dataset capturado.")
    parser.add_argument("--root", default="./data/raw", help="Carpeta raíz de las fotos")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR: la carpeta no existe: {root}")
        return 1

    # contador[variante][tienda] = num_fotos
    counter = defaultdict(lambda: defaultdict(int))
    total = 0

    for variante_dir in sorted(root.iterdir()):
        if not variante_dir.is_dir():
            continue
        variante = variante_dir.name

        for photo in variante_dir.glob("monster_*.jpg"):
            parts = photo.stem.split("_")
            # Formato: monster_<tienda>_<variante>_<numero>
            # tienda puede tener letras/números pero no _
            if len(parts) >= 4:
                tienda = parts[1]
                counter[variante][tienda] += 1
                total += 1

    if total == 0:
        print("No hay fotos en el dataset todavía.")
        print(f"Esperando fotos en {root}/<variante>/")
        return 0

    print("=" * 60)
    print(f"DATASET MONSTER — {total} fotos en total")
    print("=" * 60)
    print()

    for variante in sorted(counter.keys()):
        subtotal = sum(counter[variante].values())
        print(f"  {variante.upper()} ({subtotal} fotos)")
        for tienda in sorted(counter[variante].keys()):
            n = counter[variante][tienda]
            bar = "█" * min(n, 40)
            print(f"    {tienda:20s} {n:4d}  {bar}")
        print()

    # Aviso si está muy desbalanceado
    subtotales = {v: sum(t.values()) for v, t in counter.items()}
    if subtotales:
        max_v = max(subtotales.values())
        min_v = min(subtotales.values())
        if min_v > 0 and max_v / min_v > 2:
            print("⚠  Aviso: hay desbalance importante entre variantes (>2x).")
            print("   Considerad capturar más de la(s) menos representada(s).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
