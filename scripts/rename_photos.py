"""
Renombrador de fotos del dataset Monster.

Convierte fotos del móvil (IMG_1234.jpg, foto_001.HEIC, etc.) al formato
estándar del proyecto:

    monster_<tienda>_<variante>_<numero>.jpg

Uso:
    python scripts/rename_photos.py --input ./fotos_movil --tienda mercadona01 --variante original
    python scripts/rename_photos.py --input ./fotos_movil --tienda carrefour02 --variante ultra --start 50

Si --start no se especifica, mira las fotos ya existentes en data/raw/<variante>/
y empieza a numerar a continuación, evitando colisiones.

Por defecto las fotos se COPIAN (no se mueven) al destino, por seguridad.
Si quieres moverlas, usa --move.
"""

import argparse
import shutil
from pathlib import Path

VALID_VARIANTS = {"original", "ultra", "mango_loco", "pipeline_punch", "assault"}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def find_next_number(output_dir: Path, tienda: str, variante: str) -> int:
    """Busca el siguiente número libre para esta tienda/variante en output_dir."""
    pattern = f"monster_{tienda}_{variante}_*.jpg"
    existing = list(output_dir.glob(pattern))
    if not existing:
        return 1
    numbers = []
    for f in existing:
        try:
            num = int(f.stem.split("_")[-1])
            numbers.append(num)
        except ValueError:
            continue
    return max(numbers) + 1 if numbers else 1


def main():
    parser = argparse.ArgumentParser(description="Renombra fotos al formato del proyecto.")
    parser.add_argument("--input", required=True, help="Carpeta con las fotos del móvil")
    parser.add_argument("--tienda", required=True, help="Identificador de la tienda (mercadona01, carrefour02, ...)")
    parser.add_argument("--variante", required=True, help=f"Variante de Monster: {VALID_VARIANTS}")
    parser.add_argument("--start", type=int, default=None, help="Número desde el que empezar (auto si no se indica)")
    parser.add_argument("--output", default="./data/raw", help="Carpeta destino raíz (por defecto data/raw)")
    parser.add_argument("--move", action="store_true", help="Mover en vez de copiar")
    parser.add_argument("--dry-run", action="store_true", help="Simular sin tocar archivos")
    args = parser.parse_args()

    if args.variante not in VALID_VARIANTS:
        print(f"ERROR: variante '{args.variante}' no válida. Usa una de: {VALID_VARIANTS}")
        return 1

    input_dir = Path(args.input)
    output_dir = Path(args.output) / args.variante

    if not input_dir.is_dir():
        print(f"ERROR: la carpeta de entrada no existe: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Numeración inicial
    if args.start is None:
        start = find_next_number(output_dir, args.tienda, args.variante)
        print(f"Empezando desde el número {start} (auto-detectado)")
    else:
        start = args.start

    # Buscar fotos
    photos = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    )

    if not photos:
        print(f"ERROR: no hay fotos en {input_dir} con extensiones {VALID_EXTENSIONS}")
        return 1

    print(f"Encontradas {len(photos)} fotos en {input_dir}")
    print(f"Destino: {output_dir}")
    print(f"Modo: {'MOVER' if args.move else 'COPIAR'}{' (dry-run)' if args.dry_run else ''}")
    print("-" * 60)

    for i, photo in enumerate(photos, start=start):
        new_name = f"monster_{args.tienda}_{args.variante}_{i:03d}.jpg"
        new_path = output_dir / new_name

        if new_path.exists():
            print(f"  ⚠  Ya existe, salto: {new_name}")
            continue

        print(f"  {photo.name}  →  {new_name}")

        if args.dry_run:
            continue

        if args.move:
            shutil.move(str(photo), str(new_path))
        else:
            shutil.copy2(str(photo), str(new_path))

    print("-" * 60)
    print("Hecho.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
