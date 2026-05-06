"""
Descarga el Grocery Store Dataset (Klasson et al., KTH) en data/external/.

Es una alternativa a Freiburg Groceries cuando este último no está
disponible (su servidor de la Universidad de Freiburg suele caerse).

Contenido:
- ~5000 imágenes de frutas, verduras y refrigerados.
- Etiquetas jerárquicas (clase fina + clase gruesa).
- Imágenes "icónicas" de referencia para cada clase.

Uso:
    python scripts/download_klasson.py

Después de la descarga, las imágenes quedan en:
    data/external/grocery_store/dataset/...

Requisitos: tener `git` instalado (suele venir por defecto en Linux/Mac).
"""

import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/marcusklasson/GroceryStoreDataset.git"
DEST_ROOT = Path("data/external/grocery_store")


def check_git():
    """Verifica que git está disponible."""
    if shutil.which("git") is None:
        print("❌  No se encuentra `git` en el sistema.")
        print("Instálalo con:")
        print("  Ubuntu/Debian:  sudo apt install git")
        print("  Fedora:         sudo dnf install git")
        print("  Mac (homebrew): brew install git")
        return False
    return True


def main():
    if not check_git():
        return 1

    # Comprobar si ya está
    if (DEST_ROOT / "dataset").is_dir():
        print(f"Ya existe {DEST_ROOT / 'dataset'}, no descargo nada.")
        print("Si quieres redescargar, borra esa carpeta primero:")
        print(f"  rm -rf {DEST_ROOT}")
        return 0

    DEST_ROOT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Clonando repositorio:")
    print(f"  {REPO_URL}")
    print(f"Destino: {DEST_ROOT}")
    print("(esto puede tardar un par de minutos, son ~700 MB)")
    print()

    try:
        # --depth 1 evita descargar todo el historial git, solo la última versión
        result = subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(DEST_ROOT)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌  Error en git clone: {e}")
        print("Posibles causas:")
        print("  - Sin conexión a internet.")
        print("  - GitHub bloqueado en tu red.")
        print("  - El repositorio ha sido renombrado o eliminado.")
        return 1

    # Verificar
    dataset_dir = DEST_ROOT / "dataset"
    if not dataset_dir.is_dir():
        print(f"\n⚠  La descarga terminó pero no encuentro {dataset_dir}")
        print("Mira qué hay en", DEST_ROOT, "y avísame.")
        return 1

    # Estadísticas rápidas
    train_dir = dataset_dir / "train"
    if train_dir.is_dir():
        n_classes = sum(1 for d in train_dir.iterdir() if d.is_dir())
        n_images = sum(1 for _ in train_dir.rglob("*.jpg"))
        print()
        print(f"✓  Dataset listo en: {dataset_dir}")
        print(f"   Clases (en train): {n_classes}")
        print(f"   Imágenes (en train): {n_images}")
    else:
        print()
        print(f"✓  Dataset descargado en: {dataset_dir}")
        print("   (estructura inesperada, revisa el contenido a mano)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())