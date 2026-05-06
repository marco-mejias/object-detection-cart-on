"""
Descarga el dataset Freiburg Groceries usando el repositorio GitHub oficial.

El servidor de la Universidad de Freiburg suele estar caído, así que en
lugar de descargar el .tar.gz desde allí clonamos el repositorio oficial,
que incluye un script `download_dataset.py` que descarga las imágenes desde
un mirror alternativo. Si ese también falla, intentamos descargar
directamente las imágenes con el comando alternativo descrito en el README.

Uso:
    python scripts/download_freiburg_v2.py

Después de la descarga las imágenes quedan en:
    data/external/freiburg_groceries/freiburg_groceries_dataset/images/
"""

import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/PhilJd/freiburg_groceries_dataset.git"
DEST_ROOT = Path("data/external/freiburg_groceries")


def check_git():
    if shutil.which("git") is None:
        print("❌  No tienes `git` instalado.")
        print("   sudo dnf install git   (Fedora)")
        print("   sudo apt install git   (Ubuntu/Debian)")
        return False
    return True


def main():
    if not check_git():
        return 1

    images_dir = DEST_ROOT / "freiburg_groceries_dataset" / "images"
    if images_dir.is_dir():
        n_classes = sum(1 for d in images_dir.iterdir() if d.is_dir())
        n_images = sum(1 for _ in images_dir.rglob("*.png"))
        print(f"Ya existen las imágenes en {images_dir}")
        print(f"Clases: {n_classes}, imágenes: {n_images}")
        return 0

    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    # 1) Clonar el repo con --depth 1 (rápido)
    repo_dir = DEST_ROOT
    if not (repo_dir / ".git").is_dir():
        print(f"Clonando repositorio (~5 MB)...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"❌  Error clonando: {e}")
            return 1

    # 2) Ejecutar el script de descarga del propio repo, que descarga las
    #    imágenes desde el mirror que ellos mismos mantienen.
    download_script = repo_dir / "src" / "download_dataset.py"
    if not download_script.is_file():
        print(f"❌  No encuentro el script de descarga: {download_script}")
        print("Descarga manualmente siguiendo: https://github.com/PhilJd/freiburg_groceries_dataset")
        return 1

    print()
    print("Lanzando script oficial de descarga (~350 MB, varios minutos)...")
    print(f"  cd {repo_dir} && python src/download_dataset.py")
    print()

    try:
        subprocess.run(
            ["python", "src/download_dataset.py"],
            cwd=str(repo_dir),
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌  Error en la descarga: {e}")
        print()
        print("Alternativa manual:")
        print("  1. Entra en el repo:")
        print(f"     cd {repo_dir}")
        print("  2. Mira el contenido de src/download_dataset.py")
        print("  3. La URL del mirror suele estar al principio del script.")
        print("  4. Descarga manualmente con wget/curl si la URL aún sirve.")
        return 1

    # Verificar
    if not images_dir.is_dir():
        print(f"\n⚠  La descarga acabó pero no encuentro {images_dir}")
        print("Mira el contenido de", repo_dir, "para ver dónde quedaron las imágenes.")
        return 1

    n_classes = sum(1 for d in images_dir.iterdir() if d.is_dir())
    n_images = sum(1 for _ in images_dir.rglob("*.png"))
    print()
    print(f"✓  Dataset listo en: {images_dir}")
    print(f"   Clases: {n_classes}")
    print(f"   Imágenes: {n_images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
