"""
Verifica que el entorno está bien configurado.

Comprueba versión de Python, librerías instaladas, y existencia de las
carpetas del proyecto. Útil para que cada miembro del equipo confirme
que tiene todo listo después de hacer setup.

Uso:
    python scripts/check_environment.py
"""

import importlib
import sys
from pathlib import Path

REQUIRED_PACKAGES = [
    ("numpy",       "1.24"),
    ("cv2",         "4.8"),       # se importa como cv2 pero el paquete es opencv-python
    ("matplotlib",  "3.7"),
    ("sklearn",     "1.3"),       # paquete: scikit-learn
    ("skimage",     "0.21"),      # paquete: scikit-image
    ("PIL",         "10.0"),      # paquete: Pillow
    ("pandas",      "2.0"),
    ("tqdm",        "4.65"),
]

REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "data/annotations",
    "data/external",
    "src",
    "scripts",
    "results",
]


def check_python():
    print("─" * 60)
    print("PYTHON")
    print("─" * 60)
    v = sys.version_info
    print(f"  Versión: {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print("  ❌  Se recomienda Python 3.10 o superior.")
        return False
    print("  ✓  OK")
    return True


def check_packages():
    print()
    print("─" * 60)
    print("LIBRERÍAS")
    print("─" * 60)
    all_ok = True
    for module_name, min_version in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "?")
            print(f"  ✓  {module_name:15s} {version}")
        except ImportError:
            print(f"  ❌  {module_name:15s} NO INSTALADO")
            all_ok = False
    return all_ok


def check_dirs():
    print()
    print("─" * 60)
    print("ESTRUCTURA DE CARPETAS")
    print("─" * 60)
    all_ok = True
    for d in REQUIRED_DIRS:
        path = Path(d)
        if path.is_dir():
            print(f"  ✓  {d}")
        else:
            print(f"  ❌  {d}  (no existe)")
            all_ok = False
    return all_ok


def main():
    print()
    print("Verificación del entorno del proyecto")
    print()

    ok_python = check_python()
    ok_packages = check_packages()
    ok_dirs = check_dirs()

    print()
    print("─" * 60)
    if ok_python and ok_packages and ok_dirs:
        print("  ✓  TODO CORRECTO. Puedes empezar a trabajar.")
        return 0
    else:
        print("  ❌  HAY PROBLEMAS. Revisa lo que está marcado arriba.")
        if not ok_packages:
            print("     Instala lo que falta con: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
