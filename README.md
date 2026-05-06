# Detección híbrida de productos en supermercado

> **Visión por Computador · Grau en Enginyeria Informàtica · UAB · Curso 2025–2026**

Sistema híbrido que combina **visión por computador clásica** y **Deep Learning** para detectar productos en imágenes de supermercado. La parte clásica clasifica regiones por categoría (frutas, latas, botellas, cajas) y la parte de DL identifica un producto concreto: bebidas energéticas **Monster**.

## Equipo

- Artur Moret González (NIU 1641036)
- Daniel Cruz Flores (NIU 1709912)
- Marco Mejías Alés (NIU 1710748)

**Profesor:** Felipe Lumbreras Ruiz

## Estructura del repositorio

```
proyecto-vc/
├── data/
│   ├── raw/              # Fotos crudas (NO se sube a Git, ver .gitignore)
│   ├── processed/        # Imágenes ya preprocesadas
│   ├── annotations/      # Anotaciones LabelImg/CVAT (formato YOLO/COCO)
│   └── external/         # Datasets públicos descargados
├── notebooks/            # Experimentación rápida
├── src/                  # Código modular del pipeline
│   ├── preprocessing/    # CLAHE, white balance, denoising
│   ├── region_proposal/  # HSV, contornos, morfología
│   ├── features/         # Histogramas, HOG, LBP
│   ├── classification/   # SVM, Random Forest
│   ├── deep_learning/    # CNN para Monster
│   └── utils/            # Helpers compartidos
├── scripts/              # Scripts auxiliares (renombrar fotos, etc.)
├── results/
│   ├── figures/          # Gráficas y visualizaciones
│   ├── models/           # Modelos entrenados (NO se sube a Git)
│   └── logs/             # Logs de entrenamiento
└── docs/                 # Documentación adicional
```

## Setup inicial (solo la primera vez)

### 1. Clonar el repositorio

```bash
git clone https://github.com/<usuario>/proyecto-vc.git
cd proyecto-vc
```

### 2. Crear entorno virtual

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verificar que todo funciona

```bash
python -c "import cv2, numpy, sklearn, skimage; print('OK')"
```

## Flujo de trabajo (Git)

- **Rama `main`**: solo código estable y revisado.
- **Ramas de feature**: cada uno trabaja en su rama (`feature/preprocesado-clahe`, `feature/region-proposal-hsv`, etc.) y al terminar abre un **Pull Request**.
- **Antes de hacer push**: probar que el código corre sin errores y que no se rompe nada en `main`.
- **Commits descriptivos**: nada de `update` o `fix bug`. Mejor: `Añadido CLAHE sobre canal L de Lab` o `Fix overflow en histograma HSV`.

## Convención de nombres de fotos

Las fotos que capturéis con el móvil deben renombrarse antes de subirlas a `data/raw/<variante>/`. Formato:

```
monster_<tienda>_<variante>_<numero>.jpg
```

**Ejemplos:**
- `monster_mercadona01_original_001.jpg`
- `monster_carrefour02_ultra_015.jpg`
- `monster_diamondia_mango_007.jpg`

Hay un script auxiliar para renombrar lotes de fotos: `scripts/rename_photos.py` (ver más abajo).

## Fases del proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| F1 | Captura inicial del dataset Monster | 🟡 En curso |
| F2 | Preprocesado clásico + region proposal | ⚪ Pendiente |
| F3 | Extracción de features + SVM/RF | ⚪ Pendiente |
| F4 | Anotación completa del dataset | ⚪ Pendiente |
| F5 | Entrenamiento del modelo DL | ⚪ Pendiente |
| F6 | Integración del pipeline | ⚪ Pendiente |
| F7 | Análisis y memoria final | ⚪ Pendiente |

## Datasets públicos a descargar

Para la parte clásica (clasificación por categoría):

- **Freiburg Groceries**: https://github.com/PhilJd/freiburg_groceries_dataset
- **Grocery Store Dataset (Klasson)**: https://github.com/marcusklasson/GroceryStoreDataset
- **SKU-110K**: https://github.com/eg4000/SKU110K_CVPR19 (solo para evaluar region proposal)
- **Fruits-360**: https://www.kaggle.com/datasets/moltean/fruits

Descargar a `data/external/` y NO subir a Git (ya está ignorado).

## Scripts auxiliares

- `scripts/rename_photos.py` — Renombra lotes de fotos a la convención del proyecto.
- `scripts/dataset_stats.py` — Cuenta cuántas fotos hay por variante y tienda.
- `scripts/check_environment.py` — Verifica que todas las dependencias funcionan.

## Notas importantes

- **NUNCA** subir fotos crudas, modelos entrenados o datasets externos a Git. El `.gitignore` ya los excluye.
- **SIEMPRE** trabajar en un entorno virtual (`venv` activado).
- Usar **rutas relativas** en el código, nunca absolutas (`./data/raw/...` en vez de `/home/usuario/...`).
- Si añadís una librería nueva: actualizar `requirements.txt` con `pip freeze | grep <libreria>`.
