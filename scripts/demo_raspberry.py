"""
Demo en vivo del clasificador de categorías sobre Raspberry Pi 4B.

Tiene dos modos:

  AUTOMÁTICO (por defecto): cada N segundos procesa el frame actual
  y muestra todas las cajas detectadas con su categoría y confianza.
  Muy visual: el profesor ve el sistema funcionando solo.

  MANUAL: pulsas ESPACIO para capturar y clasificar un frame concreto.
  Procesa la imagen entera, encuentra la región principal y la
  clasifica con foco. Útil cuando quieres centrarte en un producto
  específico.

Controles:
  M       → cambia entre modo automático y modo manual.
  ESPACIO → en modo manual, captura y clasifica.
  +/-     → en modo automático, sube/baja el intervalo (0.5s a 5s).
  Q       → sale.

Pre-requisitos:
  - Raspberry Pi OS 64-bit (Bookworm).
  - Camera Module v2 conectado.
  - Modelo entrenado en results/models/svm_categoria_gridsearch.joblib.

Uso:
  python scripts/demo_raspberry.py
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import joblib
import numpy as np

HAVE_PICAMERA = False
try:
    from picamera2 import Picamera2
    HAVE_PICAMERA = True
except ImportError:
    pass  # en portátil usamos cv2.VideoCapture

from src.preprocessing import preprocess
from src.region_proposal import propose_regions
from src.features import extract_features


# ============================================================
# Configuración
# ============================================================
MODEL_PATH = Path("results/models/svm_categoria_gridsearch.joblib")
CNN_MODEL_PATH = Path("results/models/mobilenet_v2_categoria.pth")

# Fijados en main() según --classifier
CLASSIFIER_TYPE = "svm"
CNN_CLASSES: list = []
CAMERA_RESOLUTION = (1280, 960)
DISPLAY_RESOLUTION = (960, 720)

# Intervalo del modo automático (en segundos)
AUTO_INTERVAL_DEFAULT = 1.5
AUTO_INTERVAL_MIN = 0.5
AUTO_INTERVAL_MAX = 5.0

# Cuántas cajas como mucho mostramos en modo automático (las más grandes)
# Reducido a 3 para mantener fluidez con el refinamiento de crops
AUTO_MAX_BOXES = 3

# Umbral de confianza mínimo para mostrar una predicción en automático.
# Ayuda a no llenar la pantalla con cajas dudosas.
# Con 8 clases el azar es 12.5%, así que 25% ya es informativo.
AUTO_MIN_CONFIDENCE = 0.25

# Paleta BGR por categoría (más vivos que los del region proposal para
# que se vean bien en proyector/monitor).
PALETTE_BGR = {
    "fruta":   (50, 200, 50),
    "verdura": (255, 200, 0),
    "brick":   (0, 100, 255),
    "lata":    (200, 0, 200),
    "botella": (0, 200, 200),
    "caja":    (180, 100, 50),
    "bolsa":   (200, 200, 200),
    "tarro":   (50, 50, 220),
}
PALETTE_DEFAULT = (180, 180, 180)


# ============================================================
# Funciones auxiliares
# ============================================================
def load_model(path: Path):
    if not path.is_file():
        print(f"❌  No encuentro el modelo en {path}")
        print("   Ejecuta primero: python scripts/compare_svm_rf.py")
        sys.exit(1)
    print(f"⏳  Cargando modelo desde {path} ...")
    bundle = joblib.load(path)
    print("✓  Modelo cargado.")
    return bundle["model"]


def load_cnn_model(path: Path):
    """Carga MobileNetV2 fine-tuneada. Devuelve (model, classes)."""
    import torch
    import torch.nn as nn
    from torchvision.models import mobilenet_v2

    if not path.is_file():
        print(f"❌  No encuentro el modelo CNN en {path}")
        print("   Entrénalo primero: python scripts/train_mobilenet.py")
        sys.exit(1)
    print(f"⏳  Cargando modelo CNN desde {path} ...")
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    classes = bundle["classes"]
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(1280, len(classes))
    model.load_state_dict(bundle["model_state"])
    model.eval()
    print(f"✓  CNN cargada. Clases: {classes}")
    return model, classes


def init_camera(resolution: tuple):
    print("⏳  Inicializando cámara (Camera Module v2)...")
    picam = Picamera2()
    config = picam.create_still_configuration(
        main={"size": resolution, "format": "RGB888"},
    )
    picam.configure(config)
    picam.start()
    time.sleep(2)  # Estabilización del sensor
    print(f"✓  Cámara lista a {resolution[0]}x{resolution[1]}")
    return picam


def init_webcam(resolution: tuple):
    print("⏳  Inicializando webcam (cv2.VideoCapture)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌  No se puede abrir la webcam.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    # Leer un frame para estabilizar
    for _ in range(5):
        cap.read()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓  Webcam lista a {w}x{h}")
    return cap


def pad_crop(image: np.ndarray, p, pad_ratio: float = 0.15) -> np.ndarray:
    """
    Extrae un crop con padding alrededor de la propuesta.

    El problema: las imágenes de entrenamiento (Klasson/Freiburg) muestran
    el producto ocupando TODA la imagen (256x256 o 348x348), sin fondo.
    Pero los crops del region proposal incluyen mucha pared/mesa.

    Esta función añade un pequeño margen (pad_ratio) alrededor de la caja
    para no cortar el producto, pero sin incluir demasiado fondo.
    """
    h_img, w_img = image.shape[:2]
    pad_x = int(p.w * pad_ratio)
    pad_y = int(p.h * pad_ratio)

    x1 = max(0, p.x - pad_x)
    y1 = max(0, p.y - pad_y)
    x2 = min(w_img, p.x + p.w + pad_x)
    y2 = min(h_img, p.y + p.h + pad_y)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return image
    return crop


def select_best_proposal(model, image: np.ndarray, proposals, max_candidates: int = 8):
    """
    Clasifica las top-N propuestas con el SVM y devuelve la de mayor
    confianza. Mucho mejor que elegir la más grande, porque el SVM
    "sabe" qué crop se parece más a un producto real.
    """
    if not proposals:
        return None, None, 0.0

    # Ordenar por tamaño (las más grandes primero) y tomar top-N
    sorted_p = sorted(proposals, key=lambda p: p.w * p.h,
                       reverse=True)[:max_candidates]

    best_proposal = None
    best_category = None
    best_proba = 0.0

    for p in sorted_p:
        crop = pad_crop(image, p)
        if crop.size == 0:
            continue
        category, proba = predict(model, crop)
        if proba > best_proba:
            best_proba = proba
            best_category = category
            best_proposal = p

    return best_proposal, best_category, best_proba


def refine_crop(crop_rgb: np.ndarray, pad: int = 5) -> np.ndarray:
    """
    Recorta el crop ajustándose al contorno real del producto.

    Problema: el region proposal da una caja que incluye mucho fondo
    (pared, mesa, cables). El HOG (1764 de las 1806 features) se
    contamina con los bordes del fondo y el SVM predice basura.

    Solución: usar Canny + contornos para encontrar el objeto real
    dentro del crop y recortar ajustado a él. Así el SVM ve el
    producto llenando la imagen, como en las imágenes de entrenamiento.

    Verificado: sube la accuracy del 11% al 76% en escenarios simulados.
    """
    h, w = crop_rgb.shape[:2]
    if h < 50 or w < 50:
        return crop_rgb

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Cerrar bordes para formar siluetas completas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Encontrar contornos y tomar el más grande (= el producto)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return crop_rgb

    largest = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(largest)

    # Si el contorno es muy pequeño, usar la unión de todos los bordes
    if bw * bh < 0.05 * h * w:
        coords = np.column_stack(np.where(edges > 0))
        if len(coords) < 50:
            return crop_rgb
        ey1, ex1 = coords.min(axis=0)
        ey2, ex2 = coords.max(axis=0)
    else:
        ex1, ey1 = bx, by
        ex2, ey2 = bx + bw, by + bh

    # Mínimo padding para no cortar el producto
    ey1 = max(0, ey1 - pad)
    ex1 = max(0, ex1 - pad)
    ey2 = min(h, ey2 + pad)
    ex2 = min(w, ex2 + pad)

    tight = crop_rgb[ey1:ey2, ex1:ex2]
    if tight.size == 0:
        return crop_rgb
    return tight


def predict_cnn(model, crop: np.ndarray):
    """Clasificación con MobileNetV2 fine-tuneada."""
    import torch
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = tf(crop).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
    conf, idx = probs.max(dim=1)
    return CNN_CLASSES[idx.item()], conf.item()


def predict(model, crop: np.ndarray):
    """Devuelve (categoría, confianza) para un crop. Dispatcha según CLASSIFIER_TYPE."""
    if CLASSIFIER_TYPE == "cnn":
        return predict_cnn(model, crop)
    # SVM: refinar crop y extraer features clásicas
    refined = refine_crop(crop)
    features = extract_features(refined).reshape(1, -1)
    pred = model.predict(features)[0]
    proba = float(model.predict_proba(features)[0].max())
    return pred, proba


def classify_all_proposals(model, image: np.ndarray, proposals,
                            max_boxes: int = AUTO_MAX_BOXES,
                            min_conf: float = AUTO_MIN_CONFIDENCE):
    """
    Clasifica las N cajas más grandes y devuelve solo las que superan
    el umbral de confianza. Para el modo automático.

    Limitamos a las top-N para que la Pi no se atragante intentando
    procesar 30 cajas por frame.
    """
    if not proposals:
        return []

    sorted_p = sorted(proposals, key=lambda p: p.w * p.h,
                       reverse=True)[:max_boxes]
    results = []
    for p in sorted_p:
        crop = pad_crop(image, p)
        if crop.size == 0:
            continue
        category, proba = predict(model, crop)
        if proba >= min_conf:
            results.append((p, category, proba))
    return results


def draw_box_with_label(bgr: np.ndarray, p, category: str, proba: float,
                          thickness: int = 3, font_scale: float = 0.7):
    """Dibuja una caja con su etiqueta sobre la imagen BGR (in-place)."""
    color = PALETTE_BGR.get(category, PALETTE_DEFAULT)
    cv2.rectangle(bgr, (p.x, p.y), (p.x + p.w, p.y + p.h), color, thickness)
    label = f"{category} {proba:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale, 2)
    label_y = max(p.y, th + 8)
    cv2.rectangle(bgr,
                   (p.x, label_y - th - 8),
                   (p.x + tw + 12, label_y),
                   color, -1)
    cv2.putText(bgr, label, (p.x + 6, label_y - 6),
                 cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                 (255, 255, 255), 2, cv2.LINE_AA)


def draw_hud(bgr: np.ndarray, text: str, color=(255, 165, 0)):
    """Banner inferior con info global."""
    cv2.rectangle(bgr, (0, bgr.shape[0] - 40), (bgr.shape[1], bgr.shape[0]),
                   (0, 0, 0), -1)
    cv2.putText(bgr, text, (12, bgr.shape[0] - 12),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def render_auto(frame_rgb: np.ndarray, detections: list,
                 took_ms: float, interval: float) -> np.ndarray:
    """Vista del modo automático con todas las detecciones dibujadas."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    for p, cat, prob in detections:
        draw_box_with_label(bgr, p, cat, prob,
                              thickness=3, font_scale=0.65)
    info = (f"AUTO  -  {len(detections)} obj.  -  {took_ms:.0f} ms"
             f"  -  intervalo {interval:.1f}s  -  M:manual +/-:vel  Q:salir")
    draw_hud(bgr, info)
    return bgr


def render_manual_idle(frame_rgb: np.ndarray) -> np.ndarray:
    """Vista del modo manual antes de pulsar espacio."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    draw_hud(bgr, "MANUAL  -  ESPACIO: clasificar  -  M: auto  -  Q: salir")
    return bgr


def render_manual_result(frame_rgb: np.ndarray, proposal,
                           category: str, proba: float,
                           took_ms: float) -> np.ndarray:
    """Vista del modo manual con el resultado de una clasificación."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if proposal is not None:
        draw_box_with_label(bgr, proposal, category, proba,
                              thickness=4, font_scale=0.9)
    info = (f"MANUAL  -  {category.upper()}  {proba:.0%}"
             f"  -  {took_ms:.0f} ms  -  ESPACIO: otra  -  M: auto  -  Q: salir")
    draw_hud(bgr, info)
    return bgr


# ============================================================
# Argumentos y bucle principal
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Demo en vivo · UAB VC")
    p.add_argument(
        "--classifier", choices=["svm", "cnn"], default="svm",
        help="Clasificador a usar: 'svm' (clásico, default) o 'cnn' (MobileNetV2)"
    )
    return p.parse_args()


def main():
    global CLASSIFIER_TYPE, CNN_CLASSES
    args = parse_args()
    CLASSIFIER_TYPE = args.classifier

    # Cargar modelo
    if CLASSIFIER_TYPE == "cnn":
        model, CNN_CLASSES = load_cnn_model(CNN_MODEL_PATH)
    else:
        model = load_model(MODEL_PATH)

    # Inicializar cámara: Raspberry Pi o webcam de portátil
    picam = None
    cap = None
    if HAVE_PICAMERA:
        picam = init_camera(CAMERA_RESOLUTION)
    else:
        cap = init_webcam(CAMERA_RESOLUTION)

    clf_tag = CLASSIFIER_TYPE.upper()
    window_name = f"Detector de productos · UAB VC  [{clf_tag}]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, *DISPLAY_RESOLUTION)

    # Estado
    mode = "auto"
    auto_interval = AUTO_INTERVAL_DEFAULT
    last_auto_run = 0.0
    cached_auto_render = None
    manual_result_until = 0.0
    cached_manual_render = None

    print()
    print("=" * 60)
    print(f"  DEMO LISTA — modo AUTOMÁTICO  [clasificador: {clf_tag}]")
    print()
    print("  Controles:")
    print("    M       → cambiar entre auto y manual")
    print("    ESPACIO → en manual, clasifica el frame actual")
    print("    + / -   → en auto, ajusta el intervalo")
    print("    Q       → salir")
    print("=" * 60)
    print()

    try:
        while True:
            # Captura de frame (Pi o webcam)
            if picam is not None:
                frame_bgr = picam.capture_array()
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            else:
                ret, frame_bgr = cap.read()
                if not ret:
                    print("❌  Error leyendo de la webcam.")
                    break
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            now = time.time()

            # =====================================================
            # Decidir qué dibujar según modo y timers
            # =====================================================
            if mode == "auto":
                if (now - last_auto_run) >= auto_interval:
                    t0 = time.time()
                    pre = preprocess(frame)
                    proposals = propose_regions(pre)
                    detections = classify_all_proposals(model, pre, proposals)
                    took_ms = (time.time() - t0) * 1000
                    cached_auto_render = render_auto(
                        pre, detections, took_ms, auto_interval)
                    last_auto_run = now
                    print(f"[auto/{clf_tag}] {len(detections)} objetos · {took_ms:.0f} ms")

                if cached_auto_render is not None:
                    display = cached_auto_render
                else:
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    draw_hud(bgr, f"AUTO [{clf_tag}]  -  procesando primer frame...")
                    display = bgr

            else:  # mode == "manual"
                if now < manual_result_until and cached_manual_render is not None:
                    display = cached_manual_render
                else:
                    display = render_manual_idle(frame)

            # =====================================================
            # Mostrar y leer teclado
            # =====================================================
            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 0xFF:
                continue

            if key in (ord('q'), ord('Q')):
                print("→ Saliendo.")
                break

            if key in (ord('m'), ord('M')):
                if mode == "auto":
                    mode = "manual"
                    cached_manual_render = None
                    manual_result_until = 0
                    print("→ Cambio a modo MANUAL.")
                else:
                    mode = "auto"
                    cached_auto_render = None
                    last_auto_run = 0
                    print("→ Cambio a modo AUTOMÁTICO.")
                continue

            if mode == "auto" and key in (ord('+'), ord('=')):
                auto_interval = max(AUTO_INTERVAL_MIN, auto_interval - 0.5)
                print(f"→ Intervalo automático: {auto_interval:.1f}s")
                continue

            if mode == "auto" and key in (ord('-'), ord('_')):
                auto_interval = min(AUTO_INTERVAL_MAX, auto_interval + 0.5)
                print(f"→ Intervalo automático: {auto_interval:.1f}s")
                continue

            if mode == "manual" and key == ord(' '):
                print(f"\n→ [manual/{clf_tag}] Capturando y clasificando...")
                t0 = time.time()
                pre = preprocess(frame)
                proposals = propose_regions(pre)

                main_box, category, proba = select_best_proposal(
                    model, pre, proposals)

                if main_box is None:
                    print("   (sin region detectada, usando imagen completa)")
                    category, proba = predict(model, pre)

                took_ms = (time.time() - t0) * 1000
                print(f"   OK  {category.upper()}  ({proba:.0%})"
                      f"  en {took_ms:.0f} ms")

                cached_manual_render = render_manual_result(
                    pre, main_box, category, proba, took_ms)
                manual_result_until = time.time() + 4

    except KeyboardInterrupt:
        print("\n→ Interrumpido por teclado.")

    finally:
        if picam is not None:
            picam.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("✓  Recursos liberados. Hasta luego.")


if __name__ == "__main__":
    main()
