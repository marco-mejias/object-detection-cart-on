"""
Fine-tunea MobileNetV2 en las 8 categorías del proyecto.

Carga el dataset combinado (clases originales Klasson+Freiburg) y aplica
el mapeo a las 8 categorías definidas en src/classification/class_mapping.py.

Guarda el modelo en results/models/mobilenet_v2_categoria.pth

Uso:
    python scripts/train_mobilenet.py
    python scripts/train_mobilenet.py --epochs-head 5 --epochs-full 15

En Google Colab (recomendado para tener GPU):
    - Sube el repo o monta Drive con el dataset
    - Ejecuta este script
    - Descarga results/models/mobilenet_v2_categoria.pth
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

from src.classification.class_mapping import map_klasson_class, PROJECT_CATEGORIES


class MappedDataset(Dataset):
    """
    Envuelve un ImageFolder y remapea las etiquetas originales (68 clases)
    a las 8 categorías del proyecto usando class_mapping.py.
    Ignora las imágenes cuya clase no está en el mapeo ('otros').
    """

    def __init__(self, image_folder: datasets.ImageFolder, target_classes: list):
        self.image_folder = image_folder
        self.class_to_idx = {c: i for i, c in enumerate(target_classes)}
        self.target_classes = target_classes

        self.valid_indices: list[int] = []
        self.mapped_labels: list[int] = []

        for idx, (_, orig_label) in enumerate(image_folder.samples):
            orig_class = image_folder.classes[orig_label]
            mapped = map_klasson_class(orig_class)
            if mapped != "otros":
                self.valid_indices.append(idx)
                self.mapped_labels.append(self.class_to_idx[mapped])

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        orig_idx = self.valid_indices[idx]
        img, _ = self.image_folder[orig_idx]  # imagen ya transformada
        return img, self.mapped_labels[idx]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/external/combined",
                   help="Raíz del dataset (subdirs = clases originales)")
    p.add_argument("--epochs-head", type=int, default=5,
                   help="Epochs entrenando solo el clasificador final (backbone congelado)")
    p.add_argument("--epochs-full", type=int, default=15,
                   help="Epochs de fine-tune de toda la red")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-full", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--out", default="results/models/mobilenet_v2_categoria.pth")
    return p.parse_args()


def make_loaders(data_dir: Path, val_ratio: float, test_ratio: float, batch_size: int):
    # Augmentación agresiva para reducir el domain gap entre dataset y demo en vivo.
    # RandomResizedCrop simula distintas distancias; ColorJitter simula iluminación variable.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    classes = sorted(PROJECT_CATEGORIES)

    # Tres ImageFolders con el transform adecuado para cada split.
    train_base = datasets.ImageFolder(data_dir, transform=train_tf)
    eval_base  = datasets.ImageFolder(data_dir, transform=eval_tf)

    train_full = MappedDataset(train_base, classes)
    eval_full  = MappedDataset(eval_base,  classes)

    n = len(train_full)
    n_test  = int(n * test_ratio)
    n_val   = int(n * val_ratio)
    n_train = n - n_val - n_test

    idx = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    train_idx = idx[:n_train].tolist()
    val_idx   = idx[n_train:n_train + n_val].tolist()
    test_idx  = idx[n_train + n_val:].tolist()

    train_ds = Subset(train_full, train_idx)
    val_ds   = Subset(eval_full,  val_idx)
    test_ds  = Subset(eval_full,  test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader, classes


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
        correct += (out.argmax(1) == labels).sum().item()
        total += len(imgs)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total += len(imgs)
    return correct / total


@torch.no_grad()
def eval_test(model, loader, classes: list, device):
    """Evalúa sobre test y muestra accuracy, F1 macro y F1 por categoría."""
    from sklearn.metrics import classification_report, f1_score

    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    print("\n" + "=" * 50)
    print("EVALUACIÓN SOBRE TEST")
    print("=" * 50)
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  F1 macro:  {f1_macro:.3f}")
    print()
    print(classification_report(all_labels, all_preds,
                                 target_names=classes, digits=3))
    return acc, f1_macro


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"❌  No encuentro {data_dir}")
        print("   ¿Ejecutaste combine_datasets.py?")
        sys.exit(1)

    print(f"Cargando dataset desde {data_dir} ...")
    train_loader, val_loader, test_loader, classes = make_loaders(
        data_dir, args.val_ratio, args.test_ratio, args.batch_size)
    print(f"Clases ({len(classes)}): {classes}")
    print(f"Train: {len(train_loader.dataset)}"
          f"  Val: {len(val_loader.dataset)}"
          f"  Test: {len(test_loader.dataset)}")

    # MobileNetV2 preentrenada en ImageNet → sustituimos solo el último layer
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(1280, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # ── Fase 1: solo el clasificador (backbone congelado) ──────────────────
    print(f"\n=== Fase 1: head only ({args.epochs_head} epochs) ===")
    for p in model.features.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(model.classifier.parameters(), lr=args.lr_head)

    for epoch in range(1, args.epochs_head + 1):
        loss, acc_tr = train_epoch(model, train_loader, criterion, opt, device)
        acc_val = eval_epoch(model, val_loader, device)
        print(f"  [{epoch:2d}/{args.epochs_head}]  "
              f"loss={loss:.4f}  train={acc_tr:.3f}  val={acc_val:.3f}")

    # ── Fase 2: fine-tune completo ─────────────────────────────────────────
    print(f"\n=== Fase 2: fine-tune completo ({args.epochs_full} epochs) ===")
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.parameters(), lr=args.lr_full)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs_full)

    best_val = 0.0
    best_state = None
    for epoch in range(1, args.epochs_full + 1):
        loss, acc_tr = train_epoch(model, train_loader, criterion, opt, device)
        acc_val = eval_epoch(model, val_loader, device)
        scheduler.step()
        marker = " ✓" if acc_val > best_val else ""
        print(f"  [{epoch:2d}/{args.epochs_full}]  "
              f"loss={loss:.4f}  train={acc_tr:.3f}  val={acc_val:.3f}{marker}")
        if acc_val > best_val:
            best_val = acc_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Guardar ────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "classes": classes}, out_path)
    print(f"\n✓  Guardado en {out_path}  (mejor val acc = {best_val:.3f})")

    # Cargar el mejor estado guardado y evaluar sobre test
    model.load_state_dict(best_state)
    eval_test(model, test_loader, classes, device)

    print(f"Para usar en la demo:")
    print(f"  python scripts/demo_raspberry.py --classifier cnn")


if __name__ == "__main__":
    main()
