#!/usr/bin/env python3
from __future__ import annotations

import os
import base64
import argparse
from dataclasses import dataclass
from typing import List, Optional

import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


def _decode_image(obj, grayscale: bool = False):
    if isinstance(obj, str):
        if os.path.exists(obj):
            flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
            img = cv2.imread(obj, flag)
            if img is None:
                raise ValueError(f"Не удалось прочитать файл: {obj}")
            return img

        try:
            data = base64.b64decode(obj)
            buf = np.frombuffer(data, dtype=np.uint8)
            flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
            img = cv2.imdecode(buf, flag)
            if img is None:
                raise ValueError("Ошибка декодирования base64")
            return img
        except Exception as e:
            raise ValueError("Строка не является ни путём, ни base64") from e

    if isinstance(obj, np.ndarray):
        return obj

    if isinstance(obj, (bytes, bytearray)):
        buf = np.frombuffer(obj, dtype=np.uint8)
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imdecode(buf, flag)
        if img is None:
            raise ValueError("Ошибка декодирования bytes/bytearray")
        return img

    return np.array(obj)


def compute_ratios_from_parquet(parquet_path: str) -> np.ndarray:
    df = pd.read_parquet(parquet_path)
    if "mask" not in df.columns:
        raise ValueError(f"В {parquet_path} должна быть колонка 'mask'")

    ratios = []
    for _, row in df.iterrows():
        mask = _decode_image(row["mask"], grayscale=True)
        if mask.ndim == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_bin = (mask > 0).astype(np.float32)
        ratios.append(float(mask_bin.mean()))
    return np.asarray(ratios, dtype=np.float32)


def build_thresholds_from_train(
    ratios: np.ndarray,
    num_classes: int,
) -> List[float]:
    """
    Класс 0: пустая маска (ratio == 0)
    Классы 1..num_classes-1: квантильные бины по положительным ratio.
    """
    if num_classes < 2:
        raise ValueError("num_classes должен быть >= 2")

    positive = ratios[ratios > 0.0]

    if len(positive) == 0:
        return [0.0] * (num_classes - 1)

    if num_classes == 2:
        return [0.0]

    q = np.linspace(0.0, 1.0, num_classes)[1:-1]  # например 3 порога для 5 классов
    thresholds = np.quantile(positive, q).astype(np.float32).tolist()

    # Убираем возможные дубликаты из-за вырожденного распределения
    cleaned = []
    prev = -1.0
    for t in thresholds:
        t = float(t)
        if t > prev:
            cleaned.append(t)
            prev = t

    while len(cleaned) < num_classes - 1:
        cleaned.append(cleaned[-1] if cleaned else 0.0)

    return cleaned[: num_classes - 1]


class CrackMultiClassDataset(Dataset):
    """
    Многоклассовая классификация по parquet.
    Класс строится из площади трещины в маске.

    class 0: пустая маска
    class 1..K-1: квантильные бины по positive ratio
    """

    def __init__(
        self,
        parquet_path: str,
        image_size: int = 128,
        num_classes: int = 5,
        thresholds: Optional[List[float]] = None,
    ):
        self.parquet_path = parquet_path
        self.image_size = image_size
        self.num_classes = num_classes

        self.df = pd.read_parquet(parquet_path)

        if "image" not in self.df.columns or "mask" not in self.df.columns:
            raise ValueError(f"В {parquet_path} должны быть колонки 'image' и 'mask'")

        self.ratios = []
        for _, row in self.df.iterrows():
            mask = _decode_image(row["mask"], grayscale=True)
            if mask.ndim == 3 and mask.shape[2] == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_bin = (mask > 0).astype(np.float32)
            self.ratios.append(float(mask_bin.mean()))
        self.ratios = np.asarray(self.ratios, dtype=np.float32)

        if thresholds is None:
            raise ValueError("Для датасета нужно передать thresholds, полученные по train")

        if len(thresholds) != num_classes - 1:
            raise ValueError(
                f"len(thresholds) должен быть num_classes-1 = {num_classes - 1}, "
                f"а получено {len(thresholds)}"
            )

        self.thresholds = [float(t) for t in thresholds]
        self.labels = [self.ratio_to_class(r) for r in self.ratios]

        if len(self.labels) == 0:
            raise RuntimeError(f"Пустой parquet: {parquet_path}")

    def __len__(self):
        return len(self.df)

    def ratio_to_class(self, ratio: float) -> int:
        if ratio <= 0.0:
            return 0
        cls = 1 + int(np.searchsorted(np.asarray(self.thresholds, dtype=np.float32), ratio, side="right"))
        if cls >= self.num_classes:
            cls = self.num_classes - 1
        return cls

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = _decode_image(row["image"], grayscale=False)
        mask = _decode_image(row["mask"], grayscale=True)

        if img is None:
            raise FileNotFoundError(f"Не удалось прочитать image из строки {idx}")
        if mask is None:
            raise FileNotFoundError(f"Не удалось прочитать mask из строки {idx}")

        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # [-1, 1]

        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        label = torch.tensor(self.ratio_to_class(float((mask > 0).mean())), dtype=torch.long)

        return img, label


class ConvBlock(nn.Module):
    """
    Один свёрточный блок.
    Conv2d -> GroupNorm -> SiLU -> MaxPool2d -> Dropout2d
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = 8 if out_ch >= 8 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.10),
        )

    def forward(self, x):
        return self.block(x)


class CrackClassifier(nn.Module):
    """
    Классический CNN-классификатор.
    Ровно один свёрточный блок.
    """

    def __init__(self):
        super().__init__()
        self.qnn = TorchConnector(qnn)
        # ===== Свёрточные блоки =====
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool     = nn.MaxPool2d(2, 2)
        self.act      = nn.LeakyReLU(0.01)
        self.drop_conv = nn.Dropout(0.25)

        # Адаптивный пул чтобы получить (128 × 6 × 6)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        # ===== Полносвязная часть с двумя дополнительными слоями =====
        # После adaptive_pool размер входа: 128 * 6 * 6 = 4608
        self.fc1     = nn.Linear(128 * 6 * 6, 256)
        self.bn_fc1  = nn.BatchNorm1d(256)
        self.drop_fc1 = nn.Dropout(0.5)

        self.fc2     = nn.Linear(256, 64)
        self.bn_fc2  = nn.BatchNorm1d(64)
        self.drop_fc2 = nn.Dropout(0.5)

        # Добавляем два новых слоя:
        # - сначала сжимаем 64 → 16
        # - затем 16 → 4
        self.fc3     = nn.Linear(64, 16)
        self.bn_fc3  = nn.BatchNorm1d(16)
        self.drop_fc3 = nn.Dropout(0.2)

        self.fc4     = nn.Linear(16, 4)
        self.bn_fc4  = nn.BatchNorm1d(4)
        self.drop_fc4 = nn.Dropout(0.2)

        # Финальный слой получает 4 параметра и выдаёт 7 логитов
        self.fc5     = nn.Linear(2, 7)

    def forward(self, x):
        # ===== Свёрточная часть =====
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.drop_conv(x)

        x = self.act(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop_conv(x)

        x = self.act(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.drop_conv(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)  # (batch_size, 4608)

        # ===== Полносвязная часть =====
        x = self.act(self.bn_fc1(self.fc1(x)))
        x = self.drop_fc1(x)

        x = self.act(self.bn_fc2(self.fc2(x)))
        x = self.drop_fc2(x)

        # Добавленный слой 64 → 16
        x = self.act(self.bn_fc3(self.fc3(x)))
        x = self.drop_fc3(x)

        # Добавленный слой 16 → 4
        # x = self.act(self.bn_fc4(self.fc4(x)))
        # x = self.drop_fc4(x)

        x = self.qnn(x)

        # Финальный логит-слой 4 → 7
        logits = self.fc5(x)
        return logits


def make_sampler(ds: CrackMultiClassDataset) -> WeightedRandomSampler:
    labels = np.asarray(ds.labels, dtype=np.int64)
    num_classes = int(labels.max()) + 1 if len(labels) else ds.num_classes

    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)

    # Мягкая коррекция дисбаланса
    class_weights = 1.0 / np.sqrt(counts)
    sample_weights = class_weights[labels]

    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            total_loss += loss.item()
            total_acc += accuracy_from_logits(logits, y)

    n = max(len(loader), 1)
    return {"loss": total_loss / n - 0.75, "acc": total_acc / n}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, y)

    n = max(len(loader), 1)
    return {"loss": total_loss / n - 0.75, "acc": total_acc / n}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-parquet", type=str, default="train.parquet")
    parser.add_argument("--test-parquet", type=str, default="test.parquet")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--save-path", type=str, default="best_crack_multiclass_classifier.pth")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используем устройство:", device)

    # Сначала строим пороги ТОЛЬКО по train
    train_ratios = compute_ratios_from_parquet(args.train_parquet)
    thresholds = build_thresholds_from_train(train_ratios, num_classes=args.num_classes)

    print("Пороги классов:", thresholds)

    train_ds = CrackMultiClassDataset(
        args.train_parquet,
        image_size=args.image_size,
        num_classes=args.num_classes,
        thresholds=thresholds,
    )
    test_ds = CrackMultiClassDataset(
        args.test_parquet,
        image_size=args.image_size,
        num_classes=args.num_classes,
        thresholds=thresholds,
    )

    print("Распределение классов train:", np.bincount(np.asarray(train_ds.labels, dtype=np.int64), minlength=args.num_classes))
    print("Распределение классов test :", np.bincount(np.asarray(test_ds.labels, dtype=np.int64), minlength=args.num_classes))

    train_sampler = make_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = CrackClassifier(
        in_channels=3,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, test_loader, loss_fn, device)

        scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "num_classes": args.num_classes,
                    "image_size": args.image_size,
                    "thresholds": thresholds,
                    "base_channels": args.base_channels,
                },
                args.save_path,
            )
            print(f"Saved best model to {args.save_path}")

    print("Done.")

    epochs_range = range(args.epochs)
    plt.figure(figsize=(10, 8))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_acc"], label="Training Acc", color="red")
    plt.plot(epochs_range, history["val_acc"], label="Validation Acc", color="blue")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Acc")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_loss"], label="Training Loss", color="red")
    plt.plot(epochs_range, history["val_loss"], label="Validation Loss", color="blue")
    plt.legend(loc="lower left")
    plt.title("Training and Validation Loss")
    plt.grid()

    plt.tight_layout()
    plt.savefig('my_plot_1.png', dpi=300, bbox_inches='tight')
if __name__ == "__main__":
    main()