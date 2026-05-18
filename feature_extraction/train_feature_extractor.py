import argparse
import os
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

# 兼容路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from feature_extraction.dataset_loader import Flickr24MultiLabelDataset
from feature_extraction.feature_extract import FeatureExtractor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx, total_epochs):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (images, labels, _) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        if batch_idx % 10 == 0 or batch_idx == 1:
            print(f"[Train] Epoch {epoch_idx}/{total_epochs} Batch {batch_idx} Loss: {loss.item():.4f}")

    return total_loss / total_samples


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / total_samples


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-dir", default="dataset/mirflickr25k/mirflickr")
    parser.add_argument("--annotation-dir", default="dataset/mirflickr25k_annotations_v080")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", default="weights/feature_extractor.pth")

    args = parser.parse_args()

    set_seed(42)

    # ===== 数据集 =====
    dataset = Flickr24MultiLabelDataset(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir
    )

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # ===== 模型（固定24维）=====
    model = FeatureExtractor(feature_dim=24).to(args.device)

    # ===== 冻结大部分 VGG =====
    for param in model.features.parameters():
        param.requires_grad = False

    # ⭐ 解冻最后几层（提升效果）
    for param in model.features[-5:].parameters():
        param.requires_grad = True

    # ===== 损失函数 =====
    criterion = nn.BCEWithLogitsLoss()

    # ===== 优化器（只优化可训练参数）=====
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    best_val_loss = float("inf")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # ===== 训练循环 =====
    for epoch in range(1, args.epochs + 1):

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer,
            args.device, epoch, args.epochs
        )

        val_loss = eval_one_epoch(
            model, val_loader, criterion, args.device
        )

        print(f"\nEpoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # ===== 保存最优模型 =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save({
                "state_dict": model.state_dict(),
                "val_loss": best_val_loss
            }, args.save_path)

            print(f"✅ 保存最优模型: {args.save_path}")

    print("训练完成！")


if __name__ == "__main__":
    main()