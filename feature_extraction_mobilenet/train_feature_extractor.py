import argparse
import os
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# ===== 路径兼容 =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from feature_extraction.dataset_loader import Flickr24MultiLabelDataset


# =========================
# MobileNet模型（替换VGG）
# =========================
class FeatureExtractor(nn.Module):
    def __init__(self, feature_dim=24):
        super().__init__()

        self.model = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        )

        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, feature_dim)

    def forward(self, x):
        return self.model(x)


# =========================
# 随机种子
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Train 一轮（完全仿照你VGG写法🔥）
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx, total_epochs):
    model.train()
    total_loss = 0.0
    total_samples = 0
    num_batches = len(loader)

    for batch_idx, (images, labels, _) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # 🔥 和你VGG完全一样的输出格式
        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == num_batches:
            avg_loss = total_loss / max(1, total_samples)
            print(
                f"\r[Train] Epoch {epoch_idx}/{total_epochs} "
                f"Batch {batch_idx}/{num_batches} avg_loss={avg_loss:.6f}",
                end="",
                flush=True,
            )

    print()
    return total_loss / max(1, total_samples)


# =========================
# 验证一轮（完全一致）
# =========================
@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, epoch_idx, total_epochs):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    num_batches = len(loader)

    for batch_idx, (images, labels, _) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == num_batches:
            avg_loss = total_loss / max(1, total_samples)
            print(
                f"\r[ Val ] Epoch {epoch_idx}/{total_epochs} "
                f"Batch {batch_idx}/{num_batches} avg_loss={avg_loss:.6f}",
                end="",
                flush=True,
            )

    print()
    return total_loss / max(1, total_samples)


# =========================
# 参数解析
# =========================
def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--max-images", type=int, default=10000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", default="weights/mobilenet_flickr24.pth")

    return parser


# =========================
# 主函数
# =========================
def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    dataset = Flickr24MultiLabelDataset(
        image_dir=args.image_dir,
        annotation_dir=args.annotation_dir,
        max_images=args.max_images,
        offset=args.offset,
    )

    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    print(f"多标签训练/验证集已加载 {len(dataset)} 张图片 (offset={args.offset})")

    model = FeatureExtractor(feature_dim=24).to(args.device)

    # 🔥 冻结 backbone（非常关键，防过拟合）
    for param in model.model.features.parameters():
        param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    best_val = float("inf")
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch, args.epochs
        )

        val_loss = eval_one_epoch(
            model, val_loader, criterion, args.device, epoch, args.epochs
        )

        print(f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val,
                },
                args.save_path,
            )
            print(f"[SAVE] best checkpoint -> {args.save_path}")

    print(f"Training done. Best val_loss={best_val:.6f}")


if __name__ == "__main__":
    main()