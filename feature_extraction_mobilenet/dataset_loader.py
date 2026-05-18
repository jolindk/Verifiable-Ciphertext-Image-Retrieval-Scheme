import os
import re
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import numpy as np


FLICKR24_LABELS = [
    "sky", "clouds", "water", "sea", "river", "lake", "people", "portrait",
    "male", "female", "baby", "night", "plant_life", "tree", "flower", "animals",
    "dog", "bird", "structures", "sunset", "indoor", "transport", "car", "food",
]


def _build_common_transform(image_size=224):
    # 与 ImageNet 预训练 VGG16 的输入分布对齐
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class ImageDataset(Dataset):

    def __init__(self, image_dir, max_images=500, offset=0):
        self.image_dir = image_dir
        
        all_items = os.listdir(image_dir)
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in all_items if f.lower().endswith(image_extensions)]
        
        def extract_number(filename):
            match = re.search(r'\d+', filename)
            return int(match.group()) if match else float('inf')
        
        image_files.sort(key=extract_number)
        
        # 使用 offset 跳过前 N 张，再取 max_images 张
        self.image_files = image_files[offset : offset + max_images]
        
        print(f"已加载 {len(self.image_files)} 张图片 (offset={offset})")

        self.transform = _build_common_transform(image_size=224)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, self.image_files[idx]


def load_flickr24_gt_for_names(annotation_dir, image_names):
    """
    按 image_names 顺序构造 24 维多标签矩阵。
    gt[i, j] = 1 表示 image_names[i] 属于第 j 个标签。
    """
    name_to_idx = {n: i for i, n in enumerate(image_names)}
    gt = np.zeros((len(image_names), len(FLICKR24_LABELS)), dtype=np.float32)

    for label_idx, label in enumerate(FLICKR24_LABELS):
        path = os.path.join(annotation_dir, f"{label}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"annotation file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_id = int(line)
                img_name = f"im{img_id}.jpg"
                if img_name in name_to_idx:
                    gt[name_to_idx[img_name], label_idx] = 1.0
    return gt


class Flickr24MultiLabelDataset(Dataset):
    """
    训练/验证用：返回 (image_tensor, multi_hot_label, image_name)。
    """

    def __init__(self, image_dir, annotation_dir, max_images=10000, offset=0):
        self.image_dir = image_dir
        all_items = os.listdir(image_dir)
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        image_files = [f for f in all_items if f.lower().endswith(image_extensions)]

        def extract_number(filename):
            match = re.search(r"\d+", filename)
            return int(match.group()) if match else float("inf")

        image_files.sort(key=extract_number)
        
        # 跳过 offset 张后再取 max_images 张
        self.image_files = image_files[offset : offset + max_images]
        
        self.transform = _build_common_transform(image_size=224)
        labels_np = load_flickr24_gt_for_names(annotation_dir, self.image_files)
        self.labels = torch.from_numpy(labels_np)
        print(f"多标签训练/验证集已加载 {len(self.image_files)} 张图片 (offset={offset})")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label, img_name