import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import os

from feature_extraction.dataset_loader import ImageDataset
import feature_extraction.config as config


class FeatureExtractor(nn.Module):

    def __init__(self, feature_dim):

        super(FeatureExtractor, self).__init__()

        # 1 加载预训练VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # 2 保留卷积层
        self.features = vgg.features

        # 3 改进后的全连接层 ⭐⭐⭐
        self.classifier = nn.Sequential(

            nn.Linear(512*7*7, 1024),

            # ⭐ BatchNorm
            nn.BatchNorm1d(1024),

            nn.ReLU(inplace=True),

            # ⭐ Dropout
            nn.Dropout(p=0.5),

            nn.Linear(1024, feature_dim)

        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x


def load_feature_extractor(feature_dim, weights_path=None, device="cpu"):
    model = FeatureExtractor(feature_dim)

    if weights_path:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"weights file not found: {weights_path}")
        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)

    model.to(device)
    model.eval()
    return model


def extract_features(image_dir, weights_path=None, device="cpu", max_images=25000, offset=0):

    dataset = ImageDataset(image_dir, max_images=max_images, offset=offset)

    loader = DataLoader(dataset,
                        batch_size=config.BATCH_SIZE,
                        shuffle=False)

    model = load_feature_extractor(
        config.FEATURE_DIM,   # 推理阶段仍用训练时的维度
        weights_path=weights_path,
        device=device
    )

    feature_dict = {}

    with torch.no_grad():

        for images, names in loader:
            images = images.to(device)

            logits = model(images)

            outputs = torch.sigmoid(logits)

            binary_features = (outputs > config.THRESHOLD).int()

            for i in range(len(names)):
                feature_dict[names[i]] = binary_features[i].tolist()

    return feature_dict