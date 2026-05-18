import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# =========================
# 模型（必须和训练一致！！！）
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
# 加载模型
# =========================
def load_feature_extractor(weights_path, device):
    model = FeatureExtractor()

    state = torch.load(weights_path, map_location=device)

    if "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


def extract_features(
    image_dir,
    weights_path,
    device="cpu",
    max_images=None,
    offset=0,
    threshold=0.5
):
    import os
    from PIL import Image
    from torchvision import transforms

    model = load_feature_extractor(weights_path, device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".bmp"))
    ])

    image_paths = image_paths[offset:]
    if max_images:
        image_paths = image_paths[:max_images]

    feature_dict = {}

    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            output = model(img)

            # 二值化
            output = torch.sigmoid(output)
            binary = (output > threshold).int().cpu().numpy()[0]

            feature_dict[os.path.basename(path)] = binary

    return feature_dict