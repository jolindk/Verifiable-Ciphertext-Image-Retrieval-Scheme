# 图片尺寸
IMAGE_SIZE = 224

# 特征维度（类别数量）
# Flickr-25K: 24 个类别（论文实验维度）
FEATURE_DIM = 24

# 二值化阈值 σ
THRESHOLD = 0.3

# batch size
BATCH_SIZE = 16

# 数据路径
# 仅在直接运行 `feature_extract.py` 时使用
IMAGE_DATASET_PATH = "./dataset/mirflickr25k/mirflickr"