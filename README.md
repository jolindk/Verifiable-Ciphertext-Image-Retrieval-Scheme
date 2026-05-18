# 可验证密文图像检索方案（VEIR）

**项目简介**

本项目实现了一个面向图像检索的可验证密文图像检索系统（Verifiable Ciphertext Image Retrieval，VEIR）。系统将深度特征嵌入、位图索引与同态/可验证加密技术结合，支持在加密域上进行图像检索并对检索结果进行可验证性判定，适用于隐私保护图像检索、医疗/云端图像安全存储等场景。
**项目目录结构（节选）**

```
Verifiable-Ciphertext-Image-Retrieval-Scheme/
├─ main.py                      # 检索流程入口示例
├─ verify_result.py             # 检索结果验证脚本
├─ r_exam.py                    # 实验/示例脚本
├─ embed/                       # 图像嵌入与加密相关
│  ├─ embed.py
│  ├─ img_enc.py
│  └─ ICA.py
├─ feature_extraction/          # 特征提取（VGG / 自定义）
│  ├─ config.py
│  ├─ dataset_loader.py
│  ├─ feature_extract.py
│  └─ train_feature_extractor.py
├─ feature_extraction_mobilenet/ # MobileNet 变体的特征提取
├─ homomorphic_encryption/      # Paillier 与同态/加密匹配实现
│  ├─ paillier.py
│  ├─ encrypt_bitmap.py
│  └─ secure_match.py
├─ bitmap_index/                # 位图索引构建与保存
│  ├─ build_bitmap_index.py
│  ├─ bitmap_utils.py
│  └─ save_index.py
├─ dataset/                     # 数据集（mirflickr25k）与预处理产物
├─ results/                     # 实验输出与评估结果
├─ weights/                     # 训练好的模型权重（.pth）
└─ visualization*               # 可视化与安全性评估输出
```
**项目声明**

- 项目名称：可验证密文图像检索方案（VEIR）
- 项目作者：请在此处填写作者姓名（例如：Your Name）
- 作者单位：请在此处填写作者单位/实验室
- 开发语言：Python
- 主要依赖：PyTorch、NumPy、SciPy、Pillow、scikit-learn、matplotlib
- 核心技术：深度特征嵌入、位图索引、Paillier 同态加密、可验证检索协议

# 可验证密文图像检索方案（VEIR）

## 项目简介
本项目实现了一个面向图像检索的可验证密文图像检索（Verifiable Ciphertext Image Retrieval，VEIR）方案。系统支持在加密图像上进行特征嵌入、索引建立、加密检索以及检索结果的可验证性评估。适用于研究隐私保护检索、同态/可验证加密与图像特征匹配等方向。

## 仓库概览
- 根脚本：`main.py`, `verify_result.py`, `r_exam.py` 等用于运行与验证实验。
- 嵌入与加密：`embed/`（`embed.py`, `img_enc.py`, `ICA.py`）负责图像嵌入与加密处理。
- 特征提取：`feature_extraction/` 与 `feature_extraction_mobilenet/` 包含模型配置、数据加载、特征提取与训练脚本。
- 同态/加密工具：`homomorphic_encryption/`（`paillier.py`, `encrypt_bitmap.py`, `secure_match.py`）实现 Paillier 相关加密与加密匹配逻辑。
- 位图索引：`bitmap_index/`（`build_bitmap_index.py`, `bitmap_utils.py`, `save_index.py`）用于构建和管理位图索引以支持快速检索。



## 依赖（建议）
建议创建虚拟环境并安装以下常见依赖：

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install torch torchvision numpy scipy pillow scikit-learn matplotlib tqdm
```

此外，项目中使用了自定义的 Paillier 与位图工具，可能还需要安装其它科学计算库。请根据报错补充依赖。

## 快速开始
（以下命令为示例，运行前请根据本地环境与配置调整）

- 提取图像特征（示例）：

```bash
python feature_extraction/feature_extract.py --config feature_extraction/config.py
```

- 使用 MobileNet 变种提取特征：

```bash
python feature_extraction_mobilenet/feature_extract.py --config feature_extraction_mobilenet/config.py
```

- 训练特征提取器：

```bash
python feature_extraction/train_feature_extractor.py
```

- 构建位图索引：

```bash
python bitmap_index/build_bitmap_index.py
```

- 嵌入/加密图像：

```bash
python embed/embed.py
# 或
python embed/img_enc.py
```

- 运行检索与验证流程（示例）：

```bash
python main.py
python verify_result.py
```

具体参数和选项请查看各子目录中的脚本头部或 `config.py`。

## 数据集说明
数据位于 `dataset/mirflickr25k/`，包含原始图像、嵌入后的数据、加密数据及标注（`mirflickr25k_annotations_v080/`）。如需预处理或恢复数据，查看相应脚本和 `rdh_aux` 子目录。

## 实验与结果
- 结果文件夹：`results/` 包含若干实验输出 JSON（如 `mobilenet__threshold_0_5.json` 等）。
- 验证输出：`verification_output_single/verification_summary.json` 与 `verification_summary.csv` 提供可验证性评估摘要。

## 开发者提示
- 若使用 GPU，请确保 `torch` 可见到 CUDA 并调整脚本中的设备配置。
- 运行大型数据处理或训练时，请保证有足够磁盘空间与内存。
- 部分脚本可能使用硬编码路径或本地配置，首次运行前建议打开并阅读脚本顶部说明与 `config.py`。
