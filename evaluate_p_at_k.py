import argparse
import os
import re
import numpy as np

from feature_extraction.feature_extract import extract_features


# Flickr-25K 常用 24 个概念标签
FLICKR24_LABELS = [
    "sky", "clouds", "water", "sea", "river", "lake", "people", "portrait",
    "male", "female", "baby", "night", "plant_life", "tree", "flower", "animals",
    "dog", "bird", "structures", "sunset", "indoor", "transport", "car", "food",
]


def _image_id_from_name(name: str) -> int:
    m = re.search(r"\d+", name)
    if not m:
        raise ValueError(f"cannot parse image id from name: {name}")
    return int(m.group())


def load_flickr24_gt(annotation_dir: str, image_names):
    """
    构造 ground-truth 多标签矩阵：
    gt[i, j] = 1 表示 image_names[i] 属于第 j 个标签。
    """
    name_to_idx = {n: i for i, n in enumerate(image_names)}
    gt = np.zeros((len(image_names), len(FLICKR24_LABELS)), dtype=np.int32)

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
                    gt[name_to_idx[img_name], label_idx] = 1

    return gt


def is_relevant(q_gt, candidate_gt, min_shared_labels=2):
    """
    判断候选图像是否与查询图像相关。

    默认 min_shared_labels=2，表示：
    候选图像与查询图像至少共享 2 个真实标签，才算相关。

    如果想改回“共享 1 个标签就算相关”，可以将 min_shared_labels 改成 1。
    """
    shared_count = np.logical_and(q_gt, candidate_gt).sum()
    return shared_count >= min_shared_labels


def build_relevant_mask(gt_matrix, q_idx, min_shared_labels=2):
    """
    为某个查询图像构造相关样本布尔向量。

    relevant_mask[i] = True 表示第 i 张图像与查询图像相关。
    """
    q_gt = gt_matrix[q_idx]

    # 计算查询图像与所有候选图像共享的标签数量
    shared_counts = np.logical_and(gt_matrix, q_gt).sum(axis=1)

    # 至少共享 min_shared_labels 个标签才算相关
    relevant_mask = shared_counts >= min_shared_labels

    # 排除查询图像自身
    relevant_mask[q_idx] = False

    return relevant_mask


def evaluate_p_at_k(feature_dict, gt_matrix, k, min_shared_labels=2):
    """
    单独计算 P@K。

    正样本定义：
    候选图像与查询图像至少共享 min_shared_labels 个真实标签。
    默认 min_shared_labels=2，即至少共享 2 个标签才算相关。

    排序分数：
    二值特征点积。
    """
    image_names = list(feature_dict.keys())
    feats = np.array([feature_dict[n] for n in image_names], dtype=np.int32)
    n = feats.shape[0]

    precisions = []

    for q_idx in range(n):
        q_feat = feats[q_idx]

        scores = feats @ q_feat

        # 去掉自己，避免自己匹配自己导致结果偏高
        scores[q_idx] = -1

        topk_idx = np.argsort(-scores)[:k]

        q_gt = gt_matrix[q_idx]

        if q_gt.sum() == 0:
            continue

        positives = 0

        for idx in topk_idx:
            if is_relevant(q_gt, gt_matrix[idx], min_shared_labels=min_shared_labels):
                positives += 1

        precisions.append(positives / k)

    if not precisions:
        return 0.0

    return float(np.mean(precisions))


def compute_query_ranking_metrics(
    order,
    q_idx,
    gt_matrix,
    ks,
    min_shared_labels=2,
):
    """
    对单个 query 的排序结果计算：
    1. Precision@K
    2. Recall@K
    3. F1@K
    4. AP

    默认相关性定义：
    候选图像与查询图像至少共享 2 个真实标签。
    """
    q_gt = gt_matrix[q_idx]

    # 查询图像没有标签时跳过
    if q_gt.sum() == 0:
        return None

    # 相关样本：与 query 至少共享 min_shared_labels 个真实标签
    relevant_mask = build_relevant_mask(
        gt_matrix=gt_matrix,
        q_idx=q_idx,
        min_shared_labels=min_shared_labels,
    )

    total_relevant = int(relevant_mask.sum())

    if total_relevant == 0:
        return None

    metrics = {}

    for k in ks:
        topk_idx = order[:k]

        tp = int(relevant_mask[topk_idx].sum())

        precision = tp / k
        recall = tp / total_relevant

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        metrics[f"P@{k}"] = float(precision)
        metrics[f"R@{k}"] = float(recall)
        metrics[f"F1@{k}"] = float(f1)

    # 计算 AP
    hits = 0
    precision_sum = 0.0

    for rank, idx in enumerate(order, start=1):
        if relevant_mask[idx]:
            hits += 1
            precision_sum += hits / rank

    metrics["AP"] = float(precision_sum / total_relevant)

    return metrics


def evaluate_retrieval_metrics(
    feature_dict,
    gt_matrix,
    ks,
    min_shared_labels=2,
):
    """
    计算整体检索指标：
    1. P@K
    2. R@K
    3. F1@K
    4. mAP

    默认相关性定义：
    候选图像与查询图像至少共享 2 个真实标签。
    """
    image_names = list(feature_dict.keys())
    feats = np.array([feature_dict[n] for n in image_names], dtype=np.int32)

    n = feats.shape[0]

    accum = {}

    for k in ks:
        accum[f"P@{k}"] = []
        accum[f"R@{k}"] = []
        accum[f"F1@{k}"] = []

    ap_list = []

    valid_query_count = 0

    for q_idx in range(n):
        q_feat = feats[q_idx]

        # 二值特征点积作为相似度分数
        scores = feats @ q_feat

        # 排除查询图像自身
        scores[q_idx] = -1

        order = np.argsort(-scores)

        q_metrics = compute_query_ranking_metrics(
            order=order,
            q_idx=q_idx,
            gt_matrix=gt_matrix,
            ks=ks,
            min_shared_labels=min_shared_labels,
        )

        if q_metrics is None:
            continue

        valid_query_count += 1

        for key, value in q_metrics.items():
            if key == "AP":
                ap_list.append(value)
            else:
                accum[key].append(value)

    out = {}

    for key, values in accum.items():
        out[key] = float(np.mean(values)) if values else 0.0

    out["mAP"] = float(np.mean(ap_list)) if ap_list else 0.0
    out["valid_query_count"] = valid_query_count
    out["min_shared_labels"] = min_shared_labels

    return out


def parse_k_values(pk_values: str):
    """
    将 '10,20,30,40,50' 转换成 [10, 20, 30, 40, 50]
    """
    return [int(x.strip()) for x in pk_values.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics on Flickr-25K")

    parser.add_argument(
        "--image-dir",
        "--photo-dir",
        dest="image_dir",
        required=True,
        help="image directory",
    )

    parser.add_argument(
        "--annotation-dir",
        required=True,
        help="annotation txt directory",
    )

    parser.add_argument(
        "--weights",
        default=None,
        help="trained model weights path (.pth)",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="cpu or cuda",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=25000,
        help="number of images used for evaluation",
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="skip first N images",
    )

    parser.add_argument(
        "--pk-values",
        default="10,20,30,40,50",
        help="K values, e.g. 10,20,30,40,50",
    )

    parser.add_argument(
        "--min-shared-labels",
        type=int,
        default=2,
        help="minimum number of shared ground-truth labels to define relevance",
    )

    args = parser.parse_args()

    ks = parse_k_values(args.pk_values)

    feature_dict = extract_features(
        args.image_dir,
        weights_path=args.weights,
        device=args.device,
        max_images=args.max_images,
        offset=args.offset,
    )

    image_names = list(feature_dict.keys())

    gt = load_flickr24_gt(
        annotation_dir=args.annotation_dir,
        image_names=image_names,
    )

    metrics = evaluate_retrieval_metrics(
        feature_dict=feature_dict,
        gt_matrix=gt,
        ks=ks,
        min_shared_labels=args.min_shared_labels,
    )

    print("\n[Retrieval Metrics]")
    print(f"min_shared_labels: {args.min_shared_labels}")
    print(f"valid_query_count: {metrics.get('valid_query_count', 0)}")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()