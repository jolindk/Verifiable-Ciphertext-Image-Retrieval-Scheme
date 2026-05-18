import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np

from evaluate_p_at_k import (
    compute_query_ranking_metrics,
    evaluate_retrieval_metrics,
    load_flickr24_gt,
)

from feature_extraction.feature_extract import extract_features

from bitmap_index.build_bitmap_index import BitmapIndex
from homomorphic_encryption.encrypt_bitmap import EncryptBitmap
from homomorphic_encryption.secure_match import secure_match


def parse_k_values(pk_values: str) -> List[int]:

    return [int(x.strip()) for x in pk_values.split(",") if x.strip()]


def maybe_set_threshold(threshold):
    if threshold is None:
        return

    try:
        import feature_extraction.config as config
        config.THRESHOLD = float(threshold)
        print(f"[Config] Runtime threshold set to {config.THRESHOLD}")
    except Exception as e:
        print(f"[Warning] Failed to set runtime threshold: {e}")


def split_metrics_by_type(metrics: Dict[str, float], ks: List[int]) -> Dict:

    return {
        "precision_at_k": {
            f"P@{k}": metrics.get(f"P@{k}", None) for k in ks
        },
        "recall_at_k": {
            f"R@{k}": metrics.get(f"R@{k}", None) for k in ks
        },
        "f1_at_k": {
            f"F1@{k}": metrics.get(f"F1@{k}", None) for k in ks
        },
        "mAP": metrics.get("mAP", None),
    }


def evaluate_encrypted_retrieval_metrics_batch(
    feature_dict,
    gt_matrix: np.ndarray,
    ks: List[int],
) -> Dict[str, float]:
    """
    密文检索评价。

    排序方式：
    1. 先根据 feature_dict 构建 Bitmap Index；
    2. 对 Bitmap Index 使用 Paillier 加密；
    3. 对每个 query，在密文状态下计算匹配分数；
    4. 解密分数后排序；
    5. 用 ground-truth 标签计算 P@K、R@K、F1@K、mAP。

    注意：
    这个函数会对每个 query 都进行一次密文检索。
    如果 max-images 很大，运行会比较慢。
    """
    image_names = list(feature_dict.keys())
    feats = np.array([feature_dict[n] for n in image_names], dtype=np.int32)
    n = feats.shape[0]

    index_builder = BitmapIndex(feature_dict)
    bitmap_index = index_builder.build_index()

    enc = EncryptBitmap()
    encrypted_bitmap = enc.encrypt_index(bitmap_index)

    accum = {}
    for k in ks:
        accum[f"P@{k}"] = []
        accum[f"R@{k}"] = []
        accum[f"F1@{k}"] = []

    ap_list = []

    for q_idx in range(n):
        q_feat = feats[q_idx].tolist()

        encrypted_scores = secure_match(
            encrypted_bitmap,
            q_feat,
            enc.paillier,
            enc.pub,
        )

        scores = np.array(
            [
                int(enc.paillier.decrypt(enc.pub, enc.priv, c))
                for c in encrypted_scores
            ],
            dtype=np.int64,
        )

        # 去掉查询图像自身，避免自己匹配自己导致指标偏高
        scores[q_idx] = -1

        order = np.argsort(-scores)

        q_metrics = compute_query_ranking_metrics(order, q_idx, gt_matrix, ks)

        if q_metrics is None:
            continue

        for key, value in q_metrics.items():
            if key == "AP":
                ap_list.append(value)
            else:
                accum[key].append(value)

    out = {}

    for key, values in accum.items():
        out[key] = float(np.mean(values)) if values else 0.0

    out["mAP"] = float(np.mean(ap_list)) if ap_list else 0.0

    return out


def run_retrieval_only(args) -> Dict:

    ks = parse_k_values(args.pk_values)

    maybe_set_threshold(args.threshold)

    t0 = time.perf_counter()

    feature_dict = extract_features(
        args.photo_dir,
        weights_path=args.weights,
        device=args.device,
        max_images=args.max_images,
        offset=args.offset,
    )

    feature_extraction_ms = (time.perf_counter() - t0) * 1000.0

    out = {
        "mode": "retrieval_only",
        "photo_dir": args.photo_dir,
        "annotation_dir": args.annotation_dir,
        "weights": args.weights,
        "device": args.device,
        "max_images": args.max_images,
        "offset": args.offset,
        "threshold": args.threshold,
        "ks": ks,
        "num_images": len(feature_dict),
        "feature_extraction_ms": feature_extraction_ms,
    }

    if not args.annotation_dir or not os.path.exists(args.annotation_dir):
        out["error"] = "missing_annotation_dir"
        print("当前未检测到可用标注目录，无法计算检索评价指标。")
        print("请使用 --annotation-dir 指向 MIRFlickr-25K 的 24 个标签 txt 文件目录。")
        return out

    image_names = list(feature_dict.keys())

    # 根据当前参与评估的 image_names 构造 ground-truth 标签矩阵
    gt_matrix = load_flickr24_gt(args.annotation_dir, image_names)

    t1 = time.perf_counter()

    retrieval_metrics = evaluate_retrieval_metrics(
        feature_dict,
        gt_matrix,
        ks,
    )

    retrieval_metric_ms = (time.perf_counter() - t1) * 1000.0

    out["retrieval_metrics"] = retrieval_metrics
    out.update(split_metrics_by_type(retrieval_metrics, ks))
    out["retrieval_metric_ms"] = retrieval_metric_ms
    out["total_time_ms"] = feature_extraction_ms + retrieval_metric_ms

    print("\n[Retrieval Only]")
    print(f"images={out['num_images']}")
    print(f"feature_extraction_ms={feature_extraction_ms:.2f}")
    print(f"retrieval_metric_ms={retrieval_metric_ms:.2f}")

    for key, value in retrieval_metrics.items():
        print(f"{key}={value:.4f}")

    print(f"total_time_ms={out['total_time_ms']:.2f}")

    return out


def run_encrypted_retrieval_only(args) -> Dict:
    """
    密文检索实验。

    用于明文检索 vs 密文检索对比。
    """
    ks = parse_k_values(args.pk_values)

    maybe_set_threshold(args.threshold)

    t0 = time.perf_counter()

    feature_dict = extract_features(
        args.photo_dir,
        weights_path=args.weights,
        device=args.device,
        max_images=args.max_images,
        offset=args.offset,
    )

    feature_extraction_ms = (time.perf_counter() - t0) * 1000.0

    out = {
        "mode": "encrypted_retrieval_only",
        "photo_dir": args.photo_dir,
        "annotation_dir": args.annotation_dir,
        "weights": args.weights,
        "device": args.device,
        "max_images": args.max_images,
        "offset": args.offset,
        "threshold": args.threshold,
        "ks": ks,
        "num_images": len(feature_dict),
        "feature_extraction_ms": feature_extraction_ms,
    }

    if not args.annotation_dir or not os.path.exists(args.annotation_dir):
        out["error"] = "missing_annotation_dir"
        print("当前未检测到可用标注目录，无法计算密文检索评价指标。")
        return out

    image_names = list(feature_dict.keys())
    gt_matrix = load_flickr24_gt(args.annotation_dir, image_names)

    t1 = time.perf_counter()

    retrieval_metrics = evaluate_encrypted_retrieval_metrics_batch(
        feature_dict,
        gt_matrix,
        ks,
    )
    print("\n========== 🔍 示例检索结果 ==========")

    # 构建索引
    index_builder = BitmapIndex(feature_dict)
    bitmap_index = index_builder.build_index()

    enc = EncryptBitmap()
    encrypted_bitmap = enc.encrypt_index(bitmap_index)

    image_names = list(feature_dict.keys())

    # 选第一张作为 query（也可以改随机）
    query_idx = 324
    query_name = image_names[query_idx]
    query_feat = feature_dict[query_name]

    # 密文匹配
    encrypted_scores = secure_match(
        encrypted_bitmap,
        query_feat,
        enc.paillier,
        enc.pub,
    )

    # 解密
    scores = np.array([
        int(enc.paillier.decrypt(enc.pub, enc.priv, c))
        for c in encrypted_scores
    ])

    # 排除自己
    scores[query_idx] = -1

    # 排序
    order = np.argsort(-scores)

    print(f"查询图像: {query_name}\n")
    print("Top-10 检索结果：")

    topk_names = []
    topk_scores = []

    for i in range(10):
        idx = order[i]
        name = image_names[idx]
        score = int(scores[idx])

        topk_names.append(name)
        topk_scores.append(score)

        print(f"{i+1:02d}. {name} | 相似度: {score}")
    print("====================================\n")
        # ====== 保存TopK结果到JSON ======
    out["query_name"] = query_name
    out["topk_names"] = topk_names
    out["topk_scores"] = topk_scores

    encrypted_retrieval_metric_ms = (time.perf_counter() - t1) * 1000.0

    out["retrieval_metrics"] = retrieval_metrics
    out.update(split_metrics_by_type(retrieval_metrics, ks))
    out["encrypted_retrieval_metric_ms"] = encrypted_retrieval_metric_ms
    out["total_time_ms"] = feature_extraction_ms + encrypted_retrieval_metric_ms

    print("\n[Encrypted Retrieval Only]")
    print(f"images={out['num_images']}")
    print(f"feature_extraction_ms={feature_extraction_ms:.2f}")
    print(f"encrypted_retrieval_metric_ms={encrypted_retrieval_metric_ms:.2f}")

    for key, value in retrieval_metrics.items():
        print(f"{key}={value:.4f}")

    print(f"total_time_ms={out['total_time_ms']:.2f}")

    return out


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate MIRFlickr-25K retrieval metrics"
    )

    # 不写死数据路径，运行时必须传入
    parser.add_argument(
        "--photo-dir",
        required=True,
        help="图像数据集路径，例如 dataset/mirflickr25k/mirflickr",
    )

    parser.add_argument(
        "--annotation-dir",
        required=True,
        help="MIRFlickr-25K 标签 txt 文件目录，例如 dataset/mirflickr25k_annotations_v080",
    )

    parser.add_argument(
        "--weights",
        default=None,
        help="模型权重路径。未训练 VGG16 可以不传。",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="运行设备：cpu 或 cuda",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=25000,
        help="参与评估的图像数量",
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="跳过前 N 张图像。例如 offset=15000 表示从第 15001 张开始取。",
    )

    parser.add_argument(
        "--pk-values",
        default="10,20,30,40,50",
        help="需要计算的 K 值，例如 10,20,30,40,50",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="二值化阈值。若不传，则使用 feature_extraction/config.py 中的默认值。",
    )

    parser.add_argument(
        "--save-json",
        default="results/experiment_metrics.json",
        help="实验结果保存路径",
    )

    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="只运行明文检索评价",
    )

    parser.add_argument(
        "--encrypted-retrieval-only",
        action="store_true",
        help="只运行密文检索评价",
    )

    return parser


def main():
    args = build_parser().parse_args()

    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)

    if args.retrieval_only:
        metrics = run_retrieval_only(args)
    elif args.encrypted_retrieval_only:
        metrics = run_encrypted_retrieval_only(args)
    else:
        print("请指定运行模式：")
        print("  --retrieval-only              明文检索评价")
        print("  --encrypted-retrieval-only    密文检索评价")
        return

    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nSaved metrics JSON: {args.save_json}")


if __name__ == "__main__":
    main()