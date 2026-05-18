import hashlib
import os

import numpy as np

import embed.embed
import embed.img_enc
from bitmap_index.build_bitmap_index import BitmapIndex
from feature_extraction.feature_extract import extract_features
from homomorphic_encryption.encrypt_bitmap import EncryptBitmap
from homomorphic_encryption.secure_match import secure_match


def _md5_of_rgb_image_path(path: str) -> bytes:
    # 用像素数组计算 MD5，避免 bmp 的文件层元信息差异
    from PIL import Image

    rgb = np.array(Image.open(path).convert("RGB"))
    return hashlib.md5(rgb.tobytes()).digest()


# Flickr-25K 常用 24 个概念标签（与评估脚本一致）
FLICKR24_LABELS = [
    "sky", "clouds", "water", "sea", "river", "lake", "people", "portrait",
    "male", "female", "baby", "night", "plant_life", "tree", "flower", "animals",
    "dog", "bird", "structures", "sunset", "indoor", "transport", "car", "food",
]


def _load_flickr24_gt(annotation_dir: str, image_names):
    name_to_idx = {n: i for i, n in enumerate(image_names)}
    gt = np.zeros((len(image_names), len(FLICKR24_LABELS)), dtype=np.int32)

    for label_idx, label in enumerate(FLICKR24_LABELS):
        path = os.path.join(annotation_dir, f"{label}.txt")
        if not os.path.exists(path):
            return None
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


def _ensure_has_encrypted_images(encrypted_dir: str, photo_dir: str) -> None:
    if os.path.exists(encrypted_dir):
        for f in os.listdir(encrypted_dir):
            if f.lower().endswith(".bmp"):
                return
    os.makedirs(encrypted_dir, exist_ok=True)
    img_enc.encrypt_dataset(photo_dir, encrypted_dir)


def main():
    # ====== 论文实验设置（Flickr-25K: 24 维特征向量） ======
    photo_dir = "dataset/mirflickr25k/photo"
    annotation_dir = "dataset/mirflickr25k_annotations_v080"
    encrypted_dir = "dataset/mirflickr25k/encrypted_dataset"
    embedded_dir = "dataset/mirflickr25k/embedded_dataset"
    restored_dir = "dataset/mirflickr25k/restored_encrypted_dataset"
    decrypted_dir = "dataset/mirflickr25k/decrypted_dataset"

    k = 10

    # ====== GenIndex：提取特征并构建 Bitmap Index ======
    # 这里加上了训练好的权重路径，以使用你训练后的模型
    weights_path = "weights/feature_extractor_flickr24_v2.pth" # 替换为你最新的权重名
    if not os.path.exists(weights_path):
        weights_path = "weights/feature_extractor_flickr24.pth"
    if not os.path.exists(weights_path):
        weights_path = None
        
    feature_dict = extract_features(photo_dir, weights_path=weights_path)
    index_builder = BitmapIndex(feature_dict)
    bitmap_index = index_builder.build_index()

    # ====== EncImage / GenIAC：加密图片（如果没生成过） ======
    _ensure_has_encrypted_images(encrypted_dir, photo_dir)

    # ====== EncIndex：把 bitmap index 加密成 AHB（Paillier） ======
    enc = EncryptBitmap()
    encrypted_bitmap = enc.encrypt_index(bitmap_index)

    # ====== Embed：论文风格严格 RDH（控制信息图内嵌入，不依赖 sidecar） ======
    # 先做一轮全量预嵌入；容量不足样本会被跳过并记录，不会中断流程。
    kdh_seed = 2025
    rdh_alpha = 1
    embed.embed_iac_for_encrypted_folder(
        encrypted_dir=encrypted_dir,
        embedded_dir=embedded_dir,
        kdh_seed=kdh_seed,
        alpha=rdh_alpha,
    )

    # ====== Search：加密域检索并得到候选 top-k ======
    # 选择第一张图片作为查询图（你也可以改成自己指定的 imX.jpg）
    query_name = index_builder.image_names[0]
    query_idx = index_builder.image_names.index(query_name)
    query_vector = feature_dict[query_name]

    encrypted_scores = secure_match(encrypted_bitmap, query_vector, enc.paillier, enc.pub)
    scores = [enc.paillier.decrypt(enc.pub, enc.priv, c) for c in encrypted_scores]

    # Debug：用明文 bit 向量计算“真实相似度”，与密文解密结果对齐性检查
    # 评分定义：score = sum_i (query[i] * feature[img_idx][i])
    qv = np.array(query_vector, dtype=int)
    plain_scores = []
    for img_idx in range(index_builder.feature_matrix.shape[0]):
        plain_scores.append(int(np.dot(qv, index_builder.feature_matrix[img_idx].astype(int))))

    # 打印 query 的得分 & 是否最高
    print("Query feature ones:", int(qv.sum()))
    print("Query decrypted score:", scores[query_idx])
    print("Query plaintext score:", plain_scores[query_idx])
    print("Max decrypted score:", max(scores), "Max plaintext score:", max(plain_scores))

    # 检查 decrypted/明文是否一致（允许极小差异）
    mismatch = [(i, scores[i], plain_scores[i]) for i in range(len(scores)) if int(scores[i]) != int(plain_scores[i])]
    print("Score mismatch count:", len(mismatch))
    if mismatch[:5]:
        print("Score mismatch examples (idx, dec, plain):", mismatch[:5])

    # 排除 query 自身，避免 trivially 命中
    ranked_indices = sorted(range(len(scores)), key=lambda i: int(scores[i]), reverse=True)
    ranked_indices = [i for i in ranked_indices if i != query_idx]
    topk_indices = ranked_indices[:k]

    print("Query:", query_name)
    print("Top-k candidates (plaintext scores):")
    for idx in topk_indices:
        print("  ", index_builder.image_names[idx], scores[idx])

    gt_matrix = _load_flickr24_gt(annotation_dir, index_builder.image_names)
    if gt_matrix is not None and gt_matrix[query_idx].sum() > 0:
        hit = 0
        q_gt = gt_matrix[query_idx]
        for idx in topk_indices:
            if np.logical_and(q_gt, gt_matrix[idx]).any():
                hit += 1
        print(f"Query P@{k} (GT-based): {hit / k:.4f}")
    else:
        print("Query P@k skipped: GT annotations unavailable or query has no labels.")

    os.makedirs(restored_dir, exist_ok=True)
    verified_names = []
    skipped_capacity_names = []
    skipped_missing_names = []

    for idx in topk_indices:
        name = index_builder.image_names[idx]  # 例如 im1.jpg
        base, _ = os.path.splitext(name)  # im1

        encrypted_name = f"{base}_encrypted_64.bmp"
        embedded_name = f"{os.path.splitext(encrypted_name)[0]}_embedded.bmp"
        embedded_path = os.path.join(embedded_dir, embedded_name)
        restored_path = os.path.join(restored_dir, encrypted_name)

        if not os.path.exists(embedded_path):
            encrypted_path = os.path.join(encrypted_dir, encrypted_name)
            if not os.path.exists(encrypted_path):
                skipped_missing_names.append(name)
                print(f"Skip {name}: missing encrypted image: {encrypted_path}")
                continue
            try:
                embed.embed_iac_in_image(
                    encrypted_image_path=encrypted_path,
                    embedded_image_path=embedded_path,
                    kdh_seed=kdh_seed,
                    alpha=rdh_alpha,
                )
            except embed.RDHCapacityError as e:
                skipped_capacity_names.append(name)
                print(f"Skip {name}: {e}")
                continue

        extracted_iac_bytes, _ = embed.extract_iac_and_restore(
            embedded_image_path=embedded_path,
            restored_encrypted_image_path=restored_path,
            kdh_seed=kdh_seed,
            alpha=rdh_alpha,
        )

        restored_md5 = _md5_of_rgb_image_path(restored_path)
        if restored_md5 == extracted_iac_bytes:
            verified_names.append(name)

    print("Verified results:", verified_names)
    if skipped_capacity_names:
        print("Skipped by RDH capacity:", skipped_capacity_names)
    if skipped_missing_names:
        print("Skipped by missing encrypted image:", skipped_missing_names)


    total_candidates = len(topk_indices)
    skipped_count = len(skipped_capacity_names) + len(skipped_missing_names)
    total_verified_candidates = total_candidates - skipped_count
    passed_count = len(verified_names)
    verification_pass_rate = (
        (passed_count / total_verified_candidates) if total_verified_candidates > 0 else 0.0
    )

    print("\n[Verifiability Metrics]")
    print(f"Total retrieval candidates (Top-{k}): {total_candidates}")
    print(f"Participated in verification: {total_verified_candidates}")
    print(f"Passed verification: {passed_count}")
    print(f"Verification pass rate: {verification_pass_rate:.4f}")

    os.makedirs(decrypted_dir, exist_ok=True)
    img_enc.decrypt_dataset(restored_dir, decrypted_dir)


if __name__ == "__main__":
    main()
