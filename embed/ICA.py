import argparse
import csv
import hashlib
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt

import embed
from img_enc import scramble_and_encrypt


def parse_block_size(block_size_str: str) -> Tuple[int, int]:
    """
    将 '64,64' 转换为 (64, 64)
    """
    parts = block_size_str.split(",")
    if len(parts) != 2:
        raise ValueError("block-size 格式应为 '64,64'")
    return int(parts[0]), int(parts[1])


def natural_key(filename: str):
    """
    按文件名中的数字排序，例如 im2.jpg 排在 im10.jpg 前面。
    """
    nums = re.findall(r"\d+", filename)
    return int(nums[0]) if nums else float("inf")


def sha256_of_rgb_image(path: str) -> str:
    """
    对图像 RGB 像素内容计算 SHA-256。
    注意：这里计算的是像素数组，而不是文件字节。
    """
    rgb = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def encrypt_single_image(
    image_path: str,
    encrypted_path: str,
    seed: int,
    block_size: Tuple[int, int],
) -> None:
    """
    使用项目中的置乱 + XOR 方法加密单张图像。
    """
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    r_enc = scramble_and_encrypt(r, seed, block_size, decrypt=False)
    g_enc = scramble_and_encrypt(g, seed, block_size, decrypt=False)
    b_enc = scramble_and_encrypt(b, seed, block_size, decrypt=False)

    encrypted_rgb = np.stack([r_enc, g_enc, b_enc], axis=2).astype(np.uint8)

    os.makedirs(os.path.dirname(encrypted_path) or ".", exist_ok=True)
    Image.fromarray(encrypted_rgb).save(encrypted_path)


def decrypt_single_image(
    encrypted_path: str,
    decrypted_path: str,
    seed: int,
    block_size: Tuple[int, int],
) -> None:
    """
    将恢复后的加密图像解密为明文图像。
    注意：seed 和 block_size 必须与加密时保持一致。
    """
    img = Image.open(encrypted_path).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    r_dec = scramble_and_encrypt(r, seed, block_size, decrypt=True)
    g_dec = scramble_and_encrypt(g, seed, block_size, decrypt=True)
    b_dec = scramble_and_encrypt(b, seed, block_size, decrypt=True)

    decrypted_rgb = np.stack([r_dec, g_dec, b_dec], axis=2).astype(np.uint8)

    os.makedirs(os.path.dirname(decrypted_path) or ".", exist_ok=True)
    Image.fromarray(decrypted_rgb).save(decrypted_path)


def safe_resampling_lanczos():
    """
    兼容不同版本 Pillow。
    """
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def make_five_stage_gallery(
    original_path: str,
    encrypted_path: str,
    embedded_path: str,
    restored_path: str,
    decrypted_restored_path: str,
    output_path: str,
    title: str = "Verifiable Image Recovery Example",
) -> None:
    """
    使用 PIL 生成五阶段对比图：
    原始图像 / 加密图像 / 嵌入 IAC 图像 / 恢复加密图像 / 解密恢复图像
    """
    stage_paths = [
        original_path,
        encrypted_path,
        embedded_path,
        restored_path,
        decrypted_restored_path,
    ]

    stage_titles = [
        "Original Image",
        "Encrypted Image",
        "IAC Embedded Image",
        "Restored Encrypted Image",
        "Decrypted Restored Image",
    ]

    thumb_w, thumb_h = 240, 190
    margin = 24
    gap = 16
    title_h = 48
    label_h = 34

    canvas_w = margin * 2 + thumb_w * 5 + gap * 4
    canvas_h = margin * 2 + title_h + label_h + thumb_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, 16), title, fill=(20, 20, 20))

    y_label = margin + title_h
    y_img = y_label + label_h

    for i, (p, label) in enumerate(zip(stage_paths, stage_titles)):
        x = margin + i * (thumb_w + gap)

        img = Image.open(p).convert("RGB")
        img = ImageOps.fit(
            img,
            (thumb_w, thumb_h),
            method=safe_resampling_lanczos(),
        )

        draw.text((x + 10, y_label + 8), label, fill=(20, 20, 20))
        canvas.paste(img, (x, y_img))
        draw.rectangle(
            (x - 1, y_img - 1, x + thumb_w, y_img + thumb_h),
            outline=(160, 160, 160),
            width=1,
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas.save(output_path)


def show_five_stage_canvas(
    original_path: str,
    encrypted_path: str,
    embedded_path: str,
    restored_path: str,
    decrypted_restored_path: str,
    title: str = "Verifiable Image Recovery Process",
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    使用 matplotlib 画布显示五阶段图像。
    本地运行时会弹出窗口；Colab 中会显示在输出区域。
    """
    image_paths = [
        original_path,
        encrypted_path,
        embedded_path,
        restored_path,
        decrypted_restored_path,
    ]

    subtitles = [
        "Original Image",
        "Encrypted Image",
        "IAC Embedded Image",
        "Restored Encrypted Image",
        "Decrypted Restored Image",
    ]

    images = [Image.open(p).convert("RGB") for p in image_paths]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), dpi=160)

    for ax, img, subtitle in zip(axes, images, subtitles):
        ax.imshow(img)
        ax.set_title(subtitle, fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=15)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def make_overview_gallery(
    sample_results: List[Dict],
    output_path: str,
    max_rows: int = 2,
) -> None:
    """
    生成多张样本的总览图。
    每一行五列：
    原始图像 / 加密图像 / 嵌入 IAC 图像 / 恢复加密图像 / 解密恢复图像
    """
    selected = sample_results[:max_rows]
    if not selected:
        return

    stage_titles = [
        "Original",
        "Encrypted",
        "IAC Embedded",
        "Restored Enc",
        "Decrypted Restored",
    ]

    thumb_w, thumb_h = 190, 145
    margin = 24
    gap_x = 14
    gap_y = 42
    title_h = 60
    label_h = 28
    row_h = label_h + thumb_h + gap_y

    canvas_w = margin * 2 + thumb_w * 5 + gap_x * 4
    canvas_h = margin * 2 + title_h + row_h * len(selected)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (margin, 18),
        "Verifiability and Image Recovery Visualization",
        fill=(20, 20, 20),
    )

    y0 = margin + title_h

    for col, label in enumerate(stage_titles):
        x = margin + col * (thumb_w + gap_x)
        draw.text((x + 20, y0), label, fill=(20, 20, 20))

    for row_idx, result in enumerate(selected):
        y_img = y0 + label_h + row_idx * row_h

        paths = [
            result["original_path"],
            result["encrypted_path"],
            result["embedded_path"],
            result["restored_path"],
            result["decrypted_restored_path"],
        ]

        for col, p in enumerate(paths):
            x = margin + col * (thumb_w + gap_x)

            img = Image.open(p).convert("RGB")
            img = ImageOps.fit(
                img,
                (thumb_w, thumb_h),
                method=safe_resampling_lanczos(),
            )

            canvas.paste(img, (x, y_img))
            draw.rectangle(
                (x - 1, y_img - 1, x + thumb_w, y_img + thumb_h),
                outline=(160, 160, 160),
                width=1,
            )

        y_text = y_img + thumb_h + 6
        verify_text = "PASS" if result["verify_pass"] else "FAIL"
        recover_text = "PASS" if result["hash_match_original"] else "FAIL"

        draw.text(
            (margin, y_text),
            f"{result['image_name']} | IAC verification: {verify_text} | Plain recovery: {recover_text}",
            fill=(20, 80, 20) if result["verify_pass"] and result["hash_match_original"] else (180, 30, 30),
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas.save(output_path)


def process_one_image(
    image_path: str,
    output_root: str,
    enc_seed: int,
    block_size: Tuple[int, int],
    rdh_seed: int,
    alpha: int,
    show_canvas: bool = True,
) -> Dict:
    """
    处理单张图像：
    1. 保存原图副本
    2. 生成加密图像
    3. 嵌入 IAC
    4. 提取 IAC 并恢复加密图像
    5. 对恢复后的加密图像解密
    6. 计算 Hash 并验证
    7. 生成五阶段可视化图
    """
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]

    sample_dir = os.path.join(output_root, base_name)
    os.makedirs(sample_dir, exist_ok=True)

    original_path = os.path.join(sample_dir, "1_original.png")
    encrypted_path = os.path.join(sample_dir, "2_encrypted.bmp")
    embedded_path = os.path.join(sample_dir, "3_iac_embedded.bmp")
    restored_path = os.path.join(sample_dir, "4_restored_encrypted.bmp")
    decrypted_restored_path = os.path.join(sample_dir, "5_decrypted_restored.png")
    gallery_path = os.path.join(sample_dir, "6_verification_gallery.png")
    canvas_path = os.path.join(sample_dir, "7_verification_canvas_matplotlib.png")

    # 1. 保存原始图像副本
    Image.open(image_path).convert("RGB").save(original_path)

    # 2. 加密图像
    encrypt_single_image(
        image_path=image_path,
        encrypted_path=encrypted_path,
        seed=enc_seed,
        block_size=block_size,
    )

    encrypted_hash_before = sha256_of_rgb_image(encrypted_path)

    # 3. 嵌入 IAC
    embed.embed_iac_in_image(
        encrypted_image_path=encrypted_path,
        embedded_image_path=embedded_path,
        kdh_seed=rdh_seed,
        alpha=alpha,
    )

    # 4. 提取 IAC + 恢复加密图像
    extracted_iac_bytes, _ = embed.extract_iac_and_restore(
        embedded_image_path=embedded_path,
        restored_encrypted_image_path=restored_path,
        kdh_seed=rdh_seed,
        alpha=alpha,
    )

    extracted_iac_hex = extracted_iac_bytes.hex()
    restored_hash = sha256_of_rgb_image(restored_path)

    # 5. 将恢复后的加密图像进一步解密，得到恢复明文图像
    decrypt_single_image(
        encrypted_path=restored_path,
        decrypted_path=decrypted_restored_path,
        seed=enc_seed,
        block_size=block_size,
    )

    original_hash = sha256_of_rgb_image(original_path)
    decrypted_restored_hash = sha256_of_rgb_image(decrypted_restored_path)

    # IAC 验证：恢复后的加密图像 Hash 应与提取出的 IAC 一致
    hash_match_iac = restored_hash == extracted_iac_hex

    # 加密图像恢复验证：恢复后的加密图像应与嵌入 IAC 前的加密图像一致
    hash_match_original_encrypted = restored_hash == encrypted_hash_before

    # 明文恢复验证：解密恢复图像应与原始图像一致
    hash_match_original = original_hash == decrypted_restored_hash

    verify_pass = hash_match_iac and hash_match_original_encrypted

    # 6. 生成 PIL 五阶段拼接图
    make_five_stage_gallery(
        original_path=original_path,
        encrypted_path=encrypted_path,
        embedded_path=embedded_path,
        restored_path=restored_path,
        decrypted_restored_path=decrypted_restored_path,
        output_path=gallery_path,
        title=f"Verifiable Recovery Example: {image_name}",
    )

    # 7. 使用 matplotlib 画布显示并保存高清图
    show_five_stage_canvas(
        original_path=original_path,
        encrypted_path=encrypted_path,
        embedded_path=embedded_path,
        restored_path=restored_path,
        decrypted_restored_path=decrypted_restored_path,
        title=f"Verifiable Recovery Example: {image_name}",
        save_path=canvas_path,
        show=show_canvas,
    )

    result = {
        "image_name": image_name,
        "original_path": original_path,
        "encrypted_path": encrypted_path,
        "embedded_path": embedded_path,
        "restored_path": restored_path,
        "decrypted_restored_path": decrypted_restored_path,
        "gallery_path": gallery_path,
        "canvas_path": canvas_path,
        "original_hash": original_hash,
        "encrypted_hash_before": encrypted_hash_before,
        "extracted_iac_hash": extracted_iac_hex,
        "restored_hash": restored_hash,
        "decrypted_restored_hash": decrypted_restored_hash,
        "hash_match_iac": hash_match_iac,
        "hash_match_original_encrypted": hash_match_original_encrypted,
        "hash_match_original": hash_match_original,
        "verify_pass": verify_pass,
    }

    return result


def collect_image_paths(args) -> List[str]:
    """
    根据 image-path 或 image-dir 收集待处理图像。
    """
    if args.image_path:
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"image-path not found: {args.image_path}")
        return [args.image_path]

    if not args.image_dir:
        raise ValueError("请提供 --image-path 或 --image-dir")

    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"image-dir not found: {args.image_dir}")

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    names = [
        f for f in os.listdir(args.image_dir)
        if f.lower().endswith(exts)
    ]

    names.sort(key=natural_key)

    names = names[args.offset: args.offset + args.max_images]

    return [os.path.join(args.image_dir, n) for n in names]


def save_summary(results: List[Dict], output_dir: str) -> None:
    """
    保存验证结果为 JSON 和 CSV。
    """
    os.makedirs(output_dir, exist_ok=True)

    summary_json = os.path.join(output_dir, "verification_summary.json")
    summary_csv = os.path.join(output_dir, "verification_summary.csv")

    total = len(results)
    verified_success = sum(1 for r in results if r["verify_pass"])
    plain_recovery_success = sum(1 for r in results if r["hash_match_original"])

    verified_rate = verified_success / total if total > 0 else 0.0
    plain_recovery_rate = plain_recovery_success / total if total > 0 else 0.0

    summary = {
        "total_images": total,
        "verified_count": verified_success,
        "verified_rate": verified_rate,
        "plain_recovery_count": plain_recovery_success,
        "plain_recovery_rate": plain_recovery_rate,
        "results": results,
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_name",
            "hash_match_iac",
            "hash_match_original_encrypted",
            "hash_match_original",
            "verify_pass",
            "original_hash",
            "encrypted_hash_before",
            "extracted_iac_hash",
            "restored_hash",
            "decrypted_restored_hash",
            "gallery_path",
            "canvas_path",
        ])

        for r in results:
            writer.writerow([
                r["image_name"],
                r["hash_match_iac"],
                r["hash_match_original_encrypted"],
                r["hash_match_original"],
                r["verify_pass"],
                r["original_hash"],
                r["encrypted_hash_before"],
                r["extracted_iac_hash"],
                r["restored_hash"],
                r["decrypted_restored_hash"],
                r["gallery_path"],
                r["canvas_path"],
            ])

    print("\n========== Verification Summary ==========")
    print(f"Total images: {total}")
    print(f"IAC verified count: {verified_success}")
    print(f"IAC verified rate: {verified_rate:.4f}")
    print(f"Plain recovery count: {plain_recovery_success}")
    print(f"Plain recovery rate: {plain_recovery_rate:.4f}")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary CSV: {summary_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate IAC verification and full image recovery visualization."
    )

    parser.add_argument(
        "--image-path",
        default=None,
        help="单张原始图像路径，例如 dataset/mirflickr25k/mirflickr/im1.jpg",
    )

    parser.add_argument(
        "--image-dir",
        default=None,
        help="图像文件夹路径；如果提供该参数，则会批量处理前 max-images 张图像。",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="批量处理时的图像数量，默认 10。",
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="批量处理时跳过前 N 张图像。",
    )

    parser.add_argument(
        "--output-dir",
        default="verification_output",
        help="输出目录。",
    )

    parser.add_argument(
        "--enc-seed",
        type=int,
        default=2023,
        help="图像加密和解密使用的随机种子。",
    )

    parser.add_argument(
        "--block-size",
        default="64,64",
        help="图像加密块大小，例如 64,64。",
    )

    parser.add_argument(
        "--rdh-seed",
        type=int,
        default=2025,
        help="IAC 嵌入与提取使用的随机种子。",
    )

    parser.add_argument(
        "--alpha",
        type=int,
        default=1,
        help="IAC 嵌入参数 alpha。",
    )

    parser.add_argument(
        "--overview-rows",
        type=int,
        default=2,
        help="总览图中展示多少张样本，建议 1 或 2。",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="只保存图像，不弹出 matplotlib 画布窗口。",
    )

    args = parser.parse_args()

    block_size = parse_block_size(args.block_size)

    image_paths = collect_image_paths(args)

    os.makedirs(args.output_dir, exist_ok=True)

    results = []

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"\n[{idx}/{len(image_paths)}] Processing: {image_path}")

        try:
            result = process_one_image(
                image_path=image_path,
                output_root=args.output_dir,
                enc_seed=args.enc_seed,
                block_size=block_size,
                rdh_seed=args.rdh_seed,
                alpha=args.alpha,
                show_canvas=not args.no_show,
            )

            results.append(result)

            print(f"  hash_match_iac: {result['hash_match_iac']}")
            print(f"  hash_match_original_encrypted: {result['hash_match_original_encrypted']}")
            print(f"  hash_match_original: {result['hash_match_original']}")
            print(f"  verify_pass: {result['verify_pass']}")
            print(f"  PIL gallery: {result['gallery_path']}")
            print(f"  Matplotlib canvas: {result['canvas_path']}")

        except Exception as e:
            print(f"  Failed: {image_path}")
            print(f"  Error: {e}")

    save_summary(results, args.output_dir)

    if results:
        overview_path = os.path.join(args.output_dir, "verification_overview_gallery.png")
        make_overview_gallery(
            sample_results=results,
            output_path=overview_path,
            max_rows=args.overview_rows,
        )
        print(f"Overview gallery: {overview_path}")


if __name__ == "__main__":
    main()