import hashlib
import os
import random
import math
from typing import List, Tuple

import numpy as np
from PIL import Image

IAC_HASH_DIGEST_BYTES = hashlib.sha256().digest_size
IAC_BIT_LEN = IAC_HASH_DIGEST_BYTES * 8


class RDHCapacityError(RuntimeError):
    """Raised when strict RDH capacity is insufficient."""


def _bytes_to_bits(data: bytes) -> List[int]:
    bits: List[int] = []
    for byte in data:
        for k in range(7, -1, -1):
            bits.append((byte >> k) & 1)
    return bits


def _bits_to_bytes(bits: List[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("bits length must be multiple of 8")
    out = bytearray()
    for i in range(0, len(bits), 8):
        v = 0
        for b in bits[i : i + 8]:
            v = (v << 1) | (b & 1)
        out.append(v)
    return bytes(out)


def _xor_bits(bits: List[int], seed: int) -> List[int]:
    rnd = random.Random(seed)
    return [(b ^ rnd.randint(0, 1)) for b in bits]


def compute_iac_sha256_bits_from_image(rgb_uint8: np.ndarray) -> Tuple[bytes, List[int]]:
    sha256 = hashlib.sha256(rgb_uint8.tobytes()).digest()
    return sha256, _bytes_to_bits(sha256)


def _int_to_bits(v: int, width: int) -> List[int]:
    return [((v >> (width - 1 - i)) & 1) for i in range(width)]


def _bits_to_int(bits: List[int]) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | (b & 1)
    return v


def _channel_msb_equal(ch: np.ndarray, i: int, j: int) -> bool:
    return (int(ch[i, j]) & 0x80) == (int(ch[i + 1, j]) & 0x80)


def _is_available(r: np.ndarray, g: np.ndarray, b: np.ndarray, i: int, j: int) -> bool:
    return _channel_msb_equal(r, i, j) and _channel_msb_equal(g, i, j) and _channel_msb_equal(b, i, j)


def _scan_blocks(h: int, w: int) -> List[Tuple[int, int]]:
    blocks = []
    h_even = h - (h % 2)
    for i in range(0, h_even, 2):
        for j in range(w):
            blocks.append((i, j))
    return blocks


def _header_bit_width(num_blocks: int) -> int:
    # 对齐论文思路：Lsize 使用 log2(块数) 量级位宽表示
    return max(1, int(math.ceil(math.log2(num_blocks + 1))))


def _pick_prefix_for_capacity(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    iac_len_bits: int,
    alpha: int,
) -> Tuple[int, List[int], int, int]:
    """
    选择最短前缀长度 lsize，使其能同时承载：
    - 控制信息：header_bits 的 lsize + L 向量（lsize bits）  => control_len = header_bits + lsize
    - 密文载荷：iac_len_bits + control_len（control 原位恢复所需）
      载荷容量来自前缀内可用块：3 * alpha * avail_count
    """
    blocks = _scan_blocks(r.shape[0], r.shape[1])
    header_bits = _header_bit_width(len(blocks))
    L: List[int] = []
    avail_count = 0

    for idx, (i, j) in enumerate(blocks, start=1):
        avail = 1 if _is_available(r, g, b, i, j) else 0
        L.append(avail)
        if avail:
            avail_count += 1

        lsize = idx
        control_len = header_bits + lsize
        payload_len = iac_len_bits + control_len
        payload_cap = 3 * alpha * avail_count
        # 控制位容量：每块第一像素 RGB 的 LSB 可用 3*(8-alpha) bits
        control_cap = 3 * (8 - alpha) * lsize

        if control_cap >= control_len and payload_cap >= payload_len:
            return lsize, L, avail_count, header_bits

    total_avail = sum(L)
    total_payload_cap = 3 * alpha * total_avail
    raise RDHCapacityError(
        "insufficient embeddable capacity for strict RDH mode: "
        f"needed_payload_bits={iac_len_bits}, "
        f"total_available_blocks={total_avail}/{len(blocks)}, "
        f"total_payload_capacity={total_payload_cap}"
    )


def embed_iac_in_image(
    encrypted_image_path: str,
    embedded_image_path: str,
    kdh_seed: int = 2025,
    alpha: int = 1,
) -> None:
    """
    论文风格（无 sidecar）：
    - 图内控制参数：lsize + L
    - 图内密文载荷：Enc_kdh(IAC || control_restore_bits)
    - 恢复时不依赖外部文件
    """
    rgb = np.array(Image.open(encrypted_image_path).convert("RGB"))
    r = rgb[:, :, 0].copy()
    g = rgb[:, :, 1].copy()
    b = rgb[:, :, 2].copy()

    beta = alpha
    gamma = 8 - alpha - beta
    if alpha <= 0 or alpha > 4 or gamma < 0:
        raise ValueError("invalid alpha/beta/gamma setting for RDH")

    _, iac_bits = compute_iac_sha256_bits_from_image(rgb)
    lsize, L, _, header_bits = _pick_prefix_for_capacity(r, g, b, len(iac_bits), alpha)
    control_bits = _int_to_bits(lsize, header_bits) + L[:lsize]
    control_len = len(control_bits)

    # 记录控制区原始位，用于恢复第一像素 LSB
    control_orig_bits: List[int] = []

    blocks = _scan_blocks(r.shape[0], r.shape[1])
    control_written = 0
    for blk_idx in range(lsize):
        i, j = blocks[blk_idx]
        # 第一像素每通道低(8-alpha)位作为控制位载体（更贴近论文表述）
        channels = [int(r[i, j]), int(g[i, j]), int(b[i, j])]
        for ch_idx in range(3):
            low_bits = 8 - alpha
            old_low = channels[ch_idx] & ((1 << low_bits) - 1)
            old_low_bits = _int_to_bits(old_low, low_bits)
            for b_old in old_low_bits:
                if control_written >= control_len:
                    break
                control_orig_bits.append(b_old)
                control_written += 1
            if control_written >= control_len and len(control_orig_bits) >= control_len:
                # no more bits to write
                pass

        # 再次逐位写入，保持顺序一致
        write_cursor = 0
        for ch_idx in range(3):
            low_bits = 8 - alpha
            old_low = channels[ch_idx] & ((1 << low_bits) - 1)
            old_low_bits = _int_to_bits(old_low, low_bits)
            new_low_bits = old_low_bits[:]
            for p_idx in range(low_bits):
                global_idx = (blk_idx * 3 * low_bits) + (ch_idx * low_bits) + p_idx
                # 仅当该位置映射到控制串前缀时才写
                if global_idx >= control_len:
                    break
                new_low_bits[p_idx] = control_bits[global_idx]
            new_low = _bits_to_int(new_low_bits)
            channels[ch_idx] = (channels[ch_idx] & (~((1 << low_bits) - 1))) | new_low

        r[i, j], g[i, j], b[i, j] = channels[0], channels[1], channels[2]
        if control_written >= control_len:
            break

    payload_plain = iac_bits + control_orig_bits
    payload_cipher = _xor_bits(payload_plain, kdh_seed)

    # 载荷写入“可用块”的第二像素高 alpha 位（RGB 共 3*alpha bits/块）
    p = 0
    for blk_idx in range(lsize):
        if L[blk_idx] != 1:
            continue
        i, j = blocks[blk_idx]
        for ch in ("r", "g", "b"):
            if p >= len(payload_cipher):
                break
            if ch == "r":
                val = int(r[i + 1, j])
            elif ch == "g":
                val = int(g[i + 1, j])
            else:
                val = int(b[i + 1, j])

            bits = _int_to_bits(val, 8)
            for t in range(alpha):
                if p >= len(payload_cipher):
                    break
                bits[t] = payload_cipher[p]
                p += 1
            new_val = _bits_to_int(bits)

            if ch == "r":
                r[i + 1, j] = new_val
            elif ch == "g":
                g[i + 1, j] = new_val
            else:
                b[i + 1, j] = new_val
        if p >= len(payload_cipher):
            break

    if p < len(payload_cipher):
        raise RuntimeError("payload not fully embedded")

    out = np.stack([r, g, b], axis=2).astype(np.uint8)
    os.makedirs(os.path.dirname(embedded_image_path) or ".", exist_ok=True)
    Image.fromarray(out).save(embedded_image_path)


def extract_iac_and_restore(
    embedded_image_path: str,
    restored_encrypted_image_path: str,
    kdh_seed: int = 2025,
    alpha: int = 1,
) -> Tuple[bytes, List[int]]:
    rgb = np.array(Image.open(embedded_image_path).convert("RGB"))
    r = rgb[:, :, 0].copy()
    g = rgb[:, :, 1].copy()
    b = rgb[:, :, 2].copy()

    beta = alpha
    gamma = 8 - alpha - beta
    if alpha <= 0 or alpha > 4 or gamma < 0:
        raise ValueError("invalid alpha/beta/gamma setting for RDH")

    blocks = _scan_blocks(r.shape[0], r.shape[1])
    header_bits = _header_bit_width(len(blocks))

    # 先读取 header_bits 个控制位以获取 lsize
    first_header: List[int] = []
    for blk_idx in range(len(blocks)):
        i, j = blocks[blk_idx]
        for ch_val in (int(r[i, j]), int(g[i, j]), int(b[i, j])):
            low_bits = _int_to_bits(ch_val & ((1 << (8 - alpha)) - 1), 8 - alpha)
            for bit in low_bits:
                if len(first_header) < header_bits:
                    first_header.append(bit)
                else:
                    break
            if len(first_header) >= header_bits:
                break
        if len(first_header) >= header_bits:
            break
    if len(first_header) < header_bits:
        raise RuntimeError("cannot read lsize header")

    lsize = _bits_to_int(first_header)
    control_len = header_bits + lsize

    # 读取完整 control bits（lsize 前缀块上的第一像素 LSB）
    control_bits: List[int] = []
    for blk_idx in range(lsize):
        i, j = blocks[blk_idx]
        for ch_val in (int(r[i, j]), int(g[i, j]), int(b[i, j])):
            low_bits = _int_to_bits(ch_val & ((1 << (8 - alpha)) - 1), 8 - alpha)
            for bit in low_bits:
                if len(control_bits) < control_len:
                    control_bits.append(bit)
                else:
                    break
            if len(control_bits) >= control_len:
                break
        if len(control_bits) >= control_len:
            break
    if len(control_bits) < control_len:
        raise RuntimeError("cannot read full control bits")

    L = control_bits[header_bits:]
    payload_len = IAC_BIT_LEN + control_len

    # 从可用块中读取密文载荷
    payload_cipher: List[int] = []
    for blk_idx in range(lsize):
        if L[blk_idx] != 1:
            continue
        i, j = blocks[blk_idx]
        for ch_val in (int(r[i + 1, j]), int(g[i + 1, j]), int(b[i + 1, j])):
            bits = _int_to_bits(ch_val, 8)
            for t in range(alpha):
                if len(payload_cipher) < payload_len:
                    payload_cipher.append(bits[t])
                else:
                    break
            if len(payload_cipher) >= payload_len:
                break
        if len(payload_cipher) >= payload_len:
            break
    if len(payload_cipher) < payload_len:
        raise RuntimeError("cannot read full payload")

    payload_plain = _xor_bits(payload_cipher, kdh_seed)
    iac_bits = payload_plain[:IAC_BIT_LEN]
    control_orig_bits = payload_plain[IAC_BIT_LEN:]
    if len(control_orig_bits) != control_len:
        raise RuntimeError("control restore bits length mismatch")

    # 恢复第二像素高 alpha 位：根据可用块性质，将其还原为第一像素高 alpha 位
    for blk_idx in range(lsize):
        if L[blk_idx] != 1:
            continue
        i, j = blocks[blk_idx]
        for ch in ("r", "g", "b"):
            if ch == "r":
                p1 = _int_to_bits(int(r[i, j]), 8)
                p2 = _int_to_bits(int(r[i + 1, j]), 8)
            elif ch == "g":
                p1 = _int_to_bits(int(g[i, j]), 8)
                p2 = _int_to_bits(int(g[i + 1, j]), 8)
            else:
                p1 = _int_to_bits(int(b[i, j]), 8)
                p2 = _int_to_bits(int(b[i + 1, j]), 8)

            for t in range(alpha):
                p2[t] = p1[t]
            new_val = _bits_to_int(p2)

            if ch == "r":
                r[i + 1, j] = new_val
            elif ch == "g":
                g[i + 1, j] = new_val
            else:
                b[i + 1, j] = new_val

    # 恢复第一像素控制载体位（每通道低 8-alpha 位）
    c = 0
    for blk_idx in range(lsize):
        i, j = blocks[blk_idx]
        vals = [int(r[i, j]), int(g[i, j]), int(b[i, j])]
        for ch_idx in range(3):
            low_bits = 8 - alpha
            old_low_bits = _int_to_bits(vals[ch_idx] & ((1 << low_bits) - 1), low_bits)
            for p_idx in range(low_bits):
                if c >= control_len:
                    break
                old_low_bits[p_idx] = control_orig_bits[c]
                c += 1
            new_low = _bits_to_int(old_low_bits)
            vals[ch_idx] = (vals[ch_idx] & (~((1 << low_bits) - 1))) | new_low
            if c >= control_len:
                # 允许提前退出当前块剩余位
                pass
        r[i, j], g[i, j], b[i, j] = vals[0], vals[1], vals[2]
        if c >= control_len:
            break

    restored = np.stack([r, g, b], axis=2).astype(np.uint8)
    os.makedirs(os.path.dirname(restored_encrypted_image_path) or ".", exist_ok=True)
    Image.fromarray(restored).save(restored_encrypted_image_path)

    return _bits_to_bytes(iac_bits), iac_bits


def embed_iac_for_encrypted_folder(
    encrypted_dir: str,
    embedded_dir: str,
    kdh_seed: int = 2025,
    alpha: int = 1,
) -> None:
    os.makedirs(embedded_dir, exist_ok=True)
    total = 0
    success = 0
    skipped = 0
    for file_name in os.listdir(encrypted_dir):
        if not file_name.lower().endswith(".bmp"):
            continue
        total += 1
        in_path = os.path.join(encrypted_dir, file_name)
        out_name = os.path.splitext(file_name)[0] + "_embedded.bmp"
        out_path = os.path.join(embedded_dir, out_name)
        try:
            embed_iac_in_image(in_path, out_path, kdh_seed=kdh_seed, alpha=alpha)
            success += 1
        except RDHCapacityError as e:
            skipped += 1
            print(f"[RDH-SKIP] {file_name}: {e}")

    print(
        f"[RDH] Embedded {success}/{total} images into {embedded_dir}, "
        f"skipped={skipped}"
    )

