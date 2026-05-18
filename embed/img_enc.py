import numpy as np
from PIL import Image
import random
import os


def scramble_and_encrypt(channel_array, seed, block_size, decrypt=False):
    """
    同一个函数同时支持加密/解密：
    - 加密：先块重排（scrambling），再对每个块做 XOR 流加密
    - 解密：对每个块做 XOR（可逆），再对块重排执行逆变换
    """
    num_blocks_per_row = channel_array.shape[1] // block_size[1]
    num_blocks_per_col = channel_array.shape[0] // block_size[0]

    # 提取完整块（不含右侧/底部余数区域）
    blocks = [
        channel_array[i : i + block_size[0], j : j + block_size[1]]
        for i in range(0, num_blocks_per_col * block_size[0], block_size[0])
        for j in range(0, num_blocks_per_row * block_size[1], block_size[1])
    ]

    np.random.seed(seed)
    perm = np.random.permutation(len(blocks))  # encryption: blocks_original -> blocks_scrambled

    # 在解密模式下：输入已经是 “scrambled + XOR”的结果，当前 blocks 顺序等价于 blocks_scrambled 顺序
    blocks_scrambled = blocks if decrypt else [blocks[i] for i in perm]

    # XOR（同一套随机流，XOR 自身可逆）
    random.seed(seed)
    np.random.seed(seed)
    processed_blocks = []
    for block in blocks_scrambled:
        block_height, block_width = block.shape
        block_flat = block.flatten()
        block_bin = np.array([list(format(val, "08b")) for val in block_flat]).flatten()
        random_string = np.random.randint(2, size=block_height * block_width * 8).astype(int)
        xor = np.logical_xor(block_bin.astype(int), random_string).astype(int).astype(str)
        processed_block = (
            np.packbits(np.array([int(x) for x in xor])).reshape(block_height, block_width)
        )
        processed_blocks.append(processed_block)

    if decrypt:
        # 当前 processed_blocks 对应 blocks_scrambled_plain（= blocks_original[perm] 的明文）
        # 还原到 blocks_original 顺序
        blocks_original = [None] * len(blocks)
        # blocks_scrambled[k] = blocks_original[perm[k]]  => blocks_original[perm[k]] = blocks_scrambled[k]
        for scrambled_idx, orig_idx in enumerate(perm):
            blocks_original[orig_idx] = processed_blocks[scrambled_idx]
        blocks_final = blocks_original
    else:
        # processed_blocks 对应 scrambled order（需要继续按 row-major 放回）
        blocks_final = processed_blocks

    # 合并回主区域
    new_channel_array = np.concatenate(
        [
            np.concatenate(blocks_final[i : i + num_blocks_per_row], axis=1)
            for i in range(0, len(blocks_final), num_blocks_per_row)
        ],
        axis=0,
    )

    # 右侧余数区域递归
    if channel_array.shape[1] % block_size[1] != 0:
        right_remain = channel_array[
            : num_blocks_per_col * block_size[0], num_blocks_per_row * block_size[1] :
        ]
        right_remain_processed = scramble_and_encrypt(
            right_remain, seed, (block_size[0], right_remain.shape[1]), decrypt=decrypt
        )
        new_channel_array = np.concatenate([new_channel_array, right_remain_processed], axis=1)

    # 底部余数区域递归
    if channel_array.shape[0] % block_size[0] != 0:
        bottom_remain = channel_array[num_blocks_per_col * block_size[0] :, :]
        bottom_remain_processed = scramble_and_encrypt(
            bottom_remain, seed, (bottom_remain.shape[0], block_size[1]), decrypt=decrypt
        )
        new_channel_array = np.concatenate([new_channel_array, bottom_remain_processed], axis=0)

    return new_channel_array


def encrypt_images_in_folder(folder_path, rand_num, block_size, output_folder=None):
    if output_folder is None:
        output_folder = folder_path
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if not file_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue

        img = Image.open(file_path).convert("RGB")
        r, g, b = img.split()
        r_array = np.array(r)
        g_array = np.array(g)
        b_array = np.array(b)

        new_r = scramble_and_encrypt(r_array, rand_num, block_size, decrypt=False)
        new_g = scramble_and_encrypt(g_array, rand_num, block_size, decrypt=False)
        new_b = scramble_and_encrypt(b_array, rand_num, block_size, decrypt=False)

        new_image = Image.merge("RGB", [Image.fromarray(new_r), Image.fromarray(new_g), Image.fromarray(new_b)])

        encrypted_file_name = os.path.splitext(file_name)[0] + "_encrypted_64.bmp"
        encrypted_file_path = os.path.join(output_folder, encrypted_file_name)
        new_image.save(encrypted_file_path)


def encrypt_dataset(input_folder, output_folder):
    block_size = [64, 64]
    rand_num = 2023
    encrypt_images_in_folder(input_folder, rand_num, block_size, output_folder)


def decrypt_images_in_folder(folder_path, rand_num, block_size, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if not file_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue

        img = Image.open(file_path).convert("RGB")
        r, g, b = img.split()
        r_array = np.array(r)
        g_array = np.array(g)
        b_array = np.array(b)

        new_r = scramble_and_encrypt(r_array, rand_num, block_size, decrypt=True)
        new_g = scramble_and_encrypt(g_array, rand_num, block_size, decrypt=True)
        new_b = scramble_and_encrypt(b_array, rand_num, block_size, decrypt=True)

        new_image = Image.merge("RGB", [Image.fromarray(new_r), Image.fromarray(new_g), Image.fromarray(new_b)])

        base, ext = os.path.splitext(file_name)
        if base.endswith("_encrypted_64"):
            base = base[: -len("_encrypted_64")]
        decrypted_file_name = base + ext
        decrypted_file_path = os.path.join(output_folder, decrypted_file_name)
        new_image.save(decrypted_file_path)


def decrypt_dataset(input_folder, output_folder):
    block_size = [64, 64]
    rand_num = 2023
    decrypt_images_in_folder(input_folder, rand_num, block_size, output_folder)