def secure_match(encrypted_bitmap, query_vector, paillier, pub):

    """
    encrypted_bitmap:
        l × n 的加密位图

    query_vector:
        查询特征向量

    return:
        每张图片的加密相似度
    """

    num_images = len(encrypted_bitmap[0])
    feature_dim = len(query_vector)

    encrypted_scores = []
    # 乘法累积的单位元。对 g=n+1 的 Paillier 来说，1 等价于 Enc(0)（r=1）。
    enc0 = 1

    for img_idx in range(num_images):

        score = enc0

        for i in range(feature_dim):

            if query_vector[i] == 1:

                c = encrypted_bitmap[i][img_idx]

                score = (score * c) % pub.n_square

        encrypted_scores.append(score)

    return encrypted_scores
