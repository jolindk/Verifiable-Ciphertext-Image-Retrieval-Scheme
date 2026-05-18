import numpy as np


def query_bitmap(bitmap_index, query_vector):

    """
    query_vector:
    [1,0,1,0,1]

    return:
    匹配图片index
    """

    candidate = None

    for i,bit in enumerate(query_vector):

        if bit == 1:

            bitmap = np.array(bitmap_index[i])

            if candidate is None:

                candidate = bitmap

            else:

                candidate = candidate & bitmap

    result = []

    for idx,v in enumerate(candidate):

        if v == 1:

            result.append(idx)

    return result
