import numpy as np


class BitmapIndex:

    def __init__(self, feature_dict):

        self.feature_dict = feature_dict
        self.image_names = list(feature_dict.keys())

        self.feature_matrix = np.array(
            list(feature_dict.values())
        )

        self.num_images = self.feature_matrix.shape[0]
        self.feature_dim = self.feature_matrix.shape[1]


    def build_index(self):

        """
        构建 Bitmap Index
        """

        bitmap_index = []

        for i in range(self.feature_dim):

            column = self.feature_matrix[:, i]

            bitmap_index.append(column.tolist())

        return bitmap_index


    def print_index(self, bitmap_index):

        for i,bitmap in enumerate(bitmap_index):

            print("Feature", i, ":", bitmap)
