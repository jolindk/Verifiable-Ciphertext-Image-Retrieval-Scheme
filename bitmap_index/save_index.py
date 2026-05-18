import json


def save_bitmap(bitmap_index, path):

    with open(path,"w") as f:

        json.dump(bitmap_index,f)


def load_bitmap(path):

    with open(path,"r") as f:

        bitmap = json.load(f)

    return bitmap
