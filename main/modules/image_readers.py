import numpy as np

from PIL import Image


def read_gif(path):
    img = Image.open(path, )
    ind = 0
    sequence = []
    # img = img.convert("RGBA")
    img.seek(0)
    # fr = np.array(img, dtype=np.uint8)
    while True:
        fr = np.array(img, dtype=np.uint8).copy()
        # print(f"Read shape: {fr.shape}")
        sequence.append(fr)

        try:
            img.seek(ind)
        except EOFError:
            # print(f"Breaking at: {ind}")
            break

        # print(img.tell())
        ind += 1
    return sequence


def read_webp(path):
    img = Image.open(path)
    ind = 0
    sequence = []
    # img = img.convert("RGBA")
    img.seek(0)
    # fr = np.array(img, dtype=np.uint8)
    while True:
        fr = np.array(img, dtype=np.uint8).copy()
        # print(f"Read shape: {fr.shape}")
        sequence.append(fr)

        try:
            img.seek(ind)
        except EOFError:
            # print(f"Breaking at: {ind}")
            break

        # print(img.tell())
        ind += 1
    return sequence


if __name__ == "__main__":
    pass

    path = "3x.webp"

    seq = read_webp(path)
    print(seq)
