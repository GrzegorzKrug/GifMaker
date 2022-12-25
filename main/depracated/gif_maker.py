import matplotlib.pyplot as plt
import numpy as np
import imutils
import time
import cv2
import PIL
import sys
import os

from PIL import Image
from collections import deque

from video_gif_maker import export_gif


def conv2d_max(arr, k=3):
    out = arr.copy()
    if not k % 2:
        k += 1
        print(f"Adding 1 to kernel size: {k}")
    half = k // 2
    h, w = arr.shape

    for r, row in enumerate(arr):
        if r - half < 0 or r + half >= h:
            continue

        for c, val in enumerate(row):
            if c - half < 0 or c + half >= w:
                continue

            area = arr[r - half:r + half + 1, c - half:c + half + 1]
            v = np.max(area)

            out[r, c] = v

    return out


if __name__ == "__main__":
    src = "clipy" + os.path.sep
    dest = "gify" + os.path.sep

    cap = cv2.VideoCapture(src + "fikol.mp4")

    export_gif(cap, dest + "NKFikol.gif", st=510, end=780,
               w1=1075, w2=0, h1=250, h2=40,
               smooth=0,
               rgb_mode=True
               )

    # export_gif(cap, dest + "Neko720.gif", st=750, end=1583, x1=32, x2=66.5, y1=10, y2=79,
    #            save_interval=3, downsize=150)

    # 1180, 1400
    # export_gif(cap, dest + "NekoWioslo.gif", st=1250, end=1450, x1=26, x2=66.5, y1=10, y2=80,
    #            save_interval=3)
