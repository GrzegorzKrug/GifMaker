from modules.modifiers_sequence import cycle_slide

import numpy as np
import cv2
from PIL import Image

import imutils

rgb_image = cv2.imread("main/unknown.png")[:, :, [2, 1, 0]]
rgb_image = imutils.resize(rgb_image, height=150)
sequence_len = N = 35
rgb_sequence = [rgb_image.copy() for n in range(N)]

angles = [
    [140, 130, 90, 47, 43],
    [180, None, None, None, 0],
    [220, 230, 270, 310, 320]
]

height, width, c = rgb_image.shape
output = np.empty((sequence_len, 0, width*5, c), dtype=np.uint8)
print(output.shape)

for row in angles:
    empty_row = np.zeros(
        (sequence_len, height, width*len(row), c), dtype=np.uint8)
    for ang_i, cur_ang in enumerate(row):
        if cur_ang is not None:
            new_seq = cycle_slide(rgb_sequence, cur_ang)
        else:
            new_seq = rgb_sequence
        empty_row[:, :, ang_i*width:(ang_i*width+width)] = new_seq

    output = np.concatenate([output, empty_row], axis=1)

frames = [Image.fromarray(frame) for frame in output]
frames[0].save("test.gif", save_all=True,
               append_images=frames[1:], duration=45, loop=0)
