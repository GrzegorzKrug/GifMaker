from typing import Union

import cv2
import numpy as np

from modules.adapters import sequence_adapter


def _get_clip_src_indexes(cur_offset, size):
    """"""
    if cur_offset < 0:
        start_ind = abs(cur_offset)
        stop_ind = size
    elif cur_offset > 0:
        start_ind = 0
        stop_ind = size - cur_offset
    else:
        start_ind = 0
        stop_ind = size
    return start_ind, stop_ind


def _get_clip_dst_indexes(cur_offset, size):
    if cur_offset < 0:
        start_ind = 0
        stop_ind = size - abs(cur_offset)
    elif cur_offset > 0:
        start_ind = cur_offset
        stop_ind = size
    else:
        start_ind = 0
        stop_ind = size
    return start_ind, stop_ind


@sequence_adapter
def max_image_size(sequence: Union[list, np.ndarray], max_height=400, max_width=500, minsize=250):
    h, w, c = sequence[0].shape

    over_height = h / max_height
    over_width = w / max_width
    hw_ratio = h / w
    # print(sequence[0].shape)

    if over_height >= over_width:
        new_width = np.round(max_height / hw_ratio).astype(int)
        new_size = new_width, max_height
        # kw_str = 'height'
        # kw_val = max_height
    else:
        new_heigh = np.round(max_width * hw_ratio).astype(int)
        new_size = max_width, new_heigh
        # kw_str = 'width'
        # kw_val = max_width

    if over_height > 1 or over_width > 1:
        sequence = [cv2.resize(fr, new_size) for fr in sequence]

    return sequence


def blend_region(base, overlay):
    _, _, ch_seq = base.shape
    _, _, ch_overlay = overlay.shape

    if ch_seq == 4 or ch_overlay == 4:
        if ch_seq == 3:
            base_add_alpha = True
        else:
            base_add_alpha = False

        if ch_overlay == 3:
            overlay_add_alpha = True
        else:
            overlay_add_alpha = False

        has_alpha = True
    else:
        has_alpha = False
        base_add_alpha = False
        overlay_add_alpha = False

    if base_add_alpha:
        print("Adding alpha to base")
        alpha = np.zeros_like(base) + 255
        base = np.concatenate([base, alpha], axis=2)

    if overlay_add_alpha:
        print("Adding alpha to overlay")
        alpha = np.zeros_like(overlay) + 255
        overlay = np.concatenate([overlay, alpha], axis=2)

    if has_alpha:
        beta = overlay[:, :, 3][:, :, np.newaxis]
        alfa = 255 - beta

        beta = beta / 255
        alfa = alfa / 255

        combined = base[:, :, :3] * alfa + overlay[:, :, :3] * beta
        combined = combined.round().astype(int)

        combined_alpha = (base[:, :, 3].astype(int) + overlay[:, :, 3])[:, :, np.newaxis].astype(int)
        # combined_alpha = (overlay[:, :, 3])[:, :, np.newaxis].astype(int)

        merged = np.concatenate([combined, combined_alpha], axis=2)
        merged = np.clip(merged, 0, 255).astype(np.uint8)

    else:
        merged = overlay

    return merged


def get_overlay_indexes(base_axis_size, position, overlay_size):
    possible_pixels_on_left = position
    possible_pixels_on_right = base_axis_size - 1 - position

    # overlay_center_pixel = np.floor(overlay_size / 2).astype(int)
    overlay_center_pixel = (overlay_size - 1) // 2

    if possible_pixels_on_left > 0:
        x1 = overlay_center_pixel - possible_pixels_on_left
        if x1 < 0:
            x1 = 0
    else:
        x1 = overlay_center_pixel

    # print()
    # print(f"Possible right: {possible_pixels_on_right}")
    # print(f"Possible left: {possible_pixels_on_left}")
    # print(f"Center: {overlay_center_pixel}")

    if possible_pixels_on_right > 0:
        x2 = overlay_center_pixel + possible_pixels_on_right + 1
        if x2 > overlay_size:
            x2 = overlay_size
    else:
        x2 = overlay_center_pixel + 1

    clip_half = np.ceil((x2 - x1) / 2).astype(int)
    size = x2 - x1
    dx1 = position - clip_half + 1

    if dx1 < 0:
        dx1 = 0
    dx2 = dx1 + size

    if dx2 > base_axis_size:
        diff = dx2 - base_axis_size
        dx2 = base_axis_size
        dx1 -= diff

    return x1, x2, dx1, dx2