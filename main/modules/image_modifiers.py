import cv2
import numpy as np
import os

from PIL import Image

from functools import wraps
from scipy.signal import convolve2d
from scipy import stats as st
from typing import Union

import imutils

from main.modules.math_functions import moving_average
from yasiu_native.time import measure_real_time_decorator


def sequence_adapter(func):
    @wraps(wrapped=func)
    def wrapper(source, *a, **kw):
        if isinstance(source, np.ndarray) and len(source.shape) == 4:
            in_type = '4dnum'
        elif isinstance(source, list):
            in_type = 'list'
        else:
            in_type = 'pic'
            source = [source]

        # print(f"func: {func.__name__}, type: {type(source)} key: {in_type}")
        res = func(source, *a, **kw)

        if in_type == '4dnum':
            res = np.array(res, dtype=np.uint8)
        elif in_type == 'pic':
            return res[0]
        return res

    return wrapper


def layer_validator(fn):
    @wraps(wrapped=fn)
    def wrapper(layer_dict, destination_key, *a, **kw):
        return fn(layer_dict, destination_key, *a, **kw)

    return wrapper


class FunctionCollector:
    def __init__(self):
        self._keys = {}
        self.type_ = "base"
        # Name-function relation

        self.arguments = {}

        # Dict with variables types, limits
        # Type, minval, maxval, Number of vars, Labels

    def adder(self, fkey, *args):
        def wrapper(func):
            name = fkey.lower()
            if name in self._keys:
                raise KeyError(f"Function already registered as: {name}")
            # if len(args) != 3:
            #     raise ValueError(f"3 params required to register filter: {fkey}")
            args_ = list(args)

            for i, a in enumerate(args_):
                a = list(a)
                args_[i] = a
                assert len(a) == 5, f"Arguments does not have 4 params: {name}, but {a}"
                nvars = a[3]
                if nvars == 1:
                    assert isinstance(a[4], str), f"Variable should be single string: {name}"
                    a[4] = [a[4]]
                else:
                    assert nvars == len(a[4]), f"Variable requires list of strings: {name}"

            self._keys[name] = func
            self.arguments[name] = args_
            print(f"Register filter: {name}")

            @wraps(wrapped=func)
            def inner_wrapper(*a, **kw):
                out = func(*a, **kw)
                return out

            return inner_wrapper

        return wrapper

    def get_default(self, fkey):
        params = []
        for dtype, mmin, mmax, nvars, _ in self.arguments[fkey]:
            if nvars == 1:
                if dtype is str or dtype is bool:
                    params.append(mmin)
                elif mmin <= 0 <= mmax:
                    params.append(0)
                elif mmin <= 1 <= mmax:
                    params.append(1)
                else:
                    params.append(mmin)
            else:
                inner_p = []
                for n in range(nvars):
                    if dtype is str or dtype is bool:
                        params.append(mmin)
                    elif mmin <= 0 <= mmax:
                        inner_p.append(0)
                    elif mmin <= 1 <= mmax:
                        inner_p.append(1)
                    else:
                        inner_p.append(mmin)
                params.append(inner_p)
        return params

    # @classmethod
    # def __class_getitem__(cls, item):
    #     return cls.keys.get(item, None)
    @property
    def keys(self):
        return sorted(list(self._keys.keys()))

    def __getitem__(self, item):
        return self._keys.get(item, None)


# Filters = FunctionCollector()
SequenceModifiers = FunctionCollector()
SequenceModifiers.type_ = "modifier"


@SequenceModifiers.adder(
        'sequence sampler',
        (str, 'all', ['all', 'linear', 'ratio'], 1, 'Mode'),
        (int, 1, 99999, 1, 'N output frames'),
        (float, -999, 999, 1, 'Non linear variable'),
        (float, 0, 500, 1, "Ratio [%] sample")

)
@sequence_adapter
def sequence_sampler(image_sequence, mode='linear', frames_n=1, mode_value=0, ratio_value=100):
    # print(type(im_ob), type(im_ob) is np.ndarray)
    if len(image_sequence) == 1:
        frame = np.array(image_sequence[0], dtype=np.uint8)
        frames = [frame for _ in range(frames_n)]

    elif type(image_sequence) is np.ndarray and len(image_sequence.shape) == 3:
        # frame = np.array(im_ob, dtype=np.uint8)
        frames = [image_sequence.copy() for _ in range(frames_n)]

    elif mode == "all":
        frames = [np.array(fr, dtype=np.uint8) for fr in image_sequence]

    elif mode == 'linear':
        indexes = np.linspace(0, len(image_sequence), frames_n + 1)[:-1]
        indexes = np.floor(indexes).astype(int)
        frames = [image_sequence[ind] for ind in indexes]

    elif mode == 'ratio':
        ratio_frames = (ratio_value / 100) * len(image_sequence)
        ratio_frames = np.clip(ratio_frames, 1, np.inf).round().astype(int)
        indexes = np.linspace(0, len(image_sequence), ratio_frames + 1)[:-1]
        indexes = np.floor(indexes).astype(int)
        frames = [image_sequence[ind] for ind in indexes]

    else:
        raise ValueError("Only linear mode supported")

    return frames


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


@SequenceModifiers.adder(
        'repeat',
        (int, 1, 99999, 1, 'N times'),
)
def repeat_sequence(im_ob, repeat):
    frames = [fr for _ in range(repeat) for fr in im_ob]

    return frames


@SequenceModifiers.adder(
        'reverse',
        (bool, 0, 1, 1, "Append reversed")
)
def reverse(sequence, append=True):
    new_sequence = [fr.copy() for fr in sequence]
    out = new_sequence.copy()
    if append:
        for img in new_sequence[::-1]:
            out.append(img.copy())

    return out


@SequenceModifiers.adder(
        'Alpha cutoff',
        (float, 0, 255, 1, "Threshold")
)
@sequence_adapter
def cutoff_alpha(sequence, threshold=50):
    out = []
    for img in sequence:
        mask = img[:, :, 3] <= threshold
        img = img.copy()
        # print(mask.shape)
        img[mask, 3] = 0

        out.append(img)
    return out


@SequenceModifiers.adder(
        'slide cycle',
        # (str, 'all', ['all', 'linear'], 1, 'Mode'),
        # (int, 1, 99999, 1, 'N frames'),
        (float, 0, 360, 1, 'Direction')
)
def cycle_slide(sequence: list, angle: float):
    sequence = [fr.copy() for fr in sequence]
    height, width, c = sequence[0].shape
    N = len(sequence)
    rads = np.pi * angle / 180

    sn = np.sin(rads)
    cs = np.cos(rads)

    x_comp = 0
    y_comp = 0
    is_horiz = False

    if angle <= 45 or 315 < angle <= 360:
        if angle == 0 or angle == 360:
            yend = 0
        else:
            yend = -height * sn
        xsteps = np.linspace(0, width, N + 1)[:-1]
        ysteps = np.linspace(0, yend, N)
        is_horiz = True
        y_comp = yend

    elif angle <= 135:
        if angle == 90:
            xend = 0
        else:
            xend = -width * cs

        xsteps = np.linspace(0, xend, N)
        ysteps = np.linspace(0, height, N + 1)[:-1]
        x_comp = xend

    elif angle <= 225:
        if angle == 180:
            yend = 0
        else:
            yend = -height * sn
        xsteps = np.linspace(width, 0, N + 1)[:1]
        ysteps = np.linspace(0, yend, N)
        is_horiz = True
        # y_comp = height + yend
        # x_comp = height - yend
        x_comp = -height * sn
        y_comp = 0

    elif angle <= 315:
        if angle == 270:
            xend = 0
        else:
            xend = -width * cs
        # xsteps = np.linspace(0, xend, N + 1)[:-1]
        xsteps = np.linspace(0, xend, N)
        ysteps = np.linspace(height, 0, N + 1)[:1]
        x_comp = 0
        y_comp = -width * cs

    else:
        print("WHAT?!")
        raise ValueError(f"wrong angle: {angle}")

    xsteps = np.floor(xsteps).astype(int)
    ysteps = np.floor(ysteps).astype(int)
    x_comp = np.round(x_comp).astype(int)
    y_comp = np.round(y_comp).astype(int)
    # print(f"Ang: {angle}, X comp: {x_comp}, y comp: {y_comp}")

    for ind, (fr, x_off, y_off) in enumerate(zip(sequence, xsteps, ysteps)):
        if is_horiz:
            if x_off <= 0:
                continue
            left = fr[:, x_off:]
            right = fr[:, :x_off]

            if y_off != 0:
                left = np.roll(left, y_off - x_comp, axis=0)
                right = np.roll(right, y_off - y_comp, axis=0)

            sequence[ind] = np.concatenate([left, right], axis=1)

        else:
            if y_off <= 0:
                continue
            top = fr[y_off:, :]
            bottom = fr[:y_off, :]

            if x_off != 0:
                # print(f"X != 0: {x_off}")
                top = np.roll(top, x_off - y_comp, axis=1)
                bottom = np.roll(bottom, x_off - x_comp, axis=1)

            sequence[ind] = np.concatenate([top, bottom], axis=0)

    # sequence = [sequence[0], sequence[-1]]
    return sequence


@SequenceModifiers.adder(
        'slide delay',
        # (str, 'all', ['all', 'linear'], 1, 'Mode'),
        # (int, 1, 99999, 1, 'N frames'),
        (float, 0, 360, 1, 'Direction'),
        (float, 0, 360, 1, 'Delay')
)
def cycle_slide_delay(sequence: list, angle: float, delay: float):
    sequence = [fr.copy() for fr in sequence]
    height, width, c = sequence[0].shape
    N = len(sequence)

    rads = np.pi * angle / 180
    is_horiz = bool(round((angle % 180) / 90 - 1))

    anim_duration = 2 + delay
    slide_duration = np.round((1 / anim_duration) * len(sequence)).astype(int)
    blank_duration = len(sequence) - 2 * slide_duration

    sn = np.sin(rads)
    cs = np.cos(rads)
    HW_ratio = height / width
    WH_ratio = width / height

    if is_horiz:
        if angle < 90 or 180 <= angle <= 360:
            x_pos = -width
        else:
            x_pos = width

        missing_pixels = (width - abs(cs * width)) * HW_ratio
        y_pos = -sn * height
        y_pos = y_pos + np.sign(y_pos) * missing_pixels
    else:
        if angle <= 180:
            y_pos = -height
        else:
            y_pos = height

        missing_pixels = (height - abs(sn * height)) * WH_ratio
        x_pos = -cs * width
        x_pos = x_pos + np.sign(x_pos) * missing_pixels

    # x_pos = np.fix(x_pos).astype(int)
    # y_pos = np.fix(y_pos).astype(int)

    X = np.linspace(0, x_pos, slide_duration)
    Y = np.linspace(0, y_pos, slide_duration)

    X = np.fix(X).astype(int)
    Y = np.fix(Y).astype(int)

    output = []

    main_blank = np.zeros((*sequence[0].shape[:2], 4), dtype=np.uint8)

    "First move"
    for fr, x, y in zip(sequence, X, Y):
        blank = main_blank.copy()

        if fr.shape[2] == 3:
            full = main_blank[:, :, 0].copy()[:, :, np.newaxis] + 255
            fr = np.concatenate([fr, full], axis=2)

        x1, x2 = _get_clip_src_indexes(x, width)
        y1, y2 = _get_clip_src_indexes(y, height)

        clip = fr[y1:y2, x1:x2]

        dx1, dx2 = _get_clip_dst_indexes(x, width)
        dy1, dy2 = _get_clip_dst_indexes(y, height)

        blank[dy1:dy2, dx1:dx2] = clip
        output.append(blank)

    "Insert blank frames"
    # print(f"Adding blank frames: {blank_duration}")
    for _ in range(blank_duration):
        output.append(main_blank.copy())

    X = np.linspace(-x_pos, 0, slide_duration)
    Y = np.linspace(-y_pos, 0, slide_duration)

    X = np.fix(X).astype(int)
    Y = np.fix(Y).astype(int)

    start_ind = len(output) - slide_duration
    # print(len(output),start_ind)

    for ind, (curr_fr, x, y) in enumerate(zip(sequence, X, Y), start_ind):
        # print(ind)
        # blank = main_blank.copy()

        if curr_fr.shape[2] == 3:
            full = main_blank[:, :, 0].copy()[:, :, np.newaxis] + 255
            curr_fr = np.concatenate([curr_fr, full], axis=2)

        x1, x2 = _get_clip_src_indexes(x, width)
        y1, y2 = _get_clip_src_indexes(y, height)

        clip = curr_fr[y1:y2, x1:x2]

        dx1, dx2 = _get_clip_dst_indexes(x, width)
        dy1, dy2 = _get_clip_dst_indexes(y, height)

        output[ind][dy1:dy2, dx1:dx2] = clip
        # output.append(blank)

    return output


@SequenceModifiers.adder(
        'color blend',
        (int, 0, 255, 3, ["Red", "Green", "Blue"]),
        (float, 0, 1, 1, "Alpha"),
)
@sequence_adapter
def color_blend(sequence, color, alpha):
    h, w, c = sequence[0].shape
    if c == 4:
        color = (*color, 255)
        has_mask = True
    else:
        has_mask = False
    blank = (np.zeros_like(sequence[0]) + color).astype(np.uint8)
    output = []
    for fr in sequence:
        if has_mask:
            mask = fr[:, :, 3]
        frame = cv2.addWeighted(fr, 1 - alpha, blank, alpha, 0)

        if has_mask:
            frame[:, :, 3] = mask
        output.append(frame)
    return output


@SequenceModifiers.adder(
        'monocolor',
        (int, 0, 255, 3, ["Red", "Green", "Blue"]),
        (float, 0, 1, 1, "Alpha"),
)
@sequence_adapter
def mono_color(sequence, color, alpha):
    h, w, c = sequence[0].shape
    if c == 4:
        color = (*color, 255)
        has_mask = True
    else:
        has_mask = False

    output = []
    for fr in sequence:
        mono = fr.mean(axis=-1) / 255
        if has_mask:
            mask = fr[:, :, 3]
            bgr = np.stack([mono, mono, mono, mono], axis=-1)
        else:
            bgr = np.stack([mono, mono, mono], axis=-1)
        bgr = (bgr * color).round().astype(np.uint8)
        # print(f"Color shape: {bgr.shape}")
        # print(blank.dtype,fr.dtype)
        frame = cv2.addWeighted(fr, 1 - alpha, bgr, alpha, 0)
        if has_mask:
            frame[:, :, 3] = mask

        output.append(frame)
    return output


@SequenceModifiers.adder(
        'dynamic hue',
        # (int, 0, 255, 3, ["Red", "Green", "Blue"]),
        (float, -100, 100, 1, "Starting hue"),
        (float, -100, 100, 1, "N cycles"),
        (float, 0, 1, 1, "Alpha"),
)
@sequence_adapter
def dynamic_hue(sequence, color, n_cycles, alpha):
    size = len(sequence)
    output = [0] * size

    hues = np.linspace(0, 180, size).round().astype(int)
    colors = [0] * size
    for hi, h in enumerate(hues):
        mat = np.array([[[h, 255, 255]]], dtype=np.uint8)

        mat_rgb = cv2.cvtColor(mat, cv2.COLOR_HSV2RGB)
        colors[hi] = mat_rgb.ravel().tolist()
        # print(f"Color: {colors[hi]}")
    # colors = [cv2.cvtColor((h, 255, 255),cv2.COLOR_HSV2RGB) for h in hues]

    for fri, (fr, cr) in enumerate(zip(sequence, colors)):
        new_fr = mono_color(fr, color=cr, alpha=alpha)
        output[fri] = new_fr
    return output


@SequenceModifiers.adder(
        'time clip',
        (float, 0, 100, 1, "Start 100%"),
        (float, 0, 100, 1, "End 100%")
)
@measure_real_time_decorator
def clip_sequence(sequence, start: float, stop: float):
    size = len(sequence)
    start = np.floor(start * size / 100).astype(int)
    stop = np.floor((100 - stop) * size / 100).astype(int)

    sequence = sequence[start:stop + 1]
    return sequence


@SequenceModifiers.adder(
        'image crop',
        (float, 0, 100, 1, "Left 100%"),
        (float, 0, 100, 1, "Right 100%"),
        (float, 0, 100, 1, "Top 100%"),
        (float, 0, 100, 1, "Bottom 100%")
)
@sequence_adapter
@measure_real_time_decorator
def crop_image(sequence, left: float, right: float, top: float, bottom: float):
    orig = sequence[0]
    h, w, c = orig.shape
    top, down, left, right = np.array([top, bottom, left, right], dtype=float) / 100
    top = np.round(top * h).astype(int)
    down = np.round(down * h).astype(int)
    left = np.round(left * w).astype(int)
    right = np.round(right * w).astype(int)

    if top + bottom >= h:
        return sequence
    elif left + right >= w:
        return sequence
    sequence = [fr[top:h - down, left:w - right] for fr in sequence]

    return sequence


@SequenceModifiers.adder(
        'resize ratio',
        (str, 'outer', ['inner', 'outer'], 1, 'Type'),
        (int, 50, 10000, 1, "New Dimension"),
)
@sequence_adapter
@measure_real_time_decorator
def resize_ratio(sequence, res_typ='outer', new_dim=150):
    orig = sequence[0]
    h, w, c = orig.shape
    # print(f"Input resize: {h}, {w}")

    if h >= w and res_typ == 'outer':
        kwarg = {"height": new_dim}
    elif w >= h and res_typ == 'outer':
        kwarg = {"width": new_dim}
    elif h < w and res_typ == 'inner':
        kwarg = {"height": new_dim}
    else:
        kwarg = {"width": new_dim}

    # print(f"Resize: kwarg: {kwarg}")
    sequence = [imutils.resize(fr, **kwarg) for fr in sequence]

    return sequence


def run_thread(ang):
    path = os.path.abspath(f"func_test{os.path.sep}rita_{ang}.gif")

    sequence = [rita.copy() for r in range(50)]
    sequence = cycle_slide(sequence, ang)

    frames = [Image.fromarray(fr[:, :, [2, 1, 0]]) for fr in sequence]
    # print(frames)
    # with open(path, "wb")as fp:
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0, optimize=True)


@SequenceModifiers.adder(
        "rectangle",
        (float, -100, 100, 2, ["Center X offset", "Center Y offset"]),
        (float, 0, 50, 1, "Template size"),
        (int, 0, 255, 3, ["Red", "Green", "Blue"])
)
@sequence_adapter
@measure_real_time_decorator
def draw_rectangle(sequence, offset_start=None, window_fraction=0.1, color=None, alpha=0.6):
    if offset_start is None:
        offset_start = 0, 0
    if color is None:
        color = (0, 200, 150)

    h, w, c = sequence[0].shape

    point_start = (np.array([offset_start[0] / 200, offset_start[1] / 200], dtype=float) + 0.5)
    point_start = (point_start * (w, h)).round().astype(int)
    inner_size = min([h, w])
    window_size = np.round(1 + 2 * (window_fraction * inner_size)).astype(int)

    output = []

    for fr in sequence:
        fr = fr.copy()
        fr[point_start[1] - window_size:point_start[1] + window_size + 1,
        point_start[0] - window_size:point_start[0] + window_size + 1, :3] = color
        output.append(fr)
    return output


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


def get_move_clip_indexes(cur_offset, size):
    """
    Calculates indexes for pictures to move in axis
    Args:
        cur_offset
        size: axis dimension

    Returns:
        x1, x2, dst1, dst2

    """
    x1, x2 = _get_clip_src_indexes(cur_offset, size)
    dx1, dx2 = _get_clip_dst_indexes(cur_offset, size)
    return x1, x2, dx1, dx2


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


def convolve_pic(sequence, keep_margin, kernel, allowed_channels):
    mask_of_original_pixels = np.ones_like(sequence[0], dtype=bool)
    mask_of_original_pixels[keep_margin:-keep_margin, keep_margin:-keep_margin] = False

    picture_channels = sequence[0].shape[2]
    allowed_channels = [ch for ch in allowed_channels if ch <= picture_channels]
    if not allowed_channels:
        return sequence

    output = []
    for cur_frame in sequence:
        frame = cur_frame.copy()
        for ch_i in allowed_channels:
            channel = cur_frame[:, :, ch_i]
            new_ch = convolve2d(channel, kernel, 'same')
            new_ch = np.clip(new_ch.round(), 0, 255).astype(np.uint8)
            frame[:, :, ch_i] = new_ch

        frame[mask_of_original_pixels] = cur_frame[mask_of_original_pixels]
        output.append(frame)
    return output


def gauss_kernel(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


@SequenceModifiers.adder(
        "Mean filter",
        (int, 1, 100, 1, "Kernel radius"),
        (int, -1, 3, 1, "Channel"),
)
@sequence_adapter
@measure_real_time_decorator
def mean_filter(sequence, radius=1, channel_ind=0):
    size = 1 + 2 * radius
    # kernel = np.zeros((size, size)) / (size * size)
    # kernel = gauss_kernel(size)
    kernel = np.ones((size, size))
    kernel = kernel / kernel.sum()
    # print(f"Mean kernel: {kernel}")

    if channel_ind == 4:
        allowed_channels = [0, 1, 2, 3]
    else:
        allowed_channels = [channel_ind]

    output = convolve_pic(sequence, radius, kernel, allowed_channels)

    return output


@SequenceModifiers.adder(
        "Median filter",
        (int, 1, 100, 1, "Kernel radius"),
        (int, -1, 0, 1, "Channel")
)
@sequence_adapter
@measure_real_time_decorator
def median_filter(sequence, dist=1, channel=0):
    size = 1 + 2 * dist
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = 1 / size
    kernel[:, size // 2] = 1 / size
    kernel[size // 2, size // 2] = size
    kernel /= kernel.sum()

    if channel == -1:
        allowed_channels = [0, 1, 2, 3]
    else:
        allowed_channels = [channel]

    output = convolve_pic(sequence, dist, kernel, allowed_channels)

    return output


@SequenceModifiers.adder(
        'mask color',
        (int, 0, 255, 3, ['Red', 'Green', 'Blue']),
        (float, 1, 30000, 1, 'Distance threshold'),
        (float, 0.5, 10, 1, 'Distance exponent')
)
@sequence_adapter
@measure_real_time_decorator
def mask_color(sequence, color, max_dist=5, exponent=1):
    output = []
    for fr in sequence:
        if fr.shape[2] == 4:
            frame = fr.copy()
        else:
            full_alpha = np.ones((*fr.shape[:2], 1), dtype=np.uint8) * 255
            frame = np.concatenate([fr, full_alpha], axis=2, dtype=np.uint8)
            print(frame.shape)
        diff = np.abs(fr[:, :, :3] - color)
        diff = diff ** exponent
        error = diff.sum(axis=2)
        # print(error.shape)
        mask = error <= max_dist
        # print(f"maxdist {max_dist}, error: <{error.min():>3.2f}, {error.max():>3.2f}>")
        frame[mask, 3] = 0
        # frame = frame.astype(np.uint8)
        # print(frame.dtype)
        # mask = mask[:, :, np.newaxis]
        # frame = np.concatenate([mask, mask, mask], axis=2)

        output.append(frame)

    return output


@SequenceModifiers.adder(
        'Erode/Dilate',
        (bool, 0, 1, 1, "Dilate"),
        (int, 1, 15, 1, "How many times?"),
        (int, 1, 7, 1, "Kernel Radius"),
        (int, 0, 3, 1, "Channel"),
)
@sequence_adapter
@measure_real_time_decorator
def erode_dilate(sequence, dilate=False, repeat=1, radius=1, channel_ind=0):
    size = 2 * radius + 1
    half = radius
    kernel = np.zeros((size, size), dtype=np.uint8)
    kernel[half, :] = 1
    kernel[:, half] = 1

    out = [None] * len(sequence)

    for ind, img in enumerate(sequence):
        if len(img.shape) == 2:
            raise ValueError("Image in sequence is 2d. No alpha channel.")

        chan = img[:, :, channel_ind]

        if dilate:
            new_ch = cv2.dilate(chan, kernel, iterations=repeat)
        else:
            new_ch = cv2.erode(chan, kernel, iterations=repeat)

        new_ch = np.clip(new_ch.round(), 0, 255).astype(np.uint8)

        new_img = img.copy()
        new_img[:, :, channel_ind] = new_ch

        out[ind] = new_img

    return out


@sequence_adapter
@measure_real_time_decorator
def mask_area(sequence, pos, max_dist=5):
    return


@SequenceModifiers.adder(
        "snap point to location",
        (float, -100, 100, 2, ["Center X offset", "Center Y offset"]),
        (float, -100, 100, 2, ["Lock To X", "Lock To Y"]),
        (float, 0, 100, 1, "Frame [%] position"),
        (float, 0, 50, 1, "Template [%] size"),
        (int, 0, 100, 1, "Smoothing distance"),
        # (int, 0, 100, 1, "Keep value"),
        (bool, 0, 1, 1, "Debug")
)
@measure_real_time_decorator
def snap_point_to_location(sequence, offset_start=None, offset_end=None, start_fr=None,
                           window_fraction=None, smoothing_frames=0, debug=False):
    if offset_start is None:
        offset_start = 0, 0
    if offset_end is None:
        offset_end = 0, 0
    if start_fr is None:
        start_fr = 0
    if window_fraction is None:
        window_fraction = 10

    window_fraction /= 200

    h, w, c = sequence[0].shape
    point_end = (np.array([offset_end[0] / 200, offset_end[1] / 200], dtype=float) + 0.5)
    point_end = (point_end * (w, h)).round().astype(int)

    posx, posy, template, window_size = track_template_in_sequence(
            sequence, offset_start, start_fr, window_fraction,
            smoothing_frames,
    )

    if debug:
        output = []
        for fr, x, y in zip(sequence, posx, posy):
            fr = fr.copy()
            try:
                fr[y:y + window_size + 1, x:x + window_size + 1, ] = template
            except Exception:
                fr[y:y + window_size + 1, x:x + window_size + 1, ] = 0

            output.append(fr)

        return output

    output = []
    orig_blank = np.ones_like(sequence[0], dtype=np.uint8) * 0
    orig_blank[:, :, 3:] = 0

    for fr, x, y in zip(sequence, posx, posy):
        fr = fr.copy()
        # blank = np.ones_like(fr, dtype=np.uint8) * 255
        blank = orig_blank.copy()

        pt1 = np.array([x + window_size // 2, y + window_size // 2], dtype=int)
        # pt1 = np.array([x, y], dtype=int)

        offx, offy = point_end - pt1

        start_x, stop_x = _get_clip_src_indexes(offx, w)
        start_y, stop_y = _get_clip_src_indexes(offy, h)

        clip = fr[start_y:stop_y, start_x:stop_x, ]

        start_dest_x, stop_dest_x = _get_clip_dst_indexes(offx, w)
        start_dest_y, stop_dest_y = _get_clip_dst_indexes(offy, h)

        blank[start_dest_y:stop_dest_y, start_dest_x:stop_dest_x, ] = clip

        output.append(blank)

    return output


def track_template_in_sequence(
        sequence, offset_start, start_fr, window_fraction,
        smoothing_radius, smooth_exp=0,

):
    h, w, c = sequence[0].shape
    point_start = (np.array([offset_start[0] / 200, offset_start[1] / 200], dtype=float) + 0.5)
    point_start = (point_start * (w, h)).round().astype(int)
    inner_size = min([h, w])
    window_size = np.round(1 + 2 * (window_fraction * inner_size)).astype(int)
    sample_ind = np.floor(len(sequence) * start_fr / 100).astype(int)
    if sample_ind > len(sequence):
        sample_ind = len(sequence) - 1

    frame = sequence[sample_ind]

    template = frame[point_start[1] - window_size // 2:point_start[1] + window_size // 2 + 1,
               point_start[0] - window_size // 2:point_start[0] + window_size // 2 + 1]
    # plt.imshow(template)
    # plt.show()
    # output = []
    posx = []
    posy = []
    for fr in sequence:
        res = cv2.matchTemplate(fr, template, method=cv2.TM_SQDIFF)
        # res = cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX, -1)

        flat_ind = np.argmin(res)
        x, y = np.unravel_index(flat_ind, res.shape)

        posx.append(y)
        posy.append(x)

    if type(smoothing_radius) is int and smoothing_radius > 0:
        posx = moving_average(posx, radius=smoothing_radius, padding='try',
                              kernel_type='exp', kernel_exp=smooth_exp)
        posy = moving_average(posy, radius=smoothing_radius, padding='try',
                              kernel_type='exp', kernel_exp=smooth_exp)

        posx = np.array(posx).round().astype(int)
        posy = np.array(posy).round().astype(int)

    return posx, posy, template, window_size


@SequenceModifiers.adder(
        'squerify',
        (int, -100, 100, 2, ["X center offset", "Y center offset"]),
)
@sequence_adapter
@measure_real_time_decorator
def squerify(sequence, offset):
    pass


@SequenceModifiers.adder(
        'extend',
        (float, 0, 100, 2, ['Vertical increase', 'Horizontal increase']),
)
@sequence_adapter
@measure_real_time_decorator
def extend(sequence, increase):
    h, w, c = sequence[0].shape

    boost_y, boost_x = np.abs(increase)

    output = []
    new_h = int(round(h * (1 + boost_y / 100)))
    new_w = int(round(w * (1 + boost_x / 100)))
    blank = np.zeros((new_h, new_w, c), dtype=np.uint8)

    h_ind = (new_h - h) // 2
    w_ind = (new_w - w) // 2

    for fr in sequence:
        new_frame = blank.copy()
        new_frame[h_ind:h_ind + h, w_ind:w_ind + w] = fr

        output.append(new_frame)
    return output


@SequenceModifiers.adder('add transparency')
@sequence_adapter
@measure_real_time_decorator
def add_transparency(sequence):
    h, w, c = sequence[0].shape

    blank = np.zeros((h, w, 1), dtype=np.uint8) + 255
    output = []

    for fr in sequence:
        if fr.shape[2] == 3:
            new_frame = np.concatenate([fr, blank.copy()], axis=2)
            output.append(new_frame)
        else:
            output.append(fr)

    return output


@sequence_adapter
# @measure_time_decorator
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


def stack_channels_as_rgb(channels_list, labels, size=1.2):
    h, w, *_ = channels_list[0].shape

    rgb_list = []

    for ch, lb in zip(channels_list, labels):
        if len(ch.shape) == 2:
            ch = ch[:, :, np.newaxis]
            rgb = np.concatenate([ch, ch, ch], axis=2)

        elif ch.shape[2] == 4:
            rgb = ch[:, :, :3]

        elif ch.shape[2] == 1:
            rgb = np.concatenate([ch, ch, ch], axis=2)

        else:
            print("WHAT? What i missed?")
            print(ch.shape)

        rgb = rgb.astype(np.uint8)

        rgb = cv2.putText(
                rgb, lb, (5, 60),
                fontScale=size, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(50, 50, 0), thickness=8,
        )
        rgb = cv2.putText(
                rgb, lb, (5, 60),
                fontScale=size, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(50, 255, 0), thickness=3,
        )
        rgb_list.append(rgb)

    out = np.concatenate(rgb_list, axis=0).astype(np.uint8)
    return out


if __name__ == "__main__":
    rita = cv2.imread("../unknown.png", 1)
    aurora = cv2.imread("../aurora.png", cv2.IMREAD_UNCHANGED)

    blank = np.zeros_like(aurora)
    blank[:, :, 3] = 255

    aurora = mean_filter(aurora, channel_ind=3, radius=3, )

    out = blend_region(blank, aurora)
    cv2.imshow("Blend", out)
    cv2.waitKey()


    # cv2.imwrite(f"..{os.path.sep}test.png", aurora)
