import cv2
import numpy as np

from functools import wraps
from scipy.signal import convolve2d
from scipy import stats as st

import imutils

from modules.adapters import image_adapter, sequence_adapter
from modules.collectors import SequenceModSingleton
from modules.image_helpers import _get_clip_dst_indexes, _get_clip_src_indexes, blend_region

from yasiu_native.time import measure_real_time_decorator
from yasiu_image.modifiers import squerify
from yasiu_image.filters import mirrorAxis


def layer_validator(fn):
    @wraps(wrapped=fn)
    def wrapper(layer_dict, destination_key, *a, **kw):
        return fn(layer_dict, destination_key, *a, **kw)

    return wrapper


SequenceModifiers = SequenceModSingleton()

"=== IMPORTED WRAPS === "

mirrorAxis = image_adapter(mirrorAxis)
SequenceModifiers.adder(
    "mirror",
    (bool, 0, 1, 1, 'Vertical Mirror'),
    (float, 0, 1, 1, 'Position'),
    (bool, 0, 1, 1, 'Flip'),
    (None, True, 0.5,  False)
)(mirrorAxis)

"=== NEW FUNCTIONS ==="


@SequenceModifiers.adder(
    'Alpha cutoff',
    (float, 0, 255, 1, "Threshold"),
    (None, 50)
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
    'color blend',
    (int, 0, 255, 3, ["Red", "Green", "Blue"]),
    (float, 0, 1, 1, "Alpha"),
    (None, [140, 0, 10], 0.7)
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
    top, down, left, right = np.array(
        [top, bottom, left, right], dtype=float) / 100
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
    'resize',
    (float, 0.01, 15, 1, 'Scale Width'),
    (float, 0.01, 15, 1, 'Scale Height'),
    (None, 0.8, 0.8)
)
@sequence_adapter
@measure_real_time_decorator
def resize(sequence, scale_x=1.0, scale_y=1.0):
    orig = sequence[0]
    h, w, c = orig.shape
    new_y = np.round(h * scale_y).astype(int)
    new_x = np.round(w * scale_x).astype(int)

    if new_x < 1:
        new_x = 1
    if new_y < 1:
        new_y = 1
    # print(f"Input resize: {h}, {w}")

    # if h >= w and res_typ == 'outer':
    #     kwarg = {"height": new_dim}
    # elif w >= h and res_typ == 'outer':
    #     kwarg = {"width": new_dim}
    # elif h < w and res_typ == 'inner':
    #     kwarg = {"height": new_dim}
    # else:
    #     kwarg = {"width": new_dim}

    # print(f"Resize: kwarg: {kwarg}")
    # sequence = [cv2.resize(fr, **kwarg) for fr in sequence]
    sequence = [cv2.resize(img, (new_x, new_y)) for img in sequence]

    return sequence


@SequenceModifiers.adder(
    'resize ratio',
    (str, 'Outer', ['Inner', 'Outer'], 1, 'Type'),
    (int, 50, 10000, 1, "New Dimension"),
    (None, "Outer", 300)
)
@sequence_adapter
@measure_real_time_decorator
def resize_ratio(sequence, res_typ='Outer', new_dim=150):
    orig = sequence[0]
    h, w, c = orig.shape
    # print(f"Input resize: {h}, {w}")

    if h >= w and res_typ == 'Outer':
        kwarg = {"height": new_dim}
    elif w >= h and res_typ == 'Outer':
        kwarg = {"width": new_dim}
    elif h < w and res_typ == 'Inner':
        kwarg = {"height": new_dim}
    else:
        kwarg = {"width": new_dim}

    # print(f"Resize: kwarg: {kwarg}")
    sequence = [imutils.resize(fr, **kwarg) for fr in sequence]

    return sequence


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

    point_start = (
        np.array([offset_start[0] / 200, offset_start[1] / 200], dtype=float) + 0.5)
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


def convolve_pic(sequence, keep_margin, kernel, allowed_channels):
    mask_of_original_pixels = np.ones_like(sequence[0], dtype=bool)
    mask_of_original_pixels[keep_margin:-
                            keep_margin, keep_margin:-keep_margin] = False

    picture_channels = sequence[0].shape[2]
    allowed_channels = [
        ch for ch in allowed_channels if ch <= picture_channels]
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
    (None, 2, 0)
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
    (int, -1, 0, 1, "Channel"),
    (None, 2, 0)
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
    (float, 0.5, 10, 1, 'Distance exponent'),
    (None, [255, 255, 255], 10000, 3)
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


@SequenceModifiers.adder(
    'extend',
    (int, -500, 500, 2, ['Vertical pixel', 'Horizontal pixel']),
    (None, [10, 10])
)
@sequence_adapter
@measure_real_time_decorator
def extend(sequence, increase: tuple[float, float]):
    h, w, c = sequence[0].shape

    offset_y, offset_x = np.abs(increase)
    offset_x = int(offset_x)
    offset_y = int(offset_y)

    output = []
    new_h = h + int(abs(offset_y))
    new_w = w + int(abs(offset_x))

    blank = np.zeros((new_h, new_w, c), dtype=np.uint8)

    # h_ind = (new_h - h) // 2
    # w_ind = (new_w - w) // 2
    h_off = 0 if offset_x > 0 else abs(offset_x)
    w_off = 0 if abs(offset_y) < 0 else offset_y

    for fr in sequence:
        new_frame = blank.copy()
        new_frame[h_off:h_off + h, w_off:w_off + w] = fr
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


@SequenceModifiers.adder(
    'squerify',
    (int, -100, 100, 1, "Center offset [%]")
)
@image_adapter
def squerify_interace(image, val):
    val /= 100.0
    new_img = squerify(image, val)
    return new_img


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
