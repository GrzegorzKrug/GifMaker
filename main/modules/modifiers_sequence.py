import cv2
import numpy as np
from yasiu_native.time import measure_real_time_decorator

from main.modules.adapters import sequence_adapter
from main.modules.image_helpers import _get_clip_dst_indexes, _get_clip_src_indexes
from main.modules.math_functions import moving_average
from main.modules.collectors import SequenceModSingleton


SequenceModifiers = SequenceModSingleton()


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
        'overlap',
        (int, 0, 50, 1, "Overlap [%]")
)
@sequence_adapter
def overlap(sequence, overlap_fr):
    overlap_fr = np.clip(overlap_fr, 0, 50) / 100
    frames = (len(sequence) * overlap_fr).round().astype(int)
    if frames < 3:
        frames = 3

    left = sequence[:frames]
    right = sequence[-frames:]
    main_part = sequence[frames:-frames]

    # print(f"Orig len: {len(sequence)}")
    # print(f"Len left: {len(left)}, right: {len(right)}, main: {len(main_part)}")
    initial_seq = []
    # print(f"Overlaping frames: {frames}")

    for fr_num in range(frames):
        alpha = (fr_num / (frames - 1))
        first = left[fr_num]  # astype(float)
        second = right[fr_num]  # astype(float)
        # print(f"Alpha: {alpha}")
        # print(first.shape, second.shape)
        # print(1 - alpha)

        new_fr = cv2.addWeighted(first, alpha, second, 1 - alpha, 0).round().astype(np.uint8)
        # new_fr =
        initial_seq.append(new_fr)

    merged_seq = initial_seq + main_part

    return merged_seq


# print("IMPORTED Sequence MODS:")
# print(SequenceModifiers)

if __name__ == "__main__":
    pass
