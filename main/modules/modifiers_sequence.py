import cv2
import numpy as np
from yasiu_native.time import measure_real_time_decorator

from modules.adapters import sequence_adapter
from modules.image_helpers import _get_clip_dst_indexes, _get_clip_src_indexes
from modules.math_functions import moving_average
from modules.collectors import SequenceModSingleton


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
    frames = [fr for _ in range(repeat+1) for fr in im_ob]

    return frames


@SequenceModifiers.adder(
    'reverse',
    (bool, 0, 1, 1, "Append reversed")
)
def reverse(sequence, append=True):
    new_sequence = [fr.copy() for fr in sequence]
    if append:
        out = new_sequence.copy()
    else:
        out = []

    for img in new_sequence[::-1]:
        out.append(img.copy())

    return out


@SequenceModifiers.adder(
    'slide cycle',
    # (str, 'all', ['all', 'linear'], 1, 'Mode'),
    # (int, 1, 99999, 1, 'N frames'),
    (float, -180, 360, 1, 'Direction')
)
def cycle_slide(sequence: list, angle: float):
    if angle < 0:
        angle += 360
    sequence = [fr.copy() for fr in sequence]
    height, width, c = sequence[0].shape
    seq_size = len(sequence)
    rads = np.pi * angle / 180

    sn = np.sin(rads)
    cs = np.cos(rads)

    end_compensation_A = 0
    end_compensation_B = 0
    is_horiz = False

    yend = abs(height * sn)
    xend = abs(width * cs)

    moded_angle = angle if angle <= 180 else angle-180
    # Reduce circle to half circ for single if statment check.
    if not (45 <= moded_angle < (90+45)):
        "Horiz movement is on edges"
        is_horiz = True

    if is_horiz:
        end_compensation_A = abs(np.round(height*sn).astype(int))
        if 180 < angle < 270 or 0 < angle < 90:
            end_compensation_B = end_compensation_A
            end_compensation_A = 0
    else:
        end_compensation_B = abs(np.round(width*cs).astype(int))
        if 0 < angle < 90 or 180 < angle < 270:
            end_compensation_A = end_compensation_B
            end_compensation_B = 0

    if 90 <= angle <= 270:
        "Left move"
        xsteps = np.linspace(xend, 0, seq_size)
    else:
        xsteps = np.linspace(0, xend, seq_size)

    if 0 <= angle <= 180:
        "Up move"
        ysteps = np.linspace(yend, 0, seq_size)
    else:
        ysteps = np.linspace(0, yend, seq_size)

    "Define full cycle move on axis"
    if is_horiz:
        if (90+45) <= angle < (180+45):
            # LEFT MOVE
            xsteps = np.linspace(0, width, seq_size)
        else:
            xsteps = np.linspace(width, 0, seq_size)
    else:
        if (0+45) <= angle < (90+45):
            ysteps = np.linspace(0, height, seq_size)
        else:
            ysteps = np.linspace(height, 0, seq_size)

    xsteps = np.round(xsteps).astype(int)
    ysteps = np.round(ysteps).astype(int)

    print(f"Angle: {angle}")
    # print(f"Moded angle: {moded_angle}")
    # print(f"Horizonal fix?: {is_horiz}")
    # print("Przesunięcia X:")
    # print(xsteps)
    # print("Przesunięcia Y:")
    # print(ysteps)
    print(f"End compensation A: {end_compensation_A}")
    print(f"End compensation B: {end_compensation_B}")

    for ind, (fr, x_off, y_off) in enumerate(zip(sequence, xsteps, ysteps)):
        "2D roll with next frame compensation"
        if is_horiz:
            if x_off <= 0:
                continue
            left = fr[:, x_off:]
            right = fr[:, :x_off]

            if y_off != 0:
                left = np.roll(left, y_off - end_compensation_A, axis=0)
                right = np.roll(right, y_off - end_compensation_B, axis=0)

            sequence[ind] = np.concatenate([left, right], axis=1)

        else:
            if y_off <= 0:
                continue
            top = fr[y_off:, :]
            bottom = fr[:y_off, :]

            if x_off != 0:
                top = np.roll(top, x_off - end_compensation_B, axis=1)
                bottom = np.roll(bottom, x_off - end_compensation_A, axis=1)

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
    (float, -180, 180, 1, "Starting hue"),
    (float, -100, 100, 1, "N cycles"),
    (float, 0, 1, 1, "Alpha"),
)
@sequence_adapter
def dynamic_hue(sequence, hue_offset, n_cycles, alpha):
    seq_size = len(sequence)
    output = [0] * seq_size

    n_cycles = np.clip(int(n_cycles), 1, 100, dtype=int)

    # hues = np.linspace(0, 180, size).round().astype(int)
    hues = []
    step_s = seq_size // n_cycles
    for i in range(n_cycles):
        if i == n_cycles - 1:
            prev_sum = step_s * i
            if prev_sum <= 0:
                missing = seq_size
            else:
                missing = seq_size - prev_sum
            row = np.linspace(0, 180, missing)
        else:
            row = np.linspace(0, 180, step_s)

        hues = np.concatenate([hues, row])

    hues = (np.array(hues) + int(hue_offset)) % 180

    colors = [0] * seq_size
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
    'scroll zoom',
    (float, 1, 100, 1, "Size [%]"),
    (int, 1, 10000, 1, "Duration frame"),
    (bool, 0, 1, 1, "Slide Vertical"),
    (bool, 0, 1, 1, "Repeat sequence"),
    (bool, 0, 1, 1, "Reverse direction")
)
@measure_real_time_decorator
def scroll(sequence,  fraction: float, frames: int, isVertical: bool, loopSequence: bool, reverseDir: bool):
    fraction = fraction / 100
    if frames < 1:
        frames = 1
    out = []

    if loopSequence:
        inds = np.arange(len(sequence), dtype=int)
        howMany = np.ceil(frames/len(sequence)).astype(int)
        inds = np.tile(inds, howMany)[:frames]
        assert len(inds) == frames, "Index list size must equal frames variable"
    else:
        inds = np.linspace(0, len(sequence)-1, frames, dtype=int)

    imH, imW, c = sequence[0].shape

    hfrac = np.round(imH*fraction).astype(int)
    wfrac = np.round(imW*fraction).astype(int)
    if hfrac < 1:
        hfrac = 1
    if wfrac < 1:
        wfrac = 1

    if isVertical:
        slide_dist = imH
    else:
        slide_dist = imW

    "Last slide is overlaping so removed"
    slide_interp = np.linspace(0, slide_dist, frames+1, dtype=int)[:-1]

    if reverseDir:
        slide_interp = -slide_interp
    roll_ax = 0 if isVertical else 1

    for i, slideVal, sampleI in zip(range(frames), slide_interp, inds):
        print(f"Scrolling frame i: {i}")
        fr = sequence[sampleI]
        fr = np.roll(fr, slideVal, roll_ax,)

        if isVertical:
            print(f"Vertical clip: {hfrac} of {imH}")
            fr = fr[:hfrac, :, :]
        else:
            fr = fr[:, :wfrac, :]

        out.append(fr)

    return out


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
    point_end = (
        np.array([offset_end[0] / 200, offset_end[1] / 200], dtype=float) + 0.5)
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
    point_start = (
        np.array([offset_start[0] / 200, offset_start[1] / 200], dtype=float) + 0.5)
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

        new_fr = cv2.addWeighted(first, alpha, second,
                                 1 - alpha, 0).round().astype(np.uint8)
        # new_fr =
        initial_seq.append(new_fr)

    merged_seq = initial_seq + main_part

    return merged_seq


# print("IMPORTED Sequence MODS:")
# print(SequenceModifiers)

if __name__ == "__main__":
    pass
