from .image_modifiers import (
    FunctionCollector, track_template_in_sequence, get_overlay_indexes,
    sequence_sampler, blend_region,

)

from .image_layer import Layer

import numpy as np
import cv2


PipeLineModifers = FunctionCollector()
PipeLineModifers.type_ = 'pipe'


@PipeLineModifers.adder(
        "merge to new",
        (int, 0, 100, 1, "Output layer key"),
        (int, 0, 100, 1, "Base layer key"),
        (int, 0, 100, 1, "Overlay layer key"),

)
# @sequence_adapter
def merge_to_new(output_frames, layer_dict, dst_layer_key, base_key, overlay_key):
    if dst_layer_key in layer_dict:
        dst_layer = layer_dict[dst_layer_key]
        assert dst_layer.pipeline_updates, "This layer must have pipe update flag!"
    else:
        layer_dict[dst_layer_key] = Layer(pipe_update=True)

    layer = layer_dict[dst_layer_key]
    layer.orig_frames = layer_dict[base_key].output_frames
    layer.apply_mods()


@PipeLineModifers.adder(
        "Snap pic to Tracked region",
        (int, 0, 100, 1, "Base layer key"),
        (int, 0, 100, 1, "Overlay layer key"),
        (int, 0, 100, 1, "Output layer key"),
        (float, -100, 100, 2, ["Sample X offset", "Sample Y offset"]),
        (float, -100, 100, 2, ["Target X offset", "Target Y offset"]),

        (float, 0, 100, 1, "Frame position"),
        (float, 0, 50, 1, "Template size"),
        (int, 0, 100, 1, "Smoothing distance"),
        (float, 0, 100, 1, "Smoothing kernel exponent"),
        (float, 0, 100, 1, "Overlay scale"),
        (bool, 0, 1, 1, "Debug")
)
def snap_to_tracked_region(output_frames, layer_dict,
                           base_layer_key, overlay_key, dst_layer_key,
                           offset_start, offset_end,

                           start_fr=None,
                           window_fraction=None,
                           smoothing_frames=None,
                           smoothing_val=1,
                           overlay_scale=1,
                           debug=False,
                           ):
    if dst_layer_key in layer_dict:
        dst_layer = layer_dict[dst_layer_key]
        assert dst_layer.pipeline_updates, "This layer must have pipe update flag!"
    else:
        layer_dict[dst_layer_key] = Layer(pipe_update=True)

    dst_layer = layer_dict[dst_layer_key]
    base = layer_dict[base_layer_key]
    overlay = layer_dict[overlay_key]

    h, w, _ = base.output_frames[0].shape
    offset_end = (np.array(offset_end) / 200 * (w, h)).round().astype(int)

    posx, posy, template, window_pixel_size = track_template_in_sequence(
            base.output_frames, offset_start, start_fr, window_fraction,
            smoothing_frames, smoothing_val
    )
    # snap_point_to_location(sequence, offset_start=None, offset_end=None, start_fr=None,

    over_frame = overlay.output_frames[0]

    base_h, base_w, *_ = base.output_frames[0].shape
    height, width, *_ = over_frame.shape

    seq = [fr.copy() for fr in base.output_frames]
    if len(overlay.output_frames) == 1:
        overlay_frames = [over_frame] * len(seq)
    else:
        overlay_frames = sequence_sampler(overlay.output_frames, mode='linear', frames_n=len(seq))

    if debug:
        height, width, *_ = template.shape
        for ind, (pic, x, y, over_frame) in enumerate(zip(seq, posx, posy, overlay_frames)):
            x1, x2, dx1, dx2 = get_overlay_indexes(base_w, x, width)
            y1, y2, dy1, dy2 = get_overlay_indexes(base_h, y, height)

            merged = pic.copy()
            merged[dy1:dy2, dx1:dx2] = [0, 0, 0, 255]

            # x1, x2, dx1, dx2 = get_overlay_indexes(base_w, x, width - 2)
            # y1, y2, dy1, dy2 = get_overlay_indexes(base_h, y, height 2)
            # print(template.shape, template[1:-1,1:-1].shape)
            merged[dy1 + 1:dy2 - 1, dx1 + 1:dx2 - 1] = template[y1 + 1:y2 - 1, x1 + 1:x2 - 1]
            seq[ind] = merged
    else:
        for ind, (pic, x, y, over_frame) in enumerate(zip(seq, posx, posy, overlay_frames)):
            x += offset_end[0]
            y += offset_end[1]

            x1, x2, dx1, dx2 = get_overlay_indexes(base_w, x, width)
            y1, y2, dy1, dy2 = get_overlay_indexes(base_h, y, height)

            clip = over_frame[y1:y2, x1: x2]
            b_clip = pic[dy1:dy2, dx1:dx2]

            merged = blend_region(b_clip, clip)

            seq[ind][dy1:dy2, dx1:dx2] = merged

    dst_layer.orig_frames = seq
    dst_layer.apply_mods()


@PipeLineModifers.adder(
        "merge to output",
        (int, 0, 100, 1, "Overlay layer key"),

)
# @sequence_adapter
def merge_to_output(output_frames, layer_dict, overlay_key):
    return output_frames


@PipeLineModifers.adder(
        "replace output",
        (int, 0, 100, 1, "New layer key"),

)
# @sequence_adapter
def replace_output(output_frames, layer_dict, overlay_key):
    return layer_dict[overlay_key].output_frames
