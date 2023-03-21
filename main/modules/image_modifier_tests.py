import pytest

from main.modules.modifiers_sequence import (
    cycle_slide, cycle_slide_delay,
    mono_color,
)
from main.modules.image_helpers import (
    _get_clip_dst_indexes, _get_clip_src_indexes, get_overlay_indexes,
    max_image_size,
)
from modifiers_image import *


rita = cv2.imread("../unknown.png", 1)
rita_t = rita


# rita_t = cv2.imread("unknown.png", 1)


@pytest.fixture
def fixture_rita():
    print("Running fixture!")
    return [rita.copy()]


@pytest.fixture
def fixture_rita_t():
    print("Running fixture2!")
    h, w, c = rita_t.shape
    rita = rita_t.reshape([w, h, c])
    return [rita]


def test_resize_1(fixture_rita):
    resize_ratio(fixture_rita, 'inner', 100)


def test_resize_2(fixture_rita):
    resize_ratio(fixture_rita, 'outer', 100)


def test_resize_3(fixture_rita_t):
    resize_ratio(fixture_rita_t, 'inner', 100)


def test_resize_4(fixture_rita_t):
    resize_ratio(fixture_rita_t, 'outer', 100)


testdata = [ang for ang in range(0, 360, 30)]


@pytest.mark.parametrize("ang", testdata)
def test_cycle_slide(ang):
    img = cv2.imread("../unknown.png", 1)
    rita = [img.copy() for i in range(30)]
    seq = cycle_slide(rita, ang)
    assert not (seq[0] != rita[0]).any(), "First frame does not match"
    assert not (seq[0] != rita[-1]).any(), "Last frame does not match"


data = [(ang, dl) for ang in range(0, 360, 30) for dl in [1, 2.5, 5, 10]]


@pytest.mark.parametrize("angle,delay", data)
def test_slide_delay(angle, delay):
    img = cv2.imread("../unknown.png", 1)
    source = [img.copy() for i in range(60)]
    seq = cycle_slide_delay(source, angle, delay)

    if type(seq) is list:
        assert seq[0].shape[2] == 4, "Must return alpha channel"
    elif len(seq.shape) == 3:
        assert seq.shape[2] == 4, "Must return alpha channel"
    else:
        assert seq.shape[3] == 4, "Must return alpha channel"

    assert not (seq[0][:, :, :3] != source[0][:, :, :3]).any(), "First frame does not match"
    assert not (seq[0][:, :, :3] != source[-1][:, :, :3]).any(), "Last frame does not match"


def test_indexes_1():
    size = 10
    a, b = _get_clip_src_indexes(0, size)
    assert a == 0
    assert b == 10


def test_indexes_dest_1():
    size = 10
    a, b = _get_clip_dst_indexes(0, size)
    assert a == 0
    assert b == 10


def test_indexes_2():
    size = 10
    a, b = _get_clip_src_indexes(-1, size)
    assert a == 1
    assert b == 10


def test_indexes_dest_2():
    size = 10
    a, b = _get_clip_dst_indexes(-1, size)
    assert a == 0
    assert b == 9


def test_indexes_dest_2b():
    size = 10
    a, b = _get_clip_dst_indexes(-2, size)
    assert a == 0
    assert b == 8


def test_indexes_3():
    size = 10
    a, b = _get_clip_src_indexes(1, size)
    assert a == 0
    assert b == 9


def test_indexes_dest_3():
    size = 10
    a, b = _get_clip_dst_indexes(1, size)
    assert a == 1
    assert b == 10


image_sources = [rita, [rita], rita[np.newaxis, :, :, :]]


@pytest.mark.parametrize("source", image_sources, ids=['pic', 'list', 'numpy 4D seq'])
def test_pic_video_input_1(source):
    res = mono_color(source, [0, 0, 0], 0)
    assert type(source) is type(res), "Must return same type"


@pytest.mark.parametrize("source", image_sources, ids=['pic', 'list', 'numpy 4D seq'])
def test_pic_video_input_2(source):
    res = color_blend(source, [0, 0, 0], 0)
    assert type(source) is type(res), "Must return same type"


@pytest.mark.parametrize("source", image_sources, ids=['pic', 'list', 'numpy 4D seq'])
def test_pic_video_input_3(source):
    res = crop_image(source, 0, 0, 0, 0)
    assert type(source) is type(res), "Must return same type"


@pytest.mark.parametrize("source", image_sources, ids=['pic', 'list', 'numpy 4D seq'])
def test_pic_video_input_4(source):
    res = max_image_size(source)
    assert type(source) is type(res), "Must return same type"


@pytest.mark.parametrize("source", image_sources, ids=['pic', 'list', 'numpy 4D seq'])
def test_pic_video_input_5(source):
    res = mean_filter(source, )
    assert type(source) is type(res), "Must return same type"


@pytest.mark.parametrize("source", image_sources, ids=['pic', 'list', 'numpy 4D seq'])
def test_pic_video_input_6(source):
    res = median_filter(source)
    assert type(source) is type(res), "Must return same type"


@pytest.mark.parametrize("source", image_sources, ids=['pic', 'list', 'numpy 4D seq'])
def test_pic_video_input_7(source):
    res = mask_color(source, (0, 0, 0), 20)
    assert type(source) is type(res), "Must return same type"

    if type(res) is list:
        assert res[0].shape[2] == 4, "Must return alpha channel"
    elif len(res.shape) == 3:
        assert res.shape[2] == 4, "Must return alpha channel"
    else:
        assert res.shape[3] == 4, "Must return alpha channel"


@pytest.mark.parametrize("source", image_sources, ids=['pic', 'list', 'numpy 4D seq'])
def test_pic_video_input_8_mask_area(source):
    res = mask_area(source, (0, 0, 0), 20)
    assert type(source) is type(res), "Must return same type"
    if type(res) is list:
        assert res[0].shape[2] == 4, "Must return alpha channel"
    elif len(res.shape) == 3:
        assert res.shape[2] == 4, "Must return alpha channel"
    else:
        assert res.shape[3] == 4, "Must return alpha channel"


@pytest.mark.parametrize("source", image_sources, ids=['pic', 'list', 'numpy 4D seq'])
def test_pic_video_input_9(source):
    res = add_transparency(source)
    assert type(source) is type(res), "Must return same type"

    if type(res) is list:
        assert res[0].shape[2] == 4, "Must return alpha channel"
    elif len(res.shape) == 3:
        assert res.shape[2] == 4, "Must return alpha channel"
    else:
        assert res.shape[3] == 4, "Must return alpha channel"


mod_keys = SequenceModifiers.keys


@pytest.mark.parametrize("key", mod_keys, ids=mod_keys)
def test_modifier_default_keys(key):
    SequenceModifiers.get_default(key)


"ARG: Size, Pos, OverSize, X1,X2, Ds1, Ds2"
data = [
        ([10, 2, 2], (0, 2, 2, 4), 'First Even'),
        ([10, 2, 3], (0, 3, 1, 4), 'First Odd'),
        ([10, 0, 2], (0, 2, 0, 2), 'Lower Clip Even'),
        ([10, 0, 3], (1, 3, 0, 2), 'Lower Clip Odd'),
        ([10, 9, 2], (0, 1, 9, 10), 'Upper Clip Even'),
        ([10, 9, 3], (0, 2, 8, 10), 'Upper Clip Odd'),
        ([10, 0, 1], (0, 1, 0, 1), 'Lower Clip Single'),
        ([10, 5, 1], (0, 1, 5, 6), 'Center Clip Single'),
        ([10, 9, 1], (0, 1, 9, 10), 'Upper Clip Single'),

        ([3, 1, 5], (1, 4, 0, 3), 'Bigger'),

]

ids = [a[2] for a in data]
data = [(a[0], a[1]) for a in data]


@pytest.mark.parametrize("test_case", data, ids=ids)
def test_overlay_indexes(test_case):
    arg, expected = test_case
    out = get_overlay_indexes(*arg)
    assert out == expected, out


# def test_overlay_indexes_2():
#     out = get_overlay_indexes(10, 0, 2)
#     assert out == (1, 2, 0, 1), out
#
#
# def test_overlay_indexes_3():
#     out = get_overlay_indexes(10, 0, 3)
#     assert out == (1, 3, 0, 2), out
#
#
# def test_overlay_indexes_4():
#     out = get_overlay_indexes(10, 1, 3)
#     assert out == (0, 3, 0, 3), out
#
#
# def test_overlay_indexes_5():
#     out = get_overlay_indexes(10, 2, 3)
#     assert out == (0, 3, 1, 4), out
#
#
# def test_overlay_indexes_6():
#     out = get_overlay_indexes(10, 2, 4)
#     assert out == (0, 4, 1, 5), out
#
#
# def test_overlay_indexes_7():
#     out = get_overlay_indexes(5, 1, 3)
#     assert out == (0, 3, 0, 3), out
#
#
# def test_overlay_indexes_8():
#     out = get_overlay_indexes(6, 3, 4)
#     "2,3,4,5"
#     assert out == (0, 4, 2, 6), out
