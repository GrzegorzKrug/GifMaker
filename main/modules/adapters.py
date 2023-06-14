from functools import wraps

import numpy as np


def sequence_adapter(func):
    """Converts inputs to sequence of images"""
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


def image_adapter(func):
    """Splits input to image ob"""
    @wraps(wrapped=func)
    def wrapper(source, *a, **kw):
        if isinstance(source, np.ndarray) and len(source.shape) == 4:
            "4D numpy"
            res_list = [func(img, *a, **kw) for img in source]
            res_list = [img.reshape(1, *img.shape) for img in res_list]
            res = np.concatenate(res_list, axis=0)

        elif isinstance(source, list):
            "List of pictures"
            res = [func(img, *a, **kw) for img in source]
        else:
            "Picture"
            res = func(source, *a, **kw)
        return res

    return wrapper