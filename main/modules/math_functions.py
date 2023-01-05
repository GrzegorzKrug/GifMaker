import numpy as np
from scipy.signal import convolve


def moving_average(array, radius=1, padding="try", kernel_type="avg", kernel_exp=2):
    """
    Description

    :param array: int

    :param radius:
    radius
     radius of kernel, `kernel size = 2 * radius + 1`

    :param padding: str
    :type padding: str

    Padding - str
     full - extends results to padded numbers

     same - calculates average for every number, uses 0 vals

     valid- returns only average number that have full frame of values

     try - calculates number using every value it has without padding

     keep - keep raw values that can not be calculated

    :param kernel_type:
    kernel_type: str
        avg - all values have equal weight

        linear - linear weights, same as exp type, with `exponent weight = 1`
         example weights before norm:
          [1, 2, 3, 2, 1]

        exp - raising weight to given power in `kernel_exp` variable
         example weights before norm with kernel_exp = 2:
          [1, 4, 9, 4, 1]

    :param kernel_exp:
    kernel_exp - float, must be in range <0, inf>
     non linearity exponent before normalization
     only for `kernel_type='exp'`
     kernel_value ^ exponent

    :return:

    """
    smoothing_frames = 1 + 2 * abs(radius)
    if smoothing_frames > len(array):
        return array

    if kernel_type == "avg":
        kernel = np.ones(smoothing_frames) / smoothing_frames

    elif kernel_type == "linear":
        kernel = np.arange(radius + 1) + 1
        kernel = np.concatenate([kernel[:-1], np.flip(kernel)])
        kernel = kernel / kernel.sum()

    else:
        "Exponential"
        kernel = np.arange(radius + 1) + 1
        kernel = np.concatenate([kernel[:-1], np.flip(kernel)])
        kernel = kernel ** kernel_exp
        kernel = kernel / kernel.sum()

    if padding == "same":
        out = convolve(array, kernel, 'same')

    elif padding == "try" or padding == 'keep':
        out = convolve(array, kernel, 'valid')

        if padding == "try":
            left, right = convolve_array_edges(array, radius)
            # a, b, c = len(left), len(out), len(right)
            # print(radius,a + b + c, a, b, c)
            # print(f"\tKernel: {kernel.shape}")

        else:
            left = array[:radius]
            right = array[-radius:]
            # print(len(left), len(out), len(right))

        out = np.concatenate([left, out, right])

    elif padding == "full":
        out = convolve(array, kernel, 'full')

    else:
        "valid"
        out = convolve(array, kernel, 'valid')

    # posx[:smoothing_frames] = tempx[:smoothing_frames]
    return out


def convolve_array_edges(arr, radius):
    """
    Convolve array using available numbers

    Args:
        arr:
        radius:

    Returns
     left - convoluted list of left side

     right - convoluted list of right side

    """
    # size = 1 + 2 * abs(radius)

    left = np.zeros(radius)
    right = np.zeros(radius)

    if radius > 0:
        csum = 0
        for i, num in enumerate(arr[:radius * 2]):
            csum += num
            if i >= radius:
                left[i - radius] = csum / (i + 1)

        csum = 0
        for i, num in enumerate(arr[:-radius * 2 - 1:-1]):
            csum += num
            if i >= radius:
                right[i - radius] = csum / (i + 1)

    return left, right
