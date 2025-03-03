"""
Copyright 2017-2024 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import contextlib
import os
import warnings

import numpy as np
import skimage.util

from .typing import Iterable


def silent(func):
    """
    Decorator that mutes the standard error stream of the decorated function.
    """

    def wrapper(*args, **kwargs):
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stderr(fnull):
                return func(*args, **kwargs)

    return wrapper


def convert_image_to_format_of(image, format_image):
    """
    Convert the first image to the format of the second image.
    """

    # There is nothing to do with the image if the formats match.
    if format_image.dtype == image.dtype:
        return image

    # Convert the image to uint8 if this is the format of the second image.
    elif format_image.dtype == np.uint8:
        return skimage.util.img_as_ubyte(image)

    # Convert the image to uint16 if this is the format of the second image.
    elif format_image.dtype == np.uint16:
        return skimage.util.img_as_uint(image)

    # Convert the image to int16 if this is the format of the second image.
    elif format_image.dtype == np.int16:
        return skimage.util.img_as_int(image)

    # Convert the image to float32 if this is the format of the second image.
    elif format_image.dtype == np.float32:
        return skimage.util.img_as_float32(image)

    # Convert the image to float64 if this is the format of the second image.
    elif format_image.dtype == np.float64:
        return skimage.util.img_as_float64(image)

    # Other formats are not supported yet (e.g., float16).
    else:
        raise ValueError(f'Unsupported image data type: {format_image.dtype}')


def move_char(s: str, pos_src: int, pos_dst: int) -> str:
    s_list = list(s)
    c = s_list.pop(pos_src)
    if pos_dst < 0:
        pos_dst = len(s_list) + pos_dst + 1
    s_list.insert(pos_dst, c)
    return ''.join(s_list)


def swap_char(s: str, pos1: int, pos2: int) -> str:
    """
    Swaps the characters at positions `pos1` and `pos2` in the string `s`.

    .. deprecated:: 0.3.2
    """

    warnings.warn(
        'swap_char function is deprecated and will be removed in a future release.',
        DeprecationWarning,
        stacklevel=2
    )

    s_list = list(s)
    s_list[pos1], s_list[pos2] = s_list[pos2], s_list[pos1]
    return ''.join(s_list)


def str_without_positions(s: str, positions: Iterable[int]) -> str:
    """
    Returns the string `s` with the `characters` removed from it.
    """
    for pos in sorted(positions, reverse=True):
        s = s[:pos] + s[pos + 1:]
    return s
