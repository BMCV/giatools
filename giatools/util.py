"""
Copyright 2017-2025 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import contextlib as _contextlib
import functools as _functools
import inspect as _inspect
import os as _os

import numpy as _np
import skimage.util as _skimage_util

from . import typing as _typing


def silent(func):
    """
    Decorator that mutes the standard error stream of the decorated function.
    """

    @_functools.wraps(func)  # propagate function signature so Sphinx can handle it
    def wrapper(*args, **kwargs):
        with open(_os.devnull, 'w') as fnull:
            with _contextlib.redirect_stderr(fnull):
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
    elif format_image.dtype == _np.uint8:
        return _skimage_util.img_as_ubyte(image)

    # Convert the image to uint16 if this is the format of the second image.
    elif format_image.dtype == _np.uint16:
        return _skimage_util.img_as_uint(image)

    # Convert the image to int16 if this is the format of the second image.
    elif format_image.dtype == _np.int16:
        return _skimage_util.img_as_int(image)

    # Convert the image to float32 if this is the format of the second image.
    elif format_image.dtype == _np.float32:
        return _skimage_util.img_as_float32(image)

    # Convert the image to float64 if this is the format of the second image.
    elif format_image.dtype == _np.float64:
        return _skimage_util.img_as_float64(image)

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


def str_without_positions(s: str, positions: _typing.Iterable[int]) -> str:
    """
    Returns the string `s` with the `characters` removed from it.
    """
    for pos in sorted(positions, reverse=True):
        s = s[:pos] + s[pos + 1:]
    return s


def distance_to_external_frame():
    """
    Returns the number of stack levels until the first frame of the user's code.
    """
    for depth, frame_info in enumerate(_inspect.stack()[1:], start=1):
        frame = frame_info.frame
        module = _inspect.getmodule(frame)
        module_name = module.__name__ if module else None

        if module_name.split('.')[0] != 'giatools':
            return depth

    return None
