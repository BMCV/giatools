"""
Copyright 2017-2025 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import numpy as _np

from . import (
    image as _image,
    typing as _T,
)


def iterate_jointly(
    img: _image.Image,
    axes: str,
) -> _T.Iterator[_T.Tuple[_T.Tuple[_T.Union[int, slice], ...], _T.NDArray]]:

    if len(axes) == 0 or not frozenset(axes).issubset(frozenset(img.axes)):
        raise ValueError(f'Cannot iterate jointly over axes "{axes}" of image with axes "{img.axes}"')

    # Prepare slicing
    ndindex, s_ = list(), list()
    for axis_idx, axis in enumerate(img.axes):
        if axis in axes:
            s_.append(None)
        else:
            s_.append(len(ndindex))
            ndindex.append(img.data.shape[axis_idx])

    # Iterate the given `axes` jointly
    for pos in _np.ndindex(*ndindex):

        # Build slice
        sl = _np.s_[*[(slice(None) if s is None else pos[s]) for s in s_]]  # not supported in Python <3.11

        # Extract array
        arr = img.data[sl]
        assert arr.ndim == len(axes)  # sanity check, should always be True

        # Yield slice and array
        yield sl, arr
