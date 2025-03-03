"""
Copyright 2017-2025 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

from typing import (
    Optional,
    Self,
)

import numpy as np

from . import io
from . import util


class Image:
    """
    Represents an image (image pixel/voxel data and the corresponding axes metadata).
    """

    def __init__(self, data: np.ndarray, axes: str, original_axes: Optional[str] = None):
        self.data = data
        self.axes = axes
        self.original_axes = original_axes

    @staticmethod
    def read(*args, normalize_axes: str = 'TZYXC', **kwargs) -> Self:
        """
        Read an image from file and normalize the image axes like `normalize_axes`.

        See :func:`giatools.io.imreadraw` for details how axes are determined and treated.
        """
        data, axes = io.imreadraw(*args, **kwargs)
        img = Image(data, axes, original_axes=axes)
        return img.normalize_axes_like(normalize_axes)

    def squeeze_like(self, axes: str) -> Self:
        """
        Squeeze the axes of the image to match the axes.

        Raises:
            ValueError: If one of the axis cannot be squeezed.
            AssertionError: If `axes` is not a subset of the image axes.
        """
        assert (
            frozenset(axes) <= frozenset(self.axes)
        ), f'Cannot squeeze axes "{axes}" from image with axes "{self.axes}"'
        s = tuple(axis_pos for axis_pos, axis in enumerate(self.axes) if axis not in axes)
        squeezed_axes = util.str_without_positions(self.axes, s)
        squeezed_image = Image(data=self.data.squeeze(axis=s), axes=squeezed_axes, original_axes=self.original_axes)
        return squeezed_image.reorder_axes_like(axes)

    def reorder_axes_like(self, axes: str) -> Self:
        """
        Reorder the axes of the image to match the given order.
        """
        assert (
            frozenset(axes) == frozenset(self.axes) and len(frozenset(axes)) == len(axes)
        ), f'Cannot reorder axes like "{axes}" of image with axes "{self.axes}"'
        reordered_data = self.data
        reordered_axes = self.axes
        for dst, axis in enumerate(axes):
            src = reordered_axes.index(axis)
            if src != dst:
                reordered_data = np.moveaxis(reordered_data, src, dst)
                reordered_axes = util.move_char(reordered_axes, src, dst)
        assert reordered_axes == axes, f'Failed to reorder axes "{self.axes}" to "{axes}", got "{reordered_axes}"'
        return Image(data=reordered_data, axes=axes, original_axes=self.original_axes)

    def normalize_axes_like(self, axes: str) -> Self:
        """
        Normalize the axes of the image.

        Raises:
            AssertionError: If `axes` is ambiguous.
            ValueError: If one of the axis cannot be squeezed.
        """
        assert len(frozenset(axes)) == len(axes), f'Axes "{axes}" is ambiguous'

        # Add missing axes
        complete_data = self.data
        complete_axes = self.axes
        for axis in axes:
            if axis not in self.axes:
                complete_data = complete_data[..., None]
                complete_axes += axis

        # Squeeze spurious axes and establish order
        return Image(data=complete_data, axes=complete_axes, original_axes=self.original_axes).squeeze_like(axes)
