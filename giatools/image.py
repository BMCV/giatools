"""
Copyright 2017-2025 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import sys

import numpy as np

from . import (
    io,
    util,
)
from .typing import (
    Dict,
    Iterator,
    Optional,
    Self,
    Tuple,
    Union,
)

default_normalized_axes = 'QTZYXC'
"""
The default axes used for normalization in :meth:`Image.read`.
"""


class Image:
    """
    Represents an image (image pixel/voxel data and the corresponding axes metadata).
    """

    data: np.ndarray
    """
    The image data as a numpy array.
    """

    axes: str
    """
    The axes of the image data as a string.
    """

    original_axes: Optional[str]
    """
    The original axes of the image data as a string, if available. This is useful for keeping track of the original
    axes when normalizing or reordering axes.
    """

    metadata: Dict
    """
    Additional metadata of the image.

    The following metadata keys are covered by tests:

    - **resolution**, `Tuple[float, float]`: Pixels per unit in X and Y dimensions.
    - **z_spacing**, `float`: The pixel spacing in the Z dimension (if applicable).
    - **unit**, `str`: The unit of measurement (e.g., nn, um, mm, cm, m, km).
    """

    def __init__(
        self,
        data: np.ndarray,
        axes: str,
        original_axes: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        self.data = data
        self.axes = axes
        self.original_axes = original_axes
        self.metadata = dict() if metadata is None else metadata

    @staticmethod
    def read(*args, normalize_axes: str = default_normalized_axes, **kwargs) -> Self:
        """
        Read an image from file and normalize the image axes like `normalize_axes`.

        See :func:`giatools.io.imreadraw` for details how axes are determined and treated.
        """
        data, axes, metadata = io.imreadraw(*args, **kwargs)
        img = Image(data, axes, original_axes=axes, metadata=metadata)
        return img.normalize_axes_like(normalize_axes)

    def write(
        self,
        filepath: str,
        backend: io.BackendType = 'auto',
    ) -> Self:
        """
        Write the image to a file.
        """
        full_metadata = dict(axes=self.axes) | (self.metadata if self.metadata else dict())
        io.imwrite(self.data, filepath, backend=backend, metadata=full_metadata)
        return self

    def squeeze_like(self, axes: str) -> Self:
        """
        Squeeze the axes of the image to match the axes.

        This image is not changed in place, a new image is returned (without copying the data). The new image
        references the original metadata.

        Raises:
            ValueError: If one of the axis cannot be squeezed or `axes` is not a subset of the image axes.
        """

        if not (frozenset(axes) <= frozenset(self.axes)):
            raise ValueError(f'Cannot squeeze axes "{axes}" from image with axes "{self.axes}"')

        s = tuple(axis_pos for axis_pos, axis in enumerate(self.axes) if axis not in axes)
        squeezed_axes = util.str_without_positions(self.axes, s)
        squeezed_image = Image(
            data=self.data.squeeze(axis=s),
            axes=squeezed_axes,
            original_axes=self.original_axes,
            metadata=self.metadata,
        )
        return squeezed_image.reorder_axes_like(axes)

    def reorder_axes_like(self, axes: str) -> Self:
        """
        Reorder the axes of the image to match the given order.

        This image is not changed in place, a new image is returned (without copying the data). The new image
        references the original metadata.

        Raises:
            ValueError: If there are spurious, missing, or ambiguous axes.
        """
        if not (frozenset(axes) == frozenset(self.axes) and len(frozenset(axes)) == len(axes)):
            raise ValueError(f'Cannot reorder axes like "{axes}" of image with axes "{self.axes}"')

        reordered_data = self.data
        reordered_axes = self.axes
        for dst, axis in enumerate(axes):
            src = reordered_axes.index(axis)
            if src != dst:
                reordered_data = np.moveaxis(reordered_data, src, dst)
                reordered_axes = util.move_char(reordered_axes, src, dst)
        assert reordered_axes == axes, f'Failed to reorder axes "{self.axes}" to "{axes}", got "{reordered_axes}"'
        return Image(data=reordered_data, axes=axes, original_axes=self.original_axes, metadata=self.metadata)

    def normalize_axes_like(self, axes: str) -> Self:
        """
        Normalize the axes of the image.

        This image is not changed in place, a new image is returned (without copying the data). The new image
        references the original metadata.

        Raises:
            AssertionError: If `axes` is ambiguous.
            ValueError: If one of the axis cannot be squeezed.
        """
        if not (len(frozenset(axes)) == len(axes)):
            raise AssertionError(f'Axes "{axes}" is ambiguous')

        # Add missing axes
        complete_data = self.data
        complete_axes = self.axes
        for axis in axes:
            if axis not in self.axes:
                complete_data = complete_data[..., None]
                complete_axes += axis

        # Squeeze spurious axes and establish order
        return Image(
            data=complete_data,
            axes=complete_axes,
            original_axes=self.original_axes,
            metadata=self.metadata,
        ).squeeze_like(axes)

    def squeeze(self) -> Self:
        """
        Squeeze all singleton axes of the image.

        This image is not changed in place, a new image is returned (without copying the data). The new image
        references the original metadata.
        """
        squeezed_axes = ''.join(np.array(list(self.axes))[np.array(self.data.shape) > 1])
        return self.squeeze_like(squeezed_axes)

    def iterate_jointly(self, axes: str = 'YX') -> Iterator[Tuple[Tuple[Union[int, slice], ...], np.ndarray]]:
        """
        Iterates over all slices of the image along the given axes.

        This method yields tuples of slices and the corresponding image data. This method is useful for, for example,
        applying 2-D operations to all YX slices of a 3-D image or time series.

        .. note::

            This method requires **Python 3.11** or later.
        """
        if sys.version_info < (3, 11):
            raise RuntimeError('Image.iterate_jointly requires Python 3.11 or later')

        if len(axes) == 0 or not frozenset(axes).issubset(frozenset(self.axes)):
            raise ValueError(f'Cannot iterate jointly over axes "{axes}" of image with axes "{self.axes}"')

        # Prepare slicing
        ndindex, s_ = list(), list()
        for axis_idx, axis in enumerate(self.axes):
            if axis in axes:
                s_.append(None)
            else:
                s_.append(len(ndindex))
                ndindex.append(self.data.shape[axis_idx])

        # Iterate the given `axes` jointly
        for pos in np.ndindex(*ndindex):

            # Build slice
            sl = np.s_[*[(slice(None) if s is None else pos[s]) for s in s_]]  # not supported in Python <3.11

            # Extract array
            arr = self.data[sl]
            assert arr.ndim == len(axes)  # sanity check, should always be True

            # Yield slice and array
            yield sl, arr
