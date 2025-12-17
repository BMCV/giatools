"""
Copyright 2017-2025 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import sys as _sys

import numpy as _np

from . import (
    metadata as _metadata,
    util as _util,
)
from .typing import (
    Iterator,
    NDArray,
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

    axes: str
    """
    The axes of the image data as a string.

    Example:

        .. runblock:: pycon

            >>> from giatools import Image
            >>> image = Image.read('data/input7_uint8_zcyx.tiff')
            >>> print(image.axes)
    """

    data: NDArray
    """
    The image data as a NumPy array or a Dask array.

    Example:

        .. runblock:: pycon

            >>> from giatools import Image
            >>> image = Image.read('data/input7_uint8_zcyx.tiff')
            >>> print(image.data.shape)
    """

    metadata: _metadata.Metadata
    """
    Additional metadata of the image.

    Example:

        .. runblock:: pycon

            >>> from giatools import Image
            >>> image = Image.read('data/input7_uint8_zcyx.tiff')
            >>> print(image.metadata.pixel_size)
            >>> print(image.metadata.z_spacing)
            >>> print(image.metadata.unit)
    """

    original_axes: Optional[str]
    """
    The original axes of the image data as a string, if available. This is useful for keeping track of the original
    axes when normalizing or reordering axes.

    Example:

        .. runblock:: pycon

            >>> from giatools import Image
            >>> image = Image.read('data/input7_uint8_zcyx.tiff')
            >>> print(image.original_axes)
    """

    def __init__(
        self,
        data: NDArray,
        axes: str,
        metadata: Optional[_metadata.Metadata] = None,
        original_axes: Optional[str] = None,
    ):
        self.data = data
        self.axes = axes
        self.original_axes = original_axes
        self.metadata = _metadata.Metadata() if metadata is None else metadata

    @staticmethod
    def read(filepath: str, *args, normalize_axes: Optional[str] = default_normalized_axes, **kwargs) -> Self:
        """
        Read an image from file and normalize the image axes like `normalize_axes`. Normalization will be (almost)
        skipped if `normalize_axes` is `None`.

        See :func:`giatools.io.imreadraw` for details how axes are determined and treated.
        """
        from .io import imreadraw
        data, axes, metadata = imreadraw(filepath, *args, **kwargs)
        img = Image(data, axes, original_axes=axes, metadata=_metadata.Metadata(**metadata))
        if normalize_axes is None:
            return img
        else:
            return img.normalize_axes_like(normalize_axes)

    def write(
        self,
        filepath: str,
        backend: str = 'auto',
    ) -> Self:
        """
        Write the image to a file.

        Raises:
            ValueError: If the number of axes does not match the number of data dimensions.
        """
        if len(self.axes) != len(self.data.shape):
            raise ValueError(
                f'Number of axes "{self.axes}" does not match number of data dimensions {self.data.shape}'
            )
        from .io import imwrite
        imwrite(self.data, filepath, axes=self.axes, metadata=self.metadata, backend=backend)
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
        squeezed_axes = _util.str_without_positions(self.axes, s)
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
                reordered_data = _np.moveaxis(reordered_data, src, dst)
                reordered_axes = _util.move_char(reordered_axes, src, dst)
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
        squeezed_axes = ''.join(_np.array(list(self.axes))[_np.array(self.data.shape) > 1])
        return self.squeeze_like(squeezed_axes)

    def iterate_jointly(self, axes: str = 'YX') -> Iterator[Tuple[Tuple[Union[int, slice], ...], NDArray]]:
        """
        Iterate over all slices of the image along the given axes.

        This method yields tuples of slices and the corresponding image data. This method is useful for, for example,
        applying 2-D operations to all YX slices of a 3-D image or time series.

        .. note::

            This method requires **Python 3.11** or later.
        """
        if _sys.version_info < (3, 11):
            raise RuntimeError('Image.iterate_jointly requires Python 3.11 or later')
        else:
            from . import image_py311
            return image_py311.iterate_jointly(self, axes)

    def is_isotropic(self, tol=1e-2) -> Optional[bool]:
        """
        Determine whether the image pixels/voxels are isotropic.

        Returns:
            `True` if the image is isotropic, `False` if not, or `None` if the resolution is not fully known.

        Example:

            .. runblock:: pycon

                >>> from giatools import Image
                >>> import numpy as np
                >>> image = Image(np.zeros((10, 20, 30)), axes='CYX')
                >>> print(image.is_isotropic())
                >>> image.metadata.pixel_size = (1.0, 1.1)
                >>> print(image.is_isotropic())
                >>> image.metadata.pixel_size = (1.0, 1.0)
                >>> print(image.is_isotropic())
                >>> image.axes = 'ZYX'
                >>> print(image.is_isotropic())
                >>> image.metadata.z_spacing = 1.0
                >>> print(image.is_isotropic())
        """
        if self.metadata.pixel_size is None or ('Z' in self.axes and self.metadata.z_spacing is None):
            return None

        # Determine the pixel/voxel size
        voxel_size = list(self.metadata.pixel_size)
        if 'Z' in self.axes:
            voxel_size += [self.metadata.z_spacing]

        # Check for isotropy
        anisotropy = max(voxel_size) / min(voxel_size)
        return anisotropy <= 1 + tol
