"""
Copyright 2017-2025 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import sys as _sys

import numpy as _np
import numpy.typing as _npt

from . import (
    metadata as _metadata,
    typing as _T,
    util as _util,
)

default_normalized_axes: str = 'QTZYXC'
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

        >>> from giatools import Image
        >>> image = Image.read('data/input7_uint8_zcyx.tiff')
        >>> print(image.axes)
        QTZYXC
    """

    data: _T.NDArray
    """
    The image data as a NumPy array or a Dask array.

    Example:

        >>> from giatools import Image
        >>> image = Image.read('data/input7_uint8_zcyx.tiff')
        >>> print(image.data.shape)
        (1, 1, 25, 50, 50, 2)
    """

    metadata: _metadata.Metadata
    """
    Additional metadata of the image.

    Example:

        >>> from giatools import Image
        >>> image = Image.read('data/input7_uint8_zcyx.tiff')
        >>> print(image.metadata.pixel_size)
        (0.43564006198286803, 0.43564006198286803)
        >>> print(image.metadata.z_spacing)
        0.05445500181716341
        >>> print(image.metadata.unit)
        um
    """

    original_axes: _T.Optional[str]
    """
    The original axes of the image data as a string, if available. This is useful for keeping track of the original
    axes when normalizing or reordering axes.

    Example:

        >>> from giatools import Image
        >>> image = Image.read('data/input7_uint8_zcyx.tiff')
        >>> print(image.original_axes)
        ZCYX
    """

    def __init__(
        self,
        data: _T.NDArray,
        axes: str,
        metadata: _T.Optional[_metadata.Metadata] = None,
        original_axes: _T.Optional[str] = None,
    ):
        self.data = data
        self.axes = axes
        self.original_axes = original_axes
        self.metadata = _metadata.Metadata() if metadata is None else metadata

    @property
    def original_shape(self) -> _T.Optional[_T.Tuple[int, ...]]:
        """
        The shape of the original image data, if `original_axes` is available.

        Example:

            >>> from giatools import Image
            >>> image = Image.read('data/input7_uint8_zcyx.tiff')
            >>> print(image.original_shape)
            (25, 2, 50, 50)
        """
        if self.original_axes is None:
            return None
        else:
            return tuple(
                [self.data.shape[self.axes.index(axis)] for axis in self.original_axes],
            )

    @property
    def shape(self) -> _T.Tuple[int, ...]:
        """
        The shape of the image data.

        This is a short-hand for `image.data.shape`, provided for consistency with the :py:attr:`original_shape`
        attribute.

        Example:

            >>> from giatools import Image
            >>> image = Image.read('data/input7_uint8_zcyx.tiff')
            >>> print(image.shape)
            (1, 1, 25, 50, 50, 2)
        """
        return self.data.shape

    @staticmethod
    def read(
        filepath: _T.PathLike,
        *args: _T.Any,
        normalize_axes: _T.Optional[str] = default_normalized_axes,
        **kwargs: _T.Any,
    ) -> _T.Self:
        """
        Read an image from file and normalize the image axes like `normalize_axes`. Normalization will be (almost)
        skipped if `normalize_axes` is `None`.

        See :func:`giatools.io.imreadraw` for details how axes are determined and treated.
        """
        from .io import imreadraw
        data, axes, metadata = imreadraw(filepath, *args, **kwargs)
        img = Image(data, axes, metadata, original_axes=axes)
        if normalize_axes is None:
            return img
        else:
            return img.normalize_axes_like(normalize_axes)

    def write(self, filepath: _T.PathLike, backend: str = 'auto') -> _T.Self:
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

    def squeeze_like(self, axes: str) -> _T.Self:
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

    def reorder_axes_like(self, axes: str) -> _T.Self:
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

    def normalize_axes_like(self, axes: str) -> _T.Self:
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

    def squeeze(self) -> _T.Self:
        """
        Squeeze all singleton axes of the image.

        This image is not changed in place, a new image is returned (without copying the data). The new image
        references the original metadata.
        """
        squeezed_axes = ''.join(_np.array(list(self.axes))[_np.array(self.data.shape) > 1])
        return self.squeeze_like(squeezed_axes)

    def iterate_jointly(
        self, axes: str = 'YX',
    ) -> _T.Iterator[_T.Tuple[_T.Tuple[_T.Union[int, slice], ...], _T.Self]]:
        """
        Iterate over all slices of the image along the given axes.

        This method yields tuples `(slice, section)` where `slice` is the slice in the source data array (this image)
        and `section` is the corresponding image section. This method is useful, for example, for applying 2-D
        operations to all slices of a 3-D image or time series. The order of axes in the yielded image `section`
        corresponds to the order in the ``axes`` parameter.

        .. note::

            This method requires **Python 3.11** or later.

        Example:

            >>> from giatools import Image
            >>> image = Image.read('data/input7_uint8_zcyx.tiff')
            >>> for _, section in image.iterate_jointly('XY'):
            ...     print(
            ...         section.data.shape,
            ...         section.axes,
            ...         section.original_axes,
            ...     )
            ...     break
            (50, 50) XY YX

        Raises:
            RuntimeError: If Python version is less than 3.11.
            ValueError: If `axes` contains invalid axes (must be a non-empty subset of the image axes).
        """
        if _sys.version_info < (3, 11):
            raise RuntimeError('Image.iterate_jointly requires Python 3.11 or later')
        else:
            from . import image_py311
            return image_py311.iterate_jointly(self, axes)

    def get_anisotropy(self, axes: _T.Optional[str] = None, eps: float = 1e-8) -> _T.Optional[_T.Tuple[float, ...]]:
        """
        Get the anisotropy of the image pixels/voxels.

        If `axes` is given, only the specified `axes` are considered for the anisotropy computation. Otherwise, all
        spatial axes are considered. The pixels/voxels of the image (along the specified `axes`) are isotropic if all
        returned anisotropy factors are (approximately) equal to 1.0.

        The `eps` parameter specifies a threshold for deciding whether a pixel/voxel size is too close to zero. If any
        pixel/voxel size along the considered axes has absolute value smaller than `eps` (or is `None`), the resolution
        is treated as unknown and the method returns `None`.

        Returns:
            A tuple of anisotropy factors for the specified axes (or the spatial axes of this image if `axes` is
            `None`), or `None` if the resolution is not fully known.

        Example:

            >>> from giatools import Image
            >>> import numpy as np
            >>> image = Image(np.zeros((10, 20, 30)), axes='CYX')
            >>> print(image.get_anisotropy())
            None
            >>>
            >>> image.metadata.pixel_size = (1.0, 1.2)
            >>> print(image.get_anisotropy())
            (1.0954451150103324, 0.9128709291752769)
            >>> print(image.get_anisotropy('XY'))
            (0.9128709291752769, 1.0954451150103324)
            >>> image.metadata.pixel_size = (1.0, 1.0)
            >>> print(image.get_anisotropy())
            (1.0, 1.0)
            >>>
            >>> image.axes = 'ZYX'
            >>> print(image.get_anisotropy())
            None
            >>> print(image.get_anisotropy(axes='YX'))
            (1.0, 1.0)
            >>> image.metadata.z_spacing = 1.0
            >>> print(image.get_anisotropy())
            (1.0, 1.0, 1.0)

        Scaling the pixel/voxel size of each axis by the reciprocal of the corresponding anisotropy factor yields the
        isotropic pixel/voxel size.

        Example:

            >>> from giatools import Image
            >>> import numpy as np
            >>> image = Image(np.zeros((10, 20, 30)), axes='CYX')
            >>> image.metadata.pixel_size = (1.0, 1.2)  # X, Y
            >>> image.metadata.z_spacing = 1.1
            >>>
            >>> anisotropy = image.get_anisotropy('XYZ')
            >>> print(
            ...     image.metadata.pixel_size[0] / anisotropy[0],  # X
            ...     image.metadata.pixel_size[1] / anisotropy[1],  # Y
            ...     image.metadata.z_spacing / anisotropy[2],
            ... )
            1.0969613104865237 1.0969613104865237 1.0969613104865237
        """
        if axes is None:
            axes = self.axes
        else:
            if (len(axes) < 2 or not (frozenset(axes) <= frozenset('XYZ')) or len(frozenset(axes)) != len(axes)):
                raise ValueError(
                    f'Invalid axes "{axes}", must contain at least two axes out of "X", "Y", and "Z", '
                    'and the axes must be unique',
                )

        # Determine the pixel/voxel size
        voxel_size = list()
        for axis in (axes or self.axes):
            if axis == 'X':
                if self.metadata.pixel_size is None:
                    return None  # unknown size
                else:
                    voxel_size.append(self.metadata.pixel_size[0])
            elif axis == 'Y':
                if self.metadata.pixel_size is None:
                    return None  # unknown size
                else:
                    voxel_size.append(self.metadata.pixel_size[1])
            elif axis == 'Z':
                if self.metadata.z_spacing is None:
                    return None  # unknown size
                else:
                    voxel_size.append(self.metadata.z_spacing)

        # Check for unknown size and compute anisotropy
        if any(abs(s) < eps for s in voxel_size):
            return None  # unknown size
        else:
            denom = pow(_np.prod(voxel_size), 1 / len(voxel_size))  # geometric mean
            return tuple(_np.divide(voxel_size, denom).tolist())

    def astype(
        self,
        dtype: _npt.DTypeLike,
        force_copy: bool = False,
        resolve_floating_to: _np.dtype = _np.float64,
        resolve_integer_to: _np.dtype = _np.int64,
        resolve_signedinteger_to: _np.dtype = _np.int64,
        resolve_unsignedinteger_to: _np.dtype = _np.uint64,
    ) -> _T.Self:
        """
        Cast the image to the specific pixel/voxel data type.

        This image is not changed in place, a new image is returned. With `force_copy=False` (the default), the image
        data is copied only if necessary. The metadata is not copied (the new image references the original metadata).

        The `dtype` parameter may be an inexact type such as `np.floating`, `np.integer`, `np.signedinteger`, or
        `np.unsignedinteger`. In this case, the actual data type is resolved according to the corresponding
        `resolve_*_to` parameter.

        Raises:
            ValueError: If conversion is not possible due to overflows.
        """

        # Special case: No conversion needed (same dtype or subset)
        if _np.issubdtype(self.data.dtype, dtype):
            if force_copy:
                new_data = self.data.copy()
            else:
                new_data = self.data  # no conversion needed

        # Special case: Conversion to `bool`
        elif dtype == bool:
            labels = _np.unique(self.data)
            if len(labels) > 2:
                raise ValueError(
                    f'Cannot convert image data from {self.data.dtype} to bool without overflows '
                    f'(more than two unique values, found {len(labels)}).'
                )
            new_data = _np.zeros(self.data.shape, dtype=bool)
            if len(labels) == 2:
                new_data[self.data == max(labels)] = True  # set the higher label to `True`
            elif len(labels) == 1 and labels[0] != 0:
                new_data[:] = True  # single non-zero label -> all `True`

        # General case
        else:
            # Resolve inexact `dtype` requirements
            if dtype == _np.floating:
                dtype = resolve_floating_to
            elif dtype == _np.integer:
                dtype = resolve_integer_to
            elif dtype == _np.signedinteger:
                dtype = resolve_signedinteger_to
            elif dtype == _np.unsignedinteger:
                dtype = resolve_unsignedinteger_to

            # Check for overflows
            src_min = self.data.min().item()  # convert to native Python type (int, float)
            src_max = self.data.max().item()  # convert to native Python type (int, float)
            if _np.issubdtype(dtype, _np.integer):
                dst_min = _np.iinfo(dtype).min
                dst_max = _np.iinfo(dtype).max
            else:
                dst_min = _np.finfo(dtype).min.item()  # convert to native Python type (float)
                dst_max = _np.finfo(dtype).max.item()  # convert to native Python type (float)
            info_dst = _np.iinfo(dtype) if _np.issubdtype(dtype, _np.integer) else _np.finfo(dtype)
            if src_min < dst_min or src_max > dst_max:
                raise ValueError(
                    f'Cannot convert image data from {self.data.dtype} to {dtype} without overflows '
                    f'(actual source range: [{src_min}, {src_max}], '
                    f'supported destination range: [{info_dst.min}, {info_dst.max}]).'
                )

            # With `copy=True`, the `data.astype` method always returns a newly allocated array.
            # With `copy=False`, it may also return the original array.
            new_data = self.data.astype(dtype, copy=force_copy)

        # Return new image
        return Image(
            data=new_data,
            axes=self.axes,
            original_axes=self.original_axes,
            metadata=self.metadata,
        )

    def clip_to_dtype(self, dtype: _np.dtype, force_copy: bool = False) -> _T.Self:
        """
        Clips the values of this image to the valid range of the specified `dtype`.

        This image is not changed in place. If the values of this image are already within the valid range of the
        target `dtype`, the image itself is returned (unless `force_copy` is `True`). Otherwise, a new image is
        returned that has the same data type as this image. The new image references the original metadata.

        Raises:
            TypeError: If `dtype` is `bool`.
        """
        if dtype in (bool, _np.bool_):
            raise TypeError('Clipping to boolean dtype is not supported.')

        # Determine the actual range of the source image
        min_src_value = self.data.min().item()  # convert to native Python type (float, int)
        max_src_value = self.data.max().item()  # convert to native Python type (float, int)

        # Determine the valid range for the target dtype
        if _np.issubdtype(dtype, _np.integer):
            min_dst_value = _np.iinfo(dtype).min
            max_dst_value = _np.iinfo(dtype).max
        else:
            min_dst_value = _np.finfo(dtype).min.item()  # convert to native Python type (float)
            max_dst_value = _np.finfo(dtype).max.item()  # convert to native Python type (float)

        # If the image is already within the valid range and `force_copy=False`, return it as is
        if not force_copy and (min_src_value >= min_dst_value and max_src_value <= max_dst_value):
            return self

        # Otherwise, clip the image to the valid range
        else:
            clipped_data = self.data.copy()
            clipped_data[clipped_data < min_dst_value] = min_dst_value
            clipped_data[clipped_data > max_dst_value] = max_dst_value
            return Image(
                data=clipped_data,
                axes=self.axes,
                original_axes=self.original_axes,
                metadata=self.metadata,
            )
