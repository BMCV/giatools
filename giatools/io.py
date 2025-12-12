"""
Copyright 2017-2025 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import warnings

import numpy as np

from .backends import (
    backends,
    UnsupportedFileError,
)
from .typing import (
    Any,
    Dict,
    Optional,
    Tuple,
)
from .util import distance_to_external_frame


def _raise_unsupported_file_error(*args, **kwargs):
    args_str = ', '.join(repr(arg) for arg in args)
    kwargs_str = ', '.join(f'{key}={value!r}' for key, value in kwargs.items())
    suffix = ', '.join((args_str, kwargs_str))
    if suffix:
        suffix = f': {suffix}'
    raise UnsupportedFileError(f'No backend could read the image{suffix}')


def imreadraw(*args, position: int = 0, **kwargs) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """
    Wrapper for reading images, muting non-fatal errors.

    When using ``skimage.io.imread`` to read an image file, sometimes errors can be reported although the image file
    will be read successfully. In those cases, Galaxy might detect the errors on stdout or stderr, and assume that the
    tool has failed: https://docs.galaxyproject.org/en/latest/dev/schema.html#error-detection To prevent this, this
    wrapper around ``skimage.io.imread`` will mute all non-fatal errors.

    Different backends are tried in succession until one is successful:

    1. `tifffile`
    2. `ome_zarr`
    3. `skimage.io.imread`

    The `tifffile` backend is likely to fail if the file is not a TIFF file. The `ome_zarr` backend is likely to fail
    if the file is not an OME-Zarr file. The `skimage.io.imread` backend is able to read a wide variety of image
    formats, but for some formats it may not be able to extract all metadata, which is why it is less preferred.

    Some image files can store multiple images (e.g., multi-series TIFF files or multi-image OME-Zarr files). In these
    cases, the desired image can be selected by specifying the `position` parameter (default: `0`, the first image). An
    `IndexError` is raised if `position` is invalid. The :py:func:`peek_num_images_in_file` function can be used to
    determine the number of images in a file.

    Returns a tuple `(im_arr, axes, metadata)` where `im_arr` is the image data as a NumPy or Dask array, `axes` are
    the axes of the image, and `metadata` is any additional metadata. Minimal normalization is performed by treating
    sample axis ``S`` as an alias for the channel axis ``C``. For images which are read by the `skimage.io.imread`
    backend, single-channel and multi-channel 2-D images are supported, assuming ``YX`` axes layout for arrays with two
    axes and ``YXC`` for arrays with three axes, respectively.
    """

    for backend in backends:
        ret = backend.read(*args, position=position, **kwargs)
        if ret is not None:
            return ret

    # Raise an error if no backend could read the image
    _raise_unsupported_file_error(*args, **kwargs)


def peek_num_images_in_file(*args, **kwargs) -> int:
    """
    Peeks the number of images that can be loaded from a file.

    Example:

        .. runblock:: pycon

            >>> from giatools.io import peek_num_images_in_file
            >>> print(
            ...     'Images in multi-series TIFF:',
            ...     peek_num_images_in_file('data/input11.ome.tiff'),
            ... )
            >>> print(
            ...     'Images in single-series TIFF:',
            ...     peek_num_images_in_file('data/input1_uint8_yx.tiff'),
            ... )
            >>> print(
            ...     'Images in PNG file:',
            ...     peek_num_images_in_file('data/input4_uint8.png'),
            ... )
    """
    for backend in backends:
        ret = backend.peek_num_images_in_file(*args, **kwargs)
        if ret is not None:
            return ret

    # Raise an error if no backend could read the image
    _raise_unsupported_file_error(*args, **kwargs)


def imwrite(im_arr: np.ndarray, filepath: str, backend: str = 'auto', metadata: Optional[dict] = None):
    """
    Save an image to a file.
    """
    supported_backends = [backend for backend in backends if backend.writer_class is not None]
    supported_backend_names = [backend.name for backend in supported_backends]
    if backend != 'auto' and backend not in supported_backend_names:
        supported_backends_str = ', '.join((f'"{backend_name}"' for backend_name in supported_backend_names))
        raise ValueError(f'Unknown backend "{backend}". Use {supported_backends_str}, or "auto".')

    if filepath.lower().endswith('.tif'):
        warnings.warn(
            '.tif extension is deprecated, use .tiff instead.',
            DeprecationWarning,
            stacklevel=distance_to_external_frame(),
        )

    # Automatically select the proper backend
    if backend == 'auto':
        for backend in supported_backends:
            if any(filepath.lower().endswith(f'.{ext}') for ext in backend.writer_class.supported_extensions):
                break
        else:
            raise UnsupportedFileError(f'No backend found to write file: {filepath}')

    # Delegate to the selected backend
    backend.write(im_arr, filepath, metadata=metadata)
