"""
Copyright 2017-2025 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import warnings as _warnings

from .. import (
    metadata as _metadata,
    typing as _T,
    util as _util,
)
from ._backends.skimage import (
    SKImageReader as _SKImageReader,
    SKImageWriter as _SKImageWriter,
)
from ._backends.tiff import (
    TiffReader as _TiffReader,
    TiffWriter as _TiffWriter,
)
from .backend import (
    Backend,
    CorruptFileError,
    IncompatibleDataError,
    UnsupportedFileError,
)

try:
    from ._backends.omezarr import (
        OMEZarrReader as _OMEZarrReader,
        OMEZarrWriter as _OMEZarrWriter,
    )
except ImportError:
    _OMEZarrReader = None  # type: ignore
    _OMEZarrWriter = None  # type: ignore

__all__ = [
    'backends',
    'CorruptFileError',
    'imreadraw',
    'imwrite',
    'IncompatibleDataError',
    'peek_num_images_in_file',
    'UnsupportedFileError',
]


#: List of the supported backends for reading and writing image files.
#:
#: For reading, the backends are tried in succession until one is successful.
#:
#: The `tifffile` backend is likely to fail if the file is not a TIFF file. The `ome_zarr` backend is likely to fail if
#: the file is not an OME-Zarr file. The `skimage.io.imread` backend is able to read a wide variety of image formats,
#: but for some formats it may not be able to extract all metadata, which is why it is less preferred.
#:
#: For writing, the appropriate backend is selected based on the file extension.
#:
#:     .. note::
#:
#:         The `omezarr` backend is only available on **Python 3.11** or later.
backends = [
    Backend('tifffile', _TiffReader, _TiffWriter),
] + (
    [
        Backend('omezarr', _OMEZarrReader, _OMEZarrWriter),
    ]
    if _OMEZarrReader is not None else []
) + [
    Backend('skimage', _SKImageReader, _SKImageWriter),
]


def _raise_unsupported_file_error(filepath: str, *args: _T.Any, **kwargs: _T.Any):
    args_str = ', '.join(repr(arg) for arg in args)
    kwargs_str = ', '.join(f'{key}={value!r}' for key, value in kwargs.items())
    details = ', '.join(filter(lambda s: len(s), (args_str, kwargs_str)))
    if details:
        details = f' with {details}'
    raise UnsupportedFileError(filepath, f'No backend could read {filepath}{details}')


def imreadraw(
    filepath: _T.PathLike, *args: _T.Any, position: int = 0, **kwargs: _T.Any,
) -> _T.Tuple[_T.NDArray, str, _metadata.Metadata]:
    """
    Wrapper for reading images, muting non-fatal errors.

    The backends defined in :py:data:`backends` are tried in succession until one is successful.

    Some image files can store multiple images (e.g., multi-series TIFF files or multi-image OME-Zarr files). In these
    cases, the desired image can be selected by specifying the `position` parameter (default: `0`, the first image). An
    `IndexError` is raised if `position` is invalid. The :py:func:`peek_num_images_in_file` function can be used to
    determine the number of images in a file.

    Returns a tuple `(data, axes, metadata)` where `data` is the image data as a NumPy or Dask array, `axes` are the
    axes of the image, and `metadata` is any additional metadata. Minimal normalization is performed by treating sample
    axis ``S`` as an alias for the channel axis ``C``. For images which are read by the `skimage.io.imread` backend,
    single-channel and multi-channel 2-D images are supported, assuming ``YX`` axes layout for arrays with two axes and
    ``YXC`` for arrays with three axes, respectively.

    Raises:
        CorruptFileError:
            If the image cannot be read by the designated backend due to corruption or an unsupported format flavor.
        FileNotFoundError:
            If the specified file does not exist.
        UnsupportedFileError:
            If no backend could read the image.
    """
    filepath = str(filepath)

    # Try each backend in succession
    for backend in backends:
        ret = backend.read(filepath, *args, position=position, **kwargs)
        if ret is not None:
            return ret

    # Raise an error if no backend could read the image
    _raise_unsupported_file_error(filepath, *args, **kwargs)


def peek_num_images_in_file(filepath: _T.PathLike, *args: _T.Any, **kwargs: _T.Any) -> int:
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

    Raises:
        FileNotFoundError:
            If the specified file does not exist.
        UnsupportedFileError:
            If no backend could read the image.
    """
    filepath = str(filepath)

    # Try each backend in succession
    for backend in backends:
        ret = backend.peek_num_images_in_file(filepath, *args, **kwargs)
        if ret is not None:
            return ret

    # Raise an error if no backend could read the image
    _raise_unsupported_file_error(filepath, *args, **kwargs)


def _select_writing_backend(filepath: str, backend_name: str) -> Backend:
    """
    Select an appropriate backend for writing the file.
    """

    # Validate the backend name
    supported_backends = [backend for backend in backends if backend.writer_class is not None]
    supported_backend_names = [backend.name for backend in supported_backends]
    if backend_name != 'auto' and backend_name not in supported_backend_names:
        supported_backends_str = ', '.join((f'"{backend_name}"' for backend_name in supported_backend_names))
        raise ValueError(f'Unknown backend "{backend_name}". Use {supported_backends_str}, or "auto".')

    # Automatically select the proper backend
    if backend_name == 'auto':
        for backend in supported_backends:
            if any(filepath.lower().endswith(f'.{ext}') for ext in backend.writer_class.supported_extensions):
                return backend

        # No backend was returned (block above), so no backend was found
        raise UnsupportedFileError(filepath, f'No backend found to write file: {filepath}')

    # Select the backend based on the given name
    else:
        return next((backend for backend in supported_backends if backend.name == backend_name))


def imwrite(
    data: _T.NDArray,
    filepath: _T.PathLike,
    axes: str,
    metadata: _metadata.Metadata,
    backend: str = 'auto',
    **kwargs: _T.Any,
):
    """
    Save an image to a file.

    Raises:
        IncompatibleDataError:
            If the image data or metadata is incompatible with the file format (inferred from the suffix of the file).
        UnsupportedFileError:
            If no backend is available to write the file format (inferred from the suffix of the file).
        ValueError:
            If `backend` is not ``"auto"`` and the specified backend is not available, or if the image data or metadata
            are invalid (e.g., `None`, invalid axes or dimensions).
    """
    filepath = str(filepath)
    if filepath.lower().endswith('.tif'):
        _warnings.warn(
            '.tif extension is deprecated, use .tiff instead.',
            DeprecationWarning,
            stacklevel=_util.distance_to_external_frame(),
        )
    _select_writing_backend(
        filepath,
        backend,
    ).write(data, filepath, axes=axes, metadata=metadata, **kwargs)
