"""
Copyright 2017-2025 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import json
import warnings

import numpy as np
import skimage.io

import giatools.util

from .typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Tuple,
)

try:
    import tifffile
except ImportError:
    tifffile = None


BackendType = Literal['auto', 'tifffile', 'skimage']


@giatools.util.silent
def imreadraw(*args, **kwargs) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """
    Wrapper for loading images, muting non-fatal errors.

    When using ``skimage.io.imread`` to read an image file, sometimes errors can be reported although the image file
    will be read successfully. In those cases, Galaxy might detect the errors on stdout or stderr, and assume that the
    tool has failed: https://docs.galaxyproject.org/en/latest/dev/schema.html#error-detection To prevent this, this
    wrapper around ``skimage.io.imread`` will mute all non-fatal errors.

    Image loading is first attempted using `tifffile` (if available, more reliable for loading TIFF files), and if
    that fails (e.g., because the file is not a TIFF file), falls back to ``skimage.io.imread``.

    Returns a tuple `(im_arr, axes, metadata)` where `im_arr` is the image data as a NumPy array, `axes` are the axes
    of the image, and `metadata` is any additional metadata. Normalization is performed, treating sample axis ``S`` as
    an alias for the channel axis ``C``. For images which cannot be read by `tifffile`, two- and three-dimensional data
    is supported. Two-dimensional images are assumed to be in ``YX`` axes order, and three-dimensional images are
    assumed to be in ``YXC`` axes order.
    """

    # First, try to read the image using `tifffile` (will only succeed if it is a TIFF file)
    if tifffile is not None:
        try:

            with tifffile.TiffFile(*args, **kwargs) as im_file:
                assert len(im_file.series) == 1, f'Image has unsupported number of series: {len(im_file.series)}'
                im_axes = im_file.series[0].axes.upper()

                # Verify that the image format is supported
                assert (
                    frozenset('YX') <= frozenset(im_axes) <= frozenset('QTZYXCS')
                ), f'Image has unsupported axes: {im_axes}'

                # Treat sample axis "S" as channel axis "C" and fail if both are present
                assert (
                    'C' not in im_axes or 'S' not in im_axes
                ), f'Image has sample and channel axes which is not supported: {im_axes}'
                im_axes = im_axes.replace('S', 'C')

                # Read the image data
                im_arr = im_file.asarray()

                # Read the metadata
                metadata = _get_tiff_metadata(im_file, im_file.series[0])

                # Return the image data, axes, and metadata
                return im_arr, im_axes, metadata

        except tifffile.TiffFileError:
            pass  # not a TIFF file

    # If the image is not a TIFF file, or `tifffile is not available`, fall back to `skimage.io.imread`
    im_arr = skimage.io.imread(*args, **kwargs)

    # Verify that the image format is supported
    assert im_arr.ndim in (2, 3), f'Image has unsupported dimension: {im_arr.ndim}'

    # Determine the axes
    if im_arr.ndim == 2:
        im_axes = 'YX'
    else:
        im_axes = 'YXC'

    # Return the image data and axes (no metadata)
    return im_arr, im_axes, dict()


def _get_tiff_metadata(tif: Any, series: Any) -> Dict[str, Any]:
    """
    Extract metadata from a `tifffile.TiffFile` object.
    """

    metadata: Dict[str, Any] = dict()

    # Extract pixel resolution, if available
    page0 = series.pages[0]
    if 'XResolution' in page0.tags and 'YResolution' in page0.tags:
        x_res = page0.tags['XResolution'].value
        y_res = page0.tags['YResolution'].value
        metadata['resolution'] = (
            x_res[0] / x_res[1],  # pixels per unit in X, numerator / denominator
            y_res[0] / y_res[1],  # pixels per unit in Y, numerator / denominator
        )

    # Read `ImageDescription` tag
    if 'ImageDescription' in page0.tags:
        description = page0.tags['ImageDescription'].value

        # Try to parse as JSON first
        try:
            description_json = json.loads(description)

            # Extract z-slice spacing, if available
            if 'spacing' in description_json:
                metadata['z_spacing'] = float(description_json['spacing'])

            # Extract unit, if available
            if 'unit' in description_json:
                metadata['unit'] = str(description_json['unit'])

        # If unsuccessful, fall back to line-by-line parsing (ImageJ-style)
        except json.JSONDecodeError:
            for line in description.splitlines():

                # Extract z-slice spacing, if available
                if line.startswith('spacing='):
                    try:
                        spacing = float(line.split('=')[1])
                        metadata['z_spacing'] = spacing
                    except ValueError:
                        pass

                # Extract unit, if available
                if line.startswith('unit='):
                    unit = line.split('=')[1]
                    if unit != 'pixel':
                        metadata['unit'] = unit

    # As a fallback, read unit from the dedicated tag, if available
    if 'unit' not in metadata and 'ResolutionUnit' in page0.tags:
        res_unit = page0.tags['ResolutionUnit'].value
        if res_unit == 2:
            metadata['unit'] = 'inch'
        elif res_unit == 3:
            metadata['unit'] = 'cm'

    # Normalize unit representation
    if metadata.get('unit', None) == r'\u00B5m':
        metadata['unit'] = 'um'

    return metadata


def imwrite(im_arr: np.ndarray, filepath: str, backend: BackendType = 'auto', metadata: Optional[dict] = None):
    """
    Save an image to a file using either `tifffile` or `skimage.io.imsave`.
    """

    if filepath.lower().endswith('.tif'):
        warnings.warn(
            '.tif extension is deprecated, use .tiff instead.',
            DeprecationWarning,
            stacklevel=giatools.util.distance_to_external_frame(),
        )

    # Automatically dispatch to the proper backend
    if backend == 'auto':
        if tifffile is not None and (filepath.lower().endswith('.tif') or filepath.lower().endswith('.tiff')):
            backend = 'tifffile'
        else:
            backend = 'skimage'

    # Dispatch via tifffile
    if backend == 'tifffile':

        # Create a copy of the metadata to avoid modifying the original
        metadata = dict(metadata) if metadata is not None else dict()

        # Update the metadata structure to what `tifffile` expects
        kwargs = dict(metadata=metadata)
        if 'resolution' in metadata:
            kwargs['resolution'] = metadata.pop('resolution')
        if 'z_spacing' in metadata:
            metadata['spacing'] = metadata.pop('z_spacing')

        # Write the image using tifffile
        tifffile.imwrite(filepath, im_arr, **kwargs)

    # Dispatch via skimage
    elif backend == 'skimage':
        skimage.io.imsave(filepath, im_arr, check_contrast=False)
    else:
        raise ValueError(f'Unknown backend: {backend}. Use "auto", "tifffile", or "skimage".')
