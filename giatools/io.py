"""
Copyright 2017-2025 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import skimage.io

import giatools.util

try:
    import tifffile
except ImportError:
    tifffile = None


@giatools.util.silent
def imreadraw(*args, **kwargs):
    """
    Wrapper for loading images, muting non-fatal errors.

    When using ``skimage.io.imread`` to read an image file, sometimes errors can be reported although the image file
    will be read successfully. In those cases, Galaxy might detect the errors on stdout or stderr, and assume that the
    tool has failed: https://docs.galaxyproject.org/en/latest/dev/schema.html#error-detection To prevent this, this
    wrapper around ``skimage.io.imread`` will mute all non-fatal errors.

    Image loading is first attempted using `tifffile` (if available, more reliable for loading TIFF files), and if
    that fails (e.g., because the file is not a TIFF file), falls back to ``skimage.io.imread``.

    Returns a tuple `(im_arr, axes)` where `im_arr` is the image data as a NumPy array, and `axes` are the axes of the
    image. Normalization is performed, treating sample axis ``S`` as an alias for the channel axis ``C``. For images
    which cannot be read by `tifffile`, two- and three-dimensional data is supported. Two-dimensional images are
    assumed to be in ``YX`` axes order, and three-dimensional images are assumed to be in ``YXC`` axes order.
    """

    # First, try to read the image using `tifffile` (will only succeed if it is a TIFF file)
    if tifffile is not None:
        try:

            with tifffile.TiffFile(*args, **kwargs) as im_file:
                assert len(im_file.series) == 1, f'Image has unsupported number of series: {len(im_file.series)}'
                im_axes = im_file.series[0].axes

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

                # Return the image data and axes
                return im_arr, im_axes

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

    # Return the image data and axes
    return im_arr, im_axes
