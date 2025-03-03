"""
Copyright 2017-2025 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import numpy as np
import skimage.io

import giatools.util

try:
    import tifffile
except ImportError:
    tifffile = None


@giatools.util.silent
def imread(*args, ret_axes: bool = False, **kwargs):
    """
    Wrapper for loading images, muting non-fatal errors, and normalizing the image axes like ``TZYXC``.

    When using ``skimage.io.imread`` to read an image file, sometimes errors can be reported although the image file
    will be read successfully. In those cases, Galaxy might detect the errors on stdout or stderr, and assume that the
    tool has failed: https://docs.galaxyproject.org/en/latest/dev/schema.html#error-detection To prevent this, this
    wrapper around ``skimage.io.imread`` will mute all non-fatal errors.

    Image loading is first attempted using `tifffile` (if available, more reliable for loading TIFF files), and if
    that fails (e.g., because the file is not a TIFF file), falls back to ``skimage.io.imread``.

    The image axes are normalized like ``TZYXC``, treating sample axis ``S`` as an alias for the channel axis ``C``.
    For images which cannot be read by `tifffile`, two- and three-dimensional data is supported. Two-dimensional images
    are assumed to be in ``YX`` axes order, and three-dimensional images are assumed to be in ``YXC`` axes order.

    Returns:
        The returned object depends on the value of the `ret_axes` parameter:
        - If `ret_axes` is `True`, a tuple `(im_arr, axes)` is returned, where `im_arr` is the image data as a
          five-dimensional NumPy array, and `axes` is the original axes of the image.
        - If `ret_axes` is `False`, only the image data as a five-dimensional NumPy array is returned.
    """

    # Helper function to return the result based on the value of `ret_axes`
    def result(im_arr: np.ndarray, axes: str, ret_axes: bool):
        if ret_axes:
            return im_arr, axes
        else:
            return im_arr

    # First, try to read the image using `tifffile` (will only succeed if it is a TIFF file)
    if tifffile is not None:
        try:

            with tifffile.TiffFile(*args, **kwargs) as im_file:
                assert len(im_file.series) == 1, f'Image has unsupported number of series: {len(im_file.series)}'
                original_axes = im_file.series[0].axes
                im_axes = original_axes

                # Verify that the image format is supported
                assert (
                    frozenset('YX') <= frozenset(im_axes) <= frozenset('TZYXCS')
                ), f'Image has unsupported axes: {im_axes}'

                # Treat sample axis "S" as channel axis "C" and fail if both are present
                assert (
                    'C' not in im_axes or 'S' not in im_axes
                ), f'Image has sample and channel axes which is not supported: {im_axes}'
                im_axes = im_axes.replace('S', 'C')

                # Read the image data
                im_arr = im_file.asarray()

                # Step 1. In the three steps below, the optional axes are added, of if they arent't there yet:

                # (1.1) Append "C" axis if not present yet
                if im_axes.find('C') == -1:
                    im_arr = im_arr[..., None]
                    im_axes += 'C'

                # (1.2) Append "Z" axis if not present yet
                if im_axes.find('Z') == -1:
                    im_arr = im_arr[..., None]
                    im_axes += 'Z'

                # (1.3) Append "T" axis if not present yet
                if im_axes.find('T') == -1:
                    im_arr = im_arr[..., None]
                    im_axes += 'T'

                # Step 2. All supported axes are there now. Normalize the order of the axes:

                # (2.1) Normalize order of axes "Y" and "X"
                ypos = im_axes.find('Y')
                xpos = im_axes.find('X')
                if ypos > xpos:
                    im_arr = im_arr.swapaxes(ypos, xpos)
                    im_axes = giatools.util.swap_char(im_axes, xpos, ypos)

                # (2.2) Normalize the position of the "C" axis (should be last)
                cpos = im_axes.find('C')
                if cpos < len(im_axes) - 1:
                    im_arr = np.moveaxis(im_arr, cpos, -1)
                    im_axes = giatools.util.move_char(im_axes, cpos, -1)

                # (2.3) Normalize the position of the "T" axis (should be first)
                tpos = im_axes.find('T')
                if tpos != 0:
                    im_arr = np.moveaxis(im_arr, tpos, 0)
                    im_axes = giatools.util.move_char(im_axes, tpos, 0)

                # (2.4) Normalize the position of the "Z" axis (should be second)
                zpos = im_axes.find('Z')
                if zpos != 1:
                    im_arr = np.moveaxis(im_arr, zpos, 1)
                    im_axes = giatools.util.move_char(im_axes, zpos, 1)

                # Verify that the normalizations were successful
                assert im_axes == 'TZYXC', f'Image axis normalization failed: {im_axes}'
                return result(im_arr, original_axes, ret_axes)

        except tifffile.TiffFileError:
            pass  # not a TIFF file

    # If the image is not a TIFF file, or `tifffile is not available`, fall back to `skimage.io.imread`
    im_arr = skimage.io.imread(*args, **kwargs)

    # Verify that the image format is supported
    assert im_arr.ndim in (2, 3), f"Image has unsupported dimension: {im_arr.ndim}"

    # Normalize the axes
    if im_arr.ndim == 2:  # Append "C" axis if not present yet
        im_arr = im_arr[..., None]
        original_axes = 'YX'
    else:
        original_axes = 'YXC'
    im_arr = im_arr[None, None, ...]  # Prepend "T" and "Z" axes

    # Verify that the normalizations were successful
    assert im_arr.ndim == 5, "Image axis normalization failed"
    return result(im_arr, original_axes, ret_axes)
