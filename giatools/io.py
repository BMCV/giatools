"""
Copyright 2017-2024 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

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
def imread(*args, **kwargs):
    """
    Wrapper for loading images which mutes non-fatal errors.

    When using ``skimage.io.imread`` to read an image file, sometimes errors can be reported although the image file
    will be read successfully. In those cases, Galaxy might detect the errors on stdout or stderr, and assume that the
    tool has failed: https://docs.galaxyproject.org/en/latest/dev/schema.html#error-detection To prevent this, this
    wrapper around ``skimage.io.imread`` will mute all non-fatal errors.

    Image loading is first attempted using ``tifffile`` (if available, more reliable for loading TIFF files), and if
    that fails (e.g., because the file is not a TIFF file), falls back to ``skimage.io.imread``.
    """

    # First, try to read the image using `tifffile` (will only succeed if it is a TIFF file)
    if tifffile is not None:
        try:
            return tifffile.imread(*args, **kwargs)
        except tifffile.TiffFileError:
            pass  # not a TIFF file

    # If the image is not a TIFF file, or `tifffile is not available`, fall back to `skimage.io.imread`
    return skimage.io.imread(*args, **kwargs)
