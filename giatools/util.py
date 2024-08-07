"""
Copyright 2017-2024 Leonid Kostrykin, Biomedical Computer Vision Group, Heidelberg University.

Distributed under the MIT license.
See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
"""

import numpy as np
import skimage.util


def convert_image_to_format_of(image, format_image):
    """
    Convert the first image to the format of the second image.
    """

    # There is nothing to do with the image if the formats match.
    if format_image.dtype == image.dtype:
        return image

    # Convert the image to uint8 if this is the format of the second image.
    elif format_image.dtype == np.uint8:
        return skimage.util.img_as_ubyte(image)

    # Convert the image to uint16 if this is the format of the second image.
    elif format_image.dtype == np.uint16:
        return skimage.util.img_as_uint(image)

    # Convert the image to int16 if this is the format of the second image.
    elif format_image.dtype == np.int16:
        return skimage.util.img_as_int(image)

    # Convert the image to float32 if this is the format of the second image.
    elif format_image.dtype == np.float32:
        return skimage.util.img_as_float32(image)

    # Convert the image to float64 if this is the format of the second image.
    elif format_image.dtype == np.float64:
        return skimage.util.img_as_float64(image)

    # Other formats are not supported yet (e.g., float16).
    else:
        raise ValueError(f'Unsupported image data type: {format_image.dtype}')
