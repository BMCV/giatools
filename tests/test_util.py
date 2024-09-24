import itertools
import unittest

import numpy as np
import skimage.util

import giatools.util


class convert_image_to_format_of(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        np.random.seed(0)
        img_uint8 = (np.random.rand(16, 16) * 255).round().astype(np.uint8)
        self.testdata = [
            skimage.util.img_as_ubyte(img_uint8),    # uint8
            skimage.util.img_as_uint(img_uint8),     # uint16
            skimage.util.img_as_int(img_uint8),      # int16
            skimage.util.img_as_float32(img_uint8),  # float32
            skimage.util.img_as_float64(img_uint8),  # float64
        ]

    def test_self_conversion(self):
        for img in self.testdata:
            actual = giatools.util.convert_image_to_format_of(img, img)
            self.assertIs(actual, img)

    def test_cross_conversion(self):
        for src_img, dst_img in itertools.product(self.testdata, self.testdata):
            with self.subTest(f'{src_img.dtype} -> {dst_img.dtype}'):
                actual = giatools.util.convert_image_to_format_of(src_img, dst_img)
                self.assertEqual(actual.dtype, dst_img.dtype)
                self.assertTrue(np.allclose(actual, dst_img, rtol=1e-2))
