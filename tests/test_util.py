import itertools
import sys
import unittest

import numpy as np
import skimage.util

import giatools.util
import tests.tools


class silent(unittest.TestCase):

    def test_silent(self):
        @giatools.util.silent
        def func():
            print('Test', file=sys.stderr)
            raise ValueError('This is a test error message')
        with tests.tools.CaptureStderr() as stderr:
            with self.assertRaises(ValueError):
                func()
        self.assertEqual(str(stderr), '')


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


class move_char(unittest.TestCase):

    def test(self):
        self.assertEqual(giatools.util.move_char('ABC', 0, 1), 'BAC')
        self.assertEqual(giatools.util.move_char('ABC', 0, 2), 'BCA')
        self.assertEqual(giatools.util.move_char('ABC', 1, 0), 'BAC')
        self.assertEqual(giatools.util.move_char('ABC', 1, 2), 'ACB')
        self.assertEqual(giatools.util.move_char('ABC', 2, 0), 'CAB')
        self.assertEqual(giatools.util.move_char('ABC', 2, 1), 'ACB')
        self.assertEqual(giatools.util.move_char('ABC', 0, 0), 'ABC')
        self.assertEqual(giatools.util.move_char('ABC', 1, 1), 'ABC')
        self.assertEqual(giatools.util.move_char('ABC', 2, 2), 'ABC')
        self.assertEqual(giatools.util.move_char('ABC', 0, -1), 'BCA')
        self.assertEqual(giatools.util.move_char('ABC', 1, -1), 'ACB')
        self.assertEqual(giatools.util.move_char('ABC', 2, -1), 'ABC')
        self.assertEqual(giatools.util.move_char('ABC', 0, -2), 'BAC')
        self.assertEqual(giatools.util.move_char('ABC', 1, -2), 'ABC')
        self.assertEqual(giatools.util.move_char('ABC', 2, -2), 'ACB')
