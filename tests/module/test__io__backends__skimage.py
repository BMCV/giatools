"""
Module tests for the `giatools.io._backends.skimage` module.
"""

import unittest

import giatools.io
import giatools.io._backends.skimage

from ..tools import validate_metadata


class SKImageReader(unittest.TestCase):

    def test__valid__png(self):
        with giatools.io._backends.skimage.SKImageReader('tests/data/input4_uint8.png') as reader:
            self.assertEqual(reader.get_num_images(), 1)
            im = reader.select_image(0)
            self.assertEqual(reader.get_axes(im), 'YXC')
            validate_metadata(self, reader.get_image_metadata(im))
            arr = reader.get_image_data(im)
            self.assertEqual(arr.shape, (10, 10, 3))
            self.assertEqual(round(arr.mean(), 2), 130.04)

    def test__valid__tiff(self):
        with giatools.io._backends.skimage.SKImageReader('tests/data/input1_uint8_yx.tiff') as reader:
            self.assertEqual(reader.get_num_images(), 1)
            im = reader.select_image(0)
            self.assertEqual(reader.get_axes(im), 'YX')
            validate_metadata(self, reader.get_image_metadata(im))
            arr = reader.get_image_data(im)
            self.assertEqual(arr.shape, (265, 329))
            self.assertEqual(round(arr.mean(), 2), 63.67)

    def test__invalid(self):
        with giatools.io._backends.skimage.SKImageReader('tests/data/input7_uint8_zcyx.tif') as reader:
            self.assertEqual(reader.get_num_images(), 1)
            with self.assertRaises(giatools.io.UnsupportedFileError):
                reader.select_image(0)
