import unittest

import numpy as np

import giatools.io

from ..tools import (
    minimum_python_version,
    random_io_test,
)


class peek_num_images_in_file(unittest.TestCase):

    @random_io_test(shape=(10, 10, 3), dtype=np.uint8, ext='not-an-image')
    def test__invalid_file(self, filepath, data):
        with open(filepath, 'w') as f:
            f.write(str(data))
        with self.assertRaisesRegex(giatools.io.UnsupportedFileError, f'No backend could read {filepath}'):
            giatools.io.peek_num_images_in_file(filepath)

    def test__tiff_multiseries(self):
        num_images = giatools.io.peek_num_images_in_file('tests/data/input11.ome.tiff')
        self.assertEqual(num_images, 6)

    def test__tiff_single_series(self):
        num_images = giatools.io.peek_num_images_in_file('tests/data/input1_uint8_yx.tiff')
        self.assertEqual(num_images, 1)

    def test__png(self):
        num_images = giatools.io.peek_num_images_in_file('tests/data/input4_uint8.png')
        self.assertEqual(num_images, 1)

    @minimum_python_version(3, 11)
    def test__omezarr(self):
        num_images = giatools.io.peek_num_images_in_file('tests/data/input12.zarr')
        self.assertEqual(num_images, 1)
