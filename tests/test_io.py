import unittest

import giatools.io


class imread(unittest.TestCase):

    def test_input1(self):
        img = giatools.io.imread('tests/data/input1.tif')
        self.assertEqual(img.mean(), 63.66848655158571)

    def test_input2(self):
        img = giatools.io.imread('tests/data/input2.tif')
        self.assertEqual(img.mean(), 9.543921821305842)

    def test_input3(self):
        """
        This is a multi-page TIFF, that sometimes fails to load properly with ``skimage.io.imread``, but works with
        ``tifffile``.

        For details see: https://github.com/BMCV/galaxy-image-analysis/pull/132#issuecomment-2371561435
        """
        img = giatools.io.imread('tests/data/input3.tif')
        self.assertEqual(img.shape, (5, 198, 356))
        self.assertEqual(img.mean(), 1259.6755334241288)
