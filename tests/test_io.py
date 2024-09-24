import unittest
import unittest.mock

import giatools.io

# This tests require that the `tifffile` package is installed.
assert giatools.io.tifffile is not None


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
        `tifffile`.

        For details see: https://github.com/BMCV/galaxy-image-analysis/pull/132#issuecomment-2371561435
        """
        img = giatools.io.imread('tests/data/input3.tif')
        self.assertEqual(img.shape, (5, 198, 356))
        self.assertEqual(img.mean(), 1259.6755334241288)

    def test_input4(self):
        """
        This is an RGB PNG file, that cannot be loaded with `tifffile`, but works with ``skimage.io.imread``.
        """
        img = giatools.io.imread('tests/data/input4.png')
        self.assertEqual(img.shape, (10, 10, 3))
        self.assertEqual(img.mean(), 130.04)

    @unittest.mock.patch('skimage.io.imread')
    @unittest.mock.patch('giatools.io.tifffile', None)
    def test_without_tifffile(self, mock_skimage_io_imread):
        """
        Test that loading an image without `tifffile` installed falls back to ``skimage.io.imread``.
        """
        img = giatools.io.imread('tests/data/input1.tif')
        mock_skimage_io_imread.assert_called_once_with('tests/data/input1.tif')
        self.assertIs(img, mock_skimage_io_imread.return_value)
