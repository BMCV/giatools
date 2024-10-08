import unittest
import unittest.mock

import numpy as np

import giatools.io

# This tests require that the `tifffile` package is installed.
assert giatools.io.tifffile is not None


class imread__with_tifffile(unittest.TestCase):

    def setUp(self):
        # Verify that the `tifffile` package is installed
        assert giatools.io.tifffile is not None

    def test__input1(self):
        img = giatools.io.imread('tests/data/input1_uint8_yx.tif')
        self.assertEqual(img.mean(), 63.66848655158571)
        self.assertEqual(img.shape, (1, 1, 265, 329, 1))

    def test__input2(self):
        img = giatools.io.imread('tests/data/input2_uint8_yx.tif')
        self.assertEqual(img.mean(), 9.543921821305842)
        self.assertEqual(img.shape, (1, 1, 96, 97, 1))

    def test__input3(self):
        """
        Test a multi-page TIFF, that sometimes fails to load properly with ``skimage.io.imread``, but works with
        `tifffile`.

        For details see: https://github.com/BMCV/galaxy-image-analysis/pull/132#issuecomment-2371561435
        """
        img = giatools.io.imread('tests/data/input3_uint16_zyx.tif')
        self.assertEqual(img.shape, (1, 5, 198, 356, 1))
        self.assertEqual(img.mean(), 1259.6755334241288)

    def test__input4(self):
        """
        Test an RGB PNG file, that cannot be loaded with `tifffile`, but works with ``skimage.io.imread``.
        """
        img = giatools.io.imread('tests/data/input4_uint8.png')
        self.assertEqual(img.shape, (1, 1, 10, 10, 3))
        self.assertEqual(img.mean(), 130.04)

    def test__input5(self):
        """
        Test TIFF file with ``CYX`` axes.
        """
        img = giatools.io.imread('tests/data/input5_uint8_cyx.tif')
        self.assertEqual(img.shape, (1, 1, 8, 16, 2))
        self.assertEqual(img.mean(), 22.25390625)

    def test__input6(self):
        """
        Test TIFF file with ``ZYX`` axes.
        """
        img = giatools.io.imread('tests/data/input6_uint8_zyx.tif')
        self.assertEqual(img.shape, (1, 25, 8, 16, 1))
        self.assertEqual(img.mean(), 26.555)

    def test__input7(self):
        """
        Test TIFF file with ``ZCYX`` axes.
        """
        img = giatools.io.imread('tests/data/input7_uint8_zcyx.tif')
        self.assertEqual(img.shape, (1, 25, 50, 50, 2))
        self.assertEqual(img.mean(), 14.182152)

    def test__input8(self):
        """
        Test TIFF file with ``TYX`` axes.
        """
        img = giatools.io.imread('tests/data/input8_uint16_tyx.tif')
        self.assertEqual(img.shape, (5, 1, 49, 56, 1))
        self.assertEqual(img.mean(), 5815.486880466472)


@unittest.mock.patch('skimage.io.imread')
@unittest.mock.patch('giatools.io.tifffile', None)
class imread__without_tifffile(unittest.TestCase):
    """
    Test loading an image without `tifffile` installed.
    """

    def test__yx(self, mock_skimage_io_imread):
        """
        Test fallback to ``skimage.io.imread`` with a two-dimensional image.
        """
        np.random.seed(0)
        mock_skimage_io_imread.return_value = np.random.rand(5, 5)
        img = giatools.io.imread('tests/data/input1.tif')
        mock_skimage_io_imread.assert_called_once_with('tests/data/input1.tif')
        np.testing.assert_array_equal(img, mock_skimage_io_imread.return_value[None, None, :, :, None])

    def test__yxc(self, mock_skimage_io_imread):
        """
        Test fallback to ``skimage.io.imread`` with a three-dimensional image.
        """
        np.random.seed(0)
        mock_skimage_io_imread.return_value = np.random.rand(5, 5, 3)
        img = giatools.io.imread('tests/data/input1.tif')
        mock_skimage_io_imread.assert_called_once_with('tests/data/input1.tif')
        np.testing.assert_array_equal(img, mock_skimage_io_imread.return_value[None, None, ...])

    def test__unsupported_dimensions(self, mock_skimage_io_imread):
        """
        Test fallback to ``skimage.io.imread`` and failure if the image has more than 3 dimensions.
        """
        np.random.seed(0)
        mock_skimage_io_imread.return_value = np.random.rand(1, 1, 5, 5, 3)
        with self.assertRaises(AssertionError):
            giatools.io.imread('tests/data/input1.tif')
        mock_skimage_io_imread.assert_called_once_with('tests/data/input1.tif')
