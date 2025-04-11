import unittest
import unittest.mock

import numpy as np

import giatools.io

# This tests require that the `tifffile` package is installed.
assert giatools.io.tifffile is not None


class imreadraw__with_tifffile(unittest.TestCase):

    def setUp(self):
        # Verify that the `tifffile` package is installed
        assert giatools.io.tifffile is not None

    def test__input1(self):
        img, axes = giatools.io.imreadraw('tests/data/input1_uint8_yx.tif')
        self.assertEqual(img.mean(), 63.66848655158571)
        self.assertEqual(img.shape, (265, 329))
        self.assertEqual(axes, 'YX')

    def test__input2(self):
        img, axes = giatools.io.imreadraw('tests/data/input2_uint8_yx.tif')
        self.assertEqual(img.mean(), 9.543921821305842)
        self.assertEqual(img.shape, (96, 97))
        self.assertEqual(axes, 'YX')

    def test__input3(self):
        """
        Test a multi-page TIFF, that sometimes fails to load properly with ``skimage.io.imread``, but works with
        `tifffile`.

        For details see: https://github.com/BMCV/galaxy-image-analysis/pull/132#issuecomment-2371561435
        """
        img, axes = giatools.io.imreadraw('tests/data/input3_uint16_zyx.tif')
        self.assertEqual(img.shape, (5, 198, 356))
        self.assertEqual(img.mean(), 1259.6755334241288)
        self.assertEqual(axes, 'ZYX')

    def test__input4(self):
        """
        Test an RGB PNG file, that cannot be loaded with `tifffile`, but works with ``skimage.io.imread``.
        """
        img, axes = giatools.io.imreadraw('tests/data/input4_uint8.png')
        self.assertEqual(img.shape, (10, 10, 3))
        self.assertEqual(img.mean(), 130.04)
        self.assertEqual(axes, 'YXC')

    def test__input5(self):
        """
        Test TIFF file with ``CYX`` axes.
        """
        img, axes = giatools.io.imreadraw('tests/data/input5_uint8_cyx.tif')
        self.assertEqual(img.shape, (2, 8, 16))
        self.assertEqual(img.mean(), 22.25390625)
        self.assertEqual(axes, 'CYX')

    def test__input6(self):
        """
        Test TIFF file with ``ZYX`` axes.
        """
        img, axes = giatools.io.imreadraw('tests/data/input6_uint8_zyx.tif')
        self.assertEqual(img.shape, (25, 8, 16))
        self.assertEqual(img.mean(), 26.555)
        self.assertEqual(axes, 'ZYX')

    def test__input7(self):
        """
        Test TIFF file with ``ZCYX`` axes.
        """
        img, axes = giatools.io.imreadraw('tests/data/input7_uint8_zcyx.tif')
        self.assertEqual(img.shape, (25, 2, 50, 50))
        self.assertEqual(img.mean(), 14.182152)
        self.assertEqual(axes, 'ZCYX')

    def test__input8(self):
        """
        Test TIFF file with ``TYX`` axes.
        """
        img, axes = giatools.io.imreadraw('tests/data/input8_uint16_tyx.tif')
        self.assertEqual(img.shape, (5, 49, 56))
        self.assertEqual(img.mean(), 5815.486880466472)
        self.assertEqual(axes, 'TYX')

    def test__input9(self):
        """
        Test TIFF file with ``QYX`` axes.
        """
        img, axes = giatools.io.imreadraw('tests/data/input9_qyx.tif')
        self.assertEqual(img.shape, (2, 256, 256))
        self.assertEqual(img.mean(), 0.05388291)
        self.assertEqual(axes, 'QYX')


@unittest.mock.patch('skimage.io.imread')
@unittest.mock.patch('giatools.io.tifffile', None)
class imreadraw__without_tifffile(unittest.TestCase):
    """
    Test loading an image without `tifffile` installed.
    """

    def test__yx(self, mock_skimage_io_imread):
        """
        Test fallback to ``skimage.io.imread`` with a two-dimensional image.
        """
        np.random.seed(0)
        mock_skimage_io_imread.return_value = np.random.rand(5, 5)
        img, axis = giatools.io.imreadraw('tests/data/input1.tif')
        mock_skimage_io_imread.assert_called_once_with('tests/data/input1.tif')
        np.testing.assert_array_equal(img, mock_skimage_io_imread.return_value)
        self.assertEqual(axis, 'YX')

    def test__yxc(self, mock_skimage_io_imread):
        """
        Test fallback to ``skimage.io.imread`` with a three-dimensional image.
        """
        np.random.seed(0)
        mock_skimage_io_imread.return_value = np.random.rand(5, 5, 3)
        img, axis = giatools.io.imreadraw('tests/data/input1.tif')
        mock_skimage_io_imread.assert_called_once_with('tests/data/input1.tif')
        np.testing.assert_array_equal(img, mock_skimage_io_imread.return_value)
        self.assertEqual(axis, 'YXC')

    def test__unsupported_dimensions(self, mock_skimage_io_imread):
        """
        Test fallback to ``skimage.io.imread`` and failure if the image has more than 3 dimensions.
        """
        np.random.seed(0)
        mock_skimage_io_imread.return_value = np.random.rand(1, 1, 5, 5, 3)
        with self.assertRaises(AssertionError):
            giatools.io.imreadraw('tests/data/input1.tif')
        mock_skimage_io_imread.assert_called_once_with('tests/data/input1.tif')


class imread(unittest.TestCase):

    def test__deprecation(self):
        with self.assertWarns(DeprecationWarning):
            giatools.io.imread('tests/data/input1_uint8_yx.tif')
