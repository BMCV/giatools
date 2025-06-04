import os
import tempfile
import unittest
import unittest.mock

import numpy as np
import skimage.io
import tifffile

import giatools.io
from giatools.typing import Tuple

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

        Python 3.8 yields subtly different result, hence the tolerance for the `img.mean()` test.
        """
        img, axes = giatools.io.imreadraw('tests/data/input9_qyx.tif')
        self.assertEqual(img.shape, (2, 256, 256))
        self.assertAlmostEqual(img.mean(), 0.05388291, places=8)
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


class imwriteTestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def read_image(self, filepath: str) -> Tuple[np.ndarray, str]:
        """
        Read an image file and return the image data and axes.

        This is a helper function to read image files for validation.
        """
        try:
            with tifffile.TiffFile(filepath) as tif:
                axes = tif.series[0].axes
                data = tif.asarray()
        except tifffile.TiffFileError:
            data = skimage.io.imread(filepath, as_gray=False)
            axes = 'YXC'
        return data, axes

    def _test(
                self,
                data_shape: Tuple,
                axes: str,
                dtype: np.dtype,
                *,
                ext: str,
                backend: giatools.io.BackendType = 'auto',
                validate_axes: bool = True,
            ):
        data = np.random.rand(*data_shape)
        if not np.issubdtype(dtype, np.floating):
            data = (data * np.iinfo(dtype).max).astype(dtype)
        filepath = os.path.join(self.tempdir.name, f'test.{ext}')
        giatools.io.imwrite(data, filepath, backend=backend, metadata=dict(axes=axes))
        data1, axes1 = self.read_image(filepath)
        np.testing.assert_array_equal(data1, data)
        if validate_axes:
            self.assertEqual(axes1, axes)

    def test__unsupported_backend(self):
        with self.assertRaises(ValueError):
            self._test(
                data_shape=(10, 10, 2),
                axes='YXC',
                dtype=np.float32,
                ext='tif',
                backend='unsupported_backend',
            )


class imwrite__tifffile__mixin:

    def test__float32__tifffile__tif(self):
        self._test(data_shape=(10, 10, 5, 2), axes='YXZC', dtype=np.float32, ext='tif', backend='tifffile')

    def test__float32__tifffile__tiff(self):
        self._test(data_shape=(10, 10, 5, 2), axes='YXZC', dtype=np.float32, ext='tiff', backend='tifffile')


class imwrite__skimage__mixin:

    def test__float32__skimage__tif(self):
        self._test(
            data_shape=(10, 10, 5, 2),
            axes='YXZC',
            dtype=np.float32,
            ext='tif',
            backend='skimage',
            validate_axes=False,
        )

    def test__float32__skimage__tiff(self):
        self._test(
            data_shape=(10, 10, 5, 2),
            axes='YXZC',
            dtype=np.float32,
            ext='tiff',
            backend='skimage',
            validate_axes=False,
        )


class imwrite__with_tifffile(imwriteTestCase, imwrite__tifffile__mixin, imwrite__skimage__mixin):

    def setUp(self):
        super().setUp()

        # Verify that the `tifffile` package is installed
        assert giatools.io.tifffile is not None

    def test__float32__auto__tif(self):
        self._test(data_shape=(10, 10, 5, 2), axes='YXZC', dtype=np.float32, ext='tif', backend='auto')

    def test__float32__auto__tiff(self):
        self._test(data_shape=(10, 10, 5, 2), axes='YXZC', dtype=np.float32, ext='tiff', backend='auto')

    def test__uint8__auto__png(self):
        self._test(data_shape=(10, 10, 2), axes='YXC', dtype=np.uint8, ext='png', backend='auto')


@unittest.mock.patch('giatools.io.tifffile', None)
class imwrite__without_tifffile(imwriteTestCase, imwrite__skimage__mixin):

    def test__float32__auto__tif(self):
        assert giatools.io.tifffile is None  # Verify that the `tifffile` package is not installed
        self._test(
            data_shape=(10, 10, 5, 2),
            axes='YXZC',
            dtype=np.float32,
            ext='tif',
            backend='auto',
            validate_axes=False,
        )

    def test__float32__auto__tiff(self):
        assert giatools.io.tifffile is None  # Verify that the `tifffile` package is not installed
        self._test(
            data_shape=(10, 10, 5, 2),
            axes='YXZC',
            dtype=np.float32,
            ext='tiff',
            backend='auto',
            validate_axes=False,
        )

    def test__uint8__auto__png(self):
        assert giatools.io.tifffile is None  # Verify that the `tifffile` package is not installed
        self._test(data_shape=(10, 10, 2), axes='YXC', dtype=np.uint8, ext='png', backend='auto')
