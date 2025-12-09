import copy
import json
import os
import tempfile
import unittest
import unittest.mock

import numpy as np
import skimage.io
import tifffile

import giatools.io
from giatools.typing import (
    Any,
    Dict,
    Literal,
    Optional,
    Tuple,
    Union,
)

# This tests require that the `tifffile` package is installed.
assert giatools.io.tifffile is not None


class imreadraw__with_tifffile(unittest.TestCase):

    def setUp(self):
        # Verify that the `tifffile` package is installed
        assert giatools.io.tifffile is not None

    def verify_metadata(self, metadata: dict, **expected: Any):
        """
        Verify that the metadata is present and correct.
        """
        self.assertIsInstance(metadata, dict)
        for key, value in expected.items():
            if value is None:
                self.assertNotIn(key, metadata)
            else:
                self.assertIn(key, metadata)
                if isinstance(value, tuple):
                    np.testing.assert_array_almost_equal(metadata[key], value)
                else:
                    self.assertEqual(metadata[key], value)

    def test__input1(self):
        img, axes, metadata = giatools.io.imreadraw('tests/data/input1_uint8_yx.tif')
        self.assertEqual(img.mean(), 63.66848655158571)
        self.assertEqual(img.shape, (265, 329))
        self.assertEqual(axes, 'YX')
        self.verify_metadata(metadata, resolution=None, z_spacing=None, unit=None)

    def test__input2(self):
        img, axes, metadata = giatools.io.imreadraw('tests/data/input2_uint8_yx.tif')
        self.assertEqual(img.mean(), 9.543921821305842)
        self.assertEqual(img.shape, (96, 97))
        self.assertEqual(axes, 'YX')
        self.verify_metadata(metadata, resolution=None, z_spacing=None, unit=None)

    def test__input3(self):
        """
        Test a multi-page TIFF, that sometimes fails to load properly with ``skimage.io.imread``, but works with
        `tifffile`.

        For details see: https://github.com/BMCV/galaxy-image-analysis/pull/132#issuecomment-2371561435
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input3_uint16_zyx.tif')
        self.assertEqual(img.shape, (5, 198, 356))
        self.assertEqual(img.mean(), 1259.6755334241288)
        self.assertEqual(axes, 'ZYX')
        self.verify_metadata(metadata, resolution=(10000, 10000), z_spacing=None, unit='cm')

    def test__input4(self):
        """
        Test an RGB PNG file, that cannot be loaded with `tifffile`, but works with ``skimage.io.imread``.
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input4_uint8.png')
        self.assertEqual(img.shape, (10, 10, 3))
        self.assertEqual(img.mean(), 130.04)
        self.assertEqual(axes, 'YXC')
        self.verify_metadata(metadata, resolution=None, z_spacing=None, unit=None)

    def test__input5(self):
        """
        Test TIFF file with ``CYX`` axes.
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input5_uint8_cyx.tif')
        self.assertEqual(img.shape, (2, 8, 16))
        self.assertEqual(img.mean(), 22.25390625)
        self.assertEqual(axes, 'CYX')
        self.verify_metadata(metadata, resolution=(0.734551, 0.367275), z_spacing=0.05445500181716341, unit='um')

    def test__input6(self):
        """
        Test TIFF file with ``ZYX`` axes.
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input6_uint8_zyx.tif')
        self.assertEqual(img.shape, (25, 8, 16))
        self.assertEqual(img.mean(), 26.555)
        self.assertEqual(axes, 'ZYX')
        self.verify_metadata(metadata, resolution=(0.734551, 0.367275), z_spacing=0.05445500181716341, unit='um')

    def test__input7(self):
        """
        Test TIFF file with ``ZCYX`` axes.
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input7_uint8_zcyx.tif')
        self.assertEqual(img.shape, (25, 2, 50, 50))
        self.assertEqual(img.mean(), 14.182152)
        self.assertEqual(axes, 'ZCYX')
        self.verify_metadata(metadata, resolution=(2.295473, 2.295473), z_spacing=0.05445500181716341, unit='um')

    def test__input8(self):
        """
        Test TIFF file with ``TYX`` axes.
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input8_uint16_tyx.tif')
        self.assertEqual(img.shape, (5, 49, 56))
        self.assertEqual(img.mean(), 5815.486880466472)
        self.assertEqual(axes, 'TYX')
        self.verify_metadata(metadata, resolution=(1, 1), z_spacing=None, unit=None)

    def test__input9(self):
        """
        Test TIFF file with ``QYX`` axes.

        Python 3.8 yields subtly different result, hence the tolerance for the `img.mean()` test.
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input9_qyx.tif')
        self.assertEqual(img.shape, (2, 256, 256))
        self.assertAlmostEqual(img.mean(), 0.05388291, places=8)
        self.assertEqual(axes, 'QYX')
        self.verify_metadata(metadata, resolution=(1, 1))

    def test__input10(self):
        """
        Test TIFF file with ``ResolutionUnit`` tag set to 2 (inches).
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input10_resolutionunit2.tiff')
        self.assertEqual(img.shape, (64, 64))
        self.assertAlmostEqual(img.mean(), 128.549560546875, places=8)
        self.assertEqual(axes, 'YX')
        self.verify_metadata(metadata, resolution=(300, 300), unit='inch')


@unittest.mock.patch('skimage.io.imread')
@unittest.mock.patch('giatools.io.tifffile', None)
class imreadraw__without_tifffile(unittest.TestCase):
    """
    Test loading an image without `tifffile` installed.
    """

    def verify_metadata(self, metadata: dict):
        """
        Verify that the metadata dictionary is empty.
        """
        self.assertIsInstance(metadata, dict)
        self.assertEqual(len(metadata), 0)

    def test__yx(self, mock_skimage_io_imread):
        """
        Test fallback to ``skimage.io.imread`` with a two-dimensional image.
        """
        np.random.seed(0)
        mock_skimage_io_imread.return_value = np.random.rand(5, 5)
        img, axis, metadata = giatools.io.imreadraw('tests/data/input1.tif')
        mock_skimage_io_imread.assert_called_once_with('tests/data/input1.tif')
        np.testing.assert_array_equal(img, mock_skimage_io_imread.return_value)
        self.assertEqual(axis, 'YX')
        self.verify_metadata(metadata)

    def test__yxc(self, mock_skimage_io_imread):
        """
        Test fallback to ``skimage.io.imread`` with a three-dimensional image.
        """
        np.random.seed(0)
        mock_skimage_io_imread.return_value = np.random.rand(5, 5, 3)
        img, axis, metadata = giatools.io.imreadraw('tests/data/input1.tif')
        mock_skimage_io_imread.assert_called_once_with('tests/data/input1.tif')
        np.testing.assert_array_equal(img, mock_skimage_io_imread.return_value)
        self.assertEqual(axis, 'YXC')
        self.verify_metadata(metadata)

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
        metadata: Optional[Dict] = None,
        *,
        ext: str,
        backend: giatools.io.BackendType = 'auto',
        validate_axes: bool = True,
        validate_metadata: Union[bool, Literal['auto']] = 'auto',
    ):
        # Create random image data
        data = np.random.rand(*data_shape)
        if not np.issubdtype(dtype, np.floating):
            data = (data * np.iinfo(dtype).max).astype(dtype)

        # Write the image to a temporary file
        filepath = os.path.join(self.tempdir.name, f'test.{ext}')
        metadata = (dict() if metadata is None else metadata) | dict(axes=axes)
        metadata_copy = copy.deepcopy(metadata)
        giatools.io.imwrite(data, filepath, backend=backend, metadata=metadata)

        # Validate immutability of metadata
        self.assertEqual(metadata, metadata_copy)

        # Read back the image data and the axes, and validate, if applicable
        data1, axes1 = self.read_image(filepath)
        np.testing.assert_array_equal(data1, data)
        if validate_axes:
            self.assertEqual(axes1, axes)

        # Validate the metadata, if applicable
        if validate_metadata is True or (validate_metadata == 'auto' and ext in ('tif', 'tiff')):
            with tifffile.TiffFile(filepath) as im_file:
                page0 = im_file.series[0].pages[0]
                description = json.loads(page0.tags['ImageDescription'].value)
                x_res = page0.tags['XResolution'].value
                y_res = page0.tags['YResolution'].value

            if 'resolution' in metadata:
                np.testing.assert_allclose(
                    (
                        x_res[0] / x_res[1],
                        y_res[0] / y_res[1],
                    ),
                    metadata['resolution'],
                )
            if 'z_spacing' in metadata:
                self.assertEqual(float(description['spacing']), metadata['z_spacing'])
            if 'unit' in metadata:
                self.assertEqual(description['unit'], metadata['unit'])

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

    def test__float32__tifffile__tiff__metadata(self):
        self._test(
            data_shape=(10, 10, 5),
            axes='YXZ',
            dtype=np.float32,
            ext='tiff',
            backend='tifffile',
            metadata=dict(
                resolution=(0.3, 0.4),
                z_spacing=0.5,
                unit='um',
            ),
        )


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


class ModuleTestCase(unittest.TestCase):
    """
    Verify that written images can all be read back correctly.
    """

    def setUp(self):
        super().setUp()
        np.random.seed(0)
        self.tempdir = tempfile.TemporaryDirectory()

        # Verify that the `tifffile` package is installed
        assert giatools.io.tifffile is not None

    def tearDown(self):
        self.tempdir.cleanup()

    def _test(
        self,
        data_shape: Tuple,
        axes: str,
        dtype: np.dtype,
        metadata: Dict,
        *,
        ext: str,
    ):
        # Create random image data
        data = np.random.rand(*data_shape)
        if not np.issubdtype(dtype, np.floating):
            data = (data * np.iinfo(dtype).max).astype(dtype)

        # Write the image to a temporary file
        filepath = os.path.join(self.tempdir.name, f'test.{ext}')
        giatools.io.imwrite(data, filepath, metadata=metadata | dict(axes=axes))

        # Read the image back and validate
        data1, axes1, metadata1 = giatools.io.imreadraw(filepath)
        np.testing.assert_array_equal(data1, data)
        self.assertEqual(axes1, axes)
        self.assertEqual(metadata1, metadata)

    def test__tiff__float32(self):
        self._test(
            data_shape=(10, 10, 5, 2),
            axes='YXZC',
            dtype=np.float32,
            ext='tiff',
            metadata=dict(
                resolution=(0.2, 0.4),
                z_spacing=0.5,
                unit='um',
            ),
        )
