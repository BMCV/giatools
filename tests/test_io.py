import json
import os
import pathlib
import tempfile
import unittest

import attrs
import numpy as np
import scipy.ndimage as ndi
import skimage.io
import tifffile

import giatools.io
import giatools.metadata
from giatools.typing import (
    Literal,
    Optional,
    PathLike,
    Tuple,
    Type,
    Union,
)

from .tools import (
    minimum_python_version,
    random_io_test,
    validate_metadata,
    without_logging,
)


class imreadraw(unittest.TestCase):

    @random_io_test(shape=(10, 10, 3), dtype=np.uint8, ext='not-an-image')
    def test__invalid_file(self, filepath, data):
        with open(filepath, 'w') as f:
            f.write(str(data))
        with self.assertRaisesRegex(giatools.io.UnsupportedFileError, f'No backend could read {filepath}'):
            giatools.io.imreadraw(filepath)

    @random_io_test(shape=(10, 10, 3), dtype=np.uint8, ext='not-an-image')
    def test__invalid_file__with_kwargs(self, filepath, data):
        with open(filepath, 'w') as f:
            f.write(str(data))
        with self.assertRaisesRegex(
            giatools.io.UnsupportedFileError,
            fr'^No backend could read {filepath} with some_kwarg=True$',
        ):
            giatools.io.imreadraw(filepath, some_kwarg=True)

    def test__input1(self):
        filepath_str = 'tests/data/input1_uint8_yx.tiff'
        for filepath in (filepath_str, pathlib.Path(filepath_str)):
            with self.subTest(filepath_type=type(filepath)):
                img, axes, metadata = giatools.io.imreadraw(filepath)
                self.assertEqual(img.mean(), 63.66848655158571)
                self.assertEqual(img.shape, (265, 329))
                self.assertEqual(axes, 'YX')
                validate_metadata(self, metadata)

    def test__input2(self):
        img, axes, metadata = giatools.io.imreadraw('tests/data/input2_uint8_yx.tiff')
        self.assertEqual(img.mean(), 9.543921821305842)
        self.assertEqual(img.shape, (96, 97))
        self.assertEqual(axes, 'YX')
        validate_metadata(self, metadata)

    def test__input3(self):
        """
        Test a multi-page TIFF, that sometimes fails to load with ``skimage.io.imread``, but works with `tifffile`.

        For details see: https://github.com/BMCV/galaxy-image-analysis/pull/132#issuecomment-2371561435
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input3_uint16_zyx.tiff')
        self.assertEqual(img.shape, (5, 198, 356))
        self.assertEqual(img.mean(), 1259.6755334241288)
        self.assertEqual(axes, 'ZYX')
        validate_metadata(self, metadata, resolution=(10000., 10000.))

    def test__input4__png(self):
        """
        Test an RGB PNG file, that cannot be loaded with `tifffile`, but works with ``skimage.io.imread``.
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input4_uint8.png')
        self.assertEqual(img.shape, (10, 10, 3))
        self.assertEqual(round(img.mean(), 2), 130.04)
        self.assertEqual(axes, 'YXC')
        validate_metadata(self, metadata)

    def test__input4__jpg(self):
        """
        Test an JPG file, that cannot be loaded with `tifffile`, but works with ``skimage.io.imread``.
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input4_uint8.jpg')
        self.assertEqual(img.shape, (10, 10, 3))
        self.assertEqual(round(img.mean(), 2), 130.06)
        self.assertEqual(axes, 'YXC')
        validate_metadata(self, metadata)

    def test__input5(self):
        """
        Test TIFF file with ``CYX`` axes.
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input5_uint8_cyx.tiff')
        self.assertEqual(img.shape, (2, 8, 16))
        self.assertEqual(img.mean(), 22.25390625)
        self.assertEqual(axes, 'CYX')
        validate_metadata(self, metadata, resolution=(0.734551, 0.367275), z_spacing=0.05445500181716341, unit='um')

    def test__input6(self):
        """
        Test TIFF file with ``ZYX`` axes.
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input6_uint8_zyx.tiff')
        self.assertEqual(img.shape, (25, 8, 16))
        self.assertEqual(img.mean(), 26.555)
        self.assertEqual(axes, 'ZYX')
        validate_metadata(self, metadata, resolution=(0.734551, 0.367275), z_spacing=0.05445500181716341, unit='um')

    def test__input7(self):
        """
        Test TIFF file with ``ZCYX`` axes.
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input7_uint8_zcyx.tif')
        self.assertEqual(img.shape, (25, 2, 50, 50))
        self.assertEqual(img.mean(), 14.182152)
        self.assertEqual(axes, 'ZCYX')
        validate_metadata(self, metadata, resolution=(2.295473, 2.295473), z_spacing=0.05445500181716341, unit='um')

    def test__input8(self):
        """
        Test TIFF file with ``TYX`` axes.
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input8_uint16_tyx.tif')
        self.assertEqual(img.shape, (5, 49, 56))
        self.assertEqual(img.mean(), 5815.486880466472)
        self.assertEqual(axes, 'TYX')
        validate_metadata(self, metadata, resolution=(1., 1.))

    def test__input9(self):
        """
        Test TIFF file with ``QYX`` axes.
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input9_qyx.tif')
        self.assertEqual(img.shape, (2, 256, 256))
        self.assertAlmostEqual(img.mean(), 0.05388291)
        self.assertEqual(axes, 'QYX')
        validate_metadata(self, metadata, resolution=(1., 1.))

    def test__input10(self):
        """
        Test TIFF file with ``ResolutionUnit`` tag set to 2 (inches).
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input10_resolutionunit2.tiff')
        self.assertEqual(img.shape, (64, 64))
        self.assertAlmostEqual(img.mean(), 128.549560546875)
        self.assertEqual(axes, 'YX')
        validate_metadata(self, metadata, resolution=(300., 300.), unit='inch')

    def test__input11(self):
        """
        Test multi-series OME-TIFF file with OME XML metadata.
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input11.ome.tiff', position=0)
        self.assertEqual(img.shape, (4, 5, 5))
        self.assertAlmostEqual(img.mean(), 1384.33)
        self.assertEqual(axes, 'CYX')
        validate_metadata(self, metadata, resolution=(15384.615, 15384.615), z_spacing=1., unit='um')

    @minimum_python_version(3, 11)
    @without_logging
    def test__omezarr__examples__image02(self):
        """
        Test OME-Zarr file with YX axes.
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/ome-zarr-examples/image-02.zarr')
        self.assertEqual(img.shape, (200, 200))
        self.assertEqual(img.dtype, np.float64)
        self.assertAlmostEqual(round(img.mean().compute(), 2), 502.26)
        self.assertEqual(axes, 'YX')
        validate_metadata(self, metadata, resolution=(1., 1.), unit='um')

    @minimum_python_version(3, 11)
    @without_logging
    def test__omezarr__examples__image04(self):
        """
        Test OME-Zarr file with ZYX axes.
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/ome-zarr-examples/image-04.zarr')
        self.assertEqual(img.shape, (2, 64, 64))
        self.assertEqual(img.dtype, np.float64)
        self.assertAlmostEqual(img.mean().compute(), 0.0)
        self.assertEqual(axes, 'ZYX')
        validate_metadata(self, metadata, resolution=(1., 1.), z_spacing=1., unit='um')

    @minimum_python_version(3, 11)
    @without_logging
    def test__omezarr__examples__image12(self):
        """
        Test OME-Zarr file with ZYX axes, but without unit annotations.
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input12.zarr')
        self.assertEqual(img.shape, (2, 100, 100))
        self.assertEqual(img.dtype, np.bool)
        self.assertAlmostEqual(round(img.mean().compute(), 2), 0.50)
        self.assertEqual(axes, 'ZYX')
        validate_metadata(self, metadata, resolution=(1., 1.), z_spacing=1.)

    @minimum_python_version(3, 11)
    @without_logging
    def test__omezarr__examples__image13(self):
        """
        Test OME-Zarr file with CYX axes.
        """
        img, axes, metadata = giatools.io.imreadraw('tests/data/input13.zarr')
        self.assertEqual(img.shape, (2, 64, 64))
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(list(sorted(np.unique(np.asarray(img).reshape(-1)))), [0, 100, 200])
        self.assertEqual(axes, 'CYX')
        validate_metadata(self, metadata, resolution=(1., 1.))


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


class imwrite(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def _read_image(self, filepath: str) -> Tuple[np.ndarray, str]:
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
        metadata: Optional[giatools.metadata.Metadata] = None,
        *,
        filepath_type: Type[PathLike] = str,
        sigma: float = 0,
        ext: str,
        backend: str = 'auto',
        validate_axes: bool = True,
        validate_metadata: Union[bool, Literal['auto']] = 'auto',
        rms_tol: Optional[float] = None,
        **kwargs,
    ):
        # Create random image data
        data = np.random.rand(*data_shape)
        if sigma > 0:
            data = ndi.gaussian_filter(data, sigma=sigma)
        if not np.issubdtype(dtype, np.floating):
            data = (data * np.iinfo(dtype).max).astype(dtype)

        # Write the image to a temporary file
        filepath = os.path.join(self.tempdir.name, f'test.{ext}')
        metadata = giatools.metadata.Metadata() if metadata is None else metadata
        metadata_copy = attrs.asdict(metadata)
        giatools.io.imwrite(data, filepath_type(filepath), backend=backend, axes=axes, metadata=metadata, **kwargs)

        # Validate immutability of metadata
        _validate_metadata = globals()['validate_metadata']
        _validate_metadata(self, metadata, **metadata_copy)

        # Read back the image data and the axes, and validate, if applicable
        data1, axes1 = self._read_image(filepath)
        if rms_tol is None:
            np.testing.assert_array_equal(data1, data)
        else:
            self.assertLessEqual(np.sqrt((data1 - data) ** 2).mean(), rms_tol)
        if validate_axes:
            self.assertEqual(axes1, axes)

        # Validate the metadata (written as JSON), if applicable
        if validate_metadata is True or (validate_metadata == 'auto' and ext in ('tif', 'tiff')):
            with tifffile.TiffFile(filepath) as im_file:
                page0 = im_file.series[0].pages[0]
                description = json.loads(page0.tags['ImageDescription'].value)
                x_res = page0.tags['XResolution'].value
                y_res = page0.tags['YResolution'].value

            if metadata.resolution is not None:
                np.testing.assert_allclose(
                    (
                        x_res[0] / x_res[1],
                        y_res[0] / y_res[1],
                    ),
                    metadata.resolution,
                )
            if metadata.z_spacing is not None:
                self.assertEqual(float(description['spacing']), metadata.z_spacing)
            if metadata.z_position is not None:
                self.assertEqual(float(description['z_position']), metadata.z_position)
            if metadata.unit is not None:
                self.assertEqual(description['unit'], metadata.unit)

    def test__unsupported_backend(self):
        with self.assertRaises(ValueError):
            self._test(
                data_shape=(10, 10, 2),
                axes='YXC',
                dtype=np.float32,
                ext='tiff',
                backend='unsupported_backend',
            )

    def test__unsupported_file_error(self):
        with self.assertRaises(giatools.io.UnsupportedFileError):
            self._test(
                data_shape=(10, 10, 2),
                axes='YXC',
                dtype=np.float32,
                ext='unsupported_extension',
                backend='auto',
            )

    def test__incompatible_data_error(self):
        with self.assertRaises(giatools.io.IncompatibleDataError):
            self._test(
                data_shape=(10, 10, 2, 2),
                axes='YXCZ',  # Invalid axes for PNG
                dtype=np.uint8,
                ext='png',
                backend='auto',
            )

    def test__float32__tifffile__tif(self):
        with self.assertWarns(DeprecationWarning):
            self._test(data_shape=(10, 10, 5, 2), axes='YXZC', dtype=np.float32, ext='tif', backend='tifffile')

    def test__float32__tifffile__tiff(self):
        for filetype_path in (str, pathlib.Path):
            with self.subTest(filepath_type=filetype_path):
                self._test(
                    data_shape=(10, 10, 5, 2),
                    axes='YXZC',
                    dtype=np.float32,
                    ext='tiff',
                    backend='tifffile',
                    filepath_type=filetype_path,
                )

    def test__float32__tifffile__tiff__metadata(self):
        self._test(
            data_shape=(10, 10, 5),
            axes='YXZ',
            dtype=np.float32,
            ext='tiff',
            backend='tifffile',
            metadata=giatools.metadata.Metadata(
                resolution=(0.3, 0.4),
                z_spacing=0.5,
                z_position=0.8,
                unit='um',
            ),
        )

    def test__float32__skimage__tif(self):
        with self.assertWarns(DeprecationWarning):
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

    def test__float32__auto__tif(self):
        with self.assertWarns(DeprecationWarning):
            self._test(data_shape=(10, 10, 5, 2), axes='YXZC', dtype=np.float32, ext='tif', backend='auto')

    def test__float32__auto__tiff(self):
        self._test(data_shape=(10, 10, 5, 2), axes='YXZC', dtype=np.float32, ext='tiff', backend='auto')

    def test__uint8__auto__png(self):
        self._test(data_shape=(10, 10, 2), axes='YXC', dtype=np.uint8, ext='png', backend='auto')

    def test__uint8__auto__jpg(self):
        for ext in ('jpg', 'jpeg'):
            with self.subTest(ext=ext):
                self._test(
                    data_shape=(100, 150, 3),
                    axes='YXC',
                    dtype=np.uint8,
                    ext=ext,
                    backend='auto',
                    sigma=3,
                    rms_tol=0.1,
                    quality=100,
                )


class ModuleTestCase(unittest.TestCase):
    """
    Module-level tests for :mod:`giatools.io`.
    """

    def _test__write_and_read(self, axes: str, filepath: str, data: np.ndarray, metadata: dict):
        """
        Verify that written images can be read back correctly with correct data and metadata.
        """

        # Write the image and read back
        giatools.io.imwrite(
            data,
            filepath,
            axes=axes,
            metadata=giatools.metadata.Metadata(**metadata),
        )

        # Read the image back and validate
        data1, axes1, metadata1 = giatools.io.imreadraw(filepath)
        np.testing.assert_array_equal(data1, data)
        self.assertEqual(axes1, axes)
        validate_metadata(self, metadata1, **metadata)

    @random_io_test(shape=(10, 10, 5, 2), dtype=np.float32, ext='tiff')
    def test__write_and_read__tiff(self, filepath: str, data: np.ndarray):
        self._test__write_and_read(
            'YXZC',
            filepath,
            data,
            dict(resolution=(0.2, 0.4), z_spacing=0.5, z_position=0.8, unit='km'),
        )

    @minimum_python_version(3, 11)
    @random_io_test(shape=(4, 10, 10, 5, 2), dtype=np.float32, ext='zarr')
    def test__write_and_read__zarr(self, filepath: str, data: np.ndarray):
        self._test__write_and_read(
            'TCYXZ',
            filepath,
            data,
            dict(resolution=(0.2, 0.4), z_spacing=0.5, unit='km'),
        )
