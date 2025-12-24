import pathlib
import unittest

import numpy as np

import giatools.io

from ..tools import (
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
                self.assertEqual(round(img.mean(), 2), 63.67)
                self.assertEqual(img.shape, (265, 329))
                self.assertEqual(axes, 'YX')
                validate_metadata(self, metadata)

    def test__input2(self):
        img, axes, metadata = giatools.io.imreadraw('tests/data/input2_uint8_yx.tiff')
        self.assertEqual(round(img.mean(), 2), 9.54)
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
        self.assertEqual(round(img.mean(), 2), 1259.68)
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
        self.assertEqual(img.dtype, bool)
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
