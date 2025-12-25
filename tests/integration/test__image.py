"""
Integration tests for the `giatools.image` module.
"""

import os
import tempfile
import unittest
import unittest.mock

import numpy as np

import giatools.image
import giatools.metadata

from ..tools import (
    minimum_python_version,
    permute_axes,
    random_io_test,
    validate_metadata,
    without_logging,
)


class Image(unittest.TestCase):
    """
    Verify that written images can be read back correctly with correct data and metadata.

    This test case integrates ``giatools.image.Image`` with the ``giatools.io`` module.
    """

    @random_io_test(shape=(5, 10, 15), dtype=np.uint8, ext='tiff')
    def test__write_tiff_and_read(self, filepath: str, expected_data: np.ndarray):
        expected_axes = 'ZYX'
        expected_metadata = dict(resolution=(10., 10.), z_spacing=0.2, unit='nm')
        img0 = giatools.image.Image(
            data=expected_data,
            axes=expected_axes,
            metadata=giatools.metadata.Metadata(**expected_metadata),
        )

        # Write the image and read back
        img0.write(filepath)
        img1 = giatools.image.Image.read(filepath).normalize_axes_like(expected_axes)

        # Verify that data and metadata are the same
        np.testing.assert_array_equal(img1.data, expected_data)
        self.assertEqual(img1.axes, expected_axes)
        validate_metadata(self, img1.metadata, **expected_metadata)

    @minimum_python_version(3, 11)
    @without_logging
    def test__read_omezarr_and_write_tiff(self):
        """
        Verify that reading an OME-Zarr image and writing it as a TIFF works correctly.
        """
        import dask.array as da

        # Read OME-Zarr image (img1) and write it as a TIFF
        img1 = giatools.image.Image.read('tests/data/ome-zarr-examples/image-02.zarr', normalize_axes=None)
        self.assertIsInstance(img1.data, da.Array)
        with tempfile.TemporaryDirectory() as temp_path:

            # Write img1 as a TIFF file and read it back as img2
            filepath = os.path.join(temp_path, 'output.tiff')
            img1.write(filepath)
            img2 = giatools.image.Image.read(filepath, normalize_axes=None)

        # Read img2 from the TIFF file and compare img1 to img2
        self.assertEqual(img1.data.shape, img2.data.shape)
        self.assertEqual(img1.data.mean(), img2.data.mean())
        self.assertEqual(img1.original_axes, img2.original_axes)
        self.assertEqual(img1.axes, img2.axes)


class Image__get_anisotropy(unittest.TestCase):
    """
    Test the `Image.get_anisotropy` method.

    This test case integrates ``giatools.image.Image`` with the ``giatools.metadata`` module.
    """

    def setUp(self):
        super().setUp()
        self.array = unittest.mock.MagicMock(shape=(10, 20, 30))

    def test__invalid_axes(self):
        for axes in ['', ' ', 'X', 'Y', 'Z', 'XYX', 'XY ', 'XYA', 'ABCD']:
            with self.subTest(axes=axes):
                img = giatools.Image(data=self.array, axes='CYX')
                with self.assertRaises(ValueError):
                    img.get_anisotropy(axes)

    def test__cyx__unknown_resolution(self):
        img = giatools.Image(data=self.array, axes='CYX')
        self.assertIsNone(img.get_anisotropy())

    def test__cyx__anisotropic(self):
        img = giatools.Image(data=self.array, axes='CYX')  # Y, X
        img.metadata.pixel_size = (1.0, 1.1)  # X, Y
        ret = img.get_anisotropy()
        np.testing.assert_array_almost_equal(ret, (1.04880884817, 0.9534625892))  # Y, X
        self.assertAlmostEqual(1.1 / ret[0], 1.0 / ret[1])

    def test__zyx__unknown_resolution(self):
        img = giatools.Image(data=self.array, axes='ZYX')
        self.assertIsNone(img.get_anisotropy())
        img.metadata.pixel_size = (1.0, 1.1)
        self.assertIsNone(img.get_anisotropy())
        img.metadata.pixel_size = None
        img.metadata.z_spacing = 1.0
        self.assertIsNone(img.get_anisotropy())

    def test__zyx__anisotropic(self):
        img = giatools.Image(data=self.array, axes='ZYX')
        img.metadata.pixel_size = (1.0, 1.1)  # X, Y
        img.metadata.z_spacing = 1.2
        ret = img.get_anisotropy()
        np.testing.assert_array_almost_equal(ret, (1.093931, 1.00277 , 0.911609))  # Z, Y, X
        self.assertAlmostEqual(1.1 / ret[1], 1.0 / ret[2])
        self.assertAlmostEqual(1.1 / ret[1], 1.2 / ret[0])

    @permute_axes('YX', name='known_axes')
    def test__3d__yx__unknown_z(self, known_axes: str):
        img = giatools.Image(data=self.array, axes='ZYX')
        img.metadata.pixel_size = (1.0, 1.1)  # X, Y
        ret = img.get_anisotropy(axes=known_axes)
        self.assertAlmostEqual(ret[known_axes.index('Y')], 1.04880884817)
        self.assertAlmostEqual(ret[known_axes.index('X')], 0.9534625892)
        self.assertIsNone(img.get_anisotropy(axes='XZ'))
        self.assertIsNone(img.get_anisotropy(axes='YZ'))
