import os
import tempfile
import unittest
import unittest.mock

import numpy as np

import giatools.image
import giatools.metadata
from giatools.typing import Tuple

from .tools import (
    maximum_python_version,
    minimum_python_version,
    random_io_test,
    validate_metadata,
    without_logging,
)

# Define test image data
test1_data = np.random.randint(0, 255, (1, 2, 26, 32, 3), dtype=np.uint8)
test1_axes = 'TZYXC'
test1_original_axes = 'ZXYC'
test2_data = np.random.randint(0, 255, (1, 1, 32, 26, 1), dtype=np.uint8)
test2_axes = 'ZTYXC'
test2_original_axes = 'YXC'


class ModuleTestCase(unittest.TestCase):
    """
    Module-level tests for :mod:`giatools.image`.
    """

    def test__default_normalized_axes(self):
        self.assertEqual(giatools.image.default_normalized_axes, 'QTZYXC')

    @random_io_test(shape=(5, 10, 15), dtype=np.uint8, ext='tiff')
    def test__write_tiff_and_read(self, filepath: str, expected_data: np.ndarray):
        """
        Verify that written images can be read back correctly with correct data and metadata.
        """
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


class Image__read(unittest.TestCase):

    def test__input1(self):
        img = giatools.image.Image.read('tests/data/input1_uint8_yx.tiff')
        self.assertEqual(img.data.mean(), 63.66848655158571)
        self.assertEqual(img.data.shape, (1, 1, 1, 265, 329, 1))
        self.assertEqual(img.original_axes, 'YX')
        self.assertEqual(img.axes, giatools.image.default_normalized_axes)
        validate_metadata(self, img.metadata)

    def test__input1__without_normalization(self):
        img = giatools.image.Image.read('tests/data/input1_uint8_yx.tiff', normalize_axes=None)
        self.assertEqual(img.data.mean(), 63.66848655158571)
        self.assertEqual(img.data.shape, (265, 329))
        self.assertEqual(img.original_axes, 'YX')
        self.assertEqual(img.axes, 'YX')
        validate_metadata(self, img.metadata)

    def test__input2(self):
        img = giatools.image.Image.read('tests/data/input2_uint8_yx.tiff')
        self.assertEqual(img.data.mean(), 9.543921821305842)
        self.assertEqual(img.data.shape, (1, 1, 1, 96, 97, 1))
        self.assertEqual(img.original_axes, 'YX')
        self.assertEqual(img.axes, giatools.image.default_normalized_axes)
        validate_metadata(self, img.metadata)

    def test__input3(self):
        img = giatools.image.Image.read('tests/data/input3_uint16_zyx.tiff')
        self.assertEqual(img.data.shape, (1, 1, 5, 198, 356, 1))
        self.assertEqual(img.data.mean(), 1259.6755334241288)
        self.assertEqual(img.original_axes, 'ZYX')
        self.assertEqual(img.axes, giatools.image.default_normalized_axes)
        validate_metadata(self, img.metadata, resolution=(10000., 10000.))

    def test__input4(self):
        img = giatools.image.Image.read('tests/data/input4_uint8.png')
        self.assertEqual(img.data.shape, (1, 1, 1, 10, 10, 3))
        self.assertEqual(img.data.mean(), 130.04)
        self.assertEqual(img.original_axes, 'YXC')
        self.assertEqual(img.axes, giatools.image.default_normalized_axes)
        validate_metadata(self, img.metadata)

    def test__input5(self):
        img = giatools.image.Image.read('tests/data/input5_uint8_cyx.tiff')
        self.assertEqual(img.data.shape, (1, 1, 1, 8, 16, 2))
        self.assertEqual(img.data.mean(), 22.25390625)
        self.assertEqual(img.original_axes, 'CYX')
        self.assertEqual(img.axes, giatools.image.default_normalized_axes)
        validate_metadata(self, img.metadata, resolution=(0.734551, 0.367275), z_spacing=0.05445500181716341, unit='um')

    def test__input6(self):
        img = giatools.image.Image.read('tests/data/input6_uint8_zyx.tiff')
        self.assertEqual(img.data.shape, (1, 1, 25, 8, 16, 1))
        self.assertEqual(img.data.mean(), 26.555)
        self.assertEqual(img.original_axes, 'ZYX')
        self.assertEqual(img.axes, giatools.image.default_normalized_axes)
        validate_metadata(self, img.metadata, resolution=(0.734551, 0.367275), z_spacing=0.05445500181716341, unit='um')

    def test__input7(self):
        img = giatools.image.Image.read('tests/data/input7_uint8_zcyx.tif')
        self.assertEqual(img.data.shape, (1, 1, 25, 50, 50, 2))
        self.assertEqual(img.data.mean(), 14.182152)
        self.assertEqual(img.original_axes, 'ZCYX')
        self.assertEqual(img.axes, giatools.image.default_normalized_axes)
        validate_metadata(self, img.metadata, resolution=(2.295473, 2.295473), z_spacing=0.05445500181716341, unit='um')

    def test__input8(self):
        img = giatools.image.Image.read('tests/data/input8_uint16_tyx.tif')
        self.assertEqual(img.data.shape, (1, 5, 1, 49, 56, 1))
        self.assertEqual(img.data.mean(), 5815.486880466472)
        self.assertEqual(img.original_axes, 'TYX')
        self.assertEqual(img.axes, giatools.image.default_normalized_axes)
        validate_metadata(self, img.metadata, resolution=(1., 1.))

    @minimum_python_version(3, 11)
    @without_logging
    def test__omezarr__examples__image02(self):
        """
        Test OME-Zarr file with YX axes.
        """
        import dask.array as da
        img = giatools.image.Image.read('tests/data/ome-zarr-examples/image-02.zarr')
        self.assertIsInstance(img.data, da.Array)
        self.assertEqual(img.data.shape, (1, 1, 1, 200, 200, 1))
        self.assertAlmostEqual(float(img.data.mean()), 502.2611393006139)
        self.assertEqual(img.original_axes, 'YX')
        self.assertEqual(img.axes, giatools.image.default_normalized_axes)
        validate_metadata(self, img.metadata, resolution=(1., 1.), unit='um')

    @minimum_python_version(3, 11)
    @without_logging
    def test__omezarr__examples__image04(self):
        """
        Test OME-Zarr file with ZYX axes.
        """
        import dask.array as da
        img = giatools.image.Image.read('tests/data/ome-zarr-examples/image-04.zarr')
        self.assertIsInstance(img.data, da.Array)
        self.assertEqual(img.data.shape, (1, 1, 2, 64, 64, 1))
        self.assertAlmostEqual(float(img.data.mean()), 0.0)
        self.assertEqual(img.original_axes, 'ZYX')
        self.assertEqual(img.axes, giatools.image.default_normalized_axes)
        validate_metadata(self, img.metadata, resolution=(1., 1.), z_spacing=1., unit='um')


@unittest.mock.patch('giatools.io.imwrite')
class Image__write(unittest.TestCase):

    def setUp(self):
        self.img1 = giatools.image.Image(data=test1_data.copy(), axes=test1_axes, original_axes=test1_original_axes)

    def test(self, mock_imwrite):
        self.img1.write('test_output.tiff')
        mock_imwrite.assert_called_once()
        np.testing.assert_array_equal(mock_imwrite.call_args_list[0][0][0], test1_data)
        self.assertEqual(mock_imwrite.call_args_list[0][0][1], 'test_output.tiff')
        self.assertEqual(
            mock_imwrite.call_args_list[0][1],
            dict(backend='auto', axes=test1_axes, metadata=giatools.metadata.Metadata()),
        )

    def test__tifffile(self, mock_imwrite):
        self.img1.write('test_output.tiff', backend='tifffile')
        mock_imwrite.assert_called_once()
        np.testing.assert_array_equal(mock_imwrite.call_args_list[0][0][0], test1_data)
        self.assertEqual(mock_imwrite.call_args_list[0][0][1], 'test_output.tiff')
        self.assertEqual(
            mock_imwrite.call_args_list[0][1],
            dict(backend='tifffile', axes=test1_axes, metadata=giatools.metadata.Metadata()),
        )

    def test__metadata(self, mock_imwrite):
        self.img1.metadata.z_spacing = 0.5
        self.img1.write('test_output.tiff')
        mock_imwrite.assert_called_once()
        np.testing.assert_array_equal(mock_imwrite.call_args_list[0][0][0], test1_data)
        self.assertEqual(mock_imwrite.call_args_list[0][0][1], 'test_output.tiff')
        self.assertEqual(
            mock_imwrite.call_args_list[0][1],
            dict(
                backend='auto',
                axes=test1_axes,
                metadata=giatools.metadata.Metadata(z_spacing=0.5),
            ),
        )

    def test__invalid_axes(self, mock_imwrite):
        img_invalid = giatools.image.Image(data=test1_data.squeeze().copy(), axes='ZYX')
        with self.assertRaises(ValueError):
            img_invalid.write('test_output.tiff')
        mock_imwrite.assert_not_called()


class Image__data(unittest.TestCase):
    """
    Test the data property of the Image class.
    """

    @minimum_python_version(3, 11)
    def test__dask__filtering(self):
        import dask.array as da
        import scipy.ndimage as ndi
        np.random.seed(0)
        np_data = np.random.rand(40, 60)
        img = giatools.Image(data=da.from_array(np_data, chunks=(5, 5)), axes='YX')
        self.assertIsInstance(img.data, da.Array)
        np.testing.assert_almost_equal(ndi.gaussian_filter(img.data, sigma=3).mean(), 0.5, decimal=2)


class Image__reorder_axes_like(unittest.TestCase):

    def setUp(self):
        self.img1 = giatools.image.Image(data=test1_data.copy(), axes=test1_axes, original_axes=test1_original_axes)

    def test(self):
        img_reordered = self.img1.reorder_axes_like('ZTCYX')
        self.assertEqual(img_reordered.axes, 'ZTCYX')
        self.assertEqual(img_reordered.data.shape, (2, 1, 3, 26, 32))
        self.assertEqual(img_reordered.original_axes, test1_original_axes)

    def test__identity(self):
        img_reordered = self.img1.reorder_axes_like(test1_axes)
        self.assertEqual(img_reordered.axes, test1_axes)
        self.assertEqual(img_reordered.original_axes, test1_original_axes)
        self.assertIs(img_reordered.data, self.img1.data)
        self.assertIs(img_reordered.metadata, self.img1.metadata)

    def test__spurious_axis(self):
        with self.assertRaises(ValueError):
            self.img1.reorder_axes_like('ZTCYXW')

    def test__missing_axis(self):
        with self.assertRaises(ValueError):
            self.img1.reorder_axes_like('ZTCY')

    def test__ambigious_axis(self):
        with self.assertRaises(ValueError):
            self.img1.reorder_axes_like('ZTCYXX')
        with self.assertRaises(ValueError):
            self.img1.reorder_axes_like('ZTCXX')

    def test__immutability(self):
        """
        Verify that the original image is not modified.
        """
        self.img1.reorder_axes_like('ZTCYX')
        np.testing.assert_array_equal(self.img1.data, test1_data)
        self.assertEqual(self.img1.axes, test1_axes)
        self.assertEqual(self.img1.original_axes, test1_original_axes)


class Image__squeeze(unittest.TestCase):

    def setUp(self):
        self.img1 = giatools.image.Image(data=test1_data, axes=test1_axes, original_axes=test1_original_axes)
        self.img2 = giatools.image.Image(data=test2_data, axes=test2_axes, original_axes=test2_original_axes)

    def test__squeeze(self):
        img1_squeezed = self.img1.squeeze()
        self.assertEqual(img1_squeezed.axes, 'ZYXC')
        self.assertEqual(img1_squeezed.data.shape, (2, 26, 32, 3))
        self.assertEqual(img1_squeezed.original_axes, test1_original_axes)
        img2_squeezed = self.img2.squeeze()
        self.assertEqual(img2_squeezed.axes, 'YX')
        self.assertEqual(img2_squeezed.data.shape, (32, 26))
        self.assertEqual(img2_squeezed.original_axes, test2_original_axes)

    def test__immutability(self):
        """
        Verify that the original image is not modified.
        """
        self.img1.squeeze()
        np.testing.assert_array_equal(self.img1.data, test1_data)
        self.assertEqual(self.img1.axes, test1_axes)
        self.assertEqual(self.img1.original_axes, test1_original_axes)


class Image__squeeze_like(unittest.TestCase):

    def setUp(self):
        self.img1 = giatools.image.Image(data=test1_data, axes=test1_axes, original_axes=test1_original_axes)
        self.img2 = giatools.image.Image(data=test2_data, axes=test2_axes, original_axes=test2_original_axes)

    def test__identity(self):
        img_squeezed = self.img1.squeeze_like('ZCYX')
        self.assertEqual(img_squeezed.original_axes, test1_original_axes)
        self.assertTrue(np.shares_memory(img_squeezed.data, self.img1.data))
        self.assertIs(img_squeezed.metadata, self.img1.metadata)

    def test__no_squeeze(self):
        img_squeezed = self.img1.squeeze_like('ZTCYX')
        self.assertEqual(img_squeezed.axes, 'ZTCYX')
        self.assertEqual(img_squeezed.data.shape, (2, 1, 3, 26, 32))
        self.assertEqual(img_squeezed.original_axes, test1_original_axes)

    def test__squeeze_1(self):
        img_squeezed = self.img1.squeeze_like('ZCYX')
        self.assertEqual(img_squeezed.axes, 'ZCYX')
        self.assertEqual(img_squeezed.data.shape, (2, 3, 26, 32))
        self.assertEqual(img_squeezed.original_axes, test1_original_axes)

    def test__squeeze_2(self):
        img_squeezed = self.img2.squeeze_like('XYC')
        self.assertEqual(img_squeezed.axes, 'XYC')
        self.assertEqual(img_squeezed.data.shape, (26, 32, 1))
        self.assertEqual(img_squeezed.original_axes, test2_original_axes)

    def test__squeeze_3(self):
        img_squeezed = self.img2.squeeze_like('XY')
        self.assertEqual(img_squeezed.axes, 'XY')
        self.assertEqual(img_squeezed.data.shape, (26, 32))
        self.assertEqual(img_squeezed.original_axes, test2_original_axes)

    def test__squeeze_illegal_axis(self):
        with self.assertRaises(ValueError):
            self.img1.squeeze_like('TCYX')

    def test__spurious_axis(self):
        with self.assertRaises(ValueError):
            self.img1.squeeze_like('ZCYXW')

    def test__ambigious_axis(self):
        with self.assertRaises(ValueError):
            self.img1.squeeze_like('ZCYXX')

    def test__immutability(self):
        """
        Verify that the original image is not modified.
        """
        self.img1.squeeze_like('ZCYX')
        np.testing.assert_array_equal(self.img1.data, test1_data)
        self.assertEqual(self.img1.axes, test1_axes)
        self.assertEqual(self.img1.original_axes, test1_original_axes)


class Image__normalize_axes_like(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.img1 = giatools.image.Image(data=test1_data, axes=test1_axes, original_axes=test1_original_axes)
        self.img2 = giatools.image.Image(data=test2_data, axes=test2_axes, original_axes=test2_original_axes)

    def test__identity(self):
        img_normalized = self.img1.normalize_axes_like(test1_original_axes)
        self.assertEqual(img_normalized.original_axes, test1_original_axes)
        self.assertTrue(np.shares_memory(img_normalized.data, self.img1.data))
        self.assertIs(img_normalized.metadata, self.img1.metadata)

    def test1(self):
        img_normalized = self.img1.normalize_axes_like(test1_original_axes)
        self.assertEqual(img_normalized.axes, test1_original_axes)
        self.assertEqual(img_normalized.data.shape, (2, 32, 26, 3))
        self.assertEqual(img_normalized.original_axes, test1_original_axes)

    def test2(self):
        img_normalized = self.img2.normalize_axes_like(test2_original_axes)
        self.assertEqual(img_normalized.axes, test2_original_axes)
        self.assertEqual(img_normalized.data.shape, (32, 26, 1))
        self.assertEqual(img_normalized.original_axes, test2_original_axes)

    def test__immutability(self):
        """
        Verify that the original image is not modified.
        """
        self.img1.normalize_axes_like(test1_original_axes)
        np.testing.assert_array_equal(self.img1.data, test1_data)
        self.assertEqual(self.img1.axes, test1_axes)
        self.assertEqual(self.img1.original_axes, test1_original_axes)

    def test__ambiguous_axes(self):
        with self.assertRaises(AssertionError):
            self.img1.normalize_axes_like('ZTCYXX')


class Image__iterate_jointly(unittest.TestCase):

    def create_test_image(self, axes: str, shape: Tuple[int, ...]) -> giatools.image.Image:
        assert len(axes) == len(shape)
        np.random.seed(0)
        data = np.random.randint(0, 255, shape, dtype=np.uint8)
        return giatools.image.Image(data=data, axes=axes)

    @maximum_python_version(3, 10)
    def test__minimum_python_version(self):
        img = self.create_test_image('YX', (10, 11))
        with self.assertRaises(RuntimeError):
            for _ in img.iterate_jointly('YX'):
                pass

    @minimum_python_version(3, 11)
    def test__empty(self):
        img = self.create_test_image('YX', (10, 11))
        with self.assertRaises(ValueError):
            for _ in img.iterate_jointly(''):
                pass

    @minimum_python_version(3, 11)
    def test__spurious_axis(self):
        img = self.create_test_image('YX', (10, 11))
        with self.assertRaises(ValueError):
            for _ in img.iterate_jointly('YXZ'):
                pass
        with self.assertRaises(ValueError):
            for _ in img.iterate_jointly('Z'):
                pass

    def _test(self, axes: str, shape: Tuple[int, ...], joint_axes: str):
        assert set(joint_axes).issubset(set(axes))
        img = self.create_test_image(axes, shape)
        counter = np.zeros(img.data.shape, np.uint32)
        for sl, arr in img.iterate_jointly(joint_axes):
            counter[sl] += 1
            np.testing.assert_array_equal(arr, img.data[sl])
        np.testing.assert_array_equal(counter, np.ones(counter.shape, np.uint8))

    @minimum_python_version(3, 11)
    def test__img_yx__iterate_y(self):
        self._test('YX', (10, 11), 'Y')

    @minimum_python_version(3, 11)
    def test__img_yx__iterate_yx(self):
        self._test('YX', (10, 11), 'YX')

    @minimum_python_version(3, 11)
    def test__img_zyx__iterate_yx(self):
        self._test('ZYX', (1, 11, 12), 'YX')
        self._test('ZYX', (5, 11, 12), 'YX')

    @minimum_python_version(3, 11)
    def test__img_zyx__iterate_zyx(self):
        self._test('ZYX', (1, 11, 12), 'ZYX')
        self._test('ZYX', (5, 11, 12), 'ZYX')

    @minimum_python_version(3, 11)
    def test__img_tzyxc__iterate_zyx(self):
        self._test('TZYXC', (1, 1, 11, 12, 3), 'ZYX')
        self._test('TZYXC', (1, 5, 11, 12, 3), 'ZYX')
        self._test('TZYXC', (5, 1, 11, 12, 3), 'ZYX')
        self._test('TZYXC', (5, 5, 11, 12, 3), 'ZYX')

    def _test_dask(self, axes: str, shape: Tuple[int, ...], joint_axes: str, chunks: Tuple[int, ...]):
        assert set(joint_axes).issubset(set(axes))
        import dask.array as da
        img = self.create_test_image(axes, shape)
        np_data, img.data = img.data, da.from_array(img.data, chunks=chunks)
        counter = np.zeros(img.data.shape, np.uint32)
        for sl, arr in img.iterate_jointly(joint_axes):
            counter[sl] += 1
            np.testing.assert_array_equal(arr, np_data[sl])
        np.testing.assert_array_equal(counter, np.ones(counter.shape, np.uint8))

    @minimum_python_version(3, 11)
    def test__dask_array__zyx__iterate__yx(self):
        self._test_dask('ZYX', (10, 20, 30), 'YX', (2, 5, 5))


class Image__is_isotropic(unittest.TestCase):

    eps = 1e-6

    def setUp(self):
        self.array = np.zeros((10, 20, 30))

    def test__2d__unknown_resolution(self):
        img = giatools.Image(data=self.array, axes='CYX')
        self.assertIsNone(img.is_isotropic())

    def test__2d__anisotropic(self):
        img = giatools.Image(data=self.array, axes='CYX')
        img.metadata.pixel_size = (1.0, 1.01 + self.eps)
        self.assertFalse(img.is_isotropic())

    def test__2d__isotropic(self):
        img = giatools.Image(data=self.array, axes='CYX')
        img.metadata.pixel_size = (1.0, 1.01 - self.eps)
        self.assertTrue(img.is_isotropic())

    def test__3d__unknown_resolution(self):
        img = giatools.Image(data=self.array, axes='ZYX')
        self.assertIsNone(img.is_isotropic())
        img.metadata.pixel_size = (1.0, 1.01 + self.eps)
        self.assertIsNone(img.is_isotropic())
        img.metadata.pixel_size = None
        img.metadata.z_spacing = 1.0
        self.assertIsNone(img.is_isotropic())

    def test__3d__anisotropic(self):
        img = giatools.Image(data=self.array, axes='ZYX')
        img.metadata.pixel_size = (1.0, 1.0)
        img.metadata.z_spacing = 1.01 + self.eps
        self.assertFalse(img.is_isotropic())

    def test__3d__isotropic(self):
        img = giatools.Image(data=self.array, axes='ZYX')
        img.metadata.pixel_size = (1.0, 1.0)
        img.metadata.z_spacing = 1.01 - self.eps
        self.assertTrue(img.is_isotropic())
