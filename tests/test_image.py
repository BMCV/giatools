import unittest
import unittest.mock

import numpy as np

import giatools.image

# This tests require that the `tifffile` package is installed.
assert giatools.io.tifffile is not None


# Define test image data
test1_data = np.random.randint(0, 255, (1, 2, 26, 32, 3), dtype=np.uint8)
test1_axes = 'TZYXC'
test1_original_axes = 'ZXYC'
test2_data = np.random.randint(0, 255, (1, 1, 32, 26, 1), dtype=np.uint8)
test2_axes = 'ZTYXC'
test2_original_axes = 'YXC'


class Image__read(unittest.TestCase):

    def test__input1(self):
        img = giatools.image.Image.read('tests/data/input1_uint8_yx.tif')
        self.assertEqual(img.data.mean(), 63.66848655158571)
        self.assertEqual(img.data.shape, (1, 1, 1, 265, 329, 1))
        self.assertEqual(img.original_axes, 'YX')
        self.assertEqual(img.axes, 'QTZYXC')

    def test__input2(self):
        img = giatools.image.Image.read('tests/data/input2_uint8_yx.tif')
        self.assertEqual(img.data.mean(), 9.543921821305842)
        self.assertEqual(img.data.shape, (1, 1, 1, 96, 97, 1))
        self.assertEqual(img.original_axes, 'YX')
        self.assertEqual(img.axes, 'QTZYXC')

    def test__input3(self):
        img = giatools.image.Image.read('tests/data/input3_uint16_zyx.tif')
        self.assertEqual(img.data.shape, (1, 1, 5, 198, 356, 1))
        self.assertEqual(img.data.mean(), 1259.6755334241288)
        self.assertEqual(img.original_axes, 'ZYX')
        self.assertEqual(img.axes, 'QTZYXC')

    def test__input4(self):
        img = giatools.image.Image.read('tests/data/input4_uint8.png')
        self.assertEqual(img.data.shape, (1, 1, 1, 10, 10, 3))
        self.assertEqual(img.data.mean(), 130.04)
        self.assertEqual(img.original_axes, 'YXC')
        self.assertEqual(img.axes, 'QTZYXC')

    def test__input5(self):
        img = giatools.image.Image.read('tests/data/input5_uint8_cyx.tif')
        self.assertEqual(img.data.shape, (1, 1, 1, 8, 16, 2))
        self.assertEqual(img.data.mean(), 22.25390625)
        self.assertEqual(img.original_axes, 'CYX')
        self.assertEqual(img.axes, 'QTZYXC')

    def test__input6(self):
        img = giatools.image.Image.read('tests/data/input6_uint8_zyx.tif')
        self.assertEqual(img.data.shape, (1, 1, 25, 8, 16, 1))
        self.assertEqual(img.data.mean(), 26.555)
        self.assertEqual(img.original_axes, 'ZYX')
        self.assertEqual(img.axes, 'QTZYXC')

    def test__input7(self):
        img = giatools.image.Image.read('tests/data/input7_uint8_zcyx.tif')
        self.assertEqual(img.data.shape, (1, 1, 25, 50, 50, 2))
        self.assertEqual(img.data.mean(), 14.182152)
        self.assertEqual(img.original_axes, 'ZCYX')
        self.assertEqual(img.axes, 'QTZYXC')

    def test__input8(self):
        img = giatools.image.Image.read('tests/data/input8_uint16_tyx.tif')
        self.assertEqual(img.data.shape, (1, 5, 1, 49, 56, 1))
        self.assertEqual(img.data.mean(), 5815.486880466472)
        self.assertEqual(img.original_axes, 'TYX')
        self.assertEqual(img.axes, 'QTZYXC')


@unittest.mock.patch('giatools.io.imwrite')
class Image__write(unittest.TestCase):
    def setUp(self):
        self.img1 = giatools.image.Image(data=test1_data.copy(), axes=test1_axes, original_axes=test1_original_axes)

    def test(self, mock_imwrite):
        self.img1.write('test_output.tiff')
        mock_imwrite.assert_called_once()
        np.testing.assert_array_equal(mock_imwrite.call_args_list[0][0][0], test1_data)
        self.assertEqual(mock_imwrite.call_args_list[0][0][1], 'test_output.tiff')
        self.assertEqual(mock_imwrite.call_args_list[0][1], dict(backend='auto', metadata=dict(axes=test1_axes)))

    def test__tifffile(self, mock_imwrite):
        self.img1.write('test_output.tiff', backend='tifffile')
        mock_imwrite.assert_called_once()
        np.testing.assert_array_equal(mock_imwrite.call_args_list[0][0][0], test1_data)
        self.assertEqual(mock_imwrite.call_args_list[0][0][1], 'test_output.tiff')
        self.assertEqual(mock_imwrite.call_args_list[0][1], dict(backend='tifffile', metadata=dict(axes=test1_axes)))


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
        self.assertEqual(img_reordered.data.shape, test1_data.shape)
        self.assertEqual(img_reordered.original_axes, test1_original_axes)

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


class Image__squeeze_like(unittest.TestCase):

    def setUp(self):
        self.img1 = giatools.image.Image(data=test1_data, axes=test1_axes, original_axes=test1_original_axes)
        self.img2 = giatools.image.Image(data=test2_data, axes=test2_axes, original_axes=test2_original_axes)

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
