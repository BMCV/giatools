import unittest
import unittest.mock

import numpy as np

import giatools.image

# This tests require that the `tifffile` package is installed.
assert giatools.io.tifffile is not None


# Define test image data
test_data1 = np.random.randint(0, 255, (1, 2, 26, 32, 3), dtype=np.uint8)
test_axes1 = 'TZYXC'
test_data2 = np.random.randint(0, 255, (1, 1, 32, 26, 1), dtype=np.uint8)
test_axes2 = 'ZTYXC'


class Image__read(unittest.TestCase):

    def test__input1(self):
        img = giatools.image.Image.read('tests/data/input1_uint8_yx.tif')
        self.assertEqual(img.data.mean(), 63.66848655158571)
        self.assertEqual(img.data.shape, (1, 1, 265, 329, 1))
        self.assertEqual(img.axes, 'YX')


class Image__reorder_axes_like(unittest.TestCase):

    def setUp(self):
        self.img1 = giatools.image.Image(data=test_data1.copy(), axes=test_axes1)

    def test(self):
        img_reordered = self.img1.reorder_axes_like('ZTCYX')
        self.assertEqual(img_reordered.axes, 'ZTCYX')
        self.assertEqual(img_reordered.data.shape, (2, 1, 3, 26, 32))

    def test__identity(self):
        img_reordered = self.img1.reorder_axes_like(test_axes1)
        self.assertEqual(img_reordered.axes, test_axes1)
        self.assertEqual(img_reordered.data.shape, test_data1.shape)

    def test__spurious_axis(self):
        with self.assertRaises(AssertionError):
            self.img1.reorder_axes_like('ZTCYXW')

    def test__missing_axis(self):
        with self.assertRaises(AssertionError):
            self.img1.reorder_axes_like('ZTCY')

    def test__ambigious_axis(self):
        with self.assertRaises(AssertionError):
            self.img1.reorder_axes_like('ZTCYXX')
        with self.assertRaises(AssertionError):
            self.img1.reorder_axes_like('ZTCXX')


class Image__squeeze_like(unittest.TestCase):

    def setUp(self):
        self.img1 = giatools.image.Image(data=test_data1, axes=test_axes1)
        self.img2 = giatools.image.Image(data=test_data2, axes=test_axes2)

    def test__no_squeeze(self):
        img_squeezed = self.img1.squeeze_like('ZTCYX')
        self.assertEqual(img_squeezed.axes, 'ZTCYX')
        self.assertEqual(img_squeezed.data.shape, (2, 1, 3, 26, 32))

    def test__squeeze_1(self):
        img_squeezed = self.img1.squeeze_like('ZCYX')
        self.assertEqual(img_squeezed.axes, 'ZCYX')
        self.assertEqual(img_squeezed.data.shape, (2, 3, 26, 32))

    def test__squeeze_2(self):
        img_squeezed = self.img2.squeeze_like('XYC')
        self.assertEqual(img_squeezed.axes, 'XYC')
        self.assertEqual(img_squeezed.data.shape, (26, 32, 1))

    def test__squeeze_3(self):
        img_squeezed = self.img2.squeeze_like('XY')
        self.assertEqual(img_squeezed.axes, 'XY')
        self.assertEqual(img_squeezed.data.shape, (26, 32))

    def test__squeeze_illegal_axis(self):
        with self.assertRaises(ValueError):
            self.img1.squeeze_like('TCYX')

    def test__spurious_axis(self):
        with self.assertRaises(AssertionError):
            self.img1.squeeze_like('ZCYXW')

    def test__ambigious_axis(self):
        with self.assertRaises(AssertionError):
            self.img1.squeeze_like('ZCYXX')
