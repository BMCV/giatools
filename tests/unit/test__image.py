"""
Unit tests for the `giatools.image` module.
"""

import unittest
import unittest.mock

import giatools.image

from ..tools import (
    maximum_python_version,
    minimum_python_version,
)


class default_normalized_axes(unittest.TestCase):

    def test(self):
        self.assertEqual(giatools.image.default_normalized_axes, 'QTZYXC')


class ImageTestCase(unittest.TestCase):

    def setUp(self):
        super().setUp()

        self.io_imreadraw = unittest.mock.patch(
            'giatools.io.imreadraw'
        ).start()
        self.io_imwrite = unittest.mock.patch(
            'giatools.io.imwrite'
        ).start()

        self.addCleanup(unittest.mock.patch.stopall)

        self.img1_shape = (1, 2, 26, 32, 3)
        self.img1_axes = 'TZYXC'
        self.img1_original_axes = 'ZXYC'
        self.img1 = giatools.image.Image(
            data=unittest.mock.MagicMock(shape=self.img1_shape),
            axes=self.img1_axes,
            original_axes=self.img1_original_axes,
            metadata=unittest.mock.MagicMock(),
        )


class Image__read(ImageTestCase):

    @unittest.mock.patch('giatools.image.Image.normalize_axes_like')
    def test(self, mock_normalize_axes_like):
        self.io_imreadraw.return_value = (
            unittest.mock.MagicMock(),  # data array
            'YX',
            unittest.mock.MagicMock(),  # metadata
        )
        img = giatools.image.Image.read('filepath')
        mock_normalize_axes_like.assert_called_once_with(giatools.image.default_normalized_axes)
        self.assertIs(img, mock_normalize_axes_like.return_value)

    def test__without_normalization(self):
        self.io_imreadraw.return_value = (
            unittest.mock.MagicMock(),  # data array
            'YX',
            unittest.mock.MagicMock(),  # metadata
        )
        img = giatools.image.Image.read('filepath', normalize_axes=None)
        self.assertIs(img.data, self.io_imreadraw.return_value[0])
        self.assertEqual(img.axes, self.io_imreadraw.return_value[1])
        self.assertIs(img.metadata, self.io_imreadraw.return_value[2])


class Image__write(ImageTestCase):

    def test(self):
        self.img1.write('test_output.tiff')
        self.assertTrue(self.io_imwrite.called)
        self.assertIs(self.io_imwrite.call_args_list[0][0][0], self.img1.data)
        self.assertEqual(self.io_imwrite.call_args_list[0][0][1], 'test_output.tiff')
        self.assertEqual(
            self.io_imwrite.call_args_list[0][1],
            dict(backend='auto', axes=self.img1_axes, metadata=self.img1.metadata),
        )

    def test__with_backend(self):
        self.img1.write('test_output.tiff', backend='some_backend')
        self.assertTrue(self.io_imwrite.called)
        self.assertIs(self.io_imwrite.call_args_list[0][0][0], self.img1.data)
        self.assertEqual(self.io_imwrite.call_args_list[0][0][1], 'test_output.tiff')
        self.assertEqual(
            self.io_imwrite.call_args_list[0][1],
            dict(backend='some_backend', axes=self.img1_axes, metadata=self.img1.metadata),
        )

    def test__metadata(self):
        self.img1.metadata.z_spacing = 0.5
        self.img1.write('test_output.tiff')
        self.assertTrue(self.io_imwrite.called)
        self.assertIs(self.io_imwrite.call_args_list[0][0][0], self.img1.data)
        self.assertEqual(self.io_imwrite.call_args_list[0][0][1], 'test_output.tiff')
        self.assertEqual(
            self.io_imwrite.call_args_list[0][1],
            dict(
                backend='auto',
                axes=self.img1_axes,
                metadata=self.img1.metadata,
            ),
        )

    def test__invalid_axes(self):
        img_invalid = giatools.image.Image(data=unittest.mock.MagicMock(shape=(2, 26, 32, 3)), axes='ZYX')
        with self.assertRaises(ValueError):
            img_invalid.write('test_output.tiff')
        self.assertFalse(self.io_imwrite.called)


class Image__reorder_axes_like(ImageTestCase):

    def test__identity(self):
        img_reordered = self.img1.reorder_axes_like(self.img1_axes)
        self.assertEqual(img_reordered.axes, self.img1_axes)
        self.assertEqual(img_reordered.original_axes, self.img1_original_axes)
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


@unittest.mock.patch('giatools.image.Image.squeeze_like')
class Image__squeeze(ImageTestCase):

    def test__zyxc(self, mock_squeeze_like):
        img1_squeezed = self.img1.squeeze()
        mock_squeeze_like.assert_called_once_with('ZYXC')
        self.assertIs(img1_squeezed, mock_squeeze_like.return_value)

    def test__yx(self, mock_squeeze_like):
        self.img1.data.shape = (1, 1, 26, 32, 1)
        img1_squeezed = self.img1.squeeze()
        mock_squeeze_like.assert_called_once_with('YX')
        self.assertIs(img1_squeezed, mock_squeeze_like.return_value)


class Image__squeeze_like(ImageTestCase):

    def test__squeeze_illegal_axis(self):
        with self.assertRaises(ValueError):
            self.img1.squeeze_like('TCYX')

    def test__spurious_axis(self):
        with self.assertRaises(ValueError):
            self.img1.squeeze_like('ZCYXW')

    def test__ambigious_axis(self):
        with self.assertRaises(ValueError):
            self.img1.squeeze_like('ZCYXX')


class Image__normalize_axes_like(ImageTestCase):

    def test__ambiguous_axes(self):
        with self.assertRaises(AssertionError):
            self.img1.normalize_axes_like('ZTCYXX')


class Image__iterate_jointly(ImageTestCase):

    def setUp(self):
        super().setUp()
        self.img1.axes = 'YX'

    @maximum_python_version(3, 10)
    def test__minimum_python_version(self):
        with self.assertRaises(RuntimeError):
            for _ in self.img1.iterate_jointly('YX'):
                pass

    @minimum_python_version(3, 11)
    def test__empty(self):
        with self.assertRaises(ValueError):
            for _ in self.img1.iterate_jointly(''):
                pass

    @minimum_python_version(3, 11)
    def test__spurious_axis(self):
        with self.assertRaises(ValueError):
            for _ in self.img1.iterate_jointly('YXZ'):
                pass
        with self.assertRaises(ValueError):
            for _ in self.img1.iterate_jointly('Z'):
                pass
