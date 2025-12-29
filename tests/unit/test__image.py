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
        self._np = unittest.mock.patch(
            'giatools.image._np'
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


class Image__shape(ImageTestCase):

    def test(self):
        self.assertEqual(self.img1.shape, self.img1_shape)


class Image__original_shape(ImageTestCase):

    def test(self):
        self.assertEqual(self.img1.original_shape, (2, 32, 26, 3))


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


class Image__clip_to_dtype(ImageTestCase):

    def test__bool(self):
        with self.assertRaises(TypeError):
            self.img1.clip_to_dtype(bool)

    def test__to_superset_int(self):
        self.img1.data.min.return_value.item.return_value = -15
        self.img1.data.max.return_value.item.return_value = +15
        self._np.issubdtype.return_value = True  # target dtype is an integer type
        self._np.iinfo.return_value.min = -15
        self._np.iinfo.return_value.max = +15
        img_clipped = self.img1.clip_to_dtype('mocked-int-type')
        self.assertIs(img_clipped, self.img1)
        self.img1.data.copy.assert_not_called()

    def test__to_superset_float(self):
        self.img1.data.min.return_value.item.return_value = -15
        self.img1.data.max.return_value.item.return_value = +15
        self._np.issubdtype.return_value = False  # target dtype is a float type
        self._np.finfo.return_value.min.item.return_value = -15.
        self._np.finfo.return_value.max.item.return_value = +15.
        img_clipped = self.img1.clip_to_dtype('mocked-float-type')
        self.assertIs(img_clipped, self.img1)
        self.img1.data.copy.assert_not_called()
