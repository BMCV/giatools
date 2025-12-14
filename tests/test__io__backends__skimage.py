import unittest
import unittest.mock

import giatools.io
import giatools.io._backends.skimage


def mock_array(*shape, name: str = 'array'):
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            array = unittest.mock.MagicMock(shape=shape, ndim=len(shape))
            kwargs = dict(kwargs)
            kwargs[name] = array
            return test_func(*args, **kwargs)
        return wrapper
    return decorator


@unittest.mock.patch('giatools.io._backends.skimage.skimage.io.imsave')
class SKImageWriter__write(unittest.TestCase):

    def setUp(self):
        self.writer = giatools.io._backends.skimage.SKImageWriter()

    @mock_array(10, 10, 10)
    def test__png__invalid__zyx(self, mock_imsave, array):
        with self.assertRaises(giatools.io.IncompatibleDataError):
            self.writer.write(array, filepath='image.png', metadata=dict(axes='ZYX'))

    @mock_array(10, 10)
    def test__png__invalid__zx(self, mock_imsave, array):
        with self.assertRaises(giatools.io.IncompatibleDataError):
            self.writer.write(array, filepath='image.png', metadata=dict(axes='ZX'))

    @mock_array(10, 10)
    def test__png__valid__yx(self, mock_imsave, array):
        self.writer.write(array, filepath='image.png', metadata=dict(axes='YX'))
        mock_imsave.assert_called()

    @mock_array(10, 10, 3)
    def test__png__valid__yxc_rgb(self, mock_imsave, array):
        self.writer.write(array, filepath='image.png', metadata=dict(axes='YXC'))
        mock_imsave.assert_called()

    @mock_array(10, 10, 4)
    def test__png__valid__yxc_rgba(self, mock_imsave, array):
        self.writer.write(array, filepath='image.png', metadata=dict(axes='YXC'))
        mock_imsave.assert_called()

    @mock_array(10, 10, 1)
    def test__png__valid__yxc_gray(self, mock_imsave, array):
        self.writer.write(array, filepath='image.png', metadata=dict(axes='YXC'))
        mock_imsave.assert_called()
