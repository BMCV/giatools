import unittest
import unittest.mock

import giatools.io
import giatools.io._backends.skimage
import giatools.metadata

from ..tools import (
    filenames,
    mock_array,
)


class MockedTestCase(unittest.TestCase):

    def setUp(self):
        self._skimage_io = unittest.mock.patch(
            'giatools.io._backends.skimage._skimage_io'
        ).start()

        self.addCleanup(unittest.mock.patch.stopall)


class SKImageWriter__write(MockedTestCase):

    def setUp(self):
        super().setUp()
        self.writer = giatools.io._backends.skimage.SKImageWriter()

    # -----------------------------------------------------------------------------------------------------------------
    # PNG tests
    # -----------------------------------------------------------------------------------------------------------------

    @mock_array(10, 10, 10)
    @filenames('png')
    def test__png__invalid__zyx(self, array, filename):
        with self.assertRaises(giatools.io.IncompatibleDataError):
            self.writer.write(array, filepath=filename, axes='ZYX', metadata=giatools.metadata.Metadata())

    @mock_array(10, 10)
    @filenames('png')
    def test__png__invalid__zx(self, array, filename):
        with self.assertRaises(giatools.io.IncompatibleDataError):
            self.writer.write(array, filepath=filename, axes='ZX', metadata=giatools.metadata.Metadata())

    @mock_array(10, 10)
    @filenames('png')
    def test__png__valid__yx(self, array, filename):
        self.writer.write(array, filepath=filename, axes='YX', metadata=giatools.metadata.Metadata())
        self._skimage_io.imsave.assert_called()

    @mock_array(10, 10, 3)
    @filenames('png')
    def test__png__valid__yxc_rgb(self, array, filename):
        self.writer.write(array, filepath=filename, axes='YXC', metadata=giatools.metadata.Metadata())
        self._skimage_io.imsave.assert_called()

    @mock_array(10, 10, 4)
    @filenames('png')
    def test__png__valid__yxc_rgba(self, array, filename):
        self.writer.write(array, filepath=filename, axes='YXC', metadata=giatools.metadata.Metadata())
        self._skimage_io.imsave.assert_called()

    @mock_array(10, 10, 1)
    @filenames('png')
    def test__png__valid__yxc_gray(self, array, filename):
        self.writer.write(array, filepath=filename, axes='YXC', metadata=giatools.metadata.Metadata())
        self._skimage_io.imsave.assert_called()

    # -----------------------------------------------------------------------------------------------------------------
    # JPEG tests
    # -----------------------------------------------------------------------------------------------------------------

    @mock_array(10, 10, 10)
    @filenames('jpg', 'jpeg')
    def test__jpg__invalid__zyx(self, array, filename):
        with self.assertRaises(giatools.io.IncompatibleDataError):
            self.writer.write(array, filepath=filename, axes='ZYX', metadata=giatools.metadata.Metadata())

    @mock_array(10, 10, 3)
    @filenames('jpg', 'jpeg')
    def test__jpg__invalid__zxc(self, array, filename):
        with self.assertRaises(giatools.io.IncompatibleDataError):
            self.writer.write(array, filepath=filename, axes='ZXC', metadata=giatools.metadata.Metadata())

    @mock_array(10, 10)
    @filenames('jpg', 'jpeg')
    def test__jpg__invalid__yx(self, array, filename):
        with self.assertRaises(giatools.io.IncompatibleDataError):
            self.writer.write(array, filepath=filename, axes='YX', metadata=giatools.metadata.Metadata())

    @mock_array(10, 10, 1)
    @filenames('jpg', 'jpeg')
    def test__jpg__invalid__yxc_gray(self, array, filename):
        with self.assertRaises(giatools.io.IncompatibleDataError):
            self.writer.write(array, filepath=filename, axes='YXC', metadata=giatools.metadata.Metadata())

    @mock_array(10, 10, 4)
    @filenames('jpg', 'jpeg')
    def test__jpg__invalid__yxc_rgba(self, array, filename):
        with self.assertRaises(giatools.io.IncompatibleDataError):
            self.writer.write(array, filepath=filename, axes='YXC', metadata=giatools.metadata.Metadata())

    @mock_array(10, 10, 3)
    @filenames('jpg', 'jpeg')
    def test__jpg__valid__yxc_rgb(self, array, filename):
        self.writer.write(array, filepath=filename, axes='YXC', metadata=giatools.metadata.Metadata())
        self._skimage_io.imsave.assert_called()

    @mock_array(10, 10, 3)
    @filenames('jpg', 'jpeg')
    def test__jpg__valid__yxc_rgb__with_quality(self, array, filename):
        self.writer.write(array, filepath=filename, axes='YXC', metadata=giatools.metadata.Metadata(), quality=90)
        self._skimage_io.imsave.assert_called()
        self.assertEqual(self._skimage_io.imsave.call_args.kwargs['quality'], 90)
