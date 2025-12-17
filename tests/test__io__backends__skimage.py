import unittest
import unittest.mock

import giatools.io
import giatools.io._backends.skimage
import giatools.metadata

from .tools import (
    filenames,
    mock_array,
    validate_metadata,
)


@unittest.mock.patch('giatools.io._backends.skimage._skimage_io.imsave')
class SKImageWriter__write(unittest.TestCase):

    def setUp(self):
        self.writer = giatools.io._backends.skimage.SKImageWriter()

    # -----------------------------------------------------------------------------------------------------------------
    # PNG tests
    # -----------------------------------------------------------------------------------------------------------------

    @mock_array(10, 10, 10)
    @filenames('png')
    def test__png__invalid__zyx(self, mock_imsave, array, filename):
        with self.assertRaises(giatools.io.IncompatibleDataError):
            self.writer.write(array, filepath=filename, axes='ZYX', metadata=giatools.metadata.Metadata())

    @mock_array(10, 10)
    @filenames('png')
    def test__png__invalid__zx(self, mock_imsave, array, filename):
        with self.assertRaises(giatools.io.IncompatibleDataError):
            self.writer.write(array, filepath=filename, axes='ZX', metadata=giatools.metadata.Metadata())

    @mock_array(10, 10)
    @filenames('png')
    def test__png__valid__yx(self, mock_imsave, array, filename):
        self.writer.write(array, filepath=filename, axes='YX', metadata=giatools.metadata.Metadata())
        mock_imsave.assert_called()

    @mock_array(10, 10, 3)
    @filenames('png')
    def test__png__valid__yxc_rgb(self, mock_imsave, array, filename):
        self.writer.write(array, filepath=filename, axes='YXC', metadata=giatools.metadata.Metadata())
        mock_imsave.assert_called()

    @mock_array(10, 10, 4)
    @filenames('png')
    def test__png__valid__yxc_rgba(self, mock_imsave, array, filename):
        self.writer.write(array, filepath=filename, axes='YXC', metadata=giatools.metadata.Metadata())
        mock_imsave.assert_called()

    @mock_array(10, 10, 1)
    @filenames('png')
    def test__png__valid__yxc_gray(self, mock_imsave, array, filename):
        self.writer.write(array, filepath=filename, axes='YXC', metadata=giatools.metadata.Metadata())
        mock_imsave.assert_called()

    # -----------------------------------------------------------------------------------------------------------------
    # JPEG tests
    # -----------------------------------------------------------------------------------------------------------------

    @mock_array(10, 10, 10)
    @filenames('jpg', 'jpeg')
    def test__jpg__invalid__zyx(self, mock_imsave, array, filename):
        with self.assertRaises(giatools.io.IncompatibleDataError):
            self.writer.write(array, filepath=filename, axes='ZYX', metadata=giatools.metadata.Metadata())

    @mock_array(10, 10, 3)
    @filenames('jpg', 'jpeg')
    def test__jpg__invalid__zxc(self, mock_imsave, array, filename):
        with self.assertRaises(giatools.io.IncompatibleDataError):
            self.writer.write(array, filepath=filename, axes='ZXC', metadata=giatools.metadata.Metadata())

    @mock_array(10, 10)
    @filenames('jpg', 'jpeg')
    def test__jpg__invalid__yx(self, mock_imsave, array, filename):
        with self.assertRaises(giatools.io.IncompatibleDataError):
            self.writer.write(array, filepath=filename, axes='YX', metadata=giatools.metadata.Metadata())

    @mock_array(10, 10, 1)
    @filenames('jpg', 'jpeg')
    def test__jpg__invalid__yxc_gray(self, mock_imsave, array, filename):
        with self.assertRaises(giatools.io.IncompatibleDataError):
            self.writer.write(array, filepath=filename, axes='YXC', metadata=giatools.metadata.Metadata())

    @mock_array(10, 10, 4)
    @filenames('jpg', 'jpeg')
    def test__jpg__invalid__yxc_rgba(self, mock_imsave, array, filename):
        with self.assertRaises(giatools.io.IncompatibleDataError):
            self.writer.write(array, filepath=filename, axes='YXC', metadata=giatools.metadata.Metadata())

    @mock_array(10, 10, 3)
    @filenames('jpg', 'jpeg')
    def test__jpg__valid__yxc_rgb(self, mock_imsave, array, filename):
        self.writer.write(array, filepath=filename, axes='YXC', metadata=giatools.metadata.Metadata())
        mock_imsave.assert_called()

    @mock_array(10, 10, 3)
    @filenames('jpg', 'jpeg')
    def test__jpg__valid__yxc_rgb__with_quality(self, mock_imsave, array, filename):
        self.writer.write(array, filepath=filename, axes='YXC', metadata=giatools.metadata.Metadata(), quality=90)
        mock_imsave.assert_called()
        self.assertEqual(mock_imsave.call_args.kwargs['quality'], 90)


class SKImageReader(unittest.TestCase):

    def test__valid__png(self):
        with giatools.io._backends.skimage.SKImageReader('tests/data/input4_uint8.png') as reader:
            self.assertEqual(reader.get_num_images(), 1)
            im = reader.select_image(0)
            self.assertEqual(reader.get_axes(im), 'YXC')
            validate_metadata(self, reader.get_image_metadata(im))
            arr = reader.get_image_data(im)
            self.assertEqual(arr.shape, (10, 10, 3))
            self.assertEqual(round(arr.mean(), 2), 130.04)

    def test__valid__tiff(self):
        with giatools.io._backends.skimage.SKImageReader('tests/data/input1_uint8_yx.tiff') as reader:
            self.assertEqual(reader.get_num_images(), 1)
            im = reader.select_image(0)
            self.assertEqual(reader.get_axes(im), 'YX')
            validate_metadata(self, reader.get_image_metadata(im))
            arr = reader.get_image_data(im)
            self.assertEqual(arr.shape, (265, 329))
            self.assertEqual(round(arr.mean(), 2), 63.67)

    def test__invalid(self):
        with giatools.io._backends.skimage.SKImageReader('tests/data/input7_uint8_zcyx.tif') as reader:
            self.assertEqual(reader.get_num_images(), 1)
            with self.assertRaises(giatools.io.UnsupportedFileError):
                reader.select_image(0)
