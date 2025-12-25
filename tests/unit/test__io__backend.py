"""
Unit tests for the `giatools.io.backend` module.
"""

import unittest
import unittest.mock

import attrs

import giatools.io
import giatools.io.backend
import giatools.metadata

invalid_axes = (
    'YYX',   # error: Y axis given twice
    'YXCS',  # error: C and S are mutually exclusive
    'YXM',   # error: unsupported axis M
    'Y',     # error: missing X axis
    '',      # error: missing X and Y axes
)


class BackendTestCase(unittest.TestCase):

    def setUp(self):
        self.reader_cls = unittest.mock.MagicMock()
        self.writer_cls = unittest.mock.MagicMock()
        self.backend = giatools.io.backend.Backend('name', self.reader_cls, self.writer_cls)

        self._os_path_exists = unittest.mock.patch(
            'giatools.io.backend._os.path.exists'
        ).start()
        self._os_path_exists.return_value = True

        self.addCleanup(unittest.mock.patch.stopall)

    @property
    def reader_init(self):
        return self.reader_cls

    @property
    def reader_enter(self):
        return self.reader_cls.return_value.__enter__

    @property
    def reader_get_num_images(self):
        return self.reader_cls.return_value.__enter__.return_value.get_num_images

    @property
    def reader_get_axes(self):
        return self.reader_cls.return_value.__enter__.return_value.get_axes

    @property
    def reader_get_image_data(self):
        return self.reader_cls.return_value.__enter__.return_value.get_image_data

    @property
    def reader_get_image_metadata(self):
        return self.reader_cls.return_value.__enter__.return_value.get_image_metadata

    @property
    def writer_write(self):
        return self.writer_cls.return_value.write


class Backend(BackendTestCase):

    def test__str__(self):
        self.assertEqual(str(self.backend), 'name')

    def test__repr__(self):
        self.assertEqual(repr(self.backend), "<name Backend>")


class Backend__peek_num_images_in_file(BackendTestCase):

    def test__missing_file(self):
        self._os_path_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            self.backend.peek_num_images_in_file('nonexistent_file.tiff')

    def test__unsupported_file(self):
        for error_in in (
            'reader_init',
            'reader_enter',
            'reader_get_num_images',
        ):
            with self.subTest(error_in=error_in):
                mock = getattr(self, error_in)
                mock.side_effect = giatools.io.UnsupportedFileError('some_file')
                self.assertIsNone(self.backend.peek_num_images_in_file('some_file'))

    def test__valid(self):
        self.reader_get_num_images.return_value = 5
        self.assertEqual(self.backend.peek_num_images_in_file('some_file'), 5)


class Backend__read(BackendTestCase):

    def test__missing_file(self):
        self._os_path_exists.return_value = False
        with self.assertRaises(FileNotFoundError):
            self.backend.read('nonexistent_file.tiff')

    def test__unsupported_file(self):
        for error_in in (
            'reader_init',
            'reader_enter',
            'reader_get_num_images',
        ):
            with self.subTest(error_in=error_in):
                mock = getattr(self, error_in)
                mock.side_effect = giatools.io.UnsupportedFileError('some_file')
                self.assertIsNone(self.backend.read('some_file'))

    def test__invalid_position(self):
        self.reader_get_num_images.return_value = 3
        with self.assertRaises(IndexError):
            self.backend.read('some_file', position=3)

    def test__invalid_axes(self):
        self.reader_get_num_images.return_value = 1
        for axes in invalid_axes:
            with self.subTest(axes=axes):
                self.reader_get_axes.return_value = axes
                with self.assertRaises(giatools.io.CorruptFileError):
                    self.backend.read('some_file')

    def test__valid(self):
        self.reader_get_num_images.return_value = 1
        for axes in ('YXC', 'YXS'):
            self.reader_get_axes.return_value = axes
            im_arr, im_axes, metadata = self.backend.read('some_file')
            self.assertIs(im_arr, self.reader_get_image_data.return_value)
            self.assertEqual(im_axes, 'YXC')
            self.assertEqual(metadata, self.reader_get_image_metadata.return_value)


class Backend__write(BackendTestCase):

    def setUp(self):
        super().setUp()
        self.im_arr = unittest.mock.MagicMock()

    def test__invalid_axes(self):
        for axes in invalid_axes:
            with self.subTest(axes=axes):
                with self.assertRaises(ValueError):
                    self.backend.write(self.im_arr, 'some_file', axes=axes, metadata=giatools.metadata.Metadata())

    def test__missing_metadata(self):
        with self.assertRaises(ValueError):
            self.backend.write(self.im_arr, 'some_file', axes='YX', metadata=None)

    def test__valid(self):
        kwarg = unittest.mock.MagicMock()
        for axes in ('YXC', 'YXS'):
            metadata = giatools.metadata.Metadata()
            metadata_copy = attrs.asdict(metadata)
            with self.subTest(axes=axes):
                self.backend.write(self.im_arr, 'some_file', axes=axes, metadata=metadata, kwarg=kwarg)
                self.writer_write.assert_called()
                self.assertIs(self.writer_write.call_args.args[0], self.im_arr)
                self.assertIs(self.writer_write.call_args.args[1], 'some_file')
                self.assertEqual(self.writer_write.call_args.args[2], 'YXC')
                self.assertIsNot(self.writer_write.call_args.args[3], metadata)
                self.assertEqual(attrs.asdict(self.writer_write.call_args.args[3]), metadata_copy)
                self.assertEqual(self.writer_write.call_args.kwargs, dict(kwarg=kwarg))


class Reader(unittest.TestCase):

    def test_open(self):
        with self.assertRaises(NotImplementedError):
            giatools.io.backend.Reader().open()

    def test_get_num_images(self):
        with self.assertRaises(NotImplementedError):
            giatools.io.backend.Reader().get_num_images()

    def test_select_image(self):
        with self.assertRaises(NotImplementedError):
            giatools.io.backend.Reader().select_image(0)

    def test_get_axes(self):
        with self.assertRaises(NotImplementedError):
            giatools.io.backend.Reader().get_axes(unittest.mock.MagicMock())

    def test_get_image_data(self):
        with self.assertRaises(NotImplementedError):
            giatools.io.backend.Reader().get_image_data(unittest.mock.MagicMock())

    def test_get_image_metadata(self):
        with self.assertRaises(NotImplementedError):
            giatools.io.backend.Reader().get_image_metadata(unittest.mock.MagicMock())


class Writer(unittest.TestCase):

    def test_write(self):
        with self.assertRaises(NotImplementedError):
            giatools.io.backend.Writer().write(
                unittest.mock.MagicMock(),
                'some_file',
                axes='YX',
                metadata=giatools.metadata.Metadata(),
            )
