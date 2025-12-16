import unittest
import unittest.mock

import giatools.io
import giatools.io._backends.tiff

from .tools import (
    filenames,
    mock_array,
)


@unittest.mock.patch('giatools.io._backends.tiff.tifffile.imwrite')
class TiffWriter__write(unittest.TestCase):

    def setUp(self):
        self.writer = giatools.io._backends.tiff.TiffWriter()

    @mock_array(10, 10, 1)
    @filenames('tif', 'tiff')
    def test__valid(self, mock_imwrite, array, filename):
        metadata = dict(z_spacing=0.5, unit='cm', axes='YXC')
        self.writer.write(array, filepath=filename, metadata=dict(metadata))
        mock_imwrite.assert_called()
        self.assertEqual(
            mock_imwrite.call_args.kwargs['metadata'],
            dict(
                spacing=metadata['z_spacing'],
                unit=metadata['unit'],
                axes='YXC',
            ),
        )


class TiffReader__get_image_metadata(unittest.TestCase):

    def test(self):
        ...
