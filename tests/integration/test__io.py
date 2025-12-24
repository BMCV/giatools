import unittest

import numpy as np

import giatools.io
import giatools.metadata

from ..tools import (
    minimum_python_version,
    random_io_test,
    validate_metadata,
)


class WriteAndReadTestCase(unittest.TestCase):
    """
    Verify that written images can be read back correctly with correct data and metadata.
    """

    def _test__write_and_read(self, axes: str, filepath: str, data: np.ndarray, metadata: dict):

        # Write the image
        giatools.io.imwrite(
            data,
            filepath,
            axes=axes,
            metadata=giatools.metadata.Metadata(**metadata),
        )

        # Peek into the file to check number of images
        num_images = giatools.io.peek_num_images_in_file(filepath)
        self.assertEqual(num_images, 1)

        # Read the image back and validate
        data1, axes1, metadata1 = giatools.io.imreadraw(filepath)
        np.testing.assert_array_equal(data1, data)
        self.assertEqual(axes1, axes)
        validate_metadata(self, metadata1, **metadata)

    @random_io_test(shape=(10, 10, 5, 2), dtype=np.float32, ext='tiff')
    def test__write_and_read__tiff(self, filepath: str, data: np.ndarray):
        self._test__write_and_read(
            'YXZC',
            filepath,
            data,
            dict(resolution=(0.2, 0.4), z_spacing=0.5, z_position=0.8, unit='km'),
        )

    @minimum_python_version(3, 11)
    @random_io_test(shape=(4, 10, 10, 5, 2), dtype=np.float32, ext='zarr')
    def test__write_and_read__zarr(self, filepath: str, data: np.ndarray):
        for metadata in (
            dict(resolution=(0.2, 0.4), z_spacing=0.5, unit='km'),
            dict(resolution=(0.2, 0.4), z_spacing=0.5),
        ):
            self._test__write_and_read('TCYXZ', filepath, data, metadata)
