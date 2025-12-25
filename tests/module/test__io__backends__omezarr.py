"""
Module tests for the `giatools.io._backends.omezarr` module.
"""

import unittest

from ..tools import (
    minimum_python_version,
    validate_metadata,
    without_logging,
)


class OMEZarrReader(unittest.TestCase):

    @minimum_python_version(3, 11)
    @without_logging
    def test__valid__zarr(self):
        import giatools.io._backends.omezarr
        with giatools.io._backends.omezarr.OMEZarrReader('tests/data/ome-zarr-examples/image-02.zarr') as reader:
            self.assertEqual(reader.get_num_images(), 2)
            im = reader.select_image(0)
            self.assertEqual(reader.get_axes(im), 'YX')
            validate_metadata(self, reader.get_image_metadata(im), resolution=(1., 1.), unit='um')
            arr = reader.get_image_data(im)
            self.assertEqual(arr.shape, (200, 200))
            self.assertAlmostEqual(round(arr.mean().compute(), 2), 502.26)

    @minimum_python_version(3, 11)
    @without_logging
    def test__invalid__zarr(self):
        import giatools.io._backends.omezarr
        with giatools.io._backends.omezarr.OMEZarrReader('tests/data/ome-zarr-examples/image-02.zarr') as reader:
            self.assertEqual(reader.get_num_images(), 2)
            im = reader.select_image(1)
            with self.assertRaises(giatools.io.CorruptFileError):
                reader.get_axes(im)  # OME-Zarr node is missing axes information

    @minimum_python_version(3, 11)
    @without_logging
    def test__invalid(self):
        import giatools.io._backends.omezarr
        for filename in (
            'input4_uint8.png',
            'input1_uint8_yx.tiff',
        ):
            with self.subTest(filename=filename):
                with self.assertRaises(giatools.io.UnsupportedFileError):
                    with giatools.io._backends.omezarr.OMEZarrReader(f'tests/data/{filename}'):
                        pass
