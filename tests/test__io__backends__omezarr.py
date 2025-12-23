import unittest
import unittest.mock

import giatools.metadata

from .tools import (
    filenames,
    minimum_python_version,
    mock_array,
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

    @minimum_python_version(3, 11)
    @without_logging
    def test__get_image_metadata(self):
        import giatools.io._backends.omezarr
        for zarr_unit, unit in (
            ('nm', 'nm'),
            ('nanometer', 'nm'),
            ('um', 'um'),
            ('micrometer', 'um'),
            ('mm', 'mm'),
            ('millimeter', 'mm'),
        ):
            with self.subTest(zarr_unit=zarr_unit):
                image = unittest.mock.MagicMock()
                image.metadata = {
                    'axes': [
                        {'name': 'Y', 'unit': zarr_unit},
                        {'name': 'X', 'unit': zarr_unit},
                    ],
                    'coordinateTransformations': [
                        [
                            {
                                'type': 'scale',
                                'scale': [2, 2],
                            }
                        ]
                    ]
                }
                reader = giatools.io._backends.omezarr.OMEZarrReader()
                metadata = reader.get_image_metadata(image)
                self.assertEqual(metadata.unit, unit)
                self.assertEqual(metadata.resolution, (0.5, 0.5))
                self.assertIsNone(metadata.z_spacing)

    @minimum_python_version(3, 11)
    @without_logging
    def test__get_image_metadata__invalid(self):
        import giatools.io._backends.omezarr
        image = unittest.mock.MagicMock()
        image.metadata = {
            'axes': [
                {'name': 'Y', 'unit': None},
                {'name': 'X', 'unit': None},
            ],
            'coordinateTransformations': None
        }
        reader = giatools.io._backends.omezarr.OMEZarrReader()
        metadata = reader.get_image_metadata(image)
        validate_metadata(self, metadata)


@unittest.mock.patch('giatools.io._backends.omezarr._ome_zarr_writer.write_image')
class OMEZarrWriter__write(unittest.TestCase):

    @minimum_python_version(3, 11)
    def setUp(self):
        import giatools.io._backends.omezarr
        self._omezarr_backend = giatools.io._backends.omezarr
        self.writer = self._omezarr_backend.OMEZarrWriter()

    @minimum_python_version(3, 11)
    @mock_array(10, 10, 1)
    @filenames('zarr', 'ome.zarr')
    def test__yxc(self, mock_write_image, array, filename):
        metadata = dict(z_spacing=0.5, unit='cm')
        self.writer.write(array, filepath=filename, axes='YXC', metadata=giatools.metadata.Metadata(**metadata))
        mock_write_image.assert_called()
        self.assertEqual(
            mock_write_image.call_args.kwargs['axes'],
            [
                dict(name='Y', type='space', unit='cm'),
                dict(name='X', type='space', unit='cm'),
                dict(name='C', type='channel'),
            ]
        )
        self.assertEqual(
            mock_write_image.call_args.kwargs['coordinate_transformations'],
            [
                [
                    dict(type='scale', scale=[1.0, 1.0, 1.0]),
                ]
            ]
        )

    @minimum_python_version(3, 11)
    @mock_array(5, 8, 10)
    @filenames('zarr', 'ome.zarr')
    def test__zyx(self, mock_write_image, array, filename):
        metadata = dict(z_spacing=0.5, resolution=(0.4, 0.8), unit='cm')
        self.writer.write(array, filepath=filename, axes='ZYX', metadata=giatools.metadata.Metadata(**metadata))
        mock_write_image.assert_called()
        self.assertEqual(
            mock_write_image.call_args.kwargs['axes'],
            [
                dict(name='Z', type='space', unit='cm'),
                dict(name='Y', type='space', unit='cm'),
                dict(name='X', type='space', unit='cm'),
            ]
        )
        self.assertEqual(
            mock_write_image.call_args.kwargs['coordinate_transformations'],
            [
                [
                    dict(type='scale', scale=[0.5, 1.25, 2.5]),
                ]
            ]
        )

    @minimum_python_version(3, 11)
    @mock_array(12, 5, 8, 10)
    @filenames('zarr', 'ome.zarr')
    def test__tzyx(self, mock_write_image, array, filename):
        metadata = dict(z_spacing=0.5, resolution=(0.4, 0.8), unit='cm')
        self.writer.write(array, filepath=filename, axes='TZYX', metadata=giatools.metadata.Metadata(**metadata))
        mock_write_image.assert_called()
        self.assertEqual(
            mock_write_image.call_args.kwargs['axes'],
            [
                dict(name='T', type='time'),
                dict(name='Z', type='space', unit='cm'),
                dict(name='Y', type='space', unit='cm'),
                dict(name='X', type='space', unit='cm'),
            ]
        )
        self.assertEqual(
            mock_write_image.call_args.kwargs['coordinate_transformations'],
            [
                [
                    dict(type='scale', scale=[1.0, 0.5, 1.25, 2.5]),
                ]
            ]
        )
