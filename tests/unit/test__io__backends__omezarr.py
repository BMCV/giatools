"""
Unit tests for the `giatools.io._backends.omezarr` module.
"""

import os
import unittest
import unittest.mock

import giatools.io
import giatools.metadata

from ..tools import (
    filenames,
    minimum_python_version,
    mock_array,
    validate_metadata,
    without_logging,
)


class MockedTestCase(unittest.TestCase):

    def setUp(self):
        self._zarr = unittest.mock.patch(
            'giatools.io._backends.omezarr._zarr'
        ).start()
        self._ome_zarr_writer = unittest.mock.patch(
            'giatools.io._backends.omezarr._ome_zarr_writer'
        ).start()
        self._ome_zarr_io = unittest.mock.patch(
            'giatools.io._backends.omezarr._ome_zarr_io'
        ).start()

        self.addCleanup(unittest.mock.patch.stopall)

        import giatools.io._backends.omezarr
        self._omezarr_backend = giatools.io._backends.omezarr


class OMEZarrReader(unittest.TestCase):

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


class OMEZarrWriter__write(MockedTestCase):

    @minimum_python_version(3, 11)
    def setUp(self):
        super().setUp()
        self.writer = self._omezarr_backend.OMEZarrWriter()

    @minimum_python_version(3, 11)
    @mock_array(10, 10, 1)
    @filenames('zarr', 'ome.zarr')
    def test__incompatible_data_error(self, array, filename):
        self._ome_zarr_writer.write_image.side_effect = ValueError()
        with self.assertRaises(giatools.io.IncompatibleDataError):
            self.writer.write(array, filepath=filename, axes='YX', metadata=giatools.metadata.Metadata())

    @minimum_python_version(3, 11)
    @mock_array(10, 10, 1)
    @filenames('zarr', 'ome.zarr')
    def test__yxc(self, array, filename):
        metadata = dict(z_spacing=0.5, unit='cm')
        self.writer.write(array, filepath=filename, axes='YXC', metadata=giatools.metadata.Metadata(**metadata))
        self._ome_zarr_writer.write_image.assert_called()
        self.assertEqual(
            self._ome_zarr_writer.write_image.call_args.kwargs['axes'],
            [
                dict(name='Y', type='space', unit='cm'),
                dict(name='X', type='space', unit='cm'),
                dict(name='C', type='channel'),
            ]
        )
        self.assertEqual(
            self._ome_zarr_writer.write_image.call_args.kwargs['coordinate_transformations'],
            [
                [
                    dict(type='scale', scale=[1.0, 1.0, 1.0]),
                ]
            ]
        )

    @minimum_python_version(3, 11)
    @mock_array(5, 8, 10)
    @filenames('zarr', 'ome.zarr')
    def test__zyx(self, array, filename):
        metadata = dict(z_spacing=0.5, resolution=(0.4, 0.8), unit='cm')
        self.writer.write(array, filepath=filename, axes='ZYX', metadata=giatools.metadata.Metadata(**metadata))
        self._ome_zarr_writer.write_image.assert_called()
        self.assertEqual(
            self._ome_zarr_writer.write_image.call_args.kwargs['axes'],
            [
                dict(name='Z', type='space', unit='cm'),
                dict(name='Y', type='space', unit='cm'),
                dict(name='X', type='space', unit='cm'),
            ]
        )
        self.assertEqual(
            self._ome_zarr_writer.write_image.call_args.kwargs['coordinate_transformations'],
            [
                [
                    dict(type='scale', scale=[0.5, 1.25, 2.5]),
                ]
            ]
        )

    @minimum_python_version(3, 11)
    @mock_array(12, 5, 8, 10)
    @filenames('zarr', 'ome.zarr')
    def test__tzyx(self, array, filename):
        metadata = dict(z_spacing=0.5, resolution=(0.4, 0.8), unit='cm')
        self.writer.write(array, filepath=filename, axes='TZYX', metadata=giatools.metadata.Metadata(**metadata))
        self._ome_zarr_writer.write_image.assert_called()
        self.assertEqual(
            self._ome_zarr_writer.write_image.call_args.kwargs['axes'],
            [
                dict(name='T', type='time'),
                dict(name='Z', type='space', unit='cm'),
                dict(name='Y', type='space', unit='cm'),
                dict(name='X', type='space', unit='cm'),
            ]
        )
        self.assertEqual(
            self._ome_zarr_writer.write_image.call_args.kwargs['coordinate_transformations'],
            [
                [
                    dict(type='scale', scale=[1.0, 0.5, 1.25, 2.5]),
                ]
            ]
        )

    @minimum_python_version(3, 11)
    @mock_array(8, 9, 10, 2)
    @filenames('zarr', 'ome.zarr')
    def test__yxqs(self, array, filename):
        metadata = dict(resolution=(0.4, 0.8), unit='cm')
        self.writer.write(array, filepath=filename, axes='YXQS', metadata=giatools.metadata.Metadata(**metadata))
        self._ome_zarr_writer.write_image.assert_called()
        self.assertEqual(
            self._ome_zarr_writer.write_image.call_args.kwargs['axes'],
            [
                dict(name='Y', type='space', unit='cm'),
                dict(name='X', type='space', unit='cm'),
                dict(name='Q', type='unknown'),
                dict(name='S', type='channel'),
            ]
        )
        self.assertEqual(
            self._ome_zarr_writer.write_image.call_args.kwargs['coordinate_transformations'],
            [
                [
                    dict(type='scale', scale=[1.25, 2.5, 1.0, 1.0]),
                ]
            ]
        )

    @minimum_python_version(3, 11)
    @mock_array(10, 10, 1)
    @filenames('zarr', 'ome.zarr')
    def test__overwrite_file(self, array, filename):
        with open(filename, 'w') as f:
            f.write('existing file content')
        self.writer.write(array, filepath=filename, axes='YX', metadata=giatools.metadata.Metadata())
        self._ome_zarr_writer.write_image.assert_called()

    @minimum_python_version(3, 11)
    @mock_array(10, 10, 1)
    @filenames('zarr', 'ome.zarr')
    def test__overwrite_directory(self, array, filename):
        os.makedirs(filename, exist_ok=False)
        self.writer.write(array, filepath=filename, axes='YX', metadata=giatools.metadata.Metadata())
        self._ome_zarr_writer.write_image.assert_called()
