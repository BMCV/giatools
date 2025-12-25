"""
Unit tests for the `giatools.io._backends.tiff` module.
"""

import unittest
import unittest.mock

import giatools.io
import giatools.io._backends.tiff
import giatools.metadata
from giatools.typing import get_args

from ..tools import (
    filenames,
    mock_array,
)

valid_tiff_units = (
    ('inch', 'inch'),
    ('nm', 'nm'),
    ('nanometer', 'nm'),
    ('um', 'um'),
    ('µm', 'um'),
    (r'\u00B5m', 'um'),
    ('micrometer', 'um'),
    ('mm', 'mm'),
    ('millimeter', 'mm'),
    ('cm', 'cm'),
    ('centimeter', 'cm'),
    ('m', 'm'),
    ('meter', 'm'),
    ('km', 'km'),
    ('kilometers', 'km'),
)

# Consistency check
assert frozenset(u[1] for u in valid_tiff_units) == frozenset(get_args(giatools.metadata.Unit))


class MockedTestCase(unittest.TestCase):

    def setUp(self):
        self._tifffile = unittest.mock.patch(
            'giatools.io._backends.tiff._tifffile'
        ).start()

        self.addCleanup(unittest.mock.patch.stopall)


class TiffWriter__write(MockedTestCase):

    def setUp(self):
        super().setUp()
        self.writer = giatools.io._backends.tiff.TiffWriter()

    @mock_array(10, 10, 1)
    @filenames('tif', 'tiff')
    def test__valid(self, array, filename):
        metadata = dict(z_spacing=0.5, unit='cm')
        self.writer.write(array, filepath=filename, axes='YXC', metadata=giatools.metadata.Metadata(**metadata))
        self._tifffile.imwrite.assert_called()
        self.assertEqual(
            self._tifffile.imwrite.call_args.kwargs['metadata'],
            dict(
                spacing=metadata['z_spacing'],
                unit=metadata['unit'],
                axes='YXC',
            ),
        )


class TiffReader__get_image_metadata(MockedTestCase):

    def setUp(self):
        super().setUp()
        self.reader = giatools.io._backends.tiff.TiffReader('filepath').__enter__()
        self.image = self.reader.select_image(0)

    # -----------------------------------------------------------------------------------------------------------------
    # `resolution` tests
    # -----------------------------------------------------------------------------------------------------------------

    def test__resolution(self):
        self.image.pages[0].tags = dict(
            XResolution=unittest.mock.MagicMock(value=(300, 1)),
            YResolution=unittest.mock.MagicMock(value=(300, 2)),
        )
        metadata = self.reader.get_image_metadata(self.image)
        self.assertEqual(metadata.resolution, (300.0, 150.0))

    def test__resolution__unavailable1(self):
        self.image.pages[0].tags = dict(
            XResolution=unittest.mock.MagicMock(value=(300, 1)),
        )
        metadata = self.reader.get_image_metadata(self.image)
        self.assertIsNone(metadata.resolution)

    def test__resolution__unavailable2(self):
        self.image.pages[0].tags = dict()
        metadata = self.reader.get_image_metadata(self.image)
        self.assertIsNone(metadata.resolution)

    def test__resolution__invalid(self):
        self.image.pages[0].tags = dict(
            XResolution=unittest.mock.MagicMock(value=(300, 0)),
            YResolution=unittest.mock.MagicMock(value=(300, 0)),
        )
        metadata = self.reader.get_image_metadata(self.image)
        self.assertIsNone(metadata.resolution)

    # -----------------------------------------------------------------------------------------------------------------
    # Missing or invalid `ImageDescription` tests
    # -----------------------------------------------------------------------------------------------------------------

    def test__resolution_unit_inches(self):
        self.image.pages[0].tags = dict(
            ResolutionUnit=unittest.mock.MagicMock(value=2),
        )
        metadata = self.reader.get_image_metadata(self.image)
        self.assertEqual(metadata.unit, 'inch')

    def test__resolution_unit_centimeters(self):
        self.image.pages[0].tags = dict(
            ResolutionUnit=unittest.mock.MagicMock(value=3),
        )
        metadata = self.reader.get_image_metadata(self.image)
        self.assertEqual(metadata.unit, 'cm')

    def test__resolution_unit__invalid(self):
        for value in (0, 1, 4):
            with self.subTest(value=value):
                self.image.pages[0].tags = dict(
                    ResolutionUnit=unittest.mock.MagicMock(value=value),
                )
                metadata = self.reader.get_image_metadata(self.image)
                self.assertIsNone(metadata.unit)

    # -----------------------------------------------------------------------------------------------------------------
    # GIA-like `ImageDescription` tests (JSON)
    # -----------------------------------------------------------------------------------------------------------------

    def test__json_description__spacing(self):
        for value, expected_value in (
            ('0.5', 0.5), ('1', 1.0),
        ):
            with self.subTest(value=value):
                self.image.pages[0].tags = dict(
                    ImageDescription=unittest.mock.MagicMock(value=f'{{"spacing": {value}}}'),
                )
                metadata = self.reader.get_image_metadata(self.image)
                self.assertEqual(metadata.z_spacing, expected_value)

    def test__json_description__spacing__invalid(self):
        for json in (
            '{}',
            '{"spacing": "xxx"}',
            '{"spacing": null}',
        ):
            with self.subTest(json=json):
                self.image.pages[0].tags = dict(
                    ImageDescription=unittest.mock.MagicMock(value=json),
                )
                metadata = self.reader.get_image_metadata(self.image)
                self.assertIsNone(metadata.z_spacing)

    def test__json_description__z_position(self):
        for value, expected_value in (
            ('0.5', 0.5), ('1', 1.0),
        ):
            with self.subTest(value=value):
                self.image.pages[0].tags = dict(
                    ImageDescription=unittest.mock.MagicMock(value=f'{{"z_position": {value}}}'),
                )
                metadata = self.reader.get_image_metadata(self.image)
                self.assertEqual(metadata.z_position, expected_value)

    def test__json_description__z_position__invalid(self):
        for json in (
            '{}',
            '{"z_position": "xxx"}',
            '{"z_position": null}',
        ):
            with self.subTest(json=json):
                self.image.pages[0].tags = dict(
                    ImageDescription=unittest.mock.MagicMock(value=json),
                )
                metadata = self.reader.get_image_metadata(self.image)
                self.assertIsNone(metadata.z_position)

    def test__json_description__unit(self):
        for unit_input, unit_expected in valid_tiff_units:
            with self.subTest(unit_input=unit_input):
                self.image.pages[0].tags = dict(
                    ImageDescription=unittest.mock.MagicMock(value=f'{{"unit": "{unit_input}"}}'),
                )
                metadata = self.reader.get_image_metadata(self.image)
                self.assertEqual(metadata.unit, unit_expected)

    def test__json_description__unit__invalid(self):
        for json in (
            '{}',
            '{"unit": null}',
            '{"unit": 20}',
            '{"unit": "px"}',
            '{"unit": "pixel"}',
        ):
            with self.subTest(json=json):
                self.image.pages[0].tags = dict(
                    ImageDescription=unittest.mock.MagicMock(value=json),
                )
                metadata = self.reader.get_image_metadata(self.image)
                self.assertIsNone(metadata.unit)

    # -----------------------------------------------------------------------------------------------------------------
    # OME-like `ImageDescription` tests (XML)
    # -----------------------------------------------------------------------------------------------------------------

    def test__xml_description__spacing(self):
        for value, expected_value in (
            ('0.5', 0.5), ('1', 1.0),
        ):
            with self.subTest(value=value):
                self.image.pages[0].tags = dict(
                    ImageDescription=unittest.mock.MagicMock(
                        value=(
                            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
                            '<Pixels PhysicalSizeZ="{}"/>'
                            '</OME>'.format(value)
                        )
                    ),
                )
                metadata = self.reader.get_image_metadata(self.image)
                self.assertEqual(metadata.z_spacing, expected_value)

    def test__xml_description__spacing__invalid(self):
        for xml in (
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Pixels/></OME>',
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Pixels PhysicalSizeZ=""/></OME>',
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Pixels PhysicalSizeZ="xxx"/></OME>',
        ):
            with self.subTest(xml=xml):
                self.image.pages[0].tags = dict(
                    ImageDescription=unittest.mock.MagicMock(value=xml),
                )
                metadata = self.reader.get_image_metadata(self.image)
                self.assertIsNone(metadata.z_spacing)

    def test__xml_description__unit(self):
        for unit_input, unit_expected in valid_tiff_units:
            with self.subTest(unit_input=unit_input):
                self.image.pages[0].tags = dict(
                    ImageDescription=unittest.mock.MagicMock(
                        value=(
                            f'<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
                            f'<Pixels PhysicalSizeZUnit="{unit_input}"/>'
                            f'</OME>'
                        )
                    ),
                )
                metadata = self.reader.get_image_metadata(self.image)
                self.assertEqual(metadata.unit, unit_expected)

    def test__xml_description__unit__invalid(self):
        for xml in (
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Pixels/></OME>',
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Pixels PhysicalSizeZUnit="20"/></OME>',
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Pixels PhysicalSizeZUnit="px"/></OME>',
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Pixels PhysicalSizeZUnit="pixel"/></OME>',
        ):
            with self.subTest(xml=xml):
                self.image.pages[0].tags = dict(
                    ImageDescription=unittest.mock.MagicMock(value=xml),
                )
                metadata = self.reader.get_image_metadata(self.image)
                self.assertIsNone(metadata.unit)

    # -----------------------------------------------------------------------------------------------------------------
    # ImageJ-like `ImageDescription` tests (line-by-line)
    # -----------------------------------------------------------------------------------------------------------------

    def test__line_description__spacing(self):
        for value, expected_value in (
            ('0.5', 0.5), ('1', 1.0),
        ):
            with self.subTest(value=value):
                self.image.pages[0].tags = dict(
                    ImageDescription=unittest.mock.MagicMock(value=f'spacing={value}'),
                )
                metadata = self.reader.get_image_metadata(self.image)
                self.assertEqual(metadata.z_spacing, expected_value)

    def test__line_description__spacing__invalid(self):
        for line in (
            'spacing=',
            'spacing=xxx',
            'spacing=null',
        ):
            with self.subTest(line=line):
                self.image.pages[0].tags = dict(
                    ImageDescription=unittest.mock.MagicMock(value=line),
                )
                metadata = self.reader.get_image_metadata(self.image)
                self.assertIsNone(metadata.z_spacing)

    def test__line_description__unit(self):
        for unit_input, unit_expected in valid_tiff_units:
            for unit_input2 in (unit_input, f'"{unit_input}"'):
                with self.subTest(unit_input=unit_input2):
                    self.image.pages[0].tags = dict(
                        ImageDescription=unittest.mock.MagicMock(value=f'unit={unit_input2}'),
                    )
                    metadata = self.reader.get_image_metadata(self.image)
                    self.assertEqual(metadata.unit, unit_expected)

    def test__line_description__unit__invalid(self):
        for line in (
            'unit=',
            'unit=xxx',
            'unit=null',
            'unit=20',
            'unit=px',
            '{"unit": "pixel"}',
        ):
            with self.subTest(line=line):
                self.image.pages[0].tags = dict(
                    ImageDescription=unittest.mock.MagicMock(value=line),
                )
                metadata = self.reader.get_image_metadata(self.image)
                self.assertIsNone(metadata.unit)
