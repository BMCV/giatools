import unittest
import unittest.mock

import giatools.io
import giatools.io._backends.tiff

from .tools import (
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

    @unittest.mock.patch('giatools.io._backends.tiff.tifffile.TiffFile')
    def setUp(self, mock_tiff_file):
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
        self.assertEqual(metadata['resolution'], (300.0, 150.0))

    def test__resolution__unavailable1(self):
        self.image.pages[0].tags = dict(
            XResolution=unittest.mock.MagicMock(value=(300, 1)),
        )
        metadata = self.reader.get_image_metadata(self.image)
        self.assertNotIn('resolution', metadata)

    def test__resolution__unavailable2(self):
        self.image.pages[0].tags = dict()
        metadata = self.reader.get_image_metadata(self.image)
        self.assertNotIn('resolution', metadata)

    def test__resolution__invalid(self):
        self.image.pages[0].tags = dict(
            XResolution=unittest.mock.MagicMock(value=(300, 0)),
            YResolution=unittest.mock.MagicMock(value=(300, 0)),
        )
        metadata = self.reader.get_image_metadata(self.image)
        self.assertNotIn('resolution', metadata)

    # -----------------------------------------------------------------------------------------------------------------
    # Missing or invalid `ImageDescription` tests
    # -----------------------------------------------------------------------------------------------------------------

    def test__resolution_unit_inches(self):
        self.image.pages[0].tags = dict(
            ResolutionUnit=unittest.mock.MagicMock(value=2),
        )
        metadata = self.reader.get_image_metadata(self.image)
        self.assertEqual(metadata['unit'], 'inch')

    def test__resolution_unit_centimeters(self):
        self.image.pages[0].tags = dict(
            ResolutionUnit=unittest.mock.MagicMock(value=3),
        )
        metadata = self.reader.get_image_metadata(self.image)
        self.assertEqual(metadata['unit'], 'cm')

    def test__resolution_unit__invalid(self):
        for value in (0, 1, 4):
            with self.subTest(value=value):
                self.image.pages[0].tags = dict(
                    ResolutionUnit=unittest.mock.MagicMock(value=value),
                )
                metadata = self.reader.get_image_metadata(self.image)
                self.assertNotIn('unit', metadata)

    # -----------------------------------------------------------------------------------------------------------------
    # GIA-like `ImageDescription` tests (JSON)
    # -----------------------------------------------------------------------------------------------------------------

    def test__json_description__spacing(self):
        self.image.pages[0].tags = dict(
            ImageDescription=unittest.mock.MagicMock(value='{"spacing": 0.5}'),
        )
        metadata = self.reader.get_image_metadata(self.image)
        self.assertEqual(metadata['z_spacing'], 0.5)

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
                self.assertNotIn('z_spacing', metadata)

    def test__json_description__z_position(self):
        self.image.pages[0].tags = dict(
            ImageDescription=unittest.mock.MagicMock(value='{"z_position": 1.5}'),
        )
        metadata = self.reader.get_image_metadata(self.image)
        self.assertEqual(metadata['z_position'], 1.5)

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
                self.assertNotIn('z_position', metadata)

    def test__json_description__unit(self):
        for unit_input, unit_expected in valid_tiff_units:
            with self.subTest(unit_input=unit_input):
                self.image.pages[0].tags = dict(
                    ImageDescription=unittest.mock.MagicMock(value=f'{{"unit": "{unit_input}"}}'),
                )
                metadata = self.reader.get_image_metadata(self.image)
                self.assertEqual(metadata['unit'], unit_expected)

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
                self.assertNotIn('unit', metadata)

    # -----------------------------------------------------------------------------------------------------------------
    # OME-like `ImageDescription` tests (XML)
    # -----------------------------------------------------------------------------------------------------------------

    def test__xml_description__spacing(self):
        self.image.pages[0].tags = dict(
            ImageDescription=unittest.mock.MagicMock(
                value=(
                    '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
                    '<Pixels PhysicalSizeZ="0.5"/>'
                    '</OME>'
                )
            ),
        )
        metadata = self.reader.get_image_metadata(self.image)
        self.assertEqual(metadata['z_spacing'], 0.5)

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
                self.assertNotIn('z_spacing', metadata)

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
                self.assertEqual(metadata['unit'], unit_expected)

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
                self.assertNotIn('unit', metadata)

    # -----------------------------------------------------------------------------------------------------------------
    # ImageJ-like `ImageDescription` tests (line-by-line)
    # -----------------------------------------------------------------------------------------------------------------

    def test__line_description__spacing(self):
        self.image.pages[0].tags = dict(
            ImageDescription=unittest.mock.MagicMock(value='spacing=0.5'),
        )
        metadata = self.reader.get_image_metadata(self.image)
        self.assertEqual(metadata['z_spacing'], 0.5)

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
                self.assertNotIn('z_spacing', metadata)

    def test__line_description__unit(self):
        for unit_input, unit_expected in valid_tiff_units:
            for unit_input2 in (unit_input, f'"{unit_input}"'):
                with self.subTest(unit_input=unit_input2):
                    self.image.pages[0].tags = dict(
                        ImageDescription=unittest.mock.MagicMock(value=f'unit={unit_input2}'),
                    )
                    metadata = self.reader.get_image_metadata(self.image)
                    self.assertEqual(metadata['unit'], unit_expected)

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
                self.assertNotIn('unit', metadata)
