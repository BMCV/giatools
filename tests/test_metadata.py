import unittest

import giatools.metadata


class MetadataTestCase(unittest.TestCase):

    def setUp(self):
        self.metadata = giatools.metadata.Metadata()


class Metadata__resolution(MetadataTestCase):

    def test__valid(self):
        for value in [(1024.0, 768.0), None]:
            with self.subTest(value=value):
                self.metadata.resolution = value
                self.assertEqual(self.metadata.resolution, value)

    def test__invalid_length(self):
        for value in [(1024.0,), (1024.0, 768.0, 512.0)]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.metadata.resolution = value

    def test__invalid_type(self):
        for value in (
            (1024.0, 768),
            (1024, 768.0),
            (1024, 768),
            ('1024', '768'),
            (1024.0, None),
            (None, 768.0),
            (None, None),
        ):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    self.metadata.resolution = value


class Metadata__z_spacing(MetadataTestCase):

    def test__valid(self):
        for value in (1.5, None):
            with self.subTest(value=value):
                self.metadata.z_spacing = value
                self.assertEqual(self.metadata.z_spacing, value)

    def test__invalid_type(self):
        for value in ('1.5', 1):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    self.metadata.z_spacing = value


class Metadata__z_position(MetadataTestCase):

    def test__valid(self):
        for value in (1.5, None):
            with self.subTest(value=value):
                self.metadata.z_position = value
                self.assertEqual(self.metadata.z_position, value)

    def test__invalid_type(self):
        for value in ('1.5', 1):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    self.metadata.z_position = value


class Metadata__unit(MetadataTestCase):

    def test__valid(self):
        for value in ('cm', None):
            with self.subTest(value=value):
                self.metadata.unit = value
                self.assertEqual(self.metadata.unit, value)

    def test__invalid_type(self):
        for value in ('px', 1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.metadata.unit = value
