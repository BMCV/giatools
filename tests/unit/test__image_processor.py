"""
Unit tests for the `giatools.image` module.
"""

import unittest
import unittest.mock

import numpy as np

import giatools.image_processor


class ImageProcessorTestCase(unittest.TestCase):

    def setUp(self):
        super().setUp()

        self.ProcessorIteration = unittest.mock.patch(
            'giatools.image_processor.ProcessorIteration'
        ).start()

        self.addCleanup(unittest.mock.patch.stopall)


@unittest.mock.patch('giatools.image_processor.ImageProcessor.create_output_image')
class ImageProcessor__process(ImageProcessorTestCase):

    def setUp(self):
        super().setUp()
        self.image1 = unittest.mock.MagicMock(axes='YX')
        self.image_processor = giatools.image_processor.ImageProcessor(self.image1)

    def test__no_inputs(self, mock_create_output_image):
        with self.assertRaises(ValueError):
            giatools.image_processor.ImageProcessor()

    def test__inconsistent_axes(self, mock_create_output_image):
        with self.assertRaises(ValueError):
            giatools.image_processor.ImageProcessor(
                self.image1,
                unittest.mock.MagicMock(axes='XYC'),
            )

    def test__inconsistent_shape(self, mock_create_output_image):
        with self.assertRaises(ValueError):
            giatools.image_processor.ImageProcessor(
                unittest.mock.MagicMock(axes='XY', shape=(100, 80)),
                unittest.mock.MagicMock(axes='XY', shape=(80, 100)),
            )

    def test__errors(self, mock_create_output_image):
        for error in (RuntimeError, ValueError):
            with self.subTest(error=error):
                self.image1.iterate_jointly.side_effect = error()
                with self.assertRaises(error):
                    next(iter(self.image_processor.process('YX')))


class ImageProcessor__create_output_image(ImageProcessorTestCase):

    def setUp(self):
        super().setUp()
        self.image1 = unittest.mock.MagicMock(axes='YX')
        self.image_processor = giatools.image_processor.ImageProcessor(self.image1)

    def test(self):
        for key in ('output_key', 42, ('a', 'b'), None):
            with self.subTest(key=key):
                self.image_processor.create_output_image(key, dtype=np.uint8)

    def test__value_error(self):
        self.image_processor.create_output_image('output_key', dtype=np.uint8)
        for dtype in (np.uint8, np.uint16):
            with self.subTest(dtype=dtype):
                with self.assertRaises(ValueError):
                    self.image_processor.create_output_image('output_key', dtype=dtype)


class ProcessorIteration(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.processor = unittest.mock.MagicMock()
        self.output_slice = unittest.mock.MagicMock()
        self.input_section1 = unittest.mock.MagicMock()
        self.input_section2 = unittest.mock.MagicMock()
        self.input_section3 = unittest.mock.MagicMock()
        self.processor_iteration = giatools.image_processor.ProcessorIteration(
            processor=self.processor,
            input_sections={
                0: self.input_section1,
                1: self.input_section2,
                'input_key': self.input_section3,
            },
            output_slice=self.output_slice,
            joint_axes='YX',
        )

    def test__getitem__kwarg(self):
        self.assertIs(self.processor_iteration['input_key'], self.input_section3)

    def test__getitem__spurious_kwarg(self):
        with self.assertRaisesRegex(KeyError, f'No input image with key "spurious_key".'):
            self.processor_iteration['spurious_key']

    def test__getitem__posarg(self):
        self.assertIs(self.processor_iteration[0], self.input_section1)
        self.assertIs(self.processor_iteration[1], self.input_section2)
        self.assertIs(self.processor_iteration[-1], self.input_section2)
        self.assertIs(self.processor_iteration[-2], self.input_section1)

    def test__getitem__invalid_posarg(self):
        for pos in (2, 3, -3, -4):
            with self.subTest(pos=pos):
                with self.assertRaisesRegex(IndexError, f'No input image at position {pos}.'):
                    self.processor_iteration[pos]
