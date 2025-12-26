"""
Integration tests that cover the `giatools.image_processor` module.
"""

import copy
import unittest

import numpy as np

import giatools.image
import giatools.image_processor

from ..tools import permute_axes


class ImageProcessorTestCase(unittest.TestCase):

    image_filepaths = list()

    def setUp(self):
        super().setUp()
        self.images = [giatools.image.Image.read(filepath) for filepath in self.image_filepaths]

    def tearDown(self):
        super().tearDown()
        self.images.clear()


class ImageProcessor(ImageProcessorTestCase):

    image_filepaths = [
        'tests/data/input4_uint8.png',
        'tests/data/input4_uint8.jpg',
    ]

    @permute_axes('YX', name='joint_axes')
    def test__1_positional_input__1output(self, joint_axes):
        image = self.images[0]
        image_data = image.data.copy()
        image_metadata = copy.deepcopy(image.metadata)
        processor = giatools.image_processor.ImageProcessor(image)
        for section in processor.process(joint_axes):
            section['result'] = (section[0].data > section[0].data.mean())
        expected_result = np.stack(
            [
                image.data[..., c] > image.data[..., c].mean()
                for c in range(image.data.shape[-1])
            ],
            axis=-1,
        )
        np.testing.assert_array_equal(image.data, image_data)
        self.assertEqual(image.metadata, image_metadata)
        np.testing.assert_array_equal(processor.outputs['result'].data, expected_result)
        self.assertEqual(processor.outputs['result'].axes, image.axes)
        self.assertEqual(processor.outputs['result'].metadata, image.metadata)

    @permute_axes('YX', name='joint_axes')
    def test__2_positional_inputs__1output(self, joint_axes):
        processor = giatools.image_processor.ImageProcessor(*self.images[0:2])
        for section in processor.process(joint_axes):
            section['result'] = section[0].data - section[1].data
        expected_result = np.stack(
            [
                self.images[0].data[..., c] - self.images[1].data[..., c]
                for c in range(self.images[0].data.shape[-1])
            ],
            axis=-1,
        )
        np.testing.assert_array_equal(processor.outputs['result'].data, expected_result)
