"""
Integration tests that cover the `giatools.image_processor` module.
"""

import copy
import unittest

import numpy as np

import giatools.image
import giatools.image_processor

from ..tools import (
    minimum_python_version,
    permute_axes,
)


class ImageProcessorTestCase(unittest.TestCase):

    image_filepaths = list()

    def setUp(self):
        super().setUp()
        self.images = [giatools.image.Image.read(filepath) for filepath in self.image_filepaths]

    def tearDown(self):
        super().tearDown()
        self.images.clear()


class ImageProcessor__process(ImageProcessorTestCase):

    image_filepaths = [
        'tests/data/input4_uint8.png',
        'tests/data/input4_uint8.jpg',
    ]

    @minimum_python_version(3, 11)
    def test__immutability(self):
        """
        Test that the input image is not modified by the processing, and that the output image is isolated from the
        input image (changing the output image later will not change the input image).
        """
        image = self.images[0]
        image_data = image.data.copy()
        image_metadata = copy.deepcopy(image.metadata)
        processor = giatools.image_processor.ImageProcessor(image)
        for section in processor.process('XY'):
            section['result'] = section[0].data * 2

        # Verify that the input image remained unchanged
        np.testing.assert_array_equal(image.data, image_data)
        self.assertEqual(image.metadata, image_metadata)

        # Verify that changing the output image will not change the input image
        self.assertIsNot(processor.outputs['result'].metadata, image.metadata)
        self.assertFalse(np.shares_memory(processor.outputs['result'].data, image.data))

    @minimum_python_version(3, 11)
    @permute_axes('YX', name='joint_axes')
    def test__1_positional_input__1output__yx(self, joint_axes):
        processor = giatools.image_processor.ImageProcessor(self.images[0])
        for section in processor.process(joint_axes):
            section['result'] = (section[0].data > section[0].data.mean())
        expected_result = np.stack(
            [
                self.images[0].data[..., c] > self.images[0].data[..., c].mean()
                for c in range(self.images[0].data.shape[-1])
            ],
            axis=-1,
        )
        np.testing.assert_array_equal(processor.outputs['result'].data, expected_result)
        self.assertEqual(processor.outputs['result'].axes, self.images[0].axes)
        self.assertEqual(processor.outputs['result'].metadata, self.images[0].metadata)

    @minimum_python_version(3, 11)
    def test__1_positional_input__1output(self):
        for joint_axes in ('YXC', self.images[0].axes):
            with self.subTest(joint_axes=joint_axes):
                processor = giatools.image_processor.ImageProcessor(self.images[0])
                for section in processor.process(joint_axes):
                    section['result'] = (section[0].data > section[0].data.mean())
                expected_result = self.images[0].data > self.images[0].data.mean()
                np.testing.assert_array_equal(processor.outputs['result'].data, expected_result)
                self.assertEqual(processor.outputs['result'].axes, self.images[0].axes)
                self.assertEqual(processor.outputs['result'].metadata, self.images[0].metadata)

    @minimum_python_version(3, 11)
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
        self.assertEqual(processor.outputs['result'].axes, self.images[0].axes)
        self.assertEqual(processor.outputs['result'].metadata, self.images[0].metadata)

    @minimum_python_version(3, 11)
    @permute_axes('YX', name='joint_axes')
    def test__2_keyword_inputs__1output(self, joint_axes):
        processor = giatools.image_processor.ImageProcessor(input1=self.images[0], input2=self.images[1])
        for section in processor.process(joint_axes):
            section['result'] = section['input1'].data - section['input2'].data
        expected_result = np.stack(
            [
                self.images[0].data[..., c] - self.images[1].data[..., c]
                for c in range(self.images[0].data.shape[-1])
            ],
            axis=-1,
        )
        np.testing.assert_array_equal(processor.outputs['result'].data, expected_result)
        self.assertEqual(processor.outputs['result'].axes, self.images[0].axes)
        self.assertEqual(processor.outputs['result'].metadata, self.images[0].metadata)

    @minimum_python_version(3, 11)
    @permute_axes('YX', name='joint_axes')
    def test__2_mixed_inputs__2outputs(self, joint_axes):
        processor = giatools.image_processor.ImageProcessor(self.images[0], input2=self.images[1])
        for section in processor.process(joint_axes):
            section['result1'] = (section[0].data > section[0].data.mean())
            section['result2'] = section[0].data - section['input2'].data
        expected_result1 = np.stack(
            [
                self.images[0].data[..., c] > self.images[0].data[..., c].mean()
                for c in range(self.images[0].data.shape[-1])
            ],
            axis=-1,
        )
        expected_result2 = np.stack(
            [
                self.images[0].data[..., c] - self.images[1].data[..., c]
                for c in range(self.images[0].data.shape[-1])
            ],
            axis=-1,
        )
        np.testing.assert_array_equal(processor.outputs['result1'].data, expected_result1)
        np.testing.assert_array_equal(processor.outputs['result2'].data, expected_result2)
        for result in ('result1', 'result2'):
            self.assertEqual(processor.outputs[result].axes, self.images[0].axes)
            self.assertEqual(processor.outputs[result].metadata, self.images[0].metadata)
