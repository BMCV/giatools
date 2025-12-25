"""
Integration tests that cover the `giatools.image_processor` module.
"""

import copy
import unittest

import numpy as np

import giatools.image
import giatools.image_processor

from ..tools import permute_axes


class ImageProcessor(unittest.TestCase):

    @permute_axes('YX', name='joint_axes')
    def test__1input__1output(self, joint_axes):
        image = giatools.image.Image.read('tests/data/input4_uint8.png')
        image_data = image.data.copy()
        image_metadata = copy.deepcopy(image.metadata)
        processor = giatools.image_processor.ImageProcessor(image)
        for section in processor.process(joint_axes):
            section['result'] = (
                section[0].data > section[0].data.mean()
            )
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
