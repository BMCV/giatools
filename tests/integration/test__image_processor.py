"""
Integration tests that cover the `giatools.image_processor` module.
"""

import copy
import unittest

import numpy as np

import giatools.image
import giatools.image_processor
import giatools.metadata


class ImageProcessor(unittest.TestCase):

    def setUp(self):
        super().setUp()
        np.random.seed(0)

    def test__1input__1output(self):
        data1 = np.random.rand(100, 100, 3)
        metadata1 = giatools.metadata.Metadata(resolution=(1.5, 2.0), unit='cm')
        image1 = giatools.image.Image(
            data=data1.copy(),
            axes='YXC',
            metadata=copy.copy(metadata1),
        )
        processor = giatools.image_processor.ImageProcessor(image1)
        for section in processor.process(joint_axes='YX'):
            section[0] = (section[0].data > 0.5)
        np.testing.assert_array_equal(image1.data, data1)
        np.testing.assert_array_equal(image1.metadata, metadata1)
        np.testing.assert_array_equal(processor.outputs[0].data, image1.data > 0.5)
        self.assertEqual(processor.outputs[0].axes, image1.axes)
        self.assertEqual(processor.outputs[0].metadata, image1.metadata)
