import unittest
import numpy as np
import skimage.util
import giatools.util


class convert_image_to_format_of(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        np.random.seed(0)
        img_float = np.random.rand(16, 16)
        self.testdata = {
            'uint8': skimage.util.img_as_ubyte(img_float),
            'uint16': skimage.util.img_as_uint(img_float),
            'int16': skimage.util.img_as_int(img_float),
            'float32': skimage.util.img_as_float32(img_float),
            'float64': skimage.util.img_as_float64(img_float),
        }

    def test_self_conversion(self):
        for dtype, img in self.testdata.items():
            actual = giatools.util.convert_image_to_format_of(img, img)
            self.assertIs(actual, img)
            self.assertEqual(actual.dtype, getattr(np, dtype))

    def test_cross_conversion(self):
        for src_dtype, src_img in self.testdata.items():
            for dst_img in self.testdata.values():
                with self.subTest(src_dtype=src_dtype, dst_dtype=dst_img.dtype):
                    actual = giatools.util.convert_image_to_format_of(src_img, dst_img)
                    self.assertTrue(np.allclose(actual, dst_img))
                    self.assertEqual(actual.dtype, dst_img.dtype)


if __name__ == '__main__':
    unittest.main()
