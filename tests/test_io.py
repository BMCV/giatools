import unittest

import giatools.io


class imread(unittest.TestCase):

    def test_input1(self):
        img = giatools.io.imread('tests/data/input1.tif')
        self.assertEqual(img.mean(), 63.66848655158571)

    def test_input2(self):
        img = giatools.io.imread('tests/data/input2.tif')
        self.assertEqual(img.mean(), 9.543921821305842)


if __name__ == '__main__':
    unittest.main()
