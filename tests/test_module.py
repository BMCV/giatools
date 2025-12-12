import unittest

import giatools

from .tools import (
    maximum_python_version,
    minimum_python_version,
)


class require_backend(unittest.TestCase):

    @minimum_python_version(3, 11)
    def test__omezarr__available(self):
        giatools.require_backend('omezarr')

    @maximum_python_version(3, 10)
    def test__omezarr__unavailable(self):
        with self.assertRaises(ImportError):
            giatools.require_backend('omezarr')
