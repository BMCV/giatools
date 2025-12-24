import importlib
import unittest
import unittest.mock

import pandas as pd

import giatools.pandas


class find_column(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

    def test(self):
        self.assertEqual(giatools.pandas.find_column(self.df, ['a', 'c']), 'a')

    def test__missing(self):
        with self.assertRaises(KeyError):
            giatools.pandas.find_column(self.df, ['c'])

    def test__ambiguous(self):
        with self.assertRaises(KeyError):
            giatools.pandas.find_column(self.df, ['a', 'b'])


@unittest.mock.patch.dict('sys.modules', {'pandas': None})
class no_pandas(unittest.TestCase):

    def test(self):
        importlib.reload(giatools.pandas)
        self.assertEqual(giatools.pandas.DataFrame.__bases__, (object,))
