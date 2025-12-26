"""
Unit tests for the `giatools.cli` module.
"""

import unittest
import unittest.mock

import giatools.cli


class MockedTestCase(unittest.TestCase):

    def setUp(self):
        super().setUp()

        self.cli_argparse = unittest.mock.patch(
            'giatools.cli.argparse'
        ).start()
        self.cli_json = unittest.mock.patch(
            'giatools.cli.json'
        ).start()
        self.cli_types = unittest.mock.patch(
            'giatools.cli.types'
        ).start()

        self.addCleanup(unittest.mock.patch.stopall)


class ToolBaseplate(MockedTestCase):

    def setUp(self):
        super().setUp()

    def test__init__(self):
        tool = giatools.cli.ToolBaseplate()
        self.assertIs(tool.parser, self.cli_argparse.ArgumentParser.return_value)
        self.cli_argparse.ArgumentParser.return_value.add_argument.assert_called_with('params', type=str)

    def test__add_input_image__required_True(self):
        tool = giatools.cli.ToolBaseplate()
        tool.add_input_image('input1', required=True)
        self.assertEqual(tool.input_keys, ['input1'])
        tool.parser.add_argument.assert_called_with('--input1', type=str, required=True)

    def test__add_input_image__required_False(self):
        tool = giatools.cli.ToolBaseplate()
        tool.add_input_image('input1', required=False)
        self.assertEqual(tool.input_keys, ['input1'])
        tool.parser.add_argument.assert_called_with('--input1', type=str, required=False)

    def test__add_input_image__repeated(self):
        tool = giatools.cli.ToolBaseplate()
        tool.add_input_image('input1')
        self.assertEqual(tool.input_keys, ['input1'])
        tool.parser.add_argument.assert_called_with('--input1', type=str, required=True)
        tool.add_input_image('input2')
        self.assertEqual(tool.input_keys, ['input1', 'input2'])
        tool.parser.add_argument.assert_called_with('--input2', type=str, required=True)

    def test__add_input_image__value_error(self):
        tool = giatools.cli.ToolBaseplate()
        tool.add_input_image('input1')
        tool.parser.reset_mock()
        with self.assertRaises(ValueError):
            tool.add_input_image('input1')
        tool.parser.add_argument.assert_not_called()

    # TODO: Add tests for `add_output_image` method

    # TODO: Add tests for `parse_args` method

    # TODO: Add tests for `run` method
