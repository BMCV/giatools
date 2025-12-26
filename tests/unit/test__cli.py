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


class ToolBaseplate__init__(MockedTestCase):

    def setUp(self):
        super().setUp()

    def test__init__(self):
        tool = giatools.cli.ToolBaseplate()
        self.assertIs(tool.parser, self.cli_argparse.ArgumentParser.return_value)
        self.cli_argparse.ArgumentParser.return_value.add_argument.assert_called_with('params', type=str)


class ToolBaseplate__add_input_image(MockedTestCase):

    def test__required_True(self):
        tool = giatools.cli.ToolBaseplate()
        tool.add_input_image('input1', required=True)
        self.assertEqual(tool.input_keys, ['input1'])
        tool.parser.add_argument.assert_called_with('--input1', type=str, required=True)

    def test__required_False(self):
        tool = giatools.cli.ToolBaseplate()
        tool.add_input_image('input1', required=False)
        self.assertEqual(tool.input_keys, ['input1'])
        tool.parser.add_argument.assert_called_with('--input1', type=str, required=False)

    def test__repeated(self):
        tool = giatools.cli.ToolBaseplate()
        tool.add_input_image('input1')
        self.assertEqual(tool.input_keys, ['input1'])
        tool.parser.add_argument.assert_called_with('--input1', type=str, required=True)
        tool.add_input_image('input2')
        self.assertEqual(tool.input_keys, ['input1', 'input2'])
        tool.parser.add_argument.assert_called_with('--input2', type=str, required=True)

    def test__value_error(self):
        tool = giatools.cli.ToolBaseplate()
        for attr in ('input_keys', 'output_keys'):
            with self.subTest(attr=attr):
                setattr(tool, attr, ['input1'])
                tool.parser.reset_mock()
                with self.assertRaises(ValueError):
                    tool.add_input_image('input1')
                tool.parser.add_argument.assert_not_called()
                setattr(tool, attr, [])


class ToolBaseplate__add_output_image(MockedTestCase):

    def test__required_True(self):
        tool = giatools.cli.ToolBaseplate()
        tool.add_output_image('output1', required=True)
        self.assertEqual(tool.output_keys, ['output1'])
        tool.parser.add_argument.assert_called_with('--output1', type=str, required=True)

    def test__required_False(self):
        tool = giatools.cli.ToolBaseplate()
        tool.add_output_image('output1', required=False)
        self.assertEqual(tool.output_keys, ['output1'])
        tool.parser.add_argument.assert_called_with('--output1', type=str, required=False)

    def test__repeated(self):
        tool = giatools.cli.ToolBaseplate()
        tool.add_output_image('output1')
        self.assertEqual(tool.output_keys, ['output1'])
        tool.parser.add_argument.assert_called_with('--output1', type=str, required=True)
        tool.add_output_image('output2')
        self.assertEqual(tool.output_keys, ['output1', 'output2'])
        tool.parser.add_argument.assert_called_with('--output2', type=str, required=True)

    def test__value_error(self):
        tool = giatools.cli.ToolBaseplate()
        for attr in ('input_keys', 'output_keys'):
            with self.subTest(attr=attr):
                setattr(tool, attr, ['output1'])
                tool.parser.reset_mock()
                with self.assertRaises(ValueError):
                    tool.add_output_image('output1')
                tool.parser.add_argument.assert_not_called()
                setattr(tool, attr, [])


class ToolBaseplate__parse_args(MockedTestCase):

    ...  # TODO: Add tests for `parse_args` method


class ToolBaseplate__run(MockedTestCase):

    ...  # TODO: Add tests for `run` method
