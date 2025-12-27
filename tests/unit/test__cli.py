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
            'giatools.cli.argparse',
        ).start()
        self.cli_json = unittest.mock.patch(
            'giatools.cli.json',
        ).start()
        self.cli_types = unittest.mock.patch(
            'giatools.cli.types',
        ).start()
        self.cli_image = unittest.mock.patch(
            'giatools.cli._image',
        ).start()
        self.cli_image_processor = unittest.mock.patch(
            'giatools.cli._image_processor',
        ).start()
        self.builtins_print = unittest.mock.patch(
            'builtins.print',
        ).start()

        self.addCleanup(unittest.mock.patch.stopall)


class ToolBaseplate__init__(MockedTestCase):

    def setUp(self):
        super().setUp()

    def test(self):
        tool = giatools.cli.ToolBaseplate()
        self.assertIs(tool.parser, self.cli_argparse.ArgumentParser.return_value)
        self.assertIsNone(tool.args)
        tool.parser.add_argument.assert_has_calls(
            [
                unittest.mock.call('--params', type=str, required=True),
                unittest.mock.call('--verbose', action='store_true', default=False),
            ],
            any_order=True,
        )

    def test__with_args(self):
        tool = giatools.cli.ToolBaseplate('name', description='description')
        self.cli_argparse.ArgumentParser.assert_called_with('name', description='description')
        self.assertIs(tool.parser, self.cli_argparse.ArgumentParser.return_value)
        self.assertIsNone(tool.args)
        tool.parser.add_argument.assert_has_calls(
            [
                unittest.mock.call('--params', type=str, required=True),
                unittest.mock.call('--verbose', action='store_true', default=False),
            ],
            any_order=True,
        )

    def test__params_optional(self):
        tool = giatools.cli.ToolBaseplate(params_required=False)
        self.assertIs(tool.parser, self.cli_argparse.ArgumentParser.return_value)
        self.assertIsNone(tool.args)
        tool.parser.add_argument.assert_has_calls(
            [
                unittest.mock.call('--params', type=str, required=False),
                unittest.mock.call('--verbose', action='store_true', default=False),
            ],
            any_order=True,
        )


class ToolBaseplate__add_input_image(MockedTestCase):

    def setUp(self):
        super().setUp()
        self.tool = giatools.cli.ToolBaseplate()

    def test__required_True(self):
        self.tool.add_input_image('input1', required=True)
        self.assertEqual(self.tool.input_keys, ['input1'])
        self.tool.parser.add_argument.assert_called_with('--input1', type=str, required=True)

    def test__required_False(self):
        self.tool.add_input_image('input1', required=False)
        self.assertEqual(self.tool.input_keys, ['input1'])
        self.tool.parser.add_argument.assert_called_with('--input1', type=str, required=False)

    def test__repeated(self):
        self.tool.add_input_image('input1')
        self.assertEqual(self.tool.input_keys, ['input1'])
        self.tool.parser.add_argument.assert_called_with('--input1', type=str, required=True)
        self.tool.add_input_image('input2')
        self.assertEqual(self.tool.input_keys, ['input1', 'input2'])
        self.tool.parser.add_argument.assert_called_with('--input2', type=str, required=True)

    def test__value_error(self):
        for attr in ('input_keys', 'output_keys'):
            with self.subTest(attr=attr):
                setattr(self.tool, attr, ['input1'])
                self.tool.parser.reset_mock()
                with self.assertRaises(ValueError):
                    self.tool.add_input_image('input1')
                self.tool.parser.add_argument.assert_not_called()
                setattr(self.tool, attr, [])


class ToolBaseplate__add_output_image(MockedTestCase):

    def setUp(self):
        super().setUp()
        self.tool = giatools.cli.ToolBaseplate()

    def test__required_True(self):
        self.tool.add_output_image('output1', required=True)
        self.assertEqual(self.tool.output_keys, ['output1'])
        self.tool.parser.add_argument.assert_called_with('--output1', type=str, required=True)

    def test__required_False(self):
        self.tool.add_output_image('output1', required=False)
        self.assertEqual(self.tool.output_keys, ['output1'])
        self.tool.parser.add_argument.assert_called_with('--output1', type=str, required=False)

    def test__repeated(self):
        self.tool.add_output_image('output1')
        self.assertEqual(self.tool.output_keys, ['output1'])
        self.tool.parser.add_argument.assert_called_with('--output1', type=str, required=True)
        self.tool.add_output_image('output2')
        self.assertEqual(self.tool.output_keys, ['output1', 'output2'])
        self.tool.parser.add_argument.assert_called_with('--output2', type=str, required=True)

    def test__value_error(self):
        for attr in ('input_keys', 'output_keys'):
            with self.subTest(attr=attr):
                setattr(self.tool, attr, ['output1'])
                self.tool.parser.reset_mock()
                with self.assertRaises(ValueError):
                    self.tool.add_output_image('output1')
                self.tool.parser.add_argument.assert_not_called()
                setattr(self.tool, attr, [])


class ToolBaseplate__parse_args(MockedTestCase):

    def setUp(self):
        super().setUp()
        self.tool = giatools.cli.ToolBaseplate(params_required=False)

    def test(self):
        self.tool.input_keys = ['input1', 'input2']
        self.tool.output_keys = ['output1']
        self.tool.parser.parse_args.return_value = unittest.mock.Mock(
            verbose=False,
            params='params.json',
            input1='input1.png',
            input2='input2.png',
            output1='output1.png',
        )
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open()) as mock_open:
            args = self.tool.parse_args()
            self.cli_json.load.assert_called_with(mock_open())
        self.cli_image.Image.read.assert_has_calls(
            [
                unittest.mock.call('input1.png'),
                unittest.mock.call('input2.png'),
            ]
        )
        self.cli_types.SimpleNamespace.assert_called_with(
            params=self.cli_json.load.return_value,
            input_filepaths={
                'input1': 'input1.png',
                'input2': 'input2.png',
            },
            input_images={
                'input1': self.cli_image.Image.read.return_value,
                'input2': self.cli_image.Image.read.return_value,
            },
            output_filepaths={
                'output1': 'output1.png',
            },
            raw_args=self.tool.parser.parse_args.return_value,
        )
        self.assertIs(args, self.cli_types.SimpleNamespace.return_value)
        self.assertIs(args, self.tool.args)

    def test__no_inputs(self):
        self.tool.output_keys = ['output1']
        self.tool.parser.parse_args.return_value = unittest.mock.Mock(
            verbose=False,
            params='params.json',
            output1='output1.png',
        )
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open()) as mock_open:
            args = self.tool.parse_args()
            self.cli_json.load.assert_called_with(mock_open())
        self.cli_image.Image.read.assert_not_called()
        self.cli_types.SimpleNamespace.assert_called_with(
            params=self.cli_json.load.return_value,
            input_filepaths=dict(),
            input_images=dict(),
            output_filepaths={
                'output1': 'output1.png',
            },
            raw_args=self.tool.parser.parse_args.return_value,
        )
        self.assertIs(args, self.cli_types.SimpleNamespace.return_value)
        self.assertIs(args, self.tool.args)

    def test__no_outputs(self):
        self.tool.input_keys = ['input1', 'input2']
        self.tool.parser.parse_args.return_value = unittest.mock.Mock(
            verbose=False,
            params='params.json',
            input1='input1.png',
            input2='input2.png',
        )
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open()) as mock_open:
            args = self.tool.parse_args()
            self.cli_json.load.assert_called_with(mock_open())
        self.cli_image.Image.read.assert_has_calls(
            [
                unittest.mock.call('input1.png'),
                unittest.mock.call('input2.png'),
            ]
        )
        self.cli_types.SimpleNamespace.assert_called_with(
            params=self.cli_json.load.return_value,
            input_filepaths={
                'input1': 'input1.png',
                'input2': 'input2.png',
            },
            input_images={
                'input1': self.cli_image.Image.read.return_value,
                'input2': self.cli_image.Image.read.return_value,
            },
            output_filepaths=dict(),
            raw_args=self.tool.parser.parse_args.return_value,
        )
        self.assertIs(args, self.cli_types.SimpleNamespace.return_value)
        self.assertIs(args, self.tool.args)

    def test__no_inputs_or_outputs(self):
        self.tool.parser.parse_args.return_value = unittest.mock.Mock(
            verbose=False,
            params='params.json',
        )
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open()) as mock_open:
            args = self.tool.parse_args()
            self.cli_json.load.assert_called_with(mock_open())
        self.cli_image.Image.read.assert_not_called()
        self.cli_types.SimpleNamespace.assert_called_with(
            params=self.cli_json.load.return_value,
            input_filepaths=dict(),
            input_images=dict(),
            output_filepaths=dict(),
            raw_args=self.tool.parser.parse_args.return_value,
        )
        self.assertIs(args, self.cli_types.SimpleNamespace.return_value)
        self.assertIs(args, self.tool.args)

    def test__no_inputs_or_outputs__no_params(self):
        self.tool.parser.parse_args.return_value = unittest.mock.Mock(
            verbose=False,
            params=None,
        )
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open()) as mock_open:
            args = self.tool.parse_args()
            mock_open.assert_not_called()
        self.cli_image.Image.read.assert_not_called()
        self.cli_types.SimpleNamespace.assert_called_with(
            params=None,
            input_filepaths=dict(),
            input_images=dict(),
            output_filepaths=dict(),
            raw_args=self.tool.parser.parse_args.return_value,
        )
        self.assertIs(args, self.cli_types.SimpleNamespace.return_value)
        self.assertIs(args, self.tool.args)


@unittest.mock.patch('giatools.cli.ToolBaseplate.parse_args')
class ToolBaseplate__run(MockedTestCase):

    def setUp(self):
        super().setUp()
        self.joint_axes = unittest.mock.Mock()
        self.tool = giatools.cli.ToolBaseplate()
        self.tool.input_keys = ['input']
        self.tool.output_keys = ['output']
        self.args = unittest.mock.MagicMock()
        self.args.verbose = False
        self.args.input_images = {
            'input': unittest.mock.Mock(),
        }
        self.args.output_filepaths = {
            'output': unittest.mock.Mock(),
        }
        self.processor_iteration = unittest.mock.Mock()
        self.cli_image_processor.ImageProcessor.return_value.process.return_value = iter(
            [
                self.processor_iteration,  # object yielded by the only iteration
            ],
        )

    def _verify_calls(self, verbose: bool):
        self.cli_image_processor.ImageProcessor.assert_called_with(**self.args.input_images)
        self.cli_image_processor.ImageProcessor.return_value.process.assert_called_with(joint_axes=self.joint_axes)
        for key, filepath in self.args.output_filepaths.items():
            output_image = self.cli_image_processor.ImageProcessor.return_value.outputs[
                key
            ].normalize_axes_like.return_value
            output_image.write.assert_called_with(filepath)
            if verbose:
                self.builtins_print.assert_called()
                for line in (
                    f'[output] Output image shape: {output_image.data.shape}',
                    f'[output] Output image dtype: {output_image.data.dtype}',
                    f'[output] Output image axes: {output_image.axes}',
                ):
                    self.assertIn(unittest.mock.call(line), self.builtins_print.call_args_list)
        if not verbose:
            self.builtins_print.assert_not_called()

    def test__with_explicit_args(self, mock_parse_args):
        processor_iterations = list(self.tool.run(self.joint_axes, self.args))
        mock_parse_args.assert_not_called()
        self.assertEqual(processor_iterations, [self.processor_iteration])
        self._verify_calls(verbose=False)

    def test__without_explicit_args(self, mock_parse_args):
        with unittest.mock.patch.object(self.tool, 'parse_args') as mock_parse_args:
            mock_parse_args.return_value = self.args
            processor_iterations = list(self.tool.run(self.joint_axes))
            mock_parse_args.assert_called_once()
        self.assertEqual(processor_iterations, [self.processor_iteration])
        self._verify_calls(verbose=False)

    def test__without_explicit_args__with_args_attr(self, mock_parse_args):
        self.tool.args = self.args
        with unittest.mock.patch.object(self.tool, 'parse_args') as mock_parse_args:
            processor_iterations = list(self.tool.run(self.joint_axes))
            mock_parse_args.assert_not_called()
        self.assertEqual(processor_iterations, [self.processor_iteration])
        self._verify_calls(verbose=False)

    def test__verbose(self, mock_parse_args):
        self.args.verbose = True
        processor_iterations = list(self.tool.run(self.joint_axes, self.args))
        mock_parse_args.assert_not_called()
        self.assertEqual(processor_iterations, [self.processor_iteration])
        self._verify_calls(verbose=True)
