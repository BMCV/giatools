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
        self.cli_types.SimpleNamespace.side_effect = MockedTestCase.mock_namespace
        self.cli_image = unittest.mock.patch(
            'giatools.cli._image',
        ).start()
        self.cli_image.Image.read.side_effect = lambda filepath: unittest.mock.Mock(loaded_from=filepath)
        self.cli_image_processor = unittest.mock.patch(
            'giatools.cli._image_processor',
        ).start()
        self.builtins_print = unittest.mock.patch(
            'builtins.print',
        ).start()

        self.addCleanup(unittest.mock.patch.stopall)

    @staticmethod
    def mock_namespace(**kwargs):
        return unittest.mock.Mock(**kwargs)


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

    def _verify_verbose_output(self):
        if self.tool.args.verbose:
            self.builtins_print.assert_called()
            for key, input_image in self.tool.args.input_images.items():
                for line in (
                    f'[{key}] Input image axes: {input_image.original_axes}',
                    f'[{key}] Input image shape: {input_image.original_shape}',
                    f'[{key}] Input image dtype: {input_image.data.dtype}',
                ):
                    self.assertIn(unittest.mock.call(line), self.builtins_print.call_args_list)
        else:
            self.builtins_print.assert_not_called()

    def _test_default(self, verbose: bool):
        self.tool.input_keys = ['input1', 'input2']
        self.tool.output_keys = ['output1']
        self.tool.parser.parse_args.return_value = unittest.mock.Mock(
            verbose=verbose,
            params='params.json',
            input1='input1.png',
            input2='input2.png',
            output1='output1.png',
        )
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open()) as mock_open:
            args = self.tool.parse_args()
            self.cli_json.load.assert_called_with(mock_open())
        self.assertIs(args, self.tool.args)
        self.assertIs(args.params, self.cli_json.load.return_value)
        self.assertIs(args.verbose, verbose)
        self.assertEqual(args.input_filepaths, {'input1': 'input1.png', 'input2': 'input2.png'})
        self.assertEqual(args.output_filepaths, {'output1': 'output1.png'})
        self.assertIs(args.raw_args, self.tool.parser.parse_args.return_value)
        self.assertEqual(args.input_images.keys(), {'input1', 'input2'})
        self.assertEqual(args.input_images['input1'].loaded_from, 'input1.png')
        self.assertEqual(args.input_images['input2'].loaded_from, 'input2.png')
        self._verify_verbose_output()

    def test(self):
        self._test_default(verbose=False)

    def test__verbose(self):
        self._test_default(verbose=True)

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
        self.assertIs(args, self.tool.args)
        self.assertIs(args.params, self.cli_json.load.return_value)
        self.assertIs(args.verbose, False)
        self.assertEqual(args.input_filepaths, dict())
        self.assertEqual(args.output_filepaths, {'output1': 'output1.png'})
        self.assertIs(args.raw_args, self.tool.parser.parse_args.return_value)
        self.assertEqual(args.input_images, dict())
        self.cli_image.Image.read.assert_not_called()
        self._verify_verbose_output()

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
        self.assertIs(args, self.tool.args)
        self.assertIs(args.params, self.cli_json.load.return_value)
        self.assertIs(args.verbose, False)
        self.assertEqual(args.input_filepaths, {'input1': 'input1.png', 'input2': 'input2.png'})
        self.assertEqual(args.output_filepaths, dict())
        self.assertIs(args.raw_args, self.tool.parser.parse_args.return_value)
        self.assertEqual(args.input_images.keys(), {'input1', 'input2'})
        self.assertEqual(args.input_images['input1'].loaded_from, 'input1.png')
        self.assertEqual(args.input_images['input2'].loaded_from, 'input2.png')
        self._verify_verbose_output()

    def test__no_inputs_or_outputs(self):
        self.tool.parser.parse_args.return_value = unittest.mock.Mock(
            verbose=False,
            params='params.json',
        )
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open()) as mock_open:
            args = self.tool.parse_args()
            self.cli_json.load.assert_called_with(mock_open())
        self.assertIs(args, self.tool.args)
        self.assertIs(args.params, self.cli_json.load.return_value)
        self.assertIs(args.verbose, False)
        self.assertEqual(args.input_filepaths, dict())
        self.assertEqual(args.output_filepaths, dict())
        self.assertIs(args.raw_args, self.tool.parser.parse_args.return_value)
        self.assertEqual(args.input_images, dict())
        self.cli_image.Image.read.assert_not_called()
        self._verify_verbose_output()

    def test__no_inputs_or_outputs__no_params(self):
        self.tool.parser.parse_args.return_value = unittest.mock.Mock(
            verbose=False,
            params=None,
        )
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open()) as mock_open:
            args = self.tool.parse_args()
            mock_open.assert_not_called()
            self.cli_json.load.assert_not_called()
        self.assertIs(args, self.tool.args)
        self.assertIsNone(args.params)
        self.assertIs(args.verbose, False)
        self.assertEqual(args.input_filepaths, dict())
        self.assertEqual(args.output_filepaths, dict())
        self.assertIs(args.raw_args, self.tool.parser.parse_args.return_value)
        self.assertEqual(args.input_images, dict())
        self.cli_image.Image.read.assert_not_called()
        self._verify_verbose_output()


@unittest.mock.patch('giatools.cli.ToolBaseplate.write_output_images')
@unittest.mock.patch('giatools.cli.ToolBaseplate.create_processor')
class ToolBaseplate__run(MockedTestCase):

    def setUp(self):
        super().setUp()
        self.joint_axes = unittest.mock.Mock()
        self.tool = giatools.cli.ToolBaseplate()
        self.processor_iteration = unittest.mock.Mock()

    def _setup_mocks(self, mock_create_processor):
        mock_create_processor.return_value.process.return_value = iter(
            [
                self.processor_iteration,  # object yielded by the only iteration
            ],
        )

    def _verify_calls(self, mock_create_processor, mock_write_output_images, write_output_images: bool):
        mock_create_processor.assert_called_once()
        mock_create_processor.return_value.process.assert_called_with(
            joint_axes=self.joint_axes,
            output_dtype_hints=dict(),
        )
        if write_output_images:
            mock_write_output_images.assert_called_once()
        else:
            mock_write_output_images.assert_not_called()

    def test(self, mock_create_processor, mock_write_output_images):
        self._setup_mocks(mock_create_processor)
        processor_iterations = list(self.tool.run(self.joint_axes))
        self.assertEqual(processor_iterations, [self.processor_iteration])
        self._verify_calls(mock_create_processor, mock_write_output_images, write_output_images=True)

    def test__write_output_images_off(self, mock_create_processor, mock_write_output_images):
        self._setup_mocks(mock_create_processor)
        processor_iterations = list(self.tool.run(self.joint_axes, write_output_images=False))
        self.assertEqual(processor_iterations, [self.processor_iteration])
        self._verify_calls(mock_create_processor, mock_write_output_images, write_output_images=False)


@unittest.mock.patch('giatools.cli.ToolBaseplate.parse_args')
class ToolBaseplate__create_processor(MockedTestCase):

    def setUp(self):
        super().setUp()
        self.tool = giatools.cli.ToolBaseplate()
        self.tool.input_keys = ['input']
        self.args = unittest.mock.MagicMock()
        self.args.input_images = {
            'input': unittest.mock.Mock(),
        }

    def _verify(self, processor):
        self.cli_image_processor.ImageProcessor.assert_called_with(**self.args.input_images)
        self.assertIs(processor, self.tool.processor)
        self.assertIs(processor, self.cli_image_processor.ImageProcessor.return_value)

    def test__without_args_attr(self, mock_parse_args):

        def _parse_args_side_effect():
            self.tool.args = self.args
            return self.args

        mock_parse_args.side_effect = _parse_args_side_effect
        processor = self.tool.create_processor()
        mock_parse_args.assert_called_once()
        self._verify(processor)

    def test__with_args_attr(self, mock_parse_args):
        self.tool.args = self.args
        processor = self.tool.create_processor()
        mock_parse_args.assert_not_called()
        self._verify(processor)

    def test__repeated(self, mock_parse_args):
        self.cli_image_processor.ImageProcessor.side_effect = lambda *args, **kwargs: unittest.mock.Mock()
        self.tool.args = self.args
        processor1 = self.tool.create_processor()
        processor2 = self.tool.create_processor()
        mock_parse_args.assert_not_called()
        self.assertIsNot(processor1, processor2)
        self.assertIs(processor2, self.tool.processor)


class ToolBaseplate__write_output_images(MockedTestCase):

    def setUp(self):
        super().setUp()
        self.args = unittest.mock.MagicMock()
        self.args.verbose = False
        self.args.output_filepaths = {
            'output': unittest.mock.Mock(),
        }
        self.tool = giatools.cli.ToolBaseplate()
        self.tool.args = self.args
        self.tool.processor = unittest.mock.MagicMock()
        self.tool.processor.outputs = {
            'output': unittest.mock.Mock(),
        }

    def _verify_calls(self, verbose: bool):
        for key, filepath in self.args.output_filepaths.items():
            output_image = self.tool.processor.outputs[
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

    def test(self):
        self.args.verbose = False
        self.tool.write_output_images()
        self._verify_calls(verbose=False)

    def test__verbose(self):
        self.args.verbose = True
        self.tool.write_output_images()
        self._verify_calls(verbose=True)

    def test__args_is_none(self):
        self.tool.args = None
        with self.assertRaises(RuntimeError):
            self.tool.write_output_images()

    def test__processor_is_none(self):
        self.tool.processor = None
        with self.assertRaises(RuntimeError):
            self.tool.write_output_images()
