import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import giatools.cli

from ..tools import minimum_python_version


def _threshold(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    return (image1 > image2).astype(np.uint8) * 255


if __name__ == '__main__':
    tool = giatools.cli.ToolBaseplate('ToolBaseplate Test', params_required=False)
    tool.add_input_image('input1')
    tool.add_input_image('input2')
    tool.add_output_image('output')
    tool.parse_args()

    for proc in tool.run('YX'):
        proc['output'] = _threshold(proc['input1'].data, proc['input2'].data)


class ToolBaseplate(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.test_dir = pathlib.Path(__file__).parent.parent.parent

    def _run_cli(self, *extra_args: str, check_success: bool = True, **kwargs) -> subprocess.CompletedProcess:
        kwargs = dict(kwargs)
        kwargs.setdefault('capture_output', True)
        kwargs.setdefault('text', True)
        kwargs.setdefault('check', False)
        result = subprocess.run(
            [
                sys.executable, '-m', __name__, *extra_args,
            ],
            cwd=str(self.test_dir),
            **kwargs
        )
        if result.stderr:
            print(result.stderr)
        if check_success:
            self.assertEqual(result.stderr, '')
            self.assertEqual(result.returncode, 0)
        return result

    def test__help(self):
        result = self._run_cli('--help')
        for token in (
            'ToolBaseplate Test',
            '--input1 INPUT1',
            '--input2 INPUT2',
            '--output OUTPUT',
            '--params PARAMS',
            '--verbose',
        ):
            self.assertIn(token, result.stdout)

    @minimum_python_version(3, 11)
    def test(self):
        with tempfile.TemporaryDirectory() as temp_path:
            output_filepath = str(pathlib.Path(temp_path) / 'output.png')
            self._run_cli(
                '--input1', 'tests/data/input4_uint8.png',
                '--input2', 'tests/data/input4_uint8.jpg',
                '--output', output_filepath,
            )
            output_image = giatools.image.Image.read(output_filepath, normalize_axes=None)
            expected_image_data = _threshold(
                giatools.image.Image.read('tests/data/input4_uint8.png', normalize_axes=None).data,
                giatools.image.Image.read('tests/data/input4_uint8.jpg', normalize_axes=None).data,
            )
            np.testing.assert_array_equal(output_image.data, expected_image_data)
            self.assertEqual(output_image.axes, 'YXC')
