import pathlib
import subprocess
import sys
import unittest

import giatools.cli


if __name__ == '__main__':
    tool = giatools.cli.ToolBaseplate('ToolBaseplate Test', params_required=False)
    tool.add_input_image('input1')
    tool.add_input_image('input2')
    tool.add_output_image('output')
    args = tool.parse_args()

    for proc in tool.run('YX', args):
        proc['output'] = (proc['input1'] > proc['input2'])


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

    # TODO: Add test running without `--help`
