import argparse
import json
import types

from . import (
    image as _image,
    image_processor as _image_processor,
    typing as _T,
)


class ToolBaseplate:
    """
    Baseplate for command-line tools.

    Example:

        .. runblock:: pycon

            >>> import giatools
            >>> if __name__ == '__main__':
            ...     tool = giatools.ToolBaseplate(params_required=False)
            ...     tool.add_input_image('input')
            ...     tool.add_output_image('output')
            ...     for section in tool.run('ZYX'):
            ...         arr = section['input'].data
            ...         section['output'] = (arr > arr.mean())
    """

    input_keys: _T.List[str]
    """
    List of input image keys.
    """

    output_keys: _T.List[str]
    """
    List of output image keys.
    """

    args: _T.Optional[types.SimpleNamespace] = None
    """
    Command-line arguments parsed from the command line (including the loaded input images).
    """

    def __init__(self, *args, params_required=True, **kwargs):
        self.parser = argparse.ArgumentParser(*args, **kwargs)
        self.parser.add_argument('--params', type=str, required=params_required)
        self.parser.add_argument('--verbose', action='store_true', default=False)
        self.input_keys = list()
        self.output_keys = list()

    def _require_unused_key(self, key: str):
        if key in self.input_keys or key in self.output_keys:
            raise ValueError(f'Key "{key}" already exists.')

    def add_input_image(self, key: str, required: bool = True):
        """
        Add a named input image argument to the parser.

        Raises:
            ValueError: If the key is already used.
        """
        self._require_unused_key(key)
        self.input_keys.append(key)
        self.parser.add_argument(f'--{key}', type=str, required=required)

    def add_output_image(self, key: str, required: bool = True):
        """
        Add an argument for a path for an output image to the parser.

        Raises:
            ValueError: If the key is already used.
        """
        self._require_unused_key(key)
        self.output_keys.append(key)
        self.parser.add_argument(f'--{key}', type=str, required=required)

    def parse_args(self) -> types.SimpleNamespace:
        """
        Parse the command-line arguments and return a namespace that contains the JSON-encoded parameters, the input
        images, and the output image file paths. The py:data:`args` attribute is also updated.
        """
        args = self.parser.parse_args()
        input_filepaths = {key: getattr(args, key) for key in self.input_keys}
        input_images = {key: _image.Image.read(filepath) for key, filepath in input_filepaths.items()}
        output_filepaths = {key: getattr(args, key) for key in self.output_keys}
        if args.params is None:
            params = None
        else:
            with open(args.params, 'r') as fp_params:
                params = json.load(fp_params)
        self.args = types.SimpleNamespace(
            params=params,
            input_filepaths=input_filepaths,
            input_images=input_images,
            output_filepaths=output_filepaths,
            raw_args=args,
        )
        return self.args

    def run(
        self,
        joint_axes: str,
        args: _T.Optional[types.SimpleNamespace] = None,
        write_outputs: bool = True,
    ) -> _T.Iterator[_image_processor.ProcessorIteration]:
        """
        Spin up a `giatools.image_processor.ImageProcessor` with the input images parsed from the command line, and
        write the output images to the file paths specified via command line arguments (if `write_outputs` is `True`).

        The command line arguments are obtained via the :py:meth:`parse_args` method unless an explicit `args`
        namespace is provided.

        .. note::

            This method requires **Python 3.11** or later.

        Raises:
            RuntimeError: If Python version is less than 3.11.
        """
        if args is None:
            args = self.args or self.parse_args()

        processor = _image_processor.ImageProcessor(**args.input_images)
        for processor_iteration in processor.process(joint_axes=joint_axes):
            yield processor_iteration

        if write_outputs:
            for key, filepath in args.output_filepaths.items():
                output_image = processor.outputs[key].normalize_axes_like(processor.image0.original_axes)
                output_image.write(filepath)
