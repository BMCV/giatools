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

        .. literalinclude :: examples/cli.py

        .. runblock:: console

            $ python -m examples.cli --help

        .. runblock:: console

            $ python -m examples.cli --verbose --input data/input4_uint8.png --output /tmp/output.png
    """

    args: _T.Optional[types.SimpleNamespace] = None
    """
    Command-line arguments parsed from the command line (including the loaded input images).
    """

    input_keys: _T.List[str]
    """
    List of input image keys.
    """

    output_keys: _T.List[str]
    """
    List of output image keys.
    """

    processor: _T.Optional[_image_processor.ImageProcessor] = None
    """
    The `image_processor.ImageProcessor` instantiated via the :py:meth:`create_processor` method.
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

    def _read_image(self, args: types.SimpleNamespace, key: str, filepath: str) -> _image.Image:
        image = _image.Image.read(filepath)
        if args.verbose:
            print(f'[{key}] Input image axes: {image.axes}')
            print(f'[{key}] Input image shape: {image.data.shape}')
            print(f'[{key}] Input image dtype: {image.data.dtype}')
        return image

    def parse_args(self) -> types.SimpleNamespace:
        """
        Parse the command-line arguments and return a namespace that contains the JSON-encoded parameters, the input
        images, and the output image file paths. The :py:attr:`args` attribute is also populated.
        """
        args = self.parser.parse_args()
        input_filepaths = {key: getattr(args, key) for key in self.input_keys}
        input_images = {key: self._read_image(args, key, filepath) for key, filepath in input_filepaths.items()}
        output_filepaths = {key: getattr(args, key) for key in self.output_keys}
        if args.params is None:
            params = None
        else:
            with open(args.params, 'r') as fp_params:
                params = json.load(fp_params)
        self.args = types.SimpleNamespace(
            params=params,
            verbose=args.verbose,
            input_filepaths=input_filepaths,
            input_images=input_images,
            output_filepaths=output_filepaths,
            raw_args=args,
        )
        return self.args

    def run(
        self,
        joint_axes: str,
        write_output_images: bool = True,
    ) -> _T.Iterator[_image_processor.ProcessorIteration]:
        """
        Use the :py:meth:`create_processor` method to spin up an `giatools.image_processor.ImageProcessor` with the
        input images parsed from the command line, and write the output images to the file paths specified via command
        line arguments (if `write_outputs` is `True`).

        .. note::

            This method requires **Python 3.11** or later.

        Raises:
            RuntimeError: If Python version is less than 3.11.
        """
        yield from self.create_processor().process(joint_axes=joint_axes)
        if write_output_images:
            self.write_output_images()

    def create_processor(self) -> _image_processor.ImageProcessor:
        """
        Create an `giatools.image_processor.ImageProcessor` with the input images parsed from the command line. The
        `giatools.image_processor.ImageProcessor` is returned and also made available via the :py:attr:`processor`
        attribute.

        The command line arguments are obtained via the :py:meth:`parse_args` method unless the :py:attr:`args`
        attribute is already populated (which has precedence).
        """
        args = self.args or self.parse_args()
        assert self.args is not None  # sanity check
        self.processor = _image_processor.ImageProcessor(**args.input_images)
        return self.processor

    def write_output_images(self):
        """
        Write the output images to the file paths specified via command line arguments.

        The output images are obtained from the `giatools.image_processor.ImageProcessor` referenced by the
        :py:attr:`processor` attribute. The command line arguments must be provided via the :py:attr:`args` attribute.

        Raises:
            RuntimeError: If the :py:attr:`args` or :py:attr:`processor` attributes are not populated.
        """
        if self.args is None:
            raise RuntimeError('Command-line arguments have not been parsed; cannot write outputs.')
        if self.processor is None:
            raise RuntimeError('Image processor has not been created; cannot write outputs.')
        for key, filepath in self.args.output_filepaths.items():
            output_image = self.processor.outputs[key].normalize_axes_like(self.processor.image0.original_axes)
            if self.args.verbose:
                print(f'[{key}] Output image axes: {output_image.axes}')
                print(f'[{key}] Output image shape: {output_image.data.shape}')
                print(f'[{key}] Output image dtype: {output_image.data.dtype}')
            output_image.write(filepath)
