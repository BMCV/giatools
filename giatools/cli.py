import argparse
import json
import types

from . import (
    image as _image,
    image_processor as _image_processor,
    typing as _T,
)


class ToolBaseplate:

    input_keys: _T.List[str]

    output_keys: _T.List[str]

    def __init__(self, *args, **kwargs):
        self.parser = argparse.ArgumentParser(*args, **kwargs)
        self.parser.add_argument('params', type=str)
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
        """
        args = self.parser.parse_args()
        input_filepaths = {key: getattr(args, key) for key in self.input_keys}
        input_images = {key: _image.Image.read(filepath) for key, filepath in input_filepaths.items()}
        output_filepaths = {key: getattr(args, key) for key in self.output_keys}
        with open(args.params, 'r') as fp_params:
            params = json.load(fp_params)
        return types.SimpleNamespace(
            params=params,
            input_filepaths=input_filepaths,
            input_images=input_images,
            output_filepaths=output_filepaths,
            raw_args=args,
        )

    def run(
        self,
        joint_axes: str,
        args: _T.Optional[types.SimpleNamespace] = None,
    ) -> _T.Iterator[_image_processor.ProcessorIteration]:
        """
        """
        if args is None:
            args = self.parse_args()

        processor = _image_processor.ImageProcessor(**args.input_images)
        for processor_iteration in processor.process(joint_axes=joint_axes):
            yield processor_iteration

        for key, filepath in args.output_filepaths.items():
            output_image = processor.outputs[key]
            output_image.write(filepath)
