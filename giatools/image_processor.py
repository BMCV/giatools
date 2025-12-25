import copy as _copy

import numpy as _np

import giatools.typing as _T
from giatools.image import Image as _Image


class ImageProcessor:
    """
    Processes one or more images with the same shape and axes, and yields one or more output images of the same shape.
    """

    inputs: _T.Dict[_T.Union[str, int], _Image]
    """
    Dictionary of the input images, keyed by strings (keyword arguments) or integers (positional arguments).
    """

    outputs: _T.Dict[_T.Any, _Image]
    """
    Dictionary of the output images with arbitrary keys.
    """

    image0: _Image
    """
    An input image that is used to determine the shape, axes, and metadata of the output images.
    """

    def __init__(self, *args: _Image, **kwargs: _Image):
        for key, input in enumerate(args):
            self.inputs[key] = input
        self.inputs |= kwargs

        # Verify that at least one input image is provided
        if len(self.inputs) == 0:
            raise ValueError('At least one input image must be provided.')

        # Verify that all input images have the same shape and axes
        self.image0 = next(iter(self.inputs.values()))
        for input in self.inputs.values():
            if self.image0.axes != input.axes or self.image0.shape != input.shape:
                raise ValueError('All input images must have the same shape and axes.')

    def process(self, joint_axes: str) -> _T.Iterator[_T.Self]:
        """
        Iterate over all slices of the input images along the given axes, yielding :py:class:`ProcessorIteration`
        objects that provide access to the corresponding slices of the output images.

        .. note::

            This method requires **Python 3.11** or later.

        Raises:
            RuntimeError: If Python version is less than 3.11.
        """
        input_keys, input_images = zip(*self.inputs.items())
        for source_slices, sections in zip(*[input_image.iterate_jointly(joint_axes) for input_image in input_images]):
            iter = ProcessorIteration(self, dict(zip(input_keys, sections)), source_slices[0])
            yield iter

    def create_output_image(self, key: _T.Any, dtype: _np.dtype) -> _Image:
        """
        Create and return an output image with the given key and data type.

        The output image will have the same shape, axes, and metadata as the input images. The metadata is copied.

        Raises:
            ValueError: If an output image with the given key already exists.
        """
        if key not in self.outputs:
            self.outputs[key] = (
                _image := _Image(
                    data=_np.empty(self.image0.shape, dtype=dtype),
                    axes=self.image0.axes,
                    original_axes=self.image0.original_axes,
                    metadata=_copy.deepcopy(self.image0.metadata.copy()),
                )
            )
            return _image
        else:
            raise ValueError(f'Output image with key "{key}" already exists.')


class ProcessorIteration:
    """
    Represents a single iteration of an :py:class:`ImageProcessor`, providing access to corresponding input and output
    image sections.
    """

    _input_sections: _T.Dict[_T.Union[str, int], _Image]

    _output_slice: _T.NDSlice

    _processor: ImageProcessor

    def __init__(
        self,
        processor: ImageProcessor,
        input_sections: _T.Dict[_T.Union[str, int], _Image],
        output_slice: _T.NDSlice,
    ):
        self._processor = processor
        self._input_sections = input_sections
        self._output_slice = output_slice

    def __getitem__(self, key: _T.Union[str, int]) -> _Image:
        """
        Get the input image section corresponding to the given key.
        """
        return self._input_sections[key]

    def __setitem__(self, key: _T.Any, data: _T.NDArray):
        """
        Set the output image section corresponding to the given key.
        """
        if key not in self._processor.outputs:
            self._processor.create_output_image(key, data.dtype)
        self._processor.outputs[key][self._output_slice] = data
